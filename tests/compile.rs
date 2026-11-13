//! `napalm --compile`: standalone executables with the program embedded.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join(format!(
            "compile-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn napalm(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_napalm"))
            .current_dir(&self.0)
            .env("NAPALM_NO_JIT", "1")
            .args(args)
            .output()
            .unwrap()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn exe(stem: &str) -> String {
    format!("{stem}{}", std::env::consts::EXE_SUFFIX)
}

fn assert_success(output: &Output) {
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
}

#[test]
fn compiled_executable_runs_relocated_without_sources_and_receives_all_arguments() {
    let scratch = Scratch::new();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/burning");
    std::fs::copy(fixtures.join("dependency.napalm"), scratch.0.join("dependency.napalm")).unwrap();
    let source = format!(
        "{}\nSystem.print(JSON.stringify(Process.args)); System.print('standalone passed');",
        std::fs::read_to_string(fixtures.join("entry.napalm")).unwrap()
    );
    std::fs::write(scratch.0.join("entry.napalm"), source).unwrap();

    let compile = scratch.napalm(&["--compile", "entry.napalm"]);
    assert_success(&compile);
    assert!(compile.stdout.is_empty(), "--compile must not run the program");
    let artifact = std::fs::read(scratch.0.join("entry.burning")).unwrap();
    assert!(artifact.starts_with(b"NAPBURN\0"));
    let binary = std::fs::read(scratch.0.join(exe("entry"))).unwrap();
    assert!(binary.ends_with(b"NAPALM-STANDALN\0"));
    assert!(binary.windows(artifact.len()).any(|window| window == artifact));

    // Neither the sources, the artifact nor napalm itself are needed.
    let relocated = scratch.0.join("moved app");
    std::fs::create_dir(&relocated).unwrap();
    std::fs::rename(scratch.0.join(exe("entry")), relocated.join(exe("renamed"))).unwrap();
    for file in ["entry.napalm", "dependency.napalm", "entry.burning"] {
        std::fs::remove_file(scratch.0.join(file)).unwrap();
    }
    let args = ["", "two words", "petición 😀", "--help", "--build", "--compile", "--"];
    for jit in [false, true] {
        let mut command = Command::new(relocated.join(exe("renamed")));
        command.current_dir(&scratch.0).args(args);
        if jit {
            command.env_remove("NAPALM_NO_JIT");
        } else {
            command.env("NAPALM_NO_JIT", "1");
        }
        let output = command.output().unwrap();
        assert_success(&output);
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            "[\"\",\"two words\",\"petición 😀\",\"--help\",\"--build\",\"--compile\",\"--\"]\nstandalone passed\n",
            "jit={jit}"
        );
    }
    let no_args = Command::new(relocated.join(exe("renamed"))).env("NAPALM_NO_JIT", "1").output().unwrap();
    assert_success(&no_args);
    assert!(String::from_utf8_lossy(&no_args.stdout).starts_with("[]\n"));
}

#[test]
fn compiled_executable_reports_runtime_errors_like_napalm() {
    let scratch = Scratch::new();
    std::fs::write(scratch.0.join("fails.napalm"), "System.print('before'); throw 'boom';").unwrap();
    assert_success(&scratch.napalm(&["--compile", "fails.napalm"]));
    let output = Command::new(scratch.0.join(exe("fails"))).env("NAPALM_NO_JIT", "1").output().unwrap();
    let reference = scratch.napalm(&["fails.napalm"]);
    assert_eq!(output.status.code(), Some(4));
    assert_eq!(output.stdout, reference.stdout);
    assert_eq!(output.stderr, reference.stderr);
}

#[test]
fn compile_accepts_an_existing_artifact_and_reports_errors() {
    let scratch = Scratch::new();
    std::fs::write(scratch.0.join("app.napalm"), "System.print(6 * 7);").unwrap();
    assert_success(&scratch.napalm(&["--build", "app.napalm"]));
    std::fs::remove_file(scratch.0.join("app.napalm")).unwrap();
    assert_success(&scratch.napalm(&["--compile", "app.burning"]));
    let output = Command::new(scratch.0.join(exe("app"))).env("NAPALM_NO_JIT", "1").output().unwrap();
    assert_success(&output);
    assert_eq!(String::from_utf8_lossy(&output.stdout), "42\n");

    assert_eq!(scratch.napalm(&["--compile"]).status.code(), Some(2));
    assert_eq!(scratch.napalm(&["--compile", "missing.napalm"]).status.code(), Some(3));
    std::fs::write(scratch.0.join("bad.napalm"), "const = ;").unwrap();
    assert_eq!(scratch.napalm(&["--compile", "bad.napalm"]).status.code(), Some(1));
    assert!(!scratch.0.join(exe("bad")).exists());
    std::fs::write(scratch.0.join("corrupt.burning"), b"not an artifact").unwrap();
    assert_eq!(scratch.napalm(&["--compile", "corrupt.burning"]).status.code(), Some(1));
    assert!(!scratch.0.join(exe("corrupt")).exists());
}

#[test]
fn corrupt_payload_is_rejected() {
    let scratch = Scratch::new();
    std::fs::write(scratch.0.join("app.napalm"), "System.print('ok');").unwrap();
    assert_success(&scratch.napalm(&["--compile", "app.napalm"]));
    let path = scratch.0.join(exe("app"));
    let mut bytes = std::fs::read(&path).unwrap();
    // Grow the runtime_len field (trailer offset 0): lengths no longer add up.
    let trailer = bytes.len() - 64;
    bytes[trailer] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();
    let output = Command::new(&path).output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid standalone executable"));
}

/// The LLVM library travels inside the executable and is extracted to the cache
/// when the LLVM tier starts.
#[cfg(all(windows, feature = "llvm-jit"))]
#[test]
fn embedded_llvm_library_is_extracted_and_loaded() {
    let scratch = Scratch::new();
    std::fs::write(
        scratch.0.join("hot.napalm"),
        "let total = 0; for (let i = 0; i < 2000000; i += 1) { total += i % 7; } System.print(total);",
    )
    .unwrap();
    let compile = scratch.napalm(&["--compile", "hot.napalm"]);
    assert_success(&compile);
    assert!(!String::from_utf8_lossy(&compile.stderr).contains("LLVM was not embedded"));
    let cache = scratch.0.join("cache");
    for _ in 0..2 {
        let output = Command::new(scratch.0.join(exe("hot")))
            .env("NAPALM_CACHE_DIR", &cache)
            .env("NAPALM_LLVM_THRESHOLD", "1")
            .env("NAPALM_LLVM_SYNC", "1")
            .env("NAPALM_JIT_EVENTS", "1")
            .env_remove("NAPALM_NO_JIT")
            .env_remove("NAPALM_NO_LLVM")
            .env_remove("NAPALM_LLVM_DLL")
            .output()
            .unwrap();
        assert_success(&output);
        assert_eq!(String::from_utf8_lossy(&output.stdout), "5999995\n");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let loaded = stderr.lines().find(|line| line.starts_with("[napalm-llvm] loaded")).unwrap_or_else(|| panic!("{stderr}"));
        assert!(loaded.contains("cache"), "{loaded}");
    }
    let extracted: Vec<_> = std::fs::read_dir(cache.join("llvm"))
        .unwrap()
        .map(|entry| entry.unwrap().path().join("LLVM-C.dll"))
        .collect();
    assert_eq!(extracted.len(), 1);
    assert!(extracted[0].is_file());
}
