use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!(
                "burning-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn run(&self, args: &[&str]) -> Output {
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

#[test]
fn build_writes_beside_entry_without_executing_any_module() {
    let scratch = Scratch::new();
    let folder = scratch.0.join("nested folder");
    std::fs::create_dir(&folder).unwrap();
    std::fs::write(
        folder.join("dependency.napalm"),
        "throw 'import was executed';",
    )
    .unwrap();
    let source =
        "import './dependency'; System.print('entry was executed'); throw 'entry was executed';";
    std::fs::write(folder.join("fib.napalm"), source).unwrap();
    let result = scratch.run(&["--build", "nested folder/fib.napalm"]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stdout.is_empty());
    let bytes = std::fs::read(folder.join("fib.burning")).unwrap();
    assert_eq!(&bytes[..8], b"NAPBURN\0");
    assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 3);
    let mut cursor = 16;
    for tag in [b"CODE", b"META", b"BOOT"] {
        assert_eq!(&bytes[cursor..cursor + 4], tag);
        let len = u64::from_le_bytes(bytes[cursor + 4..cursor + 12].try_into().unwrap()) as usize;
        let payload = &bytes[cursor + 12..cursor + 12 + len];
        if tag == b"META" {
            let metadata: serde_json::Value = serde_json::from_slice(payload).unwrap();
            let modules = metadata["metadata"]["modules"].as_array().unwrap();
            let strings = metadata["metadata"]["strings"].as_array().unwrap();
            for expected in ["fib.napalm", "dependency.napalm"] {
                assert!(modules.iter().any(|module| {
                    let name = module["name"].as_u64().unwrap() as usize;
                    strings[name].as_str().unwrap().ends_with(expected)
                }));
            }
            assert!(metadata["functions"][0]["chunk"].get("code").is_none());
        }
        cursor += 12 + len;
    }
    assert_eq!(cursor, bytes.len());
    assert_eq!(
        std::fs::read_to_string(folder.join("fib.napalm")).unwrap(),
        source
    );
    assert!(!folder.join("dependency.burning").exists());
    std::fs::write(folder.join("fib.burning"), b"old artifact").unwrap();
    let again = scratch.run(&["--build", "nested folder/fib.napalm"]);
    assert!(again.status.success());
    assert!(
        std::fs::read(folder.join("fib.burning"))
            .unwrap()
            .starts_with(b"NAPBURN\0")
    );
    let executed = scratch.run(&["nested folder/fib.napalm"]);
    assert_eq!(executed.status.code(), Some(4));
    let artifact_execution = scratch.run(&["nested folder/fib.burning"]);
    assert_eq!(artifact_execution.status.code(), Some(4));
    assert_eq!(artifact_execution.stdout, executed.stdout);
    assert_eq!(artifact_execution.stderr, executed.stderr);
}

#[test]
fn artifact_executes_after_relocation_without_sources_and_preserves_arguments() {
    let scratch = Scratch::new();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/burning");
    std::fs::copy(
        fixtures.join("dependency.napalm"),
        scratch.0.join("dependency.napalm"),
    )
    .unwrap();
    let source = format!(
        "{}\nSystem.print(JSON.stringify(Process.args)); System.print('artifact passed');",
        std::fs::read_to_string(fixtures.join("entry.napalm")).unwrap()
    );
    std::fs::write(scratch.0.join("entry.napalm"), source).unwrap();
    let script_args = ["", "two words", "petición 😀", "--help", "--build", "--"];
    let mut source_args = vec!["entry.napalm"];
    source_args.extend(script_args);
    let expected = scratch.run(&source_args);
    assert!(
        expected.status.success(),
        "{}",
        String::from_utf8_lossy(&expected.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&expected.stdout),
        "[\"\",\"two words\",\"petición 😀\",\"--help\",\"--build\",\"--\"]\nartifact passed\n"
    );
    let build = scratch.run(&["--build", "entry.napalm"]);
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );

    let relocated = scratch.0.join("moved artifact");
    std::fs::create_dir(&relocated).unwrap();
    std::fs::rename(
        scratch.0.join("entry.burning"),
        relocated.join("renamed.burning"),
    )
    .unwrap();
    std::fs::remove_file(scratch.0.join("entry.napalm")).unwrap();
    std::fs::remove_file(scratch.0.join("dependency.napalm")).unwrap();
    // Use a different cwd, with neither source available, for both VM modes.
    for jit in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_napalm"));
        command
            .current_dir(&relocated)
            .arg("renamed.burning")
            .args(script_args);
        if jit {
            command.env_remove("NAPALM_NO_JIT");
        } else {
            command.env("NAPALM_NO_JIT", "1");
        }
        let actual = command.output().unwrap();
        assert!(
            actual.status.success(),
            "jit={jit}: {}",
            String::from_utf8_lossy(&actual.stderr)
        );
        assert_eq!(actual.stdout, expected.stdout, "jit={jit}");
    }
}

#[test]
fn artifact_loading_reports_errors_and_rejects_rebuilding() {
    let scratch = Scratch::new();
    let missing = scratch.run(&["missing.burning"]);
    assert_eq!(missing.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&missing.stderr).contains("failed to read"));
    std::fs::write(scratch.0.join("entry.napalm"), "System.print('executed');").unwrap();
    assert!(scratch.run(&["--build", "entry.napalm"]).status.success());
    let valid = std::fs::read(scratch.0.join("entry.burning")).unwrap();
    let rebuild = scratch.run(&["--build", "entry.burning"]);
    assert_eq!(rebuild.status.code(), Some(2));
    assert!(rebuild.stdout.is_empty());
    assert_eq!(
        std::fs::read(scratch.0.join("entry.burning")).unwrap(),
        valid
    );

    let mut version = valid.clone();
    version[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut code = valid.clone();
    // CODE's function count starts at 28 and first function ID at 32.
    code[32..36].copy_from_slice(&u32::MAX.to_le_bytes());
    for bytes in [
        b"not an artifact".to_vec(),
        valid[..valid.len() - 1].to_vec(),
        version,
        code,
    ] {
        std::fs::write(scratch.0.join("invalid.burning"), bytes).unwrap();
        let result = scratch.run(&["invalid.burning"]);
        assert_eq!(result.status.code(), Some(1));
        assert!(result.stdout.is_empty());
        assert!(String::from_utf8_lossy(&result.stderr).contains("invalid burning artifact"));
    }
}

#[test]
fn build_reports_usage_compilation_and_output_errors() {
    let scratch = Scratch::new();
    assert_eq!(scratch.run(&["--build"]).status.code(), Some(2));
    assert_eq!(
        scratch.run(&["--build", "missing.napalm"]).status.code(),
        Some(3)
    );
    std::fs::write(scratch.0.join("bad.napalm"), "const = ;").unwrap();
    assert_eq!(
        scratch.run(&["--build", "bad.napalm"]).status.code(),
        Some(1)
    );
    assert!(!scratch.0.join("bad.burning").exists());
    std::fs::write(scratch.0.join("bad.burning"), b"previous artifact").unwrap();
    assert_eq!(
        scratch.run(&["--build", "bad.napalm"]).status.code(),
        Some(1)
    );
    assert_eq!(
        std::fs::read(scratch.0.join("bad.burning")).unwrap(),
        b"previous artifact"
    );
    std::fs::write(scratch.0.join("good.napalm"), "System.print(42);").unwrap();
    std::fs::create_dir(scratch.0.join("good.burning")).unwrap();
    assert_eq!(
        scratch.run(&["--build", "good.napalm"]).status.code(),
        Some(3)
    );
}
