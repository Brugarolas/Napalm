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
