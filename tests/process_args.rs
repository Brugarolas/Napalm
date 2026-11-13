use std::process::{Command, Stdio};

#[test]
fn process_args_example_accepts_no_arguments_and_preserves_supplied_arguments() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let cases: &[(&[&str], &str)] = &[
        (&[], "[]"),
        (&["alpha", "beta"], r#"["alpha","beta"]"#),
        (&["only-one"], r#"["only-one"]"#),
        (&["--build"], r#"["--build"]"#),
        (
            &["", "two words", "a\"quote", "C:\\path\\", "petición 😀", "--help", "--"],
            r#"["","two words","a\"quote","C:\\path\\","petición 😀","--help","--"]"#,
        ),
    ];
    for (args, expected) in cases {
        let output = Command::new(env!("CARGO_BIN_EXE_napalm"))
            .current_dir(root)
            .arg(root.join("scripts/process_args.napalm"))
            .args(*args)
            .stdin(Stdio::null())
            .output()
            .expect("launch Napalm");
        assert!(output.status.success(), "args={args:?}, stderr={}", String::from_utf8_lossy(&output.stderr));
        assert_eq!(String::from_utf8(output.stdout).unwrap(),
            format!("{expected}\nProcess.args integration tests passed\n"), "args={args:?}");
    }
}
