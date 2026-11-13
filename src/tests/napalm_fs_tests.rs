// Exercise the public parser, runtime contracts, interpreter, and tracing JIT.
struct FsTestDirectory(std::path::PathBuf);

impl FsTestDirectory {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("napalm-fs-{}-{id}-{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn source(&self, source: &str) -> String {
        source.replace("__ROOT__", &format!("{:?}", self.0.to_str().unwrap()))
    }
}

impl Drop for FsTestDirectory {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

#[test]
fn napalm_fs_fileinfo_and_file_roundtrip() {
    let directory = FsTestDirectory::new();
    run_napalm_checked(&directory.source(r#"
        let assertion = 0;
        function expect(ok) {
            assertion++;
            if (!ok) { System.print(`filesystem assertion ${assertion} failed`); throw "filesystem assertion failed"; }
        }
        const root = __ROOT__;
        const path = Path.join(root, "roundtrip.bin");
        const file: File = Fs.open(path, "w+");
        expect(typeof file === "File" && file instanceof File && !(file instanceof FileInfo));
        expect(file.path === path && file.name === "roundtrip.bin");
        expect(file.write(Bytes.fromArray([0, 1, 255])) === 3);
        const buffer = new ByteBuffer();
        buffer.append("hello");
        expect(file.write(buffer) === 5 && buffer.length === 5);
        expect(file.flush() === null);
        const info: FileInfo = Fs.stat(path);
        expect(typeof info === "FileInfo" && info instanceof FileInfo && !(info instanceof File));
        expect(info.path === path && info.name === "roundtrip.bin" && info.size === 8);
        expect(info.isFile && !info.isDirectory && !info.isSymlink);
        expect(info.modifiedAt === null || typeof info.modifiedAt === "Date");
        expect(info.createdAt === null || typeof info.createdAt === "Date");
        expect(file.modifiedAt === null || typeof file.modifiedAt === "Date");
        expect(file.createdAt === null || typeof file.createdAt === "Date");
        const timestamp = info.modifiedAt;
        if (timestamp !== null) {
            const originalTime = timestamp.getTime();
            timestamp.setTime(0);
            expect(info.modifiedAt.getTime() === originalTime);
        }
        expect(file.seek(0) === 0);
        expect(file.read(0).isEmpty);
        expect(file.read(3).toArray()[2] === 255);
        expect(file.read(100).toUtf8() === "hello");
        expect(file.read(1).isEmpty);
        expect(file.seek(10) === 10 && file.read(1).isEmpty);
        file.write(Bytes.fromUtf8("!"));
        expect(info.size === 8 && Fs.stat(path).size === 11);
        let sum = 0;
        for (let i = 0; i < 100; i++) {
            file.seek(1);
            sum += file.read(1)[0];
            if (i % 10 === 0) System.gc();
        }
        expect(sum === 100 && info.name === "roundtrip.bin");
        expect(file.close() === null && file.close() === null);
        expect(file.path === path && file.name === "roundtrip.bin");
        expect(Fs.stat(root).isDirectory);
        Fs.removeFile(path);
        const result = "ok";
    "#), "ok", true);
}

#[test]
fn napalm_fs_open_modes_and_errors() {
    let directory = FsTestDirectory::new();
    run_napalm(&directory.source(r#"
        let assertion = 0;
        function expect(ok) {
            assertion++;
            if (!ok) { System.print(`filesystem assertion ${assertion} failed`); throw "filesystem assertion failed"; }
        }
        function rejects(action) {
            let failed = false;
            try { action(); } catch (error) { failed = true; }
            expect(failed);
        }
        const root = __ROOT__;
        const path = Path.join(root, "modes.txt");
        const missing = Path.join(root, "missing.txt");
        rejects(() => Fs.open(missing, "r"));
        rejects(() => Fs.open(missing, "r+"));
        rejects(() => Fs.open(missing, "bad"));
        expect(!Fs.exists(missing));
        for (const mode of ["r", "r+", "w", "w+", "a", "a+"]) {
            rejects(() => Fs.open(root, mode));
        }
        Fs.writeText(path, "abc");
        const readOnly = Fs.open(path, "r");
        rejects(() => readOnly.write(Bytes.fromUtf8("x")));
        expect(readOnly.read(1).toUtf8() === "a");
        readOnly.close();
        const update = Fs.open(path, "r+");
        update.seek(1);
        update.write(Bytes.fromUtf8("Z"));
        update.close();
        expect(Fs.readText(path) === "aZc");
        const append = Fs.open(path, "a");
        append.seek(0);
        append.write(Bytes.fromUtf8("!"));
        rejects(() => append.read(1));
        append.close();
        expect(Fs.readText(path) === "aZc!");
        const both = Fs.open(path, "a+");
        expect(both.read(1).toUtf8() === "a");
        both.seek(0);
        both.write(Bytes.fromUtf8("?"));
        both.seek(0);
        expect(both.read(100).toUtf8() === "aZc!?");
        both.close();
        const truncate = Fs.open(path, "w");
        expect(Fs.stat(path).size === 0);
        rejects(() => truncate.read(1));
        truncate.close();
        const file = Fs.open(path, "w+");
        rejects(() => file.write("text"));
        rejects(() => file.read(-1));
        rejects(() => file.read(0.5));
        rejects(() => file.read(268435457));
        rejects(() => file.read("1"));
        rejects(() => file.seek(-1));
        rejects(() => file.seek(1.5));
        rejects(() => file.seek(9007199254740992));
        rejects(() => file.seek("0"));
        rejects(() => { file.path = "changed"; });
        rejects(() => { file.name = "changed"; });
        rejects(() => { file.modifiedAt = null; });
        rejects(() => { file.createdAt = null; });
        const info = Fs.stat(path);
        rejects(() => { info.size = 42; });
        file.close();
        rejects(() => file.read(0));
        rejects(() => file.write(Bytes.empty()));
        rejects(() => file.seek(0));
        rejects(() => file.flush());
        rejects(() => { const date = file.modifiedAt; });
        for (const mode of ["w", "w+", "a", "a+"]) {
            const created = Fs.open(missing, mode);
            created.close();
            expect(Fs.stat(missing).isFile);
            Fs.removeFile(missing);
        }
        Fs.removeFile(path);
        const result = "ok";
    "#), "ok");
}

#[test]
fn napalm_fs_rejects_symlinks_without_truncating_target() {
    let directory = FsTestDirectory::new();
    let target = directory.0.join("target.txt");
    let link = directory.0.join("link.txt");
    std::fs::write(&target, "preserved").unwrap();
    #[cfg(unix)]
    let linked = std::os::unix::fs::symlink(&target, &link);
    #[cfg(windows)]
    let linked = std::os::windows::fs::symlink_file(&target, &link);
    #[cfg(not(any(unix, windows)))]
    return;
    #[cfg(any(unix, windows))] {
        if let Err(error) = linked {
            // Windows may require Developer Mode or a symlink privilege.
            if error.kind() == std::io::ErrorKind::PermissionDenied || error.raw_os_error() == Some(1314) {
                eprintln!("symlink test unavailable: {error}");
                return;
            }
            panic!("cannot create test symlink: {error}");
        }
        run_napalm(&directory.source(r#"
            const root = __ROOT__;
            const path = Path.join(root, "link.txt");
            const info: FileInfo = Fs.stat(path);
            let rejected = 0;
            for (const mode of ["r", "r+", "w", "w+", "a", "a+"]) {
                try { Fs.open(path, mode); } catch (error) { rejected++; }
            }
            const result = [info.isSymlink, info.isFile, rejected,
                Fs.readText(Path.join(root, "target.txt"))];
        "#), "[true, false, 6, preserved]");
        std::fs::remove_file(&target).unwrap();
        run_napalm(&directory.source(r#"
            const root = __ROOT__;
            let rejected = 0;
            for (const mode of ["w", "w+", "a", "a+"]) {
                try { Fs.open(Path.join(root, "link.txt"), mode); } catch (error) { rejected++; }
            }
            const result = [rejected, Fs.exists(Path.join(root, "target.txt"))];
        "#), "[4, false]");
    }
}

#[test]
fn napalm_fs_runtime_type_contracts() {
    let directory = FsTestDirectory::new();
    for (body, expected) in [
        (r#"const input: any = Fs.stat(root); const result: File = input;"#, "File"),
        (r#"const input: any = Fs.open(Path.join(root, "contract.txt"), "w");
            input.close(); const result: FileInfo = input;"#, "FileInfo"),
    ] {
        let source = directory.source(&format!("const root = __ROOT__; {body}"));
        let (program, entry, manifest) = compile_typescript_module(&source);
        let mut vm = source_vm(&program, &manifest);
        let error = vm.run_module(entry).expect_err("incompatible filesystem types must be rejected");
        assert!(error.message.contains(expected), "{error:?}");
    }
}

#[cfg(unix)]
#[test]
fn napalm_fs_rejects_special_files() {
    let directory = FsTestDirectory::new();
    let socket = directory.0.join("socket");
    let _listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    run_napalm(&directory.source(r#"
        const root = __ROOT__;
        let rejected = 0;
        for (const path of ["/dev/null", Path.join(root, "socket")]) {
            try { Fs.open(path, "r"); } catch (error) { rejected++; }
        }
        const result = rejected;
    "#), "2");
}

#[cfg(windows)]
#[test]
fn napalm_fs_rejects_windows_devices() {
    run_napalm(r#"
        let rejected = 0;
        for (const path of ["NUL", "\\\\.\\NUL"]) {
            try {
                const file = Fs.open(path, "r");
                file.close();
            } catch (error) { rejected++; }
        }
        const result = rejected;
    "# , "2");
}
