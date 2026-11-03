//! Links the LLVM C API used by the LLVM trace tier (`llvm-jit` feature).
//!
//! On Windows the official LLVM release ships its static libraries built with
//! the static CRT (/MT) and an rpmalloc-replaced `malloc`, which cannot be mixed
//! into a /MD Rust binary. `LLVM-C.dll` exports the complete C API with its own
//! CRT inside, so we link its import library as a *delay-loaded* DLL: the
//! runtime loads it explicitly before the first LLVM call (see
//! `napalm_tracing_jit/llvm.rs`), and a missing DLL only disables the LLVM tier.
//!
//! LLVM is located through `LLVM_SYS_231_PREFIX` (see `.cargo/config.toml`).

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLVM_SYS_231_PREFIX");
    if env::var_os("CARGO_FEATURE_LLVM_JIT").is_none() {
        return;
    }
    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => {}
        // Other hosts link libLLVM dynamically through llvm-sys itself. Look for
        // a bundled copy next to the executable first, so distributions can ship
        // libLLVM alongside napalm instead of requiring a system install.
        Ok("macos") => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
            return;
        }
        _ => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            return;
        }
    }

    let prefix = env::var_os("LLVM_SYS_231_PREFIX").map(PathBuf::from).unwrap_or_else(|| {
        panic!(
            "the `llvm-jit` feature needs LLVM 23.1: set LLVM_SYS_231_PREFIX to an extracted \
             clang+llvm-23.1.x-x86_64-pc-windows-msvc release, or build with --no-default-features"
        )
    });
    let import_lib = prefix.join("lib").join("LLVM-C.lib");
    let dll = prefix.join("bin").join("LLVM-C.dll");
    assert!(import_lib.is_file(), "missing {}", import_lib.display());
    assert!(dll.is_file(), "missing {}", dll.display());

    println!("cargo:rustc-link-search=native={}", prefix.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=LLVM-C");
    println!("cargo:rustc-link-lib=delayimp");
    println!("cargo:rustc-link-arg=/DELAYLOAD:LLVM-C.dll");
    // Fallback location used when LLVM-C.dll is neither next to the executable
    // nor on PATH (overridable at run time with NAPALM_LLVM_DLL).
    println!("cargo:rustc-env=NAPALM_LLVM_DLL={}", dll.display());

    // Bundle: copy the self-contained DLL (it imports only Windows system DLLs)
    // next to the produced napalm.exe, i.e. into target/<profile>/. A Napalm
    // distribution is then just `napalm.exe` + `LLVM-C.dll` (+ LLVM's license).
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let profile_dir = out_dir
        .ancestors()
        .find(|dir| dir.file_name().is_some_and(|name| name == "build"))
        .and_then(|build| build.parent());
    if let Some(profile_dir) = profile_dir {
        let target = profile_dir.join("LLVM-C.dll");
        let stale = match (std::fs::metadata(&target), std::fs::metadata(&dll)) {
            (Ok(have), Ok(want)) => have.len() != want.len() || have.modified().ok() < want.modified().ok(),
            _ => true,
        };
        if stale {
            if let Err(error) = std::fs::copy(&dll, &target) {
                println!("cargo:warning=cannot bundle LLVM-C.dll into {}: {error}", profile_dir.display());
            }
        }
    }
}
