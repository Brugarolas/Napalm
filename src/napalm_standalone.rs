//! Standalone executables produced by `napalm --compile`.
//!
//! A standalone executable is a byte-for-byte copy of the running Napalm
//! runtime followed by an embedded payload and a fixed trailer:
//!
//! ```text
//! [runtime image][.burning artifact][LLVM-C.dll, zstd (optional)][trailer]
//! ```
//!
//! Operating systems ignore bytes appended after an executable image, so the
//! copy still starts as the normal Napalm CLI. `main` calls [`init`] first: when
//! the last 64 bytes of its own image hold a valid trailer, it runs the embedded
//! artifact instead of parsing a command line. See docs/BURNING.md.
//!
//! The LLVM C API DLL is delay-loaded (see build.rs), so it is only needed once
//! the LLVM trace tier starts. The embedded copy is extracted then, once per
//! machine, into a cache directory named after its content hash.

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

const MAGIC: &[u8; 16] = b"NAPALM-STANDALN\0";
const VERSION: u32 = 1;
/// Five `u64` fields, version, reserved `u32`, magic.
const TRAILER_LEN: usize = 64;
/// The DLL is compressed on every `--compile`. Measured on LLVM-C.dll 23.1
/// (74 MB, multithreaded): level 12 gives 23.7 MB in 0.7 s; level 19 only gets
/// to 21.1 MB and takes 13 s. Decompression takes ~60-90 ms at any level.
const LLVM_ZSTD_LEVEL: i32 = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Trailer {
    /// Size of the runtime image; the payload starts at this offset.
    runtime_len: u64,
    burning_len: u64,
    /// Compressed size, 0 when no LLVM library is embedded.
    llvm_len: u64,
    llvm_raw_len: u64,
    /// FNV-1a 64 of the uncompressed library; names its cache directory.
    llvm_hash: u64,
}

impl Trailer {
    fn encode(&self) -> [u8; TRAILER_LEN] {
        let mut out = [0u8; TRAILER_LEN];
        let fields = [
            self.runtime_len,
            self.burning_len,
            self.llvm_len,
            self.llvm_raw_len,
            self.llvm_hash,
        ];
        for (index, value) in fields.iter().enumerate() {
            out[index * 8..index * 8 + 8].copy_from_slice(&value.to_le_bytes());
        }
        out[40..44].copy_from_slice(&VERSION.to_le_bytes());
        out[48..64].copy_from_slice(MAGIC);
        out
    }

    /// `Ok(None)`: not a standalone executable. `Err`: the magic is present
    /// but the trailer does not describe this file.
    fn decode(bytes: &[u8; TRAILER_LEN], file_len: u64) -> io::Result<Option<Self>> {
        if &bytes[48..64] != MAGIC {
            return Ok(None);
        }
        let field = |index: usize| u64::from_le_bytes(bytes[index * 8..index * 8 + 8].try_into().unwrap());
        if u32::from_le_bytes(bytes[40..44].try_into().unwrap()) != VERSION {
            return Err(invalid("unsupported standalone payload version"));
        }
        let trailer = Self {
            runtime_len: field(0),
            burning_len: field(1),
            llvm_len: field(2),
            llvm_raw_len: field(3),
            llvm_hash: field(4),
        };
        let total = [trailer.burning_len, trailer.llvm_len, TRAILER_LEN as u64]
            .iter()
            .try_fold(trailer.runtime_len, |sum, &len| sum.checked_add(len));
        if total != Some(file_len) || trailer.burning_len == 0 || (trailer.llvm_len == 0) != (trailer.llvm_raw_len == 0) {
            return Err(invalid("corrupt standalone payload trailer"));
        }
        Ok(Some(trailer))
    }
}

fn invalid(message: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

/// The payload embedded in the running executable.
pub struct Embedded {
    exe: PathBuf,
    trailer: Trailer,
    llvm_library: OnceLock<Result<PathBuf, String>>,
    /// Set once the extraction of `llvm_library` has begun.
    extracting: AtomicBool,
}

static EMBEDDED: OnceLock<Option<Embedded>> = OnceLock::new();

fn read_trailer(file: &mut File) -> io::Result<Option<Trailer>> {
    let len = file.metadata()?.len();
    if len < TRAILER_LEN as u64 {
        return Ok(None);
    }
    file.seek(SeekFrom::Start(len - TRAILER_LEN as u64))?;
    let mut bytes = [0u8; TRAILER_LEN];
    file.read_exact(&mut bytes)?;
    Trailer::decode(&bytes, len)
}

/// Detect a payload appended to the running executable. Returns whether one is
/// present. Must run before [`embedded`] reports anything; unit tests never
/// call it and therefore always behave like the plain CLI.
pub fn init() -> io::Result<bool> {
    let Ok(exe) = std::env::current_exe() else {
        let _ = EMBEDDED.set(None);
        return Ok(false);
    };
    // An unreadable image (e.g. execute-only permissions) cannot carry a
    // payload we could use anyway: behave like the normal CLI.
    let trailer = match File::open(&exe) {
        Ok(mut file) => read_trailer(&mut file)?,
        Err(_) => None,
    };
    let present = trailer.is_some();
    let _ = EMBEDDED.set(trailer.map(|trailer| Embedded {
        exe,
        trailer,
        llvm_library: OnceLock::new(),
        extracting: AtomicBool::new(false),
    }));
    Ok(present)
}

pub fn embedded() -> Option<&'static Embedded> {
    EMBEDDED.get().and_then(Option::as_ref)
}

impl Embedded {
    fn open_at(&self, offset: u64) -> io::Result<File> {
        let mut file = File::open(&self.exe)?;
        file.seek(SeekFrom::Start(offset))?;
        Ok(file)
    }

    /// The embedded `.burning` artifact, still encoded.
    pub fn read_burning(&self) -> io::Result<Vec<u8>> {
        let len = usize::try_from(self.trailer.burning_len).map_err(invalid)?;
        let mut bytes = vec![0u8; len];
        self.open_at(self.trailer.runtime_len)?.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    /// Path of the embedded LLVM library, extracting it on first use. `None`
    /// when the executable was compiled without one.
    pub fn llvm_library(&self) -> Option<&Result<PathBuf, String>> {
        if self.trailer.llvm_len == 0 {
            return None;
        }
        Some(self.llvm_library.get_or_init(|| {
            self.extracting.store(true, Ordering::Release);
            self.extract_llvm_library()
                .map_err(|error| format!("cannot extract the embedded LLVM library: {error}"))
        }))
    }

    fn extract_llvm_library(&self) -> io::Result<PathBuf> {
        let directory = cache_root().join("llvm").join(format!("{:016x}", self.trailer.llvm_hash));
        let target = directory.join(LLVM_LIBRARY_NAME);
        if fs::metadata(&target).is_ok_and(|meta| meta.len() == self.trailer.llvm_raw_len) {
            return Ok(target);
        }
        fs::create_dir_all(&directory)?;
        remove_stale_temporaries(&directory);
        // Decompress beside the target and rename, so concurrent first runs
        // never observe (or load) a partially written library.
        let temporary = directory.join(format!("{LLVM_LIBRARY_NAME}.{}.tmp", std::process::id()));
        let result = (|| {
            let source = self.open_at(self.trailer.runtime_len + self.trailer.burning_len)?;
            let mut output = File::create(&temporary)?;
            let written = io::copy(
                &mut zstd::stream::Decoder::new(source.take(self.trailer.llvm_len))?,
                &mut output,
            )?;
            if written != self.trailer.llvm_raw_len {
                return Err(invalid("embedded LLVM library has the wrong size"));
            }
            output.sync_all()?;
            drop(output);
            fs::rename(&temporary, &target)
        })();
        if let Err(error) = result {
            let _ = fs::remove_file(&temporary);
            // Another process may have won the race (and may hold the library
            // loaded, which makes replacing it fail on Windows).
            if fs::metadata(&target).is_ok_and(|meta| meta.len() == self.trailer.llvm_raw_len) {
                return Ok(target);
            }
            return Err(error);
        }
        Ok(target)
    }
}

/// The LLVM tier extracts the library on a background compiler thread, and the
/// fast process exit (`napalm_tracing_jit::finish_process`) does not wait for
/// those threads. Called before it, this lets an extraction that already
/// started finish, so it happens once per machine instead of being cut short
/// (and restarted) by every short run. No-op when nothing is being extracted.
pub fn finish_pending_extraction() {
    if let Some(embedded) = embedded() {
        if embedded.extracting.load(Ordering::Acquire) {
            embedded.llvm_library.wait();
        }
    }
}

/// Partial libraries left by processes that were killed mid-extraction. An
/// extraction takes well under a second, so anything this old is abandoned.
fn remove_stale_temporaries(directory: &Path) {
    let Ok(entries) = fs::read_dir(directory) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let abandoned = entry.metadata().ok().and_then(|meta| meta.modified().ok()).and_then(|time| time.elapsed().ok())
            .is_some_and(|age| age.as_secs() >= 60);
        if abandoned && path.extension().is_some_and(|ext| ext == "tmp") {
            let _ = fs::remove_file(path);
        }
    }
}

#[cfg(windows)]
const LLVM_LIBRARY_NAME: &str = "LLVM-C.dll";
#[cfg(not(windows))]
const LLVM_LIBRARY_NAME: &str = "libLLVM";

/// Per-user cache directory for extracted libraries.
fn cache_root() -> PathBuf {
    if let Some(explicit) = std::env::var_os("NAPALM_CACHE_DIR") {
        return PathBuf::from(explicit);
    }
    #[cfg(windows)]
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(target_os = "macos")]
    let base = std::env::var_os("HOME").map(|home| PathBuf::from(home).join("Library").join("Caches"));
    #[cfg(all(unix, not(target_os = "macos")))]
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")));
    base.unwrap_or_else(std::env::temp_dir).join("napalm")
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash = (hash ^ byte as u64).wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// The LLVM library a compiled executable should carry: the one this runtime
/// would load itself. Only Windows delay-loads LLVM; elsewhere libLLVM is a
/// normal dynamic dependency resolved by the loader before `main`.
#[cfg(all(windows, feature = "llvm-jit"))]
fn llvm_library_to_embed() -> Result<Option<PathBuf>, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(explicit) = std::env::var_os("NAPALM_LLVM_DLL") {
        candidates.push(explicit.into());
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|exe| exe.parent().map(Path::to_owned)) {
        candidates.push(dir.join(LLVM_LIBRARY_NAME));
    }
    if let Some(built) = option_env!("NAPALM_LLVM_DLL") {
        candidates.push(built.into());
    }
    match candidates.iter().find(|path| path.is_file()) {
        Some(found) => Ok(Some(found.clone())),
        None => Err(format!("{LLVM_LIBRARY_NAME} not found (tried {candidates:?})")),
    }
}

#[cfg(not(all(windows, feature = "llvm-jit")))]
fn llvm_library_to_embed() -> Result<Option<PathBuf>, String> {
    Ok(None)
}

/// What [`write_executable`] embedded.
pub struct CompileReport {
    pub output: PathBuf,
    pub size: u64,
    pub llvm: Option<PathBuf>,
    /// Why no LLVM library was embedded although this runtime uses one.
    pub llvm_warning: Option<String>,
}

/// Default output beside the entry source: `fib.napalm` -> `fib.exe` on
/// Windows, `fib` elsewhere.
pub fn default_output(source: &Path) -> PathBuf {
    source.with_extension(std::env::consts::EXE_EXTENSION)
}

/// Write `output` as a copy of the running runtime carrying `burning` (an
/// encoded, already validated artifact) and, when available, the LLVM library.
/// The file is assembled under a temporary name and renamed into place, so a
/// failure leaves an existing output untouched.
pub fn write_executable(output: &Path, burning: &[u8]) -> io::Result<CompileReport> {
    let exe = std::env::current_exe()?;
    if fs::canonicalize(output).ok().is_some_and(|path| fs::canonicalize(&exe).ok() == Some(path)) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "output would overwrite the running Napalm executable"));
    }
    let mut runtime = fs::read(&exe)?;
    // A standalone executable never compiles, but never nest payloads anyway.
    if let Some(len) = embedded().map(|embedded| embedded.trailer.runtime_len) {
        runtime.truncate(len as usize);
    }

    let (llvm_path, llvm_warning) = match llvm_library_to_embed() {
        Ok(path) => (path, None),
        Err(warning) => (None, Some(warning)),
    };
    let (llvm_frame, llvm_raw_len, llvm_hash) = match &llvm_path {
        Some(path) => {
            let raw = fs::read(path)?;
            let mut compressor = zstd::bulk::Compressor::new(LLVM_ZSTD_LEVEL)?;
            compressor.include_checksum(true)?;
            let workers = std::thread::available_parallelism().map_or(1, |n| n.get()) as u32;
            compressor.set_parameter(zstd::zstd_safe::CParameter::NbWorkers(workers))?;
            (compressor.compress(&raw)?, raw.len() as u64, fnv1a64(&raw))
        }
        None => (Vec::new(), 0, 0),
    };

    let trailer = Trailer {
        runtime_len: runtime.len() as u64,
        burning_len: burning.len() as u64,
        llvm_len: llvm_frame.len() as u64,
        llvm_raw_len,
        llvm_hash,
    };

    let file_name = output.file_name().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "output path has no file name")
    })?;
    let mut temporary_name = file_name.to_owned();
    temporary_name.push(format!(".{}.tmp", std::process::id()));
    let temporary = output.with_file_name(temporary_name);
    let result = (|| {
        let mut file = File::create(&temporary)?;
        file.write_all(&runtime)?;
        file.write_all(burning)?;
        file.write_all(&llvm_frame)?;
        file.write_all(&trailer.encode())?;
        file.sync_all()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(0o755))?;
        }
        drop(file);
        fs::rename(&temporary, output)
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(CompileReport {
        output: output.to_owned(),
        size: trailer.runtime_len + trailer.burning_len + trailer.llvm_len + TRAILER_LEN as u64,
        llvm: llvm_path,
        llvm_warning,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Trailer {
        Trailer { runtime_len: 1000, burning_len: 200, llvm_len: 30, llvm_raw_len: 90, llvm_hash: 0xdead_beef }
    }

    #[test]
    fn trailer_round_trips_and_checks_lengths() {
        let trailer = sample();
        let bytes = trailer.encode();
        assert_eq!(Trailer::decode(&bytes, 1294).unwrap(), Some(trailer));
        assert!(Trailer::decode(&bytes, 1293).is_err());
        assert!(Trailer::decode(&bytes, 1295).is_err());
    }

    #[test]
    fn plain_executables_and_corrupt_trailers() {
        assert_eq!(Trailer::decode(&[0u8; TRAILER_LEN], 5000).unwrap(), None);
        let mut version = sample().encode();
        version[40] ^= 1;
        assert!(Trailer::decode(&version, 1294).is_err());
        let overflow = Trailer { runtime_len: u64::MAX, ..sample() }.encode();
        assert!(Trailer::decode(&overflow, 1294).is_err());
        let half_llvm = Trailer { llvm_raw_len: 0, ..sample() }.encode();
        assert!(Trailer::decode(&half_llvm, 1294).is_err());
        let no_burning = Trailer { burning_len: 0, runtime_len: 1200, ..sample() }.encode();
        assert!(Trailer::decode(&no_burning, 1294).is_err());
    }
}
