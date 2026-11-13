//! Synchronous native filesystem utilities for the `Fs` global.
//!
//! The I/O layer owns all path and directory-entry data before allocating Napalm
//! values. FileInfo and File contain no managed references; their native storage
//! and open handles are released by ordinary heap cleanup. This is intentionally a
//! blocking API: callers must not use it on latency-critical event-loop threads.

use super::*;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_FS_READ_BYTES: usize = crate::napalm_bytes::MAX_BINARY_BYTES;
const MAX_FS_DIRECTORY_ENTRIES: usize = 100_000;

enum FsField {
    Text(String),
    Bool(bool),
}

#[derive(Debug)]
pub(super) struct NapalmFileInfo {
    pub(super) path: String,
    name: String,
    metadata: Metadata,
}

impl NapalmFileInfo {
    pub(super) fn external_bytes(&self) -> usize { self.path.capacity() + self.name.capacity() }
}

#[derive(Debug)]
pub(super) struct NapalmFile {
    pub(super) path: String,
    name: String,
    // Dropping an unreachable heap object also closes any remaining OS handle.
    handle: Option<File>,
    readable: bool,
    writable: bool,
}

impl NapalmFile {
    pub(super) fn external_bytes(&self) -> usize { self.path.capacity() + self.name.capacity() }

    fn handle(&mut self) -> Result<&mut File, RuntimeError> {
        self.handle.as_mut().ok_or_else(|| RuntimeError::new("File is closed"))
    }
}

fn fs_name(path: &Path) -> Result<String, RuntimeError> {
    path.file_name().unwrap_or_default().to_str().map(str::to_owned)
        .ok_or_else(|| RuntimeError::new("File name must contain valid UTF-8"))
}

fn open_regular_file(path: &Path, mode: &str) -> Result<NapalmFile, RuntimeError> {
    let (readable, writable, create, truncate, append) = match mode {
        "r" => (true, false, false, false, false),
        "r+" => (true, true, false, false, false),
        "w" => (false, true, true, true, false),
        "w+" => (true, true, true, true, false),
        "a" => (false, true, true, false, true),
        "a+" => (true, true, true, false, true),
        _ => return Err(RuntimeError::new("Fs.open mode must be r, r+, w, w+, a, or a+")),
    };
    // Reject known non-files before opening, but do not rely on this racy check:
    // no-follow flags and handle metadata below validate the actual opened object.
    match fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.is_file() || metadata.file_type().is_symlink() =>
            return Err(RuntimeError::new("Fs.open accepts only regular files, not directories or symlinks")),
        Ok(_) => {},
        Err(error) if error.kind() == io::ErrorKind::NotFound && create => {},
        Err(error) => return Err(fs_error("open", path, error)),
    }
    let name = fs_name(path)?;
    let mut options = OpenOptions::new();
    options.read(readable).write(writable).create(create).append(append);
    #[cfg(unix)] {
        use std::os::unix::fs::OpenOptionsExt;
        // NONBLOCK avoids hanging if a path is swapped for a FIFO before open.
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)] {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x00200000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    #[cfg(not(any(unix, windows)))]
    return Err(RuntimeError::new("Fs.open is unsupported on this platform"));

    let handle = options.open(path).map_err(|error| fs_error("open", path, error))?;
    let metadata = handle.metadata().map_err(|error| fs_error("open", path, error))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(RuntimeError::new("Fs.open accepts only regular files, not directories or symlinks"));
    }
    #[cfg(windows)] {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(RuntimeError::new("Fs.open does not accept reparse points"));
        }
    }
    // Never truncate until the opened handle has passed the regular-file check.
    if truncate { handle.set_len(0).map_err(|error| fs_error("open", path, error))?; }
    Ok(NapalmFile {
        path: path.to_str().ok_or_else(|| RuntimeError::new("File path must contain valid UTF-8"))?.to_owned(),
        name, handle: Some(handle), readable, writable,
    })
}

struct FsDirectoryEntry {
    name: String,
    path: String,
    is_file: bool,
    is_directory: bool,
    is_symlink: bool,
}

#[inline]
fn fs_error(operation: &str, path: &Path, error: io::Error) -> RuntimeError {
    RuntimeError::new(format!("Fs.{operation}({}): {error}", path.display()))
}

#[inline]
fn fs_system_time(result: io::Result<SystemTime>) -> Option<f64> {
    result.ok().map(|time| match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs_f64() * 1_000.0,
        Err(error) => -(error.duration().as_secs_f64() * 1_000.0),
    })
}

fn fs_read_bounded(path: &Path, operation: &str) -> Result<Vec<u8>, RuntimeError> {
    let file = File::open(path).map_err(|error| fs_error(operation, path, error))?;
    if let Ok(metadata) = file.metadata() {
        if metadata.len() > MAX_FS_READ_BYTES as u64 {
            return Err(RuntimeError::new(format!(
                "Fs.{operation}({}): file exceeds the {}-byte limit",
                path.display(), MAX_FS_READ_BYTES,
            )));
        }
    }
    // The limit is enforced again while reading, in case the file grows between
    // the metadata query and the read, or its size cannot be reported.
    let mut reader = file.take(MAX_FS_READ_BYTES as u64 + 1);
    let mut data = Vec::new();
    reader
        .read_to_end(&mut data)
        .map_err(|error| fs_error(operation, path, error))?;
    if data.len() > MAX_FS_READ_BYTES {
        return Err(RuntimeError::new(format!(
            "Fs.{operation}({}): file exceeds the {}-byte limit",
            path.display(), MAX_FS_READ_BYTES,
        )));
    }
    Ok(data)
}

impl Interpreter<'_> {
    fn fs_string_arg<'a>(&'a self, value: Value, argument: &str) -> Result<&'a str, RuntimeError> {
        let Some(HeapObject::String(value)) = self.heap.get(value) else {
            return Err(RuntimeError::new(format!(
                "TypeError: Fs {argument} must be a String",
            )));
        };
        std::str::from_utf8(value.as_ref()).map_err(|_| {
            RuntimeError::new(format!("TypeError: Fs {argument} must contain valid UTF-8"))
        })
    }

    #[inline]
    fn fs_path_arg(&self, value: Value, argument: &str) -> Result<PathBuf, RuntimeError> {
        self.fs_string_arg(value, argument).map(PathBuf::from)
    }

    /// Create a native Napalm Map without leaving unrooted intermediate values
    /// across the MMTk allocation boundary.
    fn fs_record(&mut self, fields: Vec<(&'static str, FsField)>) -> Result<Value, RuntimeError> {
        let record = self.heap.alloc(HeapObject::Map(NapalmMap::default()));
        let root = self.add_root(record);
        let inserted = (|| -> Result<(), RuntimeError> {
            for (key, field) in fields {
                let item = match field {
                    FsField::Text(text) => self.heap.alloc(HeapObject::String(
                        NapalmString::new(Arc::<[u8]>::from(text.into_bytes())),
                    )),
                    FsField::Bool(boolean) => Value::from_bool(boolean),
                };
                self.heap.remember_if_reference(record, item);
                let key = MapKey::String(MapStringKey::new(Arc::<[u8]>::from(key.as_bytes())));
                self.heap.map_set_encoded(record, key, item, None)?;
            }
            Ok(())
        })();
        self.release_root(root);
        inserted?;
        let external = self.heap.get(record).map_or(0, external_bytes_for);
        self.heap.set_external_bytes(record, external);
        self.heap.refresh_jit_view(record);
        Ok(record)
    }

    fn fs_directory_list(&mut self, entries: Vec<FsDirectoryEntry>) -> Result<Value, RuntimeError> {
        let list = self.heap.alloc(HeapObject::List(Vec::with_capacity(entries.len())));
        let root = self.add_root(list);
        let populated = (|| -> Result<(), RuntimeError> {
            for entry in entries {
                let record = self.fs_record(vec![
                    ("name", FsField::Text(entry.name)),
                    ("path", FsField::Text(entry.path)),
                    ("isFile", FsField::Bool(entry.is_file)),
                    ("isDirectory", FsField::Bool(entry.is_directory)),
                    ("isSymlink", FsField::Bool(entry.is_symlink)),
                ])?;
                self.heap.remember_if_reference(list, record);
                match self.heap.get_mut(list) {
                    Some(HeapObject::List(values)) => values.push(record),
                    _ => return Err(RuntimeError::new("Fs.readDir: result List disappeared")),
                }
            }
            Ok(())
        })();
        self.release_root(root);
        populated?;
        let external = self.heap.get(list).map_or(0, external_bytes_for);
        self.heap.set_external_bytes(list, external);
        self.heap.refresh_jit_view(list);
        Ok(list)
    }

    fn fs_operation(&mut self, native: FsNative, args: &[RegisterWord]) -> Result<Value, RuntimeError> {
        let arg = |index: usize| args.get(index).copied().map(Value::from_word).unwrap_or(Value::NULL);
        let operation = match native {
            FsNative::Open => "open",
            FsNative::ReadText => "readText",
            FsNative::ReadBytes => "readBytes",
            FsNative::WriteText => "writeText",
            FsNative::WriteBytes => "writeBytes",
            FsNative::Exists => "exists",
            FsNative::Stat => "stat",
            FsNative::ReadDir => "readDir",
            FsNative::RealPath => "realPath",
            FsNative::Mkdir => "mkdir",
            FsNative::MkdirAll => "mkdirAll",
            FsNative::RemoveFile => "removeFile",
            FsNative::Rename => "rename",
        };
        let path = self.fs_path_arg(arg(0), "path")?;

        match native {
            FsNative::Open => {
                let mode = self.fs_string_arg(arg(1), "mode")?;
                let file = open_regular_file(&path, mode)?;
                Ok(self.heap.alloc(HeapObject::File(Box::new(file))))
            }
            FsNative::ReadText => {
                let bytes = fs_read_bounded(&path, operation)?;
                let text = String::from_utf8(bytes).map_err(|error| {
                    RuntimeError::new(format!(
                        "Fs.readText({}): file is not valid UTF-8: {error}",
                        path.display(),
                    ))
                })?;
                Ok(self.heap.alloc(HeapObject::String(NapalmString::new(
                    Arc::<[u8]>::from(text.into_bytes()),
                ))))
            }
            FsNative::ReadBytes => {
                let bytes = fs_read_bounded(&path, operation)?;
                Ok(self.heap.alloc(HeapObject::Bytes(
                    NapalmBytes::from_inner(bytes::Bytes::from(bytes)),
                )))
            }
            FsNative::WriteText => {
                let text = self.fs_string_arg(arg(1), "text")?;
                if text.len() > MAX_FS_READ_BYTES {
                    return Err(RuntimeError::new("Fs.writeText: data exceeds maximum binary size"));
                }
                fs::write(&path, text.as_bytes())
                    .map_err(|error| fs_error(operation, &path, error))?;
                Ok(Value::NULL)
            }
            FsNative::WriteBytes => {
                let data = arg(1);
                let result = match self.heap.get(data) {
                    Some(HeapObject::Bytes(bytes)) => {
                        if bytes.len() > MAX_FS_READ_BYTES {
                            return Err(RuntimeError::new("Fs.writeBytes: data exceeds maximum binary size"));
                        }
                        fs::write(&path, bytes.as_slice())
                    }
                    Some(HeapObject::ByteBuffer(buffer)) => {
                        if buffer.len() > MAX_FS_READ_BYTES {
                            return Err(RuntimeError::new("Fs.writeBytes: data exceeds maximum binary size"));
                        }
                        fs::write(&path, buffer.as_slice())
                    }
                    _ => return Err(RuntimeError::new(
                        "TypeError: Fs.writeBytes expects Bytes or ByteBuffer",
                    )),
                };
                result.map_err(|error| fs_error(operation, &path, error))?;
                Ok(Value::NULL)
            }
            FsNative::Exists => {
                let exists = path.try_exists()
                    .map_err(|error| fs_error(operation, &path, error))?;
                Ok(Value::from_bool(exists))
            }
            FsNative::Stat => {
                // Intentionally lstat-style: a symbolic link is reported as a link
                // rather than silently following it to the target.
                let metadata = fs::symlink_metadata(&path)
                    .map_err(|error| fs_error(operation, &path, error))?;
                let name = fs_name(&path)?;
                let path = self.fs_string_arg(arg(0), "path")?.to_owned();
                Ok(self.heap.alloc(HeapObject::FileInfo(Box::new(NapalmFileInfo { path, name, metadata }))))
            }
            FsNative::ReadDir => {
                let iter = fs::read_dir(&path)
                    .map_err(|error| fs_error(operation, &path, error))?;
                let mut entries = Vec::new();
                for entry in iter {
                    if entries.len() >= MAX_FS_DIRECTORY_ENTRIES {
                        return Err(RuntimeError::new(format!(
                            "Fs.readDir({}): directory exceeds the {}-entry limit",
                            path.display(), MAX_FS_DIRECTORY_ENTRIES,
                        )));
                    }
                    let entry = entry.map_err(|error| fs_error(operation, &path, error))?;
                    let name = entry.file_name().into_string().map_err(|_| RuntimeError::new(
                        format!("Fs.readDir({}): an entry name is not valid UTF-8", path.display()),
                    ))?;
                    let entry_path = entry.path().into_os_string().into_string().map_err(|_| RuntimeError::new(
                        format!("Fs.readDir({}): an entry path is not valid UTF-8", path.display()),
                    ))?;
                    let kind = entry.file_type()
                        .map_err(|error| fs_error(operation, &path, error))?;
                    entries.push(FsDirectoryEntry {
                        name,
                        path: entry_path,
                        is_file: kind.is_file(),
                        is_directory: kind.is_dir(),
                        is_symlink: kind.is_symlink(),
                    });
                }
                entries.sort_unstable_by(|a, b| a.name.cmp(&b.name));
                self.fs_directory_list(entries)
            }
            FsNative::RealPath => {
                let canonical = fs::canonicalize(&path)
                    .map_err(|error| fs_error(operation, &path, error))?;
                let text = canonical.into_os_string().into_string().map_err(|_| RuntimeError::new(
                    format!("Fs.realPath({}): canonical path is not valid UTF-8", path.display()),
                ))?;
                Ok(self.heap.alloc(HeapObject::String(NapalmString::new(
                    Arc::<[u8]>::from(text.into_bytes()),
                ))))
            }
            FsNative::Mkdir => {
                fs::create_dir(&path).map_err(|error| fs_error(operation, &path, error))?;
                Ok(Value::NULL)
            }
            FsNative::MkdirAll => {
                fs::create_dir_all(&path).map_err(|error| fs_error(operation, &path, error))?;
                Ok(Value::NULL)
            }
            FsNative::RemoveFile => {
                fs::remove_file(&path).map_err(|error| fs_error(operation, &path, error))?;
                Ok(Value::NULL)
            }
            FsNative::Rename => {
                let to = self.fs_path_arg(arg(1), "destination path")?;
                fs::rename(&path, &to).map_err(|error| RuntimeError::new(format!(
                    "Fs.rename({} -> {}): {error}", path.display(), to.display(),
                )))?;
                Ok(Value::NULL)
            }
        }
    }

    fn file_operation(&mut self, native: FileNative, receiver: Value, args: &[RegisterWord]) -> Result<Value, RuntimeError> {
        let arg = || args.first().copied().map(Value::from_word).unwrap_or(Value::NULL);
        if matches!(native, FileNative::Path | FileNative::Name) {
            let text = match self.heap.get(receiver) {
                Some(HeapObject::FileInfo(info)) => if native == FileNative::Path { &info.path } else { &info.name },
                Some(HeapObject::File(file)) => if native == FileNative::Path { &file.path } else { &file.name },
                _ => return Err(RuntimeError::new("expected File or FileInfo")),
            }.clone();
            return Ok(self.heap.alloc(HeapObject::String(NapalmString::new(Arc::<[u8]>::from(text.into_bytes())))));
        }
        if matches!(native, FileNative::Size | FileNative::IsFile | FileNative::IsDirectory | FileNative::IsSymlink | FileNative::ModifiedAt | FileNative::CreatedAt) {
            let metadata = match self.heap.get_mut(receiver) {
                Some(HeapObject::FileInfo(info)) => info.metadata.clone(),
                Some(HeapObject::File(file)) => file.handle()?.metadata()
                    .map_err(|error| fs_error("metadata", Path::new(&file.path), error))?,
                _ => return Err(RuntimeError::new("expected File or FileInfo")),
            };
            return Ok(match native {
                FileNative::Size => Value::from_num(metadata.len() as f64),
                FileNative::IsFile => Value::from_bool(metadata.is_file()),
                FileNative::IsDirectory => Value::from_bool(metadata.is_dir()),
                FileNative::IsSymlink => Value::from_bool(metadata.file_type().is_symlink()),
                FileNative::ModifiedAt | FileNative::CreatedAt => {
                    match fs_system_time(if native == FileNative::ModifiedAt { metadata.modified() } else { metadata.created() }) {
                        Some(millis) => self.heap.alloc(HeapObject::Date { millis: millis.trunc() }),
                        None => Value::NULL,
                    }
                }
                _ => unreachable!(),
            });
        }
        // Copy write input before borrowing the mutable receiver. No managed
        // allocations occur until the OS operation has finished.
        let data = if native == FileNative::Write {
            Some(match self.heap.get(arg()) {
                Some(HeapObject::Bytes(bytes)) => bytes.clone(),
                Some(HeapObject::ByteBuffer(buffer)) => NapalmBytes::from_slice(buffer.as_slice()),
                _ => return Err(RuntimeError::new("File.write expects Bytes or ByteBuffer")),
            })
        } else { None };
        let number = if matches!(native, FileNative::Read | FileNative::Seek) {
            let number = arg().as_num().ok_or_else(|| RuntimeError::new("File count/offset must be a nonnegative integer"))?;
            let limit = if native == FileNative::Read { MAX_FS_READ_BYTES as f64 } else { 9_007_199_254_740_991.0 };
            if !number.is_finite() || number.fract() != 0.0 || number < 0.0 || number > limit {
                return Err(RuntimeError::new("File count/offset is outside the supported integer range"));
            }
            number as u64
        } else { 0 };
        let Some(HeapObject::File(file)) = self.heap.get_mut(receiver) else {
            return Err(RuntimeError::new("expected File"));
        };
        if native == FileNative::Close {
            file.handle.take();
            return Ok(Value::NULL);
        }
        if file.handle.is_none() { return Err(RuntimeError::new("File is closed")); }
        match native {
            FileNative::Read => {
                if !file.readable { return Err(RuntimeError::new("File was not opened for reading")); }
                let mut data = Vec::new();
                file.handle()?.take(number).read_to_end(&mut data)
                    .map_err(|error| fs_error("read", Path::new(&file.path), error))?;
                Ok(self.heap.alloc(HeapObject::Bytes(NapalmBytes::from_inner(bytes::Bytes::from(data)))))
            }
            FileNative::Write => {
                if !file.writable { return Err(RuntimeError::new("File was not opened for writing")); }
                let data = data.unwrap();
                if data.len() > MAX_FS_READ_BYTES { return Err(RuntimeError::new("File.write exceeds maximum binary size")); }
                file.handle()?.write_all(data.as_slice())
                    .map_err(|error| fs_error("write", Path::new(&file.path), error))?;
                Ok(Value::from_num(data.len() as f64))
            }
            FileNative::Seek => {
                let position = file.handle()?.seek(SeekFrom::Start(number))
                    .map_err(|error| fs_error("seek", Path::new(&file.path), error))?;
                Ok(Value::from_num(position as f64))
            }
            FileNative::Flush => {
                file.handle()?.sync_all().map_err(|error| fs_error("flush", Path::new(&file.path), error))?;
                Ok(Value::NULL)
            }
            _ => unreachable!(),
        }
    }

    pub(super) fn execute_file_native(&mut self, op: &PreparedOp, native: FileNative, receiver: RegisterWord, args: &[RegisterWord]) {
        self.mark_gc_point(op);
        self.maybe_collect();
        match self.file_operation(native, Value::from_word(receiver), args) {
            Ok(value) => unsafe {
                self.set_reg(op.a, value.word());
                self.advance(op);
            },
            Err(error) => self.fail_at(op, error.message),
        }
    }

    pub(super) fn execute_fs_native(
        &mut self,
        op: &PreparedOp,
        native: FsNative,
        args: &[RegisterWord],
    ) {
        // The original argument registers remain GC roots while creating Maps,
        // Lists, Strings and Bytes in these operations.
        self.mark_gc_point(op);
        self.maybe_collect();
        match self.fs_operation(native, args) {
            Ok(value) => unsafe {
                self.set_reg(op.a, value.word());
                self.advance(op);
            },
            Err(error) => self.fail_at(op, error.message),
        }
    }
}
