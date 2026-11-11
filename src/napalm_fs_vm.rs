//! Synchronous native filesystem utilities for the `Fs` global.
//!
//! The I/O layer owns all path and directory-entry data before allocating Napalm
//! values. Directory results and metadata are materialized as ordinary Map/List
//! objects with explicit GC roots and write barriers. This is intentionally a
//! blocking API: callers must not use it on latency-critical event-loop threads.

use super::*;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_FS_READ_BYTES: usize = crate::napalm_bytes::MAX_BINARY_BYTES;
const MAX_FS_DIRECTORY_ENTRIES: usize = 100_000;

enum FsField {
    Text(String),
    Number(f64),
    Bool(bool),
    Null,
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
fn fs_system_time(result: io::Result<SystemTime>) -> FsField {
    match result {
        Ok(time) => {
            let millis = match time.duration_since(UNIX_EPOCH) {
                Ok(duration) => duration.as_secs_f64() * 1_000.0,
                Err(error) => -(error.duration().as_secs_f64() * 1_000.0),
            };
            FsField::Number(millis)
        }
        Err(_) => FsField::Null,
    }
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
                    FsField::Number(number) => Value::from_num(number),
                    FsField::Bool(boolean) => Value::from_bool(boolean),
                    FsField::Null => Value::NULL,
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
                let file_type = metadata.file_type();
                self.fs_record(vec![
                    ("size", FsField::Number(metadata.len() as f64)),
                    ("isFile", FsField::Bool(file_type.is_file())),
                    ("isDirectory", FsField::Bool(file_type.is_dir())),
                    ("isSymlink", FsField::Bool(file_type.is_symlink())),
                    ("readonly", FsField::Bool(metadata.permissions().readonly())),
                    ("modifiedMs", fs_system_time(metadata.modified())),
                    ("createdMs", fs_system_time(metadata.created())),
                    ("accessedMs", fs_system_time(metadata.accessed())),
                ])
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
