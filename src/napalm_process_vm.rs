//! Native process environment and standard byte-stream operations.
//!
//! The standard streams are VM-rooted singleton objects. Closing one from
//! Napalm deliberately does not close stdin, stdout, or stderr at the OS level.
//! Calls are synchronous; in particular, stdin reads may block the VM thread.

use super::*;
use std::io::{self, BufRead, Read, Write};

const MAX_STREAM_BYTES: usize = crate::napalm_bytes::MAX_BINARY_BYTES;
const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(super) enum NativeInputStream {
    Stdin,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum NativeOutputStream {
    Stdout,
    Stderr,
}

impl NativeOutputStream {
    #[inline]
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Stdout => "stdout",
            Self::Stderr => "stderr",
        }
    }

    fn write(self, bytes: &[u8], complete: bool) -> io::Result<usize> {
        match self {
            Self::Stdout => {
                let handle = io::stdout();
                let mut writer = handle.lock();
                if complete {
                    writer.write_all(bytes)?;
                    Ok(bytes.len())
                } else {
                    writer.write(bytes)
                }
            }
            Self::Stderr => {
                let handle = io::stderr();
                let mut writer = handle.lock();
                if complete {
                    writer.write_all(bytes)?;
                    Ok(bytes.len())
                } else {
                    writer.write(bytes)
                }
            }
        }
    }

    fn flush(self) -> io::Result<()> {
        match self {
            Self::Stdout => io::stdout().lock().flush(),
            Self::Stderr => io::stderr().lock().flush(),
        }
    }
}

#[inline]
fn stream_error(action: &str, error: io::Error) -> RuntimeError {
    RuntimeError::new(format!("{action}: {error}"))
}

#[inline]
fn native_string(value: &NapalmString, method: &str) -> Result<String, RuntimeError> {
    String::from_utf8(value.as_ref().to_vec()).map_err(|_| {
        RuntimeError::new(format!("TypeError: {method} requires valid UTF-8"))
    })
}

#[inline]
fn checked_count(value: Value, method: &str) -> Result<usize, RuntimeError> {
    let number = value.as_num().ok_or_else(|| {
        RuntimeError::new(format!("TypeError: {method} expects a nonnegative integer Number"))
    })?;
    let count = crate::napalm_bytes::number_to_length(number)
        .map_err(|error| RuntimeError::new(format!("RangeError: {method}: {error}")))?;
    if count > MAX_STREAM_BYTES {
        return Err(RuntimeError::new(format!(
            "RangeError: {method} exceeds the {MAX_STREAM_BYTES}-byte limit",
        )));
    }
    Ok(count)
}

#[inline]
fn byte_object(data: Vec<u8>) -> HeapObject {
    HeapObject::Bytes(NapalmBytes::from_inner(bytes::Bytes::from(data)))
}

impl Interpreter<'_> {
    fn process_string_arg(&self, value: Value, method: &str) -> Result<String, RuntimeError> {
        match self.heap.get(value) {
            Some(HeapObject::String(text)) => native_string(text, method),
            _ => Err(RuntimeError::new(format!(
                "TypeError: {method} expects a String",
            ))),
        }
    }

    fn process_string(&mut self, text: String) -> Value {
        self.heap.alloc(HeapObject::String(NapalmString::new(
            Arc::<[u8]>::from(text.into_bytes()),
        )))
    }

    fn process_std_stream(&mut self, index: usize) -> Value {
        if let Some(value) = self.process_standard_streams[index] {
            return value;
        }
        let object = match index {
            0 => HeapObject::InputStream(NativeInputStream::Stdin),
            1 => HeapObject::OutputStream(NativeOutputStream::Stdout),
            2 => HeapObject::OutputStream(NativeOutputStream::Stderr),
            _ => unreachable!("standard stream index must be 0, 1, or 2"),
        };
        let value = self.heap.alloc(object);
        // Keep the stream alive for the VM lifetime, even if a script drops every
        // reference. A getter must always return the same managed stream object.
        self.add_root(value);
        self.process_standard_streams[index] = Some(value);
        value
    }

    fn process_args(&mut self) -> Value {
        // CLI convention: `napalm script.napalm [script arguments...]`; a
        // standalone executable (`--compile`) has no script path: `app [args...]`.
        // Use OsString to avoid panicking if the OS supplied non-UTF-8 argv.
        let skip = if crate::napalm_standalone::embedded().is_some() { 1 } else { 2 };
        let args: Vec<String> = std::env::args_os()
            .skip(skip)
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        let list = self.heap.alloc(HeapObject::List(Vec::with_capacity(args.len())));
        let root = self.add_root(list);
        for text in args {
            let value = self.process_string(text);
            self.heap.remember_if_reference(list, value);
            if let Some(HeapObject::List(items)) = self.heap.get_mut(list) {
                items.push(value);
            } else {
                unreachable!("rooted Process.args list must remain live");
            }
        }
        self.release_root(root);
        let external = self.heap.get(list).map_or(0, external_bytes_for);
        self.heap.set_external_bytes(list, external);
        self.heap.refresh_jit_view(list);
        list
    }

    fn process_operation(
        &mut self,
        native: ProcessNative,
        args: &[RegisterWord],
    ) -> Result<Value, RuntimeError> {
        let arg = |index: usize| args.get(index).copied().map(Value::from_word).unwrap_or(Value::NULL);
        match native {
            ProcessNative::Stdin => Ok(self.process_std_stream(0)),
            ProcessNative::Stdout => Ok(self.process_std_stream(1)),
            ProcessNative::Stderr => Ok(self.process_std_stream(2)),
            ProcessNative::Args => Ok(self.process_args()),
            ProcessNative::Cwd => {
                let cwd = std::env::current_dir().map_err(|error| stream_error("Process.cwd", error))?;
                Ok(self.process_string(cwd.to_string_lossy().into_owned()))
            }
            ProcessNative::Chdir => {
                let path = self.process_string_arg(arg(0), "Process.chdir")?;
                std::env::set_current_dir(&path)
                    .map_err(|error| stream_error("Process.chdir", error))?;
                Ok(Value::NULL)
            }
            ProcessNative::Getenv => {
                let name = self.process_string_arg(arg(0), "Process.getenv")?;
                if name.is_empty() || name.contains('=') || name.contains('\0') {
                    return Err(RuntimeError::new("TypeError: invalid environment variable name"));
                }
                match std::env::var(name) {
                    Ok(text) => Ok(self.process_string(text)),
                    Err(std::env::VarError::NotPresent) => Ok(Value::NULL),
                    Err(error) => Err(RuntimeError::new(format!("Process.getenv: {error}"))),
                }
            }
            ProcessNative::Exit => {
                let number = arg(0).as_num().ok_or_else(|| {
                    RuntimeError::new("TypeError: Process.exit expects an integer Number")
                })?;
                if !number.is_finite() || number.fract() != 0.0
                    || number < i32::MIN as f64 || number > i32::MAX as f64
                {
                    return Err(RuntimeError::new("RangeError: invalid process exit code"));
                }
                // Deliberately terminates the host process, not just a fiber.
                std::process::exit(number as i32)
            }
        }
    }

    pub(super) fn execute_process_native(
        &mut self,
        op: &PreparedOp,
        native: ProcessNative,
        args: &[RegisterWord],
    ) {
        self.mark_gc_point(op);
        self.maybe_collect();
        let result = self.process_operation(native, args);
        match result {
            Ok(value) => unsafe { self.set_reg(op.a, value.word()); self.advance(op); },
            Err(error) => self.fail_at(op, error.message),
        }
    }

    fn input_stream_operation(
        &mut self,
        native: InputStreamNative,
        receiver: Value,
        args: &[RegisterWord],
    ) -> Result<Value, RuntimeError> {
        if !matches!(self.heap.get(receiver), Some(HeapObject::InputStream(NativeInputStream::Stdin))) {
            return Err(RuntimeError::new("TypeError: InputStream method requires an InputStream"));
        }
        let arg = |index: usize| args.get(index).copied().map(Value::from_word).unwrap_or(Value::NULL);
        match native {
            InputStreamNative::Read => {
                let count = checked_count(arg(0), "InputStream.read")?;
                if count == 0 {
                    return Ok(self.heap.alloc(byte_object(Vec::new())));
                }
                // A short read is permitted. Do not preallocate the entire
                // caller-supplied maximum when stdin has only a small chunk ready.
                let mut bytes = vec![0u8; count.min(64 * 1024)];
                let handle = io::stdin();
                let read = handle.lock().read(&mut bytes)
                    .map_err(|error| stream_error("InputStream.read", error))?;
                if read == 0 {
                    return Ok(Value::NULL);
                }
                bytes.truncate(read);
                Ok(self.heap.alloc(byte_object(bytes)))
            }
            InputStreamNative::ReadExact => {
                let count = checked_count(arg(0), "InputStream.readExact")?;
                let mut bytes = vec![0u8; count];
                let handle = io::stdin();
                handle.lock().read_exact(&mut bytes)
                    .map_err(|error| stream_error("InputStream.readExact", error))?;
                Ok(self.heap.alloc(byte_object(bytes)))
            }
            InputStreamNative::ReadLine => {
                let mut bytes = Vec::new();
                let handle = io::stdin();
                let mut reader = handle.lock();
                let mut limited = (&mut reader).take(MAX_LINE_BYTES as u64 + 1);
                let consumed = limited.read_until(b'\n', &mut bytes)
                    .map_err(|error| stream_error("InputStream.readLine", error))?;
                if bytes.len() > MAX_LINE_BYTES ||
                    (bytes.len() == MAX_LINE_BYTES && !bytes.ends_with(b"\n"))
                {
                    return Err(RuntimeError::new("RangeError: InputStream.readLine exceeds the line size limit"));
                }
                if consumed == 0 {
                    return Ok(Value::NULL);
                }
                if bytes.ends_with(b"\n") {
                    bytes.pop();
                    if bytes.ends_with(b"\r") { bytes.pop(); }
                }
                let text = String::from_utf8(bytes)
                    .map_err(|_| RuntimeError::new("TypeError: InputStream.readLine received invalid UTF-8"))?;
                Ok(self.process_string(text))
            }
            // Standard input is borrowed from the process; scripts cannot close it.
            InputStreamNative::Close => Ok(Value::NULL),
        }
    }

    pub(super) fn execute_input_stream_native(
        &mut self,
        op: &PreparedOp,
        native: InputStreamNative,
        receiver: RegisterWord,
        args: &[RegisterWord],
    ) {
        self.mark_gc_point(op);
        self.maybe_collect();
        let result = self.input_stream_operation(native, Value::from_word(receiver), args);
        match result {
            Ok(value) => unsafe { self.set_reg(op.a, value.word()); self.advance(op); },
            Err(error) => self.fail_at(op, error.message),
        }
    }

    fn output_stream_operation(
        &mut self,
        native: OutputStreamNative,
        receiver: Value,
        args: &[RegisterWord],
    ) -> Result<Value, RuntimeError> {
        let Some(HeapObject::OutputStream(stream)) = self.heap.get(receiver) else {
            return Err(RuntimeError::new("TypeError: OutputStream method requires an OutputStream"));
        };
        let stream = *stream;
        let arg = |index: usize| args.get(index).copied().map(Value::from_word).unwrap_or(Value::NULL);
        match native {
            OutputStreamNative::Write | OutputStreamNative::WriteAll => {
                let data = match self.heap.get(arg(0)) {
                    Some(HeapObject::Bytes(bytes)) => bytes.as_slice(),
                    Some(HeapObject::ByteBuffer(buffer)) => buffer.as_slice(),
                    _ => return Err(RuntimeError::new(
                        "TypeError: OutputStream.write expects Bytes or ByteBuffer",
                    )),
                };
                let written = stream.write(data, native == OutputStreamNative::WriteAll)
                    .map_err(|error| stream_error("OutputStream.write", error))?;
                Ok(Value::from_num(written as f64))
            }
            OutputStreamNative::WriteText => {
                let data = match self.heap.get(arg(0)) {
                    Some(HeapObject::String(text)) => {
                        std::str::from_utf8(text.as_ref()).map_err(|_| RuntimeError::new(
                            "TypeError: OutputStream.writeText requires valid UTF-8",
                        ))?;
                        text.as_ref()
                    },
                    _ => return Err(RuntimeError::new(
                        "TypeError: OutputStream.writeText expects a String",
                    )),
                };
                let written = stream.write(data, true)
                    .map_err(|error| stream_error("OutputStream.writeText", error))?;
                Ok(Value::from_num(written as f64))
            }
            OutputStreamNative::Flush => {
                stream.flush().map_err(|error| stream_error("OutputStream.flush", error))?;
                Ok(Value::NULL)
            }
            // Standard stdout/stderr are borrowed; closing cannot disrupt LSP IO.
            OutputStreamNative::Close => Ok(Value::NULL),
        }
    }

    pub(super) fn execute_output_stream_native(
        &mut self,
        op: &PreparedOp,
        native: OutputStreamNative,
        receiver: RegisterWord,
        args: &[RegisterWord],
    ) {
        self.mark_gc_point(op);
        self.maybe_collect();
        let result = self.output_stream_operation(native, Value::from_word(receiver), args);
        match result {
            Ok(value) => unsafe { self.set_reg(op.a, value.word()); self.advance(op); },
            Err(error) => self.fail_at(op, error.message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_length_validation_rejects_non_integer_and_oversized_values() {
        let operation = "InputStream.readExact";
        for invalid in [f64::NAN, f64::INFINITY, -1.0, 0.5, 9_007_199_254_740_992.0] {
            assert!(checked_count(Value::from_num(invalid), operation).is_err());
        }
        assert_eq!(checked_count(Value::from_num(0.0), operation).unwrap(), 0);
        assert_eq!(checked_count(Value::from_num(1024.0), operation).unwrap(), 1024);
        assert!(checked_count(
            Value::from_num((MAX_STREAM_BYTES + 1) as f64), operation
        ).is_err());
    }

    #[test]
    fn standard_stream_labels_are_distinct() {
        assert_eq!(NativeOutputStream::Stdout.label(), "stdout");
        assert_eq!(NativeOutputStream::Stderr.label(), "stderr");
    }
}
