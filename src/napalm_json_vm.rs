//! Native `JSON.parse` and `JSON.stringify` bindings.
//!
//! Both operations share the fetch stack's JSON codec. Parsing uses simd-json's
//! mutable tape input; encoding uses the existing order-preserving serializer.
//! No GC references are stored inside the temporary Rust byte buffers.

use super::*;

impl Interpreter<'_> {
    pub(super) fn execute_json_native(
        &mut self,
        op: &PreparedOp,
        native: JsonNative,
        args: &[RegisterWord],
    ) {
        let value = args
            .first()
            .copied()
            .map(Value::from_word)
            .unwrap_or(Value::NULL);

        let result: Result<Value, RuntimeError> = match native {
            JsonNative::Parse => {
                // simd-json overwrites its input when building the tape, so
                // never pass storage belonging to a Napalm String or Bytes.
                let input: Result<Vec<u8>, RuntimeError> = match self.heap.get(value) {
                    Some(HeapObject::String(text)) => Ok(text.as_ref().to_vec()),
                    Some(HeapObject::Bytes(bytes)) => Ok(bytes.as_slice().to_vec()),
                    Some(HeapObject::ByteBuffer(buffer)) => Ok(buffer.as_slice().to_vec()),
                    _ => Err(RuntimeError::new(
                        "TypeError: JSON.parse expects a String, Bytes, or ByteBuffer",
                    )),
                };
                input.and_then(|input| {
                    if input.len() > crate::napalm_bytes::MAX_BINARY_BYTES {
                        return Err(RuntimeError::new("RangeError: JSON input exceeds maximum size"));
                    }
                    // Root arguments in the bytecode frame before recursive
                    // conversion may allocate and trigger a moving collection.
                    self.mark_gc_point(op);
                    self.maybe_collect();
                    self.parse_json_response_body(input)
                })
            }
            JsonNative::Stringify => {
                // Serialization reads managed values but performs no managed
                // allocation. It handles shared children and rejects cycles.
                self.serialize_fetch_json(value).and_then(|encoded| {
                    if encoded.len() > crate::napalm_bytes::MAX_BINARY_BYTES {
                        return Err(RuntimeError::new("RangeError: JSON output exceeds maximum size"));
                    }
                    self.mark_gc_point(op);
                    self.maybe_collect();
                    Ok(self.heap.alloc(HeapObject::String(NapalmString::new(
                        Arc::<[u8]>::from(encoded),
                    ))))
                })
            }
        };

        match result {
            Ok(value) => unsafe {
                self.set_reg(op.a, value.word());
                self.advance(op);
            },
            Err(error) => self.fail_at(op, error.message),
        }
    }
}
