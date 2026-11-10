//! VM bindings for native Bytes and ByteBuffer values.
//!
//! The binary data contains no Napalm GC references. Every method that returns
//! a new heap object uses the ordinary GC-aware native call boundary. Mutable
//! operations refresh the MMTk external-memory accounting after changing capacity.

use super::*;
use crate::napalm_bytes::{number_to_byte, number_to_length, NapalmByteBuffer, NapalmBytes};

enum ByteReturn {
    Immediate(Value),
    Allocate(HeapObject),
}

#[inline]
fn byte_number(value: Value) -> Result<u8, String> {
    number_to_byte(value.as_num().ok_or("expected a Number")?).map_err(|e| e.to_string())
}

#[inline]
fn byte_count(value: Value) -> Result<usize, String> {
    let number = value.as_num().ok_or("expected a nonnegative integer Number")?;
    let count = number_to_length(number).map_err(|e| e.to_string())?;
    if count > crate::napalm_bytes::MAX_BINARY_BYTES {
        return Err(format!("binary size exceeds {} bytes", crate::napalm_bytes::MAX_BINARY_BYTES));
    }
    Ok(count)
}

#[inline]
fn byte_unsigned(value: Value, maximum: u64) -> Result<u64, String> {
    let n = value.as_num().ok_or("expected an unsigned integer Number")?;
    if !n.is_finite() || n.fract() != 0.0 || n < 0.0 || n > maximum as f64 {
        return Err(format!("expected an unsigned integer between 0 and {maximum}"));
    }
    Ok(n as u64)
}

#[inline]
fn byte_index(value: Value) -> Result<usize, String> {
    number_to_length(value.as_num().ok_or("expected a byte index Number")?).map_err(|e| e.to_string())
}

impl Interpreter<'_> {
    fn byte_data(&self, value: Value) -> Result<NapalmBytes, String> {
        match self.heap.get(value) {
            Some(HeapObject::Bytes(data)) => Ok(data.clone()),
            Some(HeapObject::ByteBuffer(buffer)) => Ok(NapalmBytes::from_slice(buffer.as_slice())),
            Some(HeapObject::String(text)) => Ok(NapalmBytes::from_slice(text.as_ref())),
            _ => Err("expected Bytes, ByteBuffer, or String".into()),
        }
    }

    fn immutable_bytes(&self, receiver: Value) -> Result<&NapalmBytes, String> {
        match self.heap.get(receiver) {
            Some(HeapObject::Bytes(bytes)) => Ok(bytes),
            _ => Err("receiver must be Bytes".into()),
        }
    }

    fn mutable_buffer(&mut self, receiver: Value) -> Result<&mut NapalmByteBuffer, String> {
        match self.heap.get_mut(receiver) {
            Some(HeapObject::ByteBuffer(bytes)) => Ok(bytes),
            _ => Err("receiver must be ByteBuffer".into()),
        }
    }

    #[inline]
    fn update_buffer_accounting(&mut self, receiver: Value) {
        if let Some(HeapObject::ByteBuffer(buffer)) = self.heap.get(receiver) {
            let capacity = buffer.capacity();
            self.heap.set_external_bytes(receiver, capacity);
        }
    }

    pub(super) fn execute_bytes_native(
        &mut self,
        op: &PreparedOp,
        native: BytesNative,
        receiver_word: RegisterWord,
        args: &[RegisterWord],
    ) {
        let receiver = Value::from_word(receiver_word);
        let arg = |index: usize| args.get(index).copied().map(Value::from_word).unwrap_or(Value::NULL);
        let result = (|| -> Result<ByteReturn, String> {
            use BytesNative::*;
            match native {
                BytesEmpty => Ok(ByteReturn::Allocate(HeapObject::Bytes(NapalmBytes::empty()))),
                BytesAlloc => {
                    let len = byte_count(arg(0))?;
                    Ok(ByteReturn::Allocate(HeapObject::Bytes(
                        NapalmBytes::from_inner(bytes::Bytes::from(vec![0; len])),
                    )))
                }
                BytesFromUtf8 => {
                    let text = match self.heap.get(arg(0)) {
                        Some(HeapObject::String(text)) => text,
                        _ => return Err("Bytes.fromUtf8 expects a String".into()),
                    };
                    Ok(ByteReturn::Allocate(HeapObject::Bytes(NapalmBytes::from_slice(text.as_ref()))))
                }
                BytesFromArray => {
                    let items = match self.heap.get(arg(0)) {
                        Some(HeapObject::List(items)) => items,
                        _ => return Err("Bytes.fromArray expects an Array".into()),
                    };
                    if items.len() > crate::napalm_bytes::MAX_BINARY_BYTES {
                        return Err("array exceeds maximum binary size".into());
                    }
                    let mut data = Vec::with_capacity(items.len());
                    for &value in items {
                        data.push(byte_number(value)?);
                    }
                    Ok(ByteReturn::Allocate(HeapObject::Bytes(
                        NapalmBytes::from_inner(bytes::Bytes::from(data)),
                    )))
                }
                BufferNew => {
                    let capacity = if args.is_empty() { 0 } else { byte_count(arg(0))? };
                    let value = NapalmByteBuffer::with_capacity(capacity).map_err(|e| e.to_string())?;
                    Ok(ByteReturn::Allocate(HeapObject::ByteBuffer(value)))
                }
                BytesLength | BytesIsEmpty | BytesIndex | BytesSlice | BytesClone
                | BytesToUtf8 | BytesToArray | BytesToMutable | BytesIndexOf
                | BytesToString | BytesIterate | BytesIteratorValue => {
                    let data = self.immutable_bytes(receiver)?;
                    match native {
                        BytesLength => Ok(ByteReturn::Immediate(Value::from_num(data.len() as f64))),
                        BytesIsEmpty => Ok(ByteReturn::Immediate(Value::from_bool(data.is_empty()))),
                        BytesIndex | BytesIteratorValue => {
                            let index = byte_index(arg(0))?;
                            let value = data.at(index).map_err(|e| e.to_string())?;
                            Ok(ByteReturn::Immediate(Value::from_num(value as f64)))
                        }
                        BytesSlice => {
                            let start = byte_index(arg(0))?;
                            let end = if args.len() == 2 { byte_index(arg(1))? } else { data.len() };
                            let result = data.slice(start, end).map_err(|e| e.to_string())?;
                            Ok(ByteReturn::Allocate(HeapObject::Bytes(result)))
                        }
                        BytesClone => Ok(ByteReturn::Allocate(HeapObject::Bytes(data.clone()))),
                        BytesToUtf8 => {
                            let text = data.to_utf8().map_err(|e| e.to_string())?;
                            Ok(ByteReturn::Allocate(HeapObject::String(
                                NapalmString::new(Arc::<[u8]>::from(text.into_bytes())),
                            )))
                        }
                        BytesToArray => {
                            // Match the existing maximum List size, avoiding unbounded
                            // multiplication of the byte count by sizeof(Value).
                            if data.len() > 16_777_216 {
                                return Err("Bytes.toArray exceeds the List size limit".into());
                            }
                            let items = data.as_slice().iter().map(|&b| Value::from_num(b as f64)).collect();
                            Ok(ByteReturn::Allocate(HeapObject::List(items)))
                        }
                        BytesToMutable => Ok(ByteReturn::Allocate(HeapObject::ByteBuffer(data.to_mutable()))),
                        BytesIndexOf => {
                            let pattern = self.byte_data(arg(0))?;
                            Ok(ByteReturn::Immediate(Value::from_num(
                                data.index_of(&pattern).map_or(-1.0, |idx| idx as f64),
                            )))
                        }
                        BytesToString => Ok(ByteReturn::Allocate(HeapObject::String(
                            NapalmString::new(Arc::<[u8]>::from(format!("<Bytes length={}>", data.len()).into_bytes())),
                        ))),
                        BytesIterate => {
                            let next = if arg(0).is_null() { 0 } else { byte_index(arg(0))?.saturating_add(1) };
                            Ok(ByteReturn::Immediate(if next < data.len() { Value::from_num(next as f64) } else { Value::FALSE }))
                        }
                        _ => unreachable!(),
                    }
                }
                BufferLength | BufferCapacity | BufferIsEmpty | BufferIndex | BufferIndexSet
                | BufferAppend | BufferReserve | BufferResize | BufferWriteU8
                | BufferWriteU16BE | BufferWriteU16LE | BufferWriteU32BE | BufferWriteU32LE
                | BufferTake | BufferPeek | BufferDiscard | BufferFreeze | BufferClear
                | BufferFill | BufferIndexOf | BufferToUtf8 | BufferToString
                | BufferIterate | BufferIteratorValue => {
                    match native {
                        BufferLength => {
                            let data = self.mutable_buffer(receiver)?;
                            Ok(ByteReturn::Immediate(Value::from_num(data.len() as f64)))
                        }
                        BufferCapacity => {
                            let data = self.mutable_buffer(receiver)?;
                            Ok(ByteReturn::Immediate(Value::from_num(data.capacity() as f64)))
                        }
                        BufferIsEmpty => {
                            let data = self.mutable_buffer(receiver)?;
                            Ok(ByteReturn::Immediate(Value::from_bool(data.is_empty())))
                        }
                        BufferIndex | BufferIteratorValue => {
                            let index = byte_index(arg(0))?;
                            let result = self.mutable_buffer(receiver)?.at(index).map_err(|e| e.to_string())?;
                            Ok(ByteReturn::Immediate(Value::from_num(result as f64)))
                        }
                        BufferIndexSet => {
                            let index = byte_index(arg(0))?;
                            let value = byte_number(arg(1))?;
                            self.mutable_buffer(receiver)?.set(index, value).map_err(|e| e.to_string())?;
                            Ok(ByteReturn::Immediate(Value::from_num(value as f64)))
                        }
                        BufferAppend => {
                            // Obtain an independent immutable view before mutably borrowing the destination.
                            // For ByteBuffer/String inputs this copies once; Bytes inputs cheaply clone.
                            let source = self.byte_data(arg(0))?;
                            self.mutable_buffer(receiver)?.append(&source).map_err(|e| e.to_string())?;
                            self.update_buffer_accounting(receiver);
                            Ok(ByteReturn::Immediate(receiver))
                        }
                        BufferReserve => {
                            let additional = byte_count(arg(0))?;
                            self.mutable_buffer(receiver)?.reserve(additional).map_err(|e| e.to_string())?;
                            self.update_buffer_accounting(receiver);
                            Ok(ByteReturn::Immediate(receiver))
                        }
                        BufferResize => {
                            let size = byte_count(arg(0))?;
                            self.mutable_buffer(receiver)?.resize(size).map_err(|e| e.to_string())?;
                            self.update_buffer_accounting(receiver);
                            Ok(ByteReturn::Immediate(receiver))
                        }
                        BufferWriteU8 | BufferWriteU16BE | BufferWriteU16LE
                        | BufferWriteU32BE | BufferWriteU32LE => {
                            let max = match native {
                                BufferWriteU8 => u8::MAX as u64,
                                BufferWriteU16BE | BufferWriteU16LE => u16::MAX as u64,
                                _ => u32::MAX as u64,
                            };
                            let value = byte_unsigned(arg(0), max)?;
                            let data = self.mutable_buffer(receiver)?;
                            match native {
                                BufferWriteU8 => data.write_u8(value as u8),
                                BufferWriteU16BE => data.write_u16_be(value as u16),
                                BufferWriteU16LE => data.write_u16_le(value as u16),
                                BufferWriteU32BE => data.write_u32_be(value as u32),
                                BufferWriteU32LE => data.write_u32_le(value as u32),
                                _ => unreachable!(),
                            }.map_err(|e| e.to_string())?;
                            self.update_buffer_accounting(receiver);
                            Ok(ByteReturn::Immediate(receiver))
                        }
                        BufferTake | BufferPeek | BufferDiscard => {
                            let size = byte_count(arg(0))?;
                            let buffer = self.mutable_buffer(receiver)?;
                            match native {
                                BufferTake => {
                                    let bytes = buffer.take(size);
                                    self.update_buffer_accounting(receiver);
                                    Ok(match bytes {
                                        Some(data) => ByteReturn::Allocate(HeapObject::Bytes(data)),
                                        None => ByteReturn::Immediate(Value::NULL),
                                    })
                                }
                                BufferPeek => Ok(match buffer.peek(size) {
                                    Some(data) => ByteReturn::Allocate(HeapObject::Bytes(data)),
                                    None => ByteReturn::Immediate(Value::NULL),
                                }),
                                BufferDiscard => {
                                    buffer.discard(size).map_err(|e| e.to_string())?;
                                    self.update_buffer_accounting(receiver);
                                    Ok(ByteReturn::Immediate(receiver))
                                }
                                _ => unreachable!(),
                            }
                        }
                        BufferFreeze => {
                            let data = self.mutable_buffer(receiver)?.freeze();
                            self.update_buffer_accounting(receiver);
                            Ok(ByteReturn::Allocate(HeapObject::Bytes(data)))
                        }
                        BufferClear => {
                            self.mutable_buffer(receiver)?.clear();
                            self.update_buffer_accounting(receiver);
                            Ok(ByteReturn::Immediate(receiver))
                        }
                        BufferFill => {
                            let byte = byte_number(arg(0))?;
                            self.mutable_buffer(receiver)?.fill(byte);
                            Ok(ByteReturn::Immediate(receiver))
                        }
                        BufferIndexOf => {
                            let pattern = self.byte_data(arg(0))?;
                            let position = self.mutable_buffer(receiver)?.index_of(&pattern);
                            Ok(ByteReturn::Immediate(Value::from_num(position.map_or(-1.0, |i| i as f64))))
                        }
                        BufferToUtf8 => {
                            let bytes = self.mutable_buffer(receiver)?.as_slice();
                            let text = std::str::from_utf8(bytes).map_err(|e| format!("invalid UTF-8: {e}"))?.to_owned();
                            Ok(ByteReturn::Allocate(HeapObject::String(
                                NapalmString::new(Arc::<[u8]>::from(text.into_bytes())),
                            )))
                        }
                        BufferToString => {
                            let len = self.mutable_buffer(receiver)?.len();
                            Ok(ByteReturn::Allocate(HeapObject::String(
                                NapalmString::new(Arc::<[u8]>::from(format!("<ByteBuffer length={len}>").into_bytes())),
                            )))
                        }
                        BufferIterate => {
                            let next = if arg(0).is_null() { 0 } else { byte_index(arg(0))?.saturating_add(1) };
                            let len = self.mutable_buffer(receiver)?.len();
                            Ok(ByteReturn::Immediate(if next < len { Value::from_num(next as f64) } else { Value::FALSE }))
                        }
                        _ => unreachable!(),
                    }
                }
            }
        })();
        match result {
            Ok(ByteReturn::Immediate(result)) => unsafe {
                self.set_reg(op.a, result.word());
                self.advance(op);
            },
            Ok(ByteReturn::Allocate(object)) => {
                self.mark_gc_point(op);
                self.maybe_collect();
                let value = self.heap.alloc(object);
                unsafe {
                    self.set_reg(op.a, value.word());
                    self.advance(op);
                }
            }
            Err(message) => self.fail_at(op, message),
        }
    }
}
