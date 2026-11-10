//! Binary storage for Napalm's `Bytes` and `ByteBuffer` native classes.
//!
//! This module is independent of the VM's NaN-boxed `Value` and MMTk heap.
//! The language bindings are implemented in `napalm_bytes_vm.rs`.
//!
//! `NapalmBytes` is immutable and cheaply cloneable; its slices share storage.
//! `NapalmByteBuffer` is mutable and supports zero-copy extraction with `take`.
//! A `peek` deliberately copies: publishing an immutable alias to live mutable
//! storage would violate the immutability guarantee.

use bytes::{BufMut, Bytes, BytesMut};

/// Protects the runtime and editor hosts against accidental unbounded binary allocation.
pub const MAX_BINARY_BYTES: usize = 256 * 1024 * 1024;
use std::error::Error;
use std::fmt;
use std::str::Utf8Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ByteError {
    InvalidByte(f64),
    InvalidLength(f64),
    InvalidIndex { index: usize, length: usize },
    InvalidRange { start: usize, end: usize, length: usize },
    CapacityOverflow,
    UnexpectedEof { requested: usize, available: usize },
    InvalidUtf8(Utf8Error),
}

impl fmt::Display for ByteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidByte(value) => write!(f, "byte must be an integer from 0 through 255, got {value}"),
            Self::InvalidLength(value) => write!(f, "length must be a nonnegative safe integer, got {value}"),
            Self::InvalidIndex { index, length } => write!(f, "byte index {index} out of bounds for length {length}"),
            Self::InvalidRange { start, end, length } => write!(f, "invalid byte range {start}..{end} for length {length}"),
            Self::CapacityOverflow => write!(f, "binary buffer capacity overflow"),
            Self::UnexpectedEof { requested, available } => {
                write!(f, "requested {requested} bytes but only {available} available")
            }
            Self::InvalidUtf8(error) => write!(f, "invalid UTF-8: {error}"),
        }
    }
}

impl Error for ByteError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidUtf8(error) => Some(error),
            _ => None,
        }
    }
}

/// Validate a Napalm `Number` used as a byte. Do not silently wrap or truncate.
#[inline]
pub fn number_to_byte(value: f64) -> Result<u8, ByteError> {
    if value.is_finite() && value.fract() == 0.0 && (0.0..=255.0).contains(&value) {
        Ok(value as u8)
    } else {
        Err(ByteError::InvalidByte(value))
    }
}

/// Validate a Napalm `Number` used as a byte count or index.
///
/// IEEE-754 values above 2^53-1 cannot be represented exactly and must be
/// rejected even on a 64-bit host.
#[inline]
pub fn number_to_length(value: f64) -> Result<usize, ByteError> {
    const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    if !value.is_finite()
        || value < 0.0
        || value.fract() != 0.0
        || value > MAX_SAFE_INTEGER
        || value >= usize::MAX as f64
    {
        return Err(ByteError::InvalidLength(value));
    }
    Ok(value as usize)
}

#[inline]
fn validate_range(start: usize, end: usize, length: usize) -> Result<(), ByteError> {
    if start <= end && end <= length {
        Ok(())
    } else {
        Err(ByteError::InvalidRange { start, end, length })
    }
}

#[inline]
fn checked_capacity(len: usize, additional: usize) -> Result<(), ByteError> {
    if len.checked_add(additional).is_some_and(|size| size <= MAX_BINARY_BYTES && size <= isize::MAX as usize) {
        Ok(())
    } else {
        Err(ByteError::CapacityOverflow)
    }
}

/// Locate the first occurrence of `needle` in `haystack`.
/// The empty needle matches at byte position zero.
#[inline]
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    match needle {
        [] => Some(0),
        [byte] => haystack.iter().position(|value| value == byte),
        _ => haystack.windows(needle.len()).position(|window| window == needle),
    }
}

/// Immutable bytes; clones and slices share their backing allocation.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct NapalmBytes(Bytes);

impl fmt::Debug for NapalmBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NapalmBytes").field("length", &self.len()).finish()
    }
}

impl NapalmBytes {
    #[inline]
    pub const fn empty() -> Self {
        Self(Bytes::new())
    }

    #[inline]
    pub fn from_utf8(value: &str) -> Self {
        Self(Bytes::copy_from_slice(value.as_bytes()))
    }

    /// Transfer ownership of an existing String allocation when possible.
    #[inline]
    pub fn from_string(value: String) -> Self {
        Self(Bytes::from(value))
    }

    #[inline]
    pub fn from_slice(value: &[u8]) -> Self {
        Self(Bytes::copy_from_slice(value))
    }

    pub fn from_numbers(values: &[f64]) -> Result<Self, ByteError> {
        checked_capacity(0, values.len())?;
        let mut bytes = Vec::with_capacity(values.len());
        for &value in values {
            bytes.push(number_to_byte(value)?);
        }
        Ok(Self(Bytes::from(bytes)))
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.0.as_ref()
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<u8> {
        self.0.get(index).copied()
    }

    pub fn at(&self, index: usize) -> Result<u8, ByteError> {
        self.get(index).ok_or(ByteError::InvalidIndex { index, length: self.len() })
    }

    /// O(1) in normal cases; the returned object retains the backing allocation.
    pub fn slice(&self, start: usize, end: usize) -> Result<Self, ByteError> {
        validate_range(start, end, self.len())?;
        Ok(Self(self.0.slice(start..end)))
    }

    pub fn to_utf8(&self) -> Result<String, ByteError> {
        std::str::from_utf8(self.as_slice())
            .map(str::to_owned)
            .map_err(ByteError::InvalidUtf8)
    }

    #[inline]
    pub fn to_vec(&self) -> Vec<u8> {
        self.as_slice().to_vec()
    }

    #[inline]
    pub fn index_of(&self, needle: &Self) -> Option<usize> {
        find_bytes(self.as_slice(), needle.as_slice())
    }

    /// Makes an independent mutable copy. The source must remain immutable.
    pub fn to_mutable(&self) -> NapalmByteBuffer {
        NapalmByteBuffer(BytesMut::from(self.as_slice()))
    }

    #[inline]
    pub fn into_inner(self) -> Bytes {
        self.0
    }

    #[inline]
    pub fn from_inner(bytes: Bytes) -> Self {
        Self(bytes)
    }
}

/// Growable binary buffer backed by `bytes::BytesMut`.
#[derive(Default)]
pub struct NapalmByteBuffer(BytesMut);

impl fmt::Debug for NapalmByteBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NapalmByteBuffer")
            .field("length", &self.len())
            .field("capacity", &self.capacity())
            .finish()
    }
}

impl NapalmByteBuffer {
    #[inline]
    pub fn new() -> Self {
        Self(BytesMut::new())
    }

    pub fn with_capacity(capacity: usize) -> Result<Self, ByteError> {
        checked_capacity(0, capacity)?;
        Ok(Self(BytesMut::with_capacity(capacity)))
    }

    /// Consumes an immutable buffer and reuses its backing storage when unique.
    pub fn from_owned_bytes(bytes: NapalmBytes) -> Self {
        match bytes.into_inner().try_into_mut() {
            Ok(mutable) => Self(mutable),
            Err(immutable) => Self(BytesMut::from(immutable.as_ref())),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.0.as_ref()
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<u8> {
        self.0.get(index).copied()
    }

    pub fn at(&self, index: usize) -> Result<u8, ByteError> {
        self.get(index).ok_or(ByteError::InvalidIndex { index, length: self.len() })
    }

    pub fn set(&mut self, index: usize, value: u8) -> Result<(), ByteError> {
        let len = self.len();
        let slot = self.0.get_mut(index).ok_or(ByteError::InvalidIndex { index, length: len })?;
        *slot = value;
        Ok(())
    }

    pub fn reserve(&mut self, additional: usize) -> Result<(), ByteError> {
        checked_capacity(self.len(), additional)?;
        self.0.reserve(additional);
        Ok(())
    }

    pub fn resize(&mut self, size: usize) -> Result<(), ByteError> {
        checked_capacity(0, size)?;
        self.0.resize(size, 0);
        Ok(())
    }

    pub fn append(&mut self, data: &NapalmBytes) -> Result<(), ByteError> {
        self.append_slice(data.as_slice())
    }

    pub fn append_slice(&mut self, data: &[u8]) -> Result<(), ByteError> {
        checked_capacity(self.len(), data.len())?;
        self.0.extend_from_slice(data);
        Ok(())
    }

    pub fn append_utf8(&mut self, text: &str) -> Result<(), ByteError> {
        self.append_slice(text.as_bytes())
    }

    pub fn write_u8(&mut self, value: u8) -> Result<(), ByteError> {
        self.reserve(1)?;
        self.0.put_u8(value);
        Ok(())
    }

    pub fn write_u16_be(&mut self, value: u16) -> Result<(), ByteError> {
        self.reserve(2)?;
        self.0.put_u16(value);
        Ok(())
    }

    pub fn write_u16_le(&mut self, value: u16) -> Result<(), ByteError> {
        self.reserve(2)?;
        self.0.put_u16_le(value);
        Ok(())
    }

    pub fn write_u32_be(&mut self, value: u32) -> Result<(), ByteError> {
        self.reserve(4)?;
        self.0.put_u32(value);
        Ok(())
    }

    pub fn write_u32_le(&mut self, value: u32) -> Result<(), ByteError> {
        self.reserve(4)?;
        self.0.put_u32_le(value);
        Ok(())
    }

    pub fn index_of(&self, needle: &NapalmBytes) -> Option<usize> {
        find_bytes(self.as_slice(), needle.as_slice())
    }

    /// Returns a stable immutable copy; does not expose mutable storage.
    pub fn peek(&self, count: usize) -> Option<NapalmBytes> {
        self.as_slice().get(..count).map(NapalmBytes::from_slice)
    }

    /// Consumes a prefix without copying its bytes. Returns None if incomplete.
    pub fn take(&mut self, count: usize) -> Option<NapalmBytes> {
        (count <= self.len()).then(|| NapalmBytes(self.0.split_to(count).freeze()))
    }

    pub fn take_exact(&mut self, count: usize) -> Result<NapalmBytes, ByteError> {
        let available = self.len();
        self.take(count).ok_or(ByteError::UnexpectedEof { requested: count, available })
    }

    /// Discards a prefix without shifting the remaining data.
    pub fn discard(&mut self, count: usize) -> Result<(), ByteError> {
        let available = self.len();
        if count > available {
            return Err(ByteError::UnexpectedEof { requested: count, available });
        }
        drop(self.0.split_to(count));
        Ok(())
    }

    /// Detaches the entire readable buffer as an immutable value and empties it.
    pub fn freeze(&mut self) -> NapalmBytes {
        NapalmBytes(self.0.split().freeze())
    }

    #[inline]
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Zero-fill all initialized bytes; never touches spare capacity.
    #[inline]
    pub fn fill(&mut self, value: u8) {
        self.0.fill(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_numeric_inputs() {
        for valid in [0.0, 17.0, 255.0] {
            assert_eq!(number_to_byte(valid).unwrap(), valid as u8);
        }
        for invalid in [-1.0, 256.0, 3.25, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(number_to_byte(invalid), Err(ByteError::InvalidByte(_))));
        }
        for invalid in [-1.0, 1.1, f64::NAN, f64::INFINITY, 9_007_199_254_740_992.0] {
            assert!(matches!(number_to_length(invalid), Err(ByteError::InvalidLength(_))));
        }
        assert_eq!(number_to_length(42.0).unwrap(), 42);
    }

    #[test]
    fn immutable_bytes_are_shareable_and_slices_are_stable() {
        let original = NapalmBytes::from_utf8("hello Napalm");
        let part = original.slice(6, 12).unwrap();
        let cloned = part.clone();
        assert_eq!(part.to_utf8().unwrap(), "Napalm");
        assert_eq!(cloned, part);
        assert!(original.slice(6, 13).is_err());
        assert!(original.slice(5, 4).is_err());
        assert_eq!(original.index_of(&NapalmBytes::from_utf8("Nap")), Some(6));
        assert_eq!(original.index_of(&NapalmBytes::empty()), Some(0));
    }

    #[test]
    fn to_mutable_never_modifies_immutable_aliases() {
        let original = NapalmBytes::from_utf8("hello");
        let alias = original.clone();
        let mut mutable = original.to_mutable();
        mutable.set(0, b'H').unwrap();
        assert_eq!(mutable.freeze().to_utf8().unwrap(), "Hello");
        assert_eq!(original.to_utf8().unwrap(), "hello");
        assert_eq!(alias.to_utf8().unwrap(), "hello");
    }

    #[test]
    fn buffer_take_handles_arbitrary_chunk_boundaries() {
        let mut buf = NapalmByteBuffer::with_capacity(4).unwrap();
        buf.append_utf8("Content-").unwrap();
        assert!(buf.take(12).is_none());
        buf.append_utf8("Length: 2\r\n\r\n{}").unwrap();
        let marker = NapalmBytes::from_utf8("\r\n\r\n");
        let header_end = buf.index_of(&marker).unwrap() + marker.len();
        let header = buf.take_exact(header_end).unwrap();
        assert_eq!(header.to_utf8().unwrap(), "Content-Length: 2\r\n\r\n");
        assert_eq!(buf.take_exact(2).unwrap().to_utf8().unwrap(), "{}");
        assert!(buf.is_empty());
    }

    #[test]
    fn peek_is_a_snapshot_not_a_mutable_alias() {
        let mut buf = NapalmByteBuffer::new();
        buf.append_utf8("hello").unwrap();
        let peeked = buf.peek(5).unwrap();
        buf.set(0, b'H').unwrap();
        assert_eq!(peeked.to_utf8().unwrap(), "hello");
        assert_eq!(buf.freeze().to_utf8().unwrap(), "Hello");
    }

    #[test]
    fn extraction_survives_discard_and_growth() {
        let mut buf = NapalmByteBuffer::new();
        buf.append_utf8("abcdefgh").unwrap();
        let first = buf.take_exact(3).unwrap();
        buf.discard(2).unwrap();
        buf.append_utf8("1234567890").unwrap();
        assert_eq!(first.to_utf8().unwrap(), "abc");
        assert_eq!(buf.freeze().to_utf8().unwrap(), "fgh1234567890");
    }

    #[test]
    fn integer_writes_have_explicit_byte_order() {
        let mut buf = NapalmByteBuffer::new();
        buf.write_u8(0xaa).unwrap();
        buf.write_u16_be(0x1234).unwrap();
        buf.write_u16_le(0x1234).unwrap();
        buf.write_u32_be(0x12345678).unwrap();
        buf.write_u32_le(0x12345678).unwrap();
        assert_eq!(buf.freeze().to_vec(), [0xaa, 0x12, 0x34, 0x34, 0x12,
            0x12, 0x34, 0x56, 0x78, 0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn invalid_utf8_is_rejected() {
        let raw = NapalmBytes::from_slice(&[0xff, 0xfe]);
        assert!(matches!(raw.to_utf8(), Err(ByteError::InvalidUtf8(_))));
        assert_eq!(NapalmBytes::from_utf8("😀").len(), 4);
    }

    #[test]
    fn invalid_operations_preserve_buffer() {
        let mut buf = NapalmByteBuffer::new();
        buf.append_utf8("a").unwrap();
        assert!(buf.set(1, 99).is_err());
        assert!(buf.discard(2).is_err());
        assert!(buf.take_exact(2).is_err());
        assert_eq!(buf.freeze().to_utf8().unwrap(), "a");
    }
}
