//! Process-wide string interning for names and source literals.
//!
//! Interned: identifiers, property/class/interface/method names, source string
//! literals and object keys that enter a hidden-class shape. Not interned:
//! runtime-built strings (concatenation, fetch bodies, loop products and other
//! temporaries). They keep their plain `Arc<[u8]>` payload and are compared by
//! hash + bytes, exactly as before.
//!
//! Storage and deduplication use `string_interner` with the `BucketBackend`,
//! whose string buffers are fixed-capacity and never reallocated. The interner
//! lives in a `static` and is never dropped, cleared or shrunk, so a resolved
//! `&str` is valid for `'static`.
//!
//! The interner itself sits behind a lock, but resolving a [`Name`] does not
//! take it: every interned string is also published, with its precomputed
//! runtime content hash, into an append-only chunked table that readers index
//! with one Acquire load. The program-wide shape table and the concurrent
//! workers therefore share one symbol space without contention.
//!
//! Interned strings are never freed. That is why only bounded string families
//! (source text and shape keys, whose table is capped) are interned.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::num::NonZeroU32;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};
use std::sync::LazyLock;

use string_interner::backend::BucketBackend;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, Symbol};

const CHUNK_BITS: u32 = 12;
const CHUNK: usize = 1 << CHUNK_BITS;
const CHUNK_MASK: usize = CHUNK - 1;
const MAX_CHUNKS: usize = 1 << 12;
/// 16M distinct names: far above any program's source plus its shape keys.
const MAX_NAMES: usize = CHUNK * MAX_CHUNKS;

#[derive(Clone, Copy)]
struct Entry {
    text: &'static str,
    hash: u64,
}

struct Table {
    chunks: [AtomicPtr<Entry>; MAX_CHUNKS],
    len: AtomicU32,
}

static TABLE: Table = Table {
    chunks: [const { AtomicPtr::new(null_mut()) }; MAX_CHUNKS],
    len: AtomicU32::new(0),
};

/// Deduplication uses the runtime's AesHasher (x86 AES / ARM AES / aHash fallback).
type Interner = StringInterner<BucketBackend<SymbolU32>, crate::napalm_interpreter::runtime_hash::AesBuildHasher>;

static INTERNER: LazyLock<parking_lot::RwLock<Interner>> =
    LazyLock::new(|| parking_lot::RwLock::new(Interner::new()));

/// An interned string. Copy, 4 bytes, `Option<Name>` is also 4 bytes.
///
/// Equality and hashing use the symbol id; ordering uses the string content, so
/// sorted output does not depend on interning order.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Name(NonZeroU32);

impl Name {
    /// Intern `text`, returning the canonical name for its content.
    pub fn new(text: &str) -> Self {
        if let Some(symbol) = INTERNER.read().get(text) {
            return Self::from_symbol(symbol);
        }
        let mut interner = INTERNER.write();
        let before = interner.len();
        let symbol = interner.get_or_intern(text);
        if interner.len() != before {
            // Published while still holding the write lock: ids are dense and
            // a Name only escapes after its entry is visible.
            let resolved = interner.resolve(symbol).expect("just interned");
            // SAFETY: see the module docs. BucketBackend never moves interned
            // bytes and the static interner is never dropped or shrunk.
            let text: &'static str = unsafe { &*(resolved as *const str) };
            publish(symbol.to_usize(), Entry { text, hash: crate::napalm_interpreter::fast_hash_bytes(text.as_bytes()) });
        }
        Self::from_symbol(symbol)
    }

    /// Intern UTF-8 bytes. Non-UTF-8 runtime strings are never interned.
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        std::str::from_utf8(bytes).ok().map(Self::new)
    }

    /// The name for `text` if it is already interned; never interns.
    #[inline]
    pub fn lookup(text: &str) -> Option<Self> {
        INTERNER.read().get(text).map(Self::from_symbol)
    }

    #[inline]
    fn from_symbol(symbol: SymbolU32) -> Self {
        Self(NonZeroU32::new(symbol.to_usize() as u32 + 1).expect("symbol index fits u32"))
    }

    #[inline(always)]
    fn entry(self) -> Entry {
        let index = (self.0.get() - 1) as usize;
        let chunk = TABLE.chunks[index >> CHUNK_BITS].load(Ordering::Acquire);
        debug_assert!(!chunk.is_null() && index < TABLE.len.load(Ordering::Acquire) as usize);
        // SAFETY: a Name exists only after its entry was published.
        unsafe { *chunk.add(index & CHUNK_MASK) }
    }

    /// Lock-free resolution.
    #[inline(always)]
    pub fn as_str(self) -> &'static str { self.entry().text }

    #[inline(always)]
    pub fn as_bytes(self) -> &'static [u8] { self.entry().text.as_bytes() }

    /// The runtime content hash (`fast_hash_bytes`) of this string, computed
    /// once at interning time. Identical to hashing the bytes again.
    #[inline(always)]
    pub fn content_hash(self) -> u64 { self.entry().hash }

    /// Dense id, usable as a vector index or a compact JIT immediate.
    #[inline(always)]
    pub fn id(self) -> u32 { self.0.get() }
    /// Number of names interned so far in this process.
    pub fn count() -> usize { TABLE.len.load(Ordering::Acquire) as usize }
}

fn publish(index: usize, entry: Entry) {
    assert!(index < MAX_NAMES, "string interner capacity exceeded");
    let slot = &TABLE.chunks[index >> CHUNK_BITS];
    let mut chunk = slot.load(Ordering::Relaxed);
    if chunk.is_null() {
        let storage = Box::<[Entry]>::new_uninit_slice(CHUNK);
        chunk = Box::into_raw(storage).cast::<Entry>();
        slot.store(chunk, Ordering::Release);
    }
    unsafe { chunk.add(index & CHUNK_MASK).write(entry) };
    TABLE.len.store(index as u32 + 1, Ordering::Release);
}

impl Hash for Name {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) { state.write_u32(self.0.get()) }
}

impl PartialOrd for Name {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}

impl Ord for Name {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self == other { std::cmp::Ordering::Equal } else { self.as_str().cmp(other.as_str()) }
    }
}

impl std::ops::Deref for Name {
    type Target = str;
    #[inline(always)]
    fn deref(&self) -> &str { self.as_str() }
}

impl AsRef<str> for Name {
    #[inline(always)]
    fn as_ref(&self) -> &str { self.as_str() }
}

impl AsRef<[u8]> for Name {
    #[inline(always)]
    fn as_ref(&self) -> &[u8] { self.as_bytes() }
}

impl Default for Name {
    fn default() -> Self { Self::new("") }
}

impl From<&str> for Name {
    #[inline]
    fn from(text: &str) -> Self { Self::new(text) }
}

impl From<&String> for Name {
    #[inline]
    fn from(text: &String) -> Self { Self::new(text) }
}

impl From<String> for Name {
    #[inline]
    fn from(text: String) -> Self { Self::new(&text) }
}

impl From<compact_str::CompactString> for Name {
    #[inline]
    fn from(text: compact_str::CompactString) -> Self { Self::new(&text) }
}

impl From<Name> for String {
    #[inline]
    fn from(name: Name) -> Self { name.as_str().to_owned() }
}

impl PartialEq<str> for Name {
    #[inline]
    fn eq(&self, other: &str) -> bool { self.as_str() == other }
}

impl PartialEq<&str> for Name {
    #[inline]
    fn eq(&self, other: &&str) -> bool { self.as_str() == *other }
}

impl PartialEq<String> for Name {
    #[inline]
    fn eq(&self, other: &String) -> bool { self.as_str() == other }
}

impl PartialEq<Name> for str {
    #[inline]
    fn eq(&self, other: &Name) -> bool { self == other.as_str() }
}

impl PartialEq<Name> for &str {
    #[inline]
    fn eq(&self, other: &Name) -> bool { *self == other.as_str() }
}

impl PartialEq<Name> for String {
    #[inline]
    fn eq(&self, other: &Name) -> bool { self == other.as_str() }
}

impl fmt::Display for Name {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.as_str()) }
}

impl fmt::Debug for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Debug::fmt(self.as_str(), f) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_is_canonical_and_resolves_without_the_lock() {
        let a = Name::new("interner_test_alpha");
        let b = Name::from("interner_test_alpha".to_owned());
        assert_eq!(a, b);
        assert_eq!(a.as_str(), "interner_test_alpha");
        assert_ne!(a, Name::new("interner_test_beta"));
        assert_eq!(Name::lookup("interner_test_alpha"), Some(a));
        assert_eq!(Name::lookup("interner_test_never_interned"), None);
        assert_eq!(std::mem::size_of::<Option<Name>>(), 4);
        // Resolving while another thread holds the writer lock must not block.
        let guard = INTERNER.write();
        assert_eq!(a.as_str(), "interner_test_alpha");
        drop(guard);
    }

    #[test]
    fn content_hash_matches_runtime_hash_and_order_is_lexical() {
        let name = Name::new("interner_test_hash");
        assert_eq!(name.content_hash(), crate::napalm_interpreter::fast_hash_bytes(b"interner_test_hash"));
        let mut names = vec![Name::new("interner_z"), Name::new("interner_a"), Name::new("interner_m")];
        names.sort();
        assert_eq!(names.iter().map(|n| n.as_str()).collect::<Vec<_>>(), ["interner_a", "interner_m", "interner_z"]);
        assert_eq!(Name::from_bytes(&[0xff, 0xfe]), None);
    }

    #[test]
    fn concurrent_interning_agrees_on_one_symbol_per_string() {
        let threads: Vec<_> = (0..8).map(|t| std::thread::spawn(move || {
            (0..2000).map(|i| (i, Name::new(&format!("interner_concurrent_{}", (i * 7 + t) % 500)))).collect::<Vec<_>>()
        })).collect();
        let mut seen = std::collections::HashMap::new();
        for thread in threads {
            for (_, name) in thread.join().unwrap() {
                let previous = seen.insert(name.as_str().to_owned(), name);
                assert!(previous.is_none_or(|p| p == name));
            }
        }
        assert_eq!(seen.len(), 500);
    }
}
