//! Shared collection policy for compiler, JIT and runtime tables.
//!
//! Keep the interpreter's per-map keyed hasher, including its portable fallback.
//! Use `default()` or `with_capacity_and_hasher(capacity, Default::default())`
//! so constructors cannot silently select hashbrown's default hasher.

pub type HashMap<K, V> = hashbrown::HashMap<K, V, crate::napalm_interpreter::runtime_hash::AesBuildHasher>;
pub type HashSet<K> = hashbrown::HashSet<K, crate::napalm_interpreter::runtime_hash::AesBuildHasher>;

/// Concurrent language collections use the same AES hashing policy.
pub type ConcurrentMap<K, V> = papaya::HashMap<K, V, crate::napalm_interpreter::runtime_hash::AesBuildHasher>;
pub type ConcurrentSet<K> = papaya::HashSet<K, crate::napalm_interpreter::runtime_hash::AesBuildHasher>;
