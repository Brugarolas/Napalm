//! Shared collection policy for compiler, JIT and runtime tables.
//!
//! Keep the interpreter's per-map keyed hasher, including its portable fallback.
//! Use `default()` or `with_capacity_and_hasher(capacity, Default::default())`
//! so constructors cannot silently select hashbrown's default hasher.

use std::hash::Hash;

pub type HashMap<K, V> = hashbrown::HashMap<K, V, crate::napalm_interpreter::runtime_hash::AesBuildHasher>;
pub type HashSet<K> = hashbrown::HashSet<K, crate::napalm_interpreter::runtime_hash::AesBuildHasher>;

/// Concurrent language collections use the same AES hashing policy.
pub type ConcurrentMap<K, V> = papaya::HashMap<K, V, crate::napalm_interpreter::runtime_hash::AesBuildHasher>;
pub type ConcurrentSet<K> = papaya::HashSet<K, crate::napalm_interpreter::runtime_hash::AesBuildHasher>;

/// Inline, SIMD-probed map for compiler/JIT-internal tables whose population is
/// usually small (epoch-local lowering caches, frame-slot tables, fixed helper
/// tables). Up to `N` entries live inline with no allocation; beyond that it
/// spills transparently to a heap hashbrown table.
///
/// Policy: keys must be engine-internal (ids, slots, SSA values), never
/// user-controlled language data, because the spilled table uses FxHash rather
/// than the keyed AES hasher above. Iteration order is insertion order while
/// inline and arbitrary after a spill, so callers must not depend on it.
///
/// Size is roughly `N * (1 + size_of::<K>() + size_of::<V>())`, so keep these
/// in short-lived or per-compilation state, not in per-object runtime data.
pub type SmallMapN<const N: usize, K, V> = small_map::ASmallMap<N, K, V>;
/// Default inline capacity: two SSE2 control groups.
pub type SmallMap<K, V> = SmallMapN<64, K, V>;
/// One SSE2 control group; for tables that are almost always tiny.
pub type SmallMap16<K, V> = SmallMapN<16, K, V>;
/// Set counterpart of [`SmallMapN`] with `HashSet`-compatible signatures.
#[derive(Clone)]
pub struct SmallSetN<const N: usize, K> {
    map: SmallMapN<N, K, ()>,
}

impl<const N: usize, K> Default for SmallSetN<N, K> {
    #[inline]
    fn default() -> Self {
        Self { map: SmallMapN::default() }
    }
}

pub type SmallSet<K> = SmallSetN<64, K>;
pub type SmallSet16<K> = SmallSetN<16, K>;

impl<const N: usize, K: Eq + Hash> SmallSetN<N, K> {
    #[inline]
    pub fn new() -> Self {
        Self { map: SmallMapN::new() }
    }

    /// Returns `true` when the value was not present (same as `HashSet::insert`).
    #[inline]
    pub fn insert(&mut self, value: K) -> bool {
        self.map.insert(value, ()).is_none()
    }

    #[inline]
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        Q: ?Sized + Hash + small_map::Equivalent<K>,
    {
        self.map.contains_key(value)
    }

    /// Returns `true` when the value was present (same as `HashSet::remove`).
    #[inline]
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        Q: ?Sized + Hash + small_map::Equivalent<K>,
    {
        self.map.remove(value).is_some()
    }

    #[inline]
    pub fn retain(&mut self, mut keep: impl FnMut(&K) -> bool) {
        self.map.retain(|key, _| keep(key));
    }
}

impl<const N: usize, K> SmallSetN<N, K> {
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.map.clear();
    }

    #[inline]
    pub fn iter(&self) -> small_map::Keys<'_, N, K, ()> {
        self.map.keys()
    }
}

impl<const N: usize, K: Eq + Hash> FromIterator<K> for SmallSetN<N, K> {
    fn from_iter<I: IntoIterator<Item = K>>(iter: I) -> Self {
        let mut set = Self::new();
        set.extend(iter);
        set
    }
}

impl<const N: usize, K: Eq + Hash> Extend<K> for SmallSetN<N, K> {
    fn extend<I: IntoIterator<Item = K>>(&mut self, iter: I) {
        for value in iter {
            self.insert(value);
        }
    }
}

impl<const N: usize, K: std::fmt::Debug> std::fmt::Debug for SmallSetN<N, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.map.keys()).finish()
    }
}
