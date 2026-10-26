// src/helpers.rs
#![allow(clippy::needless_range_loop)]
use std::cmp::Eq;
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};

/// ------------------------
/// TLS PRNG with no explicit OS calls or allocations in the generation path.
/// TLS setup itself remains platform-dependent.
/// ------------------------
mod tls_prng {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static THREAD_SEQUENCE: AtomicUsize = AtomicUsize::new(0);
    const STEP: u64 = 0xA076_1D64_78BD_642F;

    #[inline]
    fn splitmix64_stateless(mut z: u64) -> u64 {
        z ^= z >> 30;
        z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z ^= z >> 27;
        z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn rdcycle_best_effort() -> u64 {
        let mut v: u64 = 0;
        #[cfg(target_arch = "x86_64")]
        unsafe {
            v ^= core::arch::x86_64::_rdtsc();
        }
        #[cfg(all(target_arch = "x86", not(target_arch = "x86_64")))]
        unsafe {
            v ^= core::arch::x86::_rdtsc() as u64;
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cntvct: u64;
            core::arch::asm!("mrs {0}, cntvct_el0", out(reg) cntvct, options(nomem, nostack, preserves_flags));
            v ^= cntvct;
        }
        v
    }

    #[inline]
    fn addr_noise() -> u64 {
        let stack_addr = {
            let local = 0u8;
            &local as *const u8 as usize as u64
        };
        let code_addr = fast_two_u64 as *const() as usize as u64;
        stack_addr.rotate_left(17) ^ code_addr.rotate_left(31)
    }

    thread_local! {
        static TL_SEED: Cell<u64> = const { Cell::new(0) };
        static TL_INITIALIZED: Cell<bool> = const { Cell::new(false) };
    }

    #[cold]
    #[inline(never)]
    fn initial_seed() -> u64 {
        // The counter distinguishes initialization events within this process.
        // It is not an entropy source or a guarantee of unique mixed seeds.
        let ticket = THREAD_SEQUENCE.fetch_add(1, Ordering::Relaxed) as u64;
        let serial = splitmix64_stateless(ticket.wrapping_add(0x9E37_79B9_7F4A_7C15));
        splitmix64_stateless(rdcycle_best_effort() ^ addr_noise() ^ serial)
    }

    #[inline]
    fn output(s: u64) -> u64 {
        let product = (s as u128) * ((s ^ 0xE703_7ED1_A0B4_28DB) as u128);
        (product as u64) ^ ((product >> 64) as u64)
    }

    /// Return two outputs from the per-thread 64-bit-state generator.
    #[inline]
    pub fn fast_two_u64() -> (u64, u64) {
        TL_SEED.with(|seed| {
            TL_INITIALIZED.with(|initialized| {
                if !initialized.get() {
                    seed.set(initial_seed());
                    initialized.set(true);
                }
            });
            // Zero is an ordinary state. Update TLS only once for the pair.
            let first = seed.get().wrapping_add(STEP);
            let second = first.wrapping_add(STEP);
            seed.set(second);
            (output(first), output(second))
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn zero_state_does_not_reseed() {
            TL_SEED.with(|seed| TL_INITIALIZED.with(|initialized| {
                let previous_seed = seed.replace(0);
                let previous_initialized = initialized.replace(true);
                let actual = fast_two_u64();
                let after = seed.get();
                seed.set(previous_seed);
                initialized.set(previous_initialized);
                assert_eq!(actual, (output(STEP), output(STEP.wrapping_add(STEP))));
                assert_eq!(after, STEP.wrapping_add(STEP));
            }));
        }
    }

}
use tls_prng::fast_two_u64;

/// ------------------------
/// Non-cryptographic AES-backed hashers.
/// Backends intentionally produce different hashes. Hash values are not stable
/// across architectures, target features, dependency versions, or this module's
/// revisions. Never persist them as identifiers.
/// AES is selected at compile time: enable the target AES feature for native
/// builds. Generic builds use the seeded aHash fallback, without runtime dispatch.
/// ------------------------
#[cfg(all(target_arch = "x86_64", target_feature = "aes"))]
mod hasher_impl {
    use super::*;
    use core::arch::x86_64::{
        __m128i, _mm_aesenc_si128, _mm_aesenclast_si128, _mm_loadu_si128, _mm_set_epi64x,
        _mm_storeu_si128, _mm_xor_si128,
    };

    pub struct AesHasher {
        hash: [__m128i; 4],
        next_lane: usize,
        used_lanes: usize,
        buffer: [u8; 16],
        len: usize,
    }

    impl AesHasher {
        #[inline]
        pub fn with_keys(k0: u64, k1: u64) -> Self {
            // The first lane holds this map's seed.
            unsafe {
                Self {
                    hash: [
                        _mm_set_epi64x(k0 as i64, k1 as i64),
                        _mm_set_epi64x((k0 ^ 0x243F_6A88_85A3_08D3) as i64, (k1 ^ 0x1319_8A2E_0370_7344) as i64),
                        _mm_set_epi64x((k0 ^ 0xA409_3822_299F_31D0) as i64, (k1 ^ 0x082E_FA98_EC4E_6C89) as i64),
                        _mm_set_epi64x((k0 ^ 0x4528_21E6_38D0_1377) as i64, (k1 ^ 0xBE54_66CF_34E9_0C6C) as i64),
                    ],
                    next_lane: 0,
                    used_lanes: 0,
                    buffer: [0; 16],
                    len: 0,
                }
            }
        }

        #[inline]
        unsafe fn round(&mut self, block_ptr: *const u8) {
            // SAFETY: The caller supplies at least 16 readable bytes.
            let block = unsafe { _mm_loadu_si128(block_ptr.cast()) };
            let lane = self.next_lane;
            self.hash[lane] = unsafe { _mm_aesenc_si128(self.hash[lane], block) };
            self.next_lane = (lane + 1) & 3;
            self.used_lanes = (self.used_lanes + 1).min(4);
        }

        #[inline]
        unsafe fn stripe(&mut self, ptr: *const u8) {
            debug_assert_eq!(self.next_lane, 0);
            // SAFETY: The caller supplies 64 readable bytes. Each lane is
            // independent, allowing the processor to overlap AES instructions.
            unsafe {
                self.hash[0] = _mm_aesenc_si128(self.hash[0], _mm_loadu_si128(ptr.cast()));
                self.hash[1] = _mm_aesenc_si128(self.hash[1], _mm_loadu_si128(ptr.add(16).cast()));
                self.hash[2] = _mm_aesenc_si128(self.hash[2], _mm_loadu_si128(ptr.add(32).cast()));
                self.hash[3] = _mm_aesenc_si128(self.hash[3], _mm_loadu_si128(ptr.add(48).cast()));
            }
            self.used_lanes = 4;
        }

        #[inline]
        fn fold128(x: __m128i) -> u64 {
            unsafe {
                let mut tmp = core::mem::MaybeUninit::<[u8; 16]>::uninit();
                _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, x);
                let bytes = tmp.assume_init();
                let a = u64::from_ne_bytes(bytes[0..8].try_into().unwrap());
                let b = u64::from_ne_bytes(bytes[8..16].try_into().unwrap());
                a ^ b
            }
        }
    }

    impl AesHasher {
        #[inline]
        fn write_small<const N: usize>(&mut self, bytes: &[u8; N]) {
            let pos = self.len & 15;
            if N < 16 - pos {
                self.buffer[pos..pos + N].copy_from_slice(bytes);
                self.len = self.len.wrapping_add(N);
            } else {
                self.write(bytes);
            }
        }
    }

    impl Hasher for AesHasher {
        #[inline]
        fn write(&mut self, mut bytes: &[u8]) {
            let buf_pos = self.len & 15;
            self.len = self.len.wrapping_add(bytes.len());
            if buf_pos != 0 {
                let take = (16 - buf_pos).min(bytes.len());
                self.buffer[buf_pos..buf_pos + take].copy_from_slice(&bytes[..take]);
                if buf_pos + take < 16 {
                    return;
                }
                bytes = &bytes[take..];
                // SAFETY: The buffer contains a complete initialized block.
                unsafe {
                    self.round(self.buffer.as_ptr());
                }
            }
            // Align the logical block number, not the input address.
            while self.next_lane != 0 && bytes.len() >= 16 {
                unsafe {
                    self.round(bytes.as_ptr());
                }
                bytes = &bytes[16..];
            }
            while bytes.len() >= 64 {
                unsafe {
                    self.stripe(bytes.as_ptr());
                }
                bytes = &bytes[64..];
            }
            while bytes.len() >= 16 {
                unsafe {
                    self.round(bytes.as_ptr());
                }
                bytes = &bytes[16..];
            }
            self.buffer[..bytes.len()].copy_from_slice(bytes);
        }

        #[inline]
        fn write_u64(&mut self, i: u64) {
            self.write_small(&i.to_ne_bytes());
        }
        #[inline]
        fn write_u32(&mut self, i: u32) {
            self.write_small(&i.to_ne_bytes());
        }
        #[inline]
        fn write_u8(&mut self, i: u8) {
            self.write_small(&[i]);
        }

        #[inline]
        fn finish(&self) -> u64 {
            unsafe {
                let mut cur = self.hash[0];
                // Merge only lanes that have absorbed complete blocks.
                for lane in 1..self.used_lanes {
                    cur = _mm_aesenc_si128(cur, self.hash[lane]);
                }
                let rem = self.len & 15;
                if rem != 0 {
                    let mut tail = [0u8; 16];
                    tail[..rem].copy_from_slice(&self.buffer[..rem]);
                    let block = _mm_loadu_si128(tail.as_ptr() as *const __m128i);
                    cur = _mm_aesenc_si128(cur, block);
                }

                // length mixing
                let len = self.len as u64;
                let len_vec = _mm_set_epi64x(!(len) as i64, len as i64);
                cur = _mm_xor_si128(cur, len_vec);

                // Finalization
                let final_key =
                    _mm_set_epi64x(0x8E51EF21FABB4522u64 as i64, 0xE43D7A0656954B6Cu64 as i64);
                // Two full rounds diffuse the final input across all 16 bytes;
                // the last substitution adds nonlinearity before folding.
                cur = _mm_aesenc_si128(cur, final_key);
                cur = _mm_aesenc_si128(cur, final_key);
                cur = _mm_aesenclast_si128(cur, final_key);

                Self::fold128(cur)
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod hasher_impl {
    use super::*;
    use core::arch::aarch64::*;

    #[inline]
    unsafe fn aes_round_arm(state: uint8x16_t, rk: uint8x16_t) -> uint8x16_t {
        // ARM absorbs the input before substitution and column mixing.
        unsafe { vaesmcq_u8(vaeseq_u8(state, rk)) }
    }
    #[inline]
    unsafe fn aes_last_round_arm(state: uint8x16_t, rk: uint8x16_t) -> uint8x16_t {
        // Final round without MixColumns
        unsafe { vaeseq_u8(state, rk) }
    }

    pub struct AesHasher {
        hash: [uint8x16_t; 4],
        next_lane: usize,
        used_lanes: usize,
        buffer: [u8; 16],
        len: usize,
    }

    impl AesHasher {
        #[inline]
        pub fn with_keys(k0: u64, k1: u64) -> Self {
            // lanes: [lo bytes of k1][lo bytes of k0]
            let mut key = [0u8; 16];
            key[..8].copy_from_slice(&k1.to_ne_bytes());
            key[8..].copy_from_slice(&k0.to_ne_bytes());
            unsafe {
                Self {
                    hash: {
                        let base = vld1q_u8(key.as_ptr());
                        [base,
                         veorq_u8(base, vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(0x1319_8A2E_0370_7344), vcreate_u64(0x243F_6A88_85A3_08D3)))),
                         veorq_u8(base, vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(0x082E_FA98_EC4E_6C89), vcreate_u64(0xA409_3822_299F_31D0)))),
                         veorq_u8(base, vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(0xBE54_66CF_34E9_0C6C), vcreate_u64(0x4528_21E6_38D0_1377))))]
                    },
                    next_lane: 0,
                    used_lanes: 0,
                    buffer: [0; 16],
                    len: 0,
                }
            }
        }
    }

    impl AesHasher {
        #[inline]
        unsafe fn round(&mut self, ptr: *const u8) {
            let lane = self.next_lane;
            // SAFETY: The caller supplies at least 16 readable bytes.
            self.hash[lane] = unsafe { aes_round_arm(self.hash[lane], vld1q_u8(ptr)) };
            self.next_lane = (lane + 1) & 3;
            self.used_lanes = (self.used_lanes + 1).min(4);
        }

        #[inline]
        unsafe fn stripe(&mut self, ptr: *const u8) {
            debug_assert_eq!(self.next_lane, 0);
            // SAFETY: The caller supplies at least 64 readable bytes.
            unsafe {
                self.hash[0] = aes_round_arm(self.hash[0], vld1q_u8(ptr));
                self.hash[1] = aes_round_arm(self.hash[1], vld1q_u8(ptr.add(16)));
                self.hash[2] = aes_round_arm(self.hash[2], vld1q_u8(ptr.add(32)));
                self.hash[3] = aes_round_arm(self.hash[3], vld1q_u8(ptr.add(48)));
            }
            self.used_lanes = 4;
        }
    }

    impl AesHasher {
        #[inline]
        fn write_small<const N: usize>(&mut self, bytes: &[u8; N]) {
            let pos = self.len & 15;
            if N < 16 - pos {
                self.buffer[pos..pos + N].copy_from_slice(bytes);
                self.len = self.len.wrapping_add(N);
            } else {
                self.write(bytes);
            }
        }
    }

    impl Hasher for AesHasher {
        #[inline]
        fn write(&mut self, mut bytes: &[u8]) {
            let buf_pos = self.len & 15;
            self.len = self.len.wrapping_add(bytes.len());
            if buf_pos != 0 {
                let take = (16 - buf_pos).min(bytes.len());
                self.buffer[buf_pos..buf_pos + take].copy_from_slice(&bytes[..take]);
                if buf_pos + take < 16 {
                    return;
                }
                bytes = &bytes[take..];
                // SAFETY: The buffer contains a complete initialized block.
                unsafe {
                    self.round(self.buffer.as_ptr());
                }
            }
            // Align the logical block number, not the input address.
            while self.next_lane != 0 && bytes.len() >= 16 {
                unsafe {
                    self.round(bytes.as_ptr());
                }
                bytes = &bytes[16..];
            }
            while bytes.len() >= 64 {
                unsafe {
                    self.stripe(bytes.as_ptr());
                }
                bytes = &bytes[64..];
            }
            while bytes.len() >= 16 {
                unsafe {
                    self.round(bytes.as_ptr());
                }
                bytes = &bytes[16..];
            }
            self.buffer[..bytes.len()].copy_from_slice(bytes);
        }

        #[inline]
        fn write_u64(&mut self, i: u64) {
            self.write_small(&i.to_ne_bytes());
        }
        #[inline]
        fn write_u32(&mut self, i: u32) {
            self.write_small(&i.to_ne_bytes());
        }
        #[inline]
        fn write_u8(&mut self, i: u8) {
            self.write_small(&[i]);
        }

        #[inline]
        fn finish(&self) -> u64 {
            unsafe {
                let mut cur = self.hash[0];
                for lane in 1..self.used_lanes {
                    cur = aes_round_arm(cur, self.hash[lane]);
                }
                let rem = self.len & 15;
                if rem != 0 {
                    let mut tail = [0u8; 16];
                    tail[..rem].copy_from_slice(&self.buffer[..rem]);
                    let block = vld1q_u8(tail.as_ptr());
                    cur = aes_round_arm(cur, block);
                }

                // length mixing
                let mut len_block = [0u8; 16];
                len_block[..8].copy_from_slice(&(self.len as u64).to_ne_bytes());
                len_block[8..].copy_from_slice(&(!(self.len as u64)).to_ne_bytes());
                let len_vec = vld1q_u8(len_block.as_ptr());
                cur = veorq_u8(cur, len_vec);

                // Finalization
                let final_key: [u8; 16] = [
                    0x8E, 0x51, 0xEF, 0x21, 0xFA, 0xBB, 0x45, 0x22, 0xE4, 0x3D, 0x7A, 0x06, 0x56,
                    0x95, 0x4B, 0x6C,
                ];
                let fk = vld1q_u8(final_key.as_ptr());
                cur = aes_round_arm(cur, fk);
                cur = aes_round_arm(cur, fk);
                cur = aes_last_round_arm(cur, fk);

                let mut out = [0u8; 16];
                vst1q_u8(out.as_mut_ptr(), cur);
                let a = u64::from_ne_bytes(out[0..8].try_into().unwrap());
                let b = u64::from_ne_bytes(out[8..16].try_into().unwrap());
                a ^ b
            }
        }
    }
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "aes"),
    all(target_arch = "aarch64", target_feature = "aes")
)))]
mod hasher_impl {
    use super::*;
    use ahash::AHasher;

    #[derive(Default)]
    pub struct AesHasher(pub AHasher);

    impl AesHasher {
        #[inline]
        pub fn with_keys(k0: u64, k1: u64) -> Self {
            Self(ahash::RandomState::with_seeds(k0, k1, 0, 0).build_hasher())
        }
    }

    impl Hasher for AesHasher {
        #[inline]
        fn write(&mut self, bytes: &[u8]) {
            self.0.write(bytes)
        }
        #[inline]
        fn write_u64(&mut self, i: u64) {
            self.0.write_u64(i)
        }
        #[inline]
        fn write_u32(&mut self, i: u32) {
            self.0.write_u32(i)
        }
        #[inline]
        fn write_u8(&mut self, i: u8) {
            self.0.write_u8(i)
        }
        #[inline]
        fn finish(&self) -> u64 {
            self.0.finish()
        }
    }
}

use hasher_impl::AesHasher;

/// ------------------------
/// BuildHasher with per-map keys generated by the thread-local PRNG.
/// ------------------------
#[derive(Clone, Copy, Debug)]
pub struct AesBuildHasher {
    k0: u64,
    k1: u64,
}
impl AesBuildHasher {
    #[inline]
    pub fn new() -> Self {
        let (k0, k1) = fast_two_u64();
        Self { k0, k1 }
    }
    #[inline]
    pub fn from_seed(k0: u64, k1: u64) -> Self {
        Self { k0, k1 }
    }
}
impl Default for AesBuildHasher {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
impl BuildHasher for AesBuildHasher {
    type Hasher = AesHasher;
    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        AesHasher::with_keys(self.k0, self.k1)
    }
}

/// ------------------------
/// Collections (hashbrown + papaya) wired to AesBuildHasher
/// ------------------------

// Concurrent map (papaya)
#[derive(Debug)]
pub struct ConcurrentHashMap<K: Eq + Hash, V>(pub papaya::HashMap<K, V, AesBuildHasher>);
impl<K: Eq + Hash, V> ConcurrentHashMap<K, V> {
    #[inline]
    pub fn new() -> Self {
        Self(papaya::HashMap::with_hasher(AesBuildHasher::new()))
    }
}
impl<K: Eq + Hash, V> Default for ConcurrentHashMap<K, V> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
impl<K: Eq + Hash, V> Deref for ConcurrentHashMap<K, V> {
    type Target = papaya::HashMap<K, V, AesBuildHasher>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<K: Eq + Hash, V> DerefMut for ConcurrentHashMap<K, V> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// hashbrown::HashMap
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashMap<K: Eq + Hash, V>(pub hashbrown::HashMap<K, V, AesBuildHasher>);
impl<K: Eq + Hash, V> HashMap<K, V> {
    #[inline]
    pub fn new() -> Self {
        Self(hashbrown::HashMap::with_hasher(AesBuildHasher::new()))
    }
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self(hashbrown::HashMap::with_capacity_and_hasher(
            cap,
            AesBuildHasher::new(),
        ))
    }

    #[inline]
    pub fn get_internals(&self) -> &hashbrown::HashMap<K, V, AesBuildHasher> {
        &self.0
    }
}
impl<K: Eq + Hash, V> Default for HashMap<K, V> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
impl<K: Eq + Hash, V> Deref for HashMap<K, V> {
    type Target = hashbrown::HashMap<K, V, AesBuildHasher>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<K: Eq + Hash, V> DerefMut for HashMap<K, V> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl<K: Eq + Hash, V> FromIterator<(K, V)> for HashMap<K, V> {
    #[inline]
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut m = hashbrown::HashMap::with_hasher(AesBuildHasher::new());
        m.extend(iter);
        Self(m)
    }
}

impl<K: Eq + Hash, V> IntoIterator for HashMap<K, V> {
    type Item = (K, V);
    type IntoIter = hashbrown::hash_map::IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

// hashbrown::HashSet
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashSet<T: Eq + Hash>(pub hashbrown::HashSet<T, AesBuildHasher>);
impl<T: Eq + Hash> HashSet<T> {
    #[inline]
    pub fn new() -> Self {
        Self(hashbrown::HashSet::with_hasher(AesBuildHasher::new()))
    }
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self(hashbrown::HashSet::with_capacity_and_hasher(
            cap,
            AesBuildHasher::new(),
        ))
    }

    #[inline]
    pub fn get_internals(&self) -> &hashbrown::HashSet<T, AesBuildHasher> {
        &self.0
    }
}
impl<T: Eq + Hash> Default for HashSet<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
impl<T: Eq + Hash> Deref for HashSet<T> {
    type Target = hashbrown::HashSet<T, AesBuildHasher>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T: Eq + Hash> DerefMut for HashSet<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl<T: Eq + Hash> FromIterator<T> for HashSet<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut s = hashbrown::HashSet::with_hasher(AesBuildHasher::new());
        s.extend(iter);
        Self(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_values_use_equal_hashes() {
        let builder = AesBuildHasher::from_seed(123, 456);
        let mut a = builder.build_hasher();
        let mut b = builder.build_hasher();
        (17u64, "example", vec![1u8, 2, 3]).hash(&mut a);
        (17u64, "example", vec![1u8, 2, 3]).hash(&mut b);
        assert_eq!(a.finish(), b.finish());
        assert_eq!(a.finish(), a.finish());
    }

    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "aes"),
        all(target_arch = "aarch64", target_feature = "aes")
    ))]
    #[test]
    fn native_streaming_and_unaligned_inputs() {
        let data: Vec<u8> = (0..1100).map(|i| (i * 37 + i / 7) as u8).collect();
        for offset in 0..16 {
            for len in [0, 1, 7, 8, 15, 16, 17, 31, 32, 47, 48, 63, 64, 65, 127, 128, 129, 1024] {
                let input = &data[offset..offset + len];
                let mut whole = AesHasher::with_keys(123, 456);
                whole.write(input);
                let expected = whole.finish();
                for split in 0..=len {
                    let mut fragmented = AesHasher::with_keys(123, 456);
                    fragmented.write(&input[..split]);
                    let checkpoint = fragmented.finish();
                    assert_eq!(checkpoint, fragmented.finish());
                    fragmented.write(&[]);
                    fragmented.write(&input[split..]);
                    assert_eq!(fragmented.finish(), expected, "offset={offset}, len={len}, split={split}");
                }
                for width in [1, 3, 7, 16, 17, 63, 64, 65] {
                    let mut fragmented = AesHasher::with_keys(123, 456);
                    for part in input.chunks(width) {
                        fragmented.write(part);
                    }
                    assert_eq!(fragmented.finish(), expected);
                }
            }
        }
    }

    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "aes"),
        all(target_arch = "aarch64", target_feature = "aes")
    ))]
    #[test]
    fn native_integer_writes_match_bytes() {
        for prefix in 0..80 {
            let mut typed = AesHasher::with_keys(123, 456);
            let mut raw = AesHasher::with_keys(123, 456);
            let bytes = vec![42; prefix];
            typed.write(&bytes);
            raw.write(&bytes);
            typed.write_u64(0x0123_4567_89AB_CDEF);
            raw.write(&0x0123_4567_89AB_CDEFu64.to_ne_bytes());
            typed.write_u32(0x1020_3040);
            raw.write(&0x1020_3040u32.to_ne_bytes());
            typed.write_u8(255);
            raw.write(&[255]);
            assert_eq!(typed.finish(), raw.finish(), "prefix={prefix}");
        }
    }

    #[test]
    fn sequential_integer_distribution_regression() {
        let mut high = [0usize; 128];
        let mut low = [0usize; 256];
        for value in 0..65536u64 {
            let mut hasher = AesHasher::with_keys(123, 456);
            hasher.write_u64(value);
            let hash = hasher.finish();
            high[(hash >> 57) as usize] += 1;
            low[(hash & 255) as usize] += 1;
        }
        assert!(high.iter().all(|&n| n > 256 && n < 768));
        assert!(low.iter().all(|&n| n > 128 && n < 384));
    }

    #[test]
    fn collections_round_trip() {
        let mut map = HashMap::with_capacity(4096);
        let mut set = HashSet::with_capacity(4096);
        for value in 0..4096u64 {
            map.insert(value, value ^ 0x55AA);
            set.insert(value);
        }
        for value in 0..4096u64 {
            assert_eq!(map.get(&value), Some(&(value ^ 0x55AA)));
            assert!(set.contains(&value));
        }
    }
}
