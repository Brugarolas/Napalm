//! Program-wide second-level cache for automatically memoized functions.
//!
//! Every VM keeps its private first-level `MemoTable`. A parent VM and the
//! `launch` workers spawned from it additionally share one `SharedMemoTables`
//! domain backed by lock-free papaya maps, so a result computed by any of those
//! VMs is reused by all of them instead of being recomputed per worker.
//!
//! Only VM-independent scalar words are published: null/bool/number Values and
//! raw Int32/F64 registers. Managed handles stay in the producing VM's private
//! table (and its GC roots), so the shared tier never needs rooting, barriers or
//! cross-heap translation.
//!
//! Soundness: auto-memoized functions are proven pure over scalar operators and
//! stock builtin dispatch. Workers install the parent's exact builtin manifest,
//! so a scalar result means the same thing in every VM of a domain. A VM whose
//! builtin dispatch is rebound after that manifest is sealed leaves the domain
//! permanently (see `Interpreter::note_builtin_rebinding`).

use super::*;
use papaya::HashMap as ConcurrentMap;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// Entries untouched for a full interval are discarded by the next sweep.
pub(super) const MEMO_MAINTENANCE_INTERVAL: Duration = Duration::from_secs(60);

#[inline]
pub(super) fn memo_entry_limit(policy: MemoizationKind) -> usize {
    if policy.is_recursive() { 65_536 } else { 4_096 }
}

/// Whether a memoized result word has the same meaning in every VM.
#[inline]
pub(super) fn is_vm_independent_memo_result(kind: RegisterKind, word: RegisterWord) -> bool {
    match kind {
        RegisterKind::Value => {
            let value = Value::from_word(word);
            !value.is_heap() && (value.is_null() || value.is_bool() || value.is_num())
        }
        RegisterKind::Int32 | RegisterKind::F64 => true,
        RegisterKind::RawPtr => false,
    }
}

struct SharedMemoEntry {
    value: RegisterWord,
    /// Latest maintenance generation that observed this entry. It is written
    /// only when it changes, so hot keys do not bounce a cache line per hit.
    generation: AtomicU32,
}

struct SharedMemoTable {
    policy: MemoizationKind,
    entries: ConcurrentMap<MemoKey, SharedMemoEntry, AesBuildHasher>,
    limit: usize,
    /// Admission counter kept outside papaya so the publish path never sums its
    /// striped length counters. Racing publishers may overshoot `limit` slightly.
    len: AtomicUsize,
    generation: AtomicU32,
    /// Earliest `SharedMemoTables::elapsed_nanos` at which a sweep may run.
    next_maintenance: AtomicU64,
}

impl SharedMemoTable {
    fn new(policy: MemoizationKind) -> Self {
        Self {
            policy,
            // Papaya allocates its table lazily, so unused functions cost no buckets.
            entries: ConcurrentMap::with_hasher(AesBuildHasher::new()),
            limit: memo_entry_limit(policy),
            len: AtomicUsize::new(0),
            generation: AtomicU32::new(0),
            next_maintenance: AtomicU64::new(MEMO_MAINTENANCE_INTERVAL.as_nanos() as u64),
        }
    }

    /// Drops entries that were not observed since the previous sweep. A single
    /// publisher wins the right to sweep for each interval; it returns whether
    /// the table has room afterwards.
    #[cold]
    fn sweep(&self, now: u64) -> bool {
        let due = self.next_maintenance.load(Ordering::Relaxed);
        if now < due
            || self
                .next_maintenance
                .compare_exchange(
                    due,
                    now.saturating_add(MEMO_MAINTENANCE_INTERVAL.as_nanos() as u64),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_err()
        {
            return false;
        }
        let live = self.generation.load(Ordering::Relaxed);
        let entries = self.entries.pin();
        entries.retain(|_, entry| entry.generation.load(Ordering::Relaxed) == live);
        // Survivors become stale unless they are observed again before the next sweep.
        self.generation.store(live.wrapping_add(1), Ordering::Relaxed);
        let len = entries.len();
        self.len.store(len, Ordering::Relaxed);
        len < self.limit
    }
}

/// Shared memo tables indexed by prepared function index. Prepared indexes are
/// positions in the immutable bytecode program, identical in every VM built from it.
pub(crate) struct SharedMemoTables {
    created: Instant,
    tables: Box<[Option<SharedMemoTable>]>,
}

impl SharedMemoTables {
    /// Builds one table per memoized function, or `None` when nothing is memoized.
    pub(super) fn new(policies: impl Iterator<Item = MemoizationKind>) -> Option<Self> {
        let tables: Box<[Option<SharedMemoTable>]> = policies
            .map(|policy| policy.enabled().then(|| SharedMemoTable::new(policy)))
            .collect();
        tables.iter().any(Option::is_some).then(|| Self {
            created: Instant::now(),
            tables,
        })
    }

    /// Whether this domain was built for a program with the same memo layout.
    pub(super) fn matches(&self, policies: impl Iterator<Item = MemoizationKind>) -> bool {
        let mut tables = self.tables.iter();
        for policy in policies {
            let same = match tables.next() {
                Some(Some(table)) => table.policy == policy,
                Some(None) => !policy.enabled(),
                None => false,
            };
            if !same {
                return false;
            }
        }
        tables.next().is_none()
    }

    #[inline]
    fn elapsed_nanos(&self) -> u64 {
        self.created.elapsed().as_nanos() as u64
    }

    #[inline]
    pub(super) fn get(&self, function: u32, key: &MemoKey) -> Option<RegisterWord> {
        let table = self.tables.get(function as usize)?.as_ref()?;
        // The admission counter can lag publication: a racing publisher can
        // observe the entry before its winner increments len. Only the table
        // itself determines whether a key is present.
        let entries = table.entries.pin();
        let entry = entries.get(key)?;
        let live = table.generation.load(Ordering::Relaxed);
        if entry.generation.load(Ordering::Relaxed) != live {
            entry.generation.store(live, Ordering::Relaxed);
        }
        Some(entry.value)
    }

    /// Publishes a VM-independent scalar result. Existing entries win: every VM
    /// computes the same value for a key, so a lost race needs no reconciliation.
    pub(super) fn publish(&self, function: u32, key: MemoKey, value: RegisterWord) {
        let Some(Some(table)) = self.tables.get(function as usize) else { return; };
        if table.len.load(Ordering::Relaxed) >= table.limit && !table.sweep(self.elapsed_nanos()) {
            return;
        }
        let entry = SharedMemoEntry {
            value,
            generation: AtomicU32::new(table.generation.load(Ordering::Relaxed)),
        };
        if table.entries.pin().try_insert(key, entry).is_ok() {
            table.len.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    pub(super) fn len(&self, function: u32) -> usize {
        self.tables
            .get(function as usize)
            .and_then(Option::as_ref)
            .map_or(0, |table| table.entries.pin().len())
    }

    #[cfg(test)]
    fn force_sweep(&self, function: u32) -> bool {
        let table = self.tables[function as usize].as_ref().expect("memoized function");
        table.next_maintenance.store(0, Ordering::Relaxed);
        table.sweep(self.elapsed_nanos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u64) -> MemoKey {
        MemoKey { words: [n, 0, 0, 0] }
    }

    #[test]
    fn shared_memo_tables_only_exist_for_memoized_functions() {
        assert!(SharedMemoTables::new([MemoizationKind::None; 3].into_iter()).is_none());
        let policies = [MemoizationKind::None, MemoizationKind::Recursive, MemoizationKind::Pure];
        let shared = SharedMemoTables::new(policies.into_iter()).unwrap();
        assert!(shared.matches(policies.into_iter()));
        assert!(!shared.matches([MemoizationKind::None, MemoizationKind::Pure, MemoizationKind::Pure].into_iter()));
        assert!(!shared.matches(policies[..2].iter().copied()));
        shared.publish(0, key(1), RegisterWord(7));
        assert_eq!(shared.get(0, &key(1)), None, "unmemoized slots never cache");
        assert_eq!(shared.get(9, &key(1)), None);
    }

    #[test]
    fn shared_memo_reads_published_entries_before_admission_count_catches_up() {
        let shared = SharedMemoTables::new([MemoizationKind::Recursive].into_iter()).unwrap();
        let table = shared.tables[0].as_ref().unwrap();
        // Pause the winning publisher between try_insert and len.fetch_add.
        assert!(table.entries.pin().try_insert(key(0), SharedMemoEntry {
            value: RegisterWord(42), generation: AtomicU32::new(0),
        }).is_ok());
        shared.publish(0, key(0), RegisterWord(99));
        assert_eq!(table.len.load(Ordering::Relaxed), 0);
        assert_eq!(shared.get(0, &key(0)), Some(RegisterWord(42)));
        table.len.fetch_add(1, Ordering::Relaxed);
        assert_eq!(shared.get(0, &key(0)), Some(RegisterWord(42)));
    }

    #[test]
    fn shared_memo_first_publisher_wins_and_is_visible_across_threads() {
        let shared = Arc::new(SharedMemoTables::new([MemoizationKind::Recursive].into_iter()).unwrap());
        let workers: Vec<_> = (0..4u64)
            .map(|worker| {
                let shared = shared.clone();
                std::thread::spawn(move || {
                    for n in 0..1_000u64 {
                        shared.publish(0, key(n), RegisterWord(n * 3));
                        assert_eq!(shared.get(0, &key(n)), Some(RegisterWord(n * 3)), "worker {worker}");
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(shared.len(0), 1_000);
        shared.publish(0, key(5), RegisterWord(99));
        assert_eq!(shared.get(0, &key(5)), Some(RegisterWord(15)));
    }

    #[test]
    fn shared_memo_respects_limit_and_sweeps_unobserved_entries() {
        let shared = SharedMemoTables::new([MemoizationKind::Pure].into_iter()).unwrap();
        let limit = memo_entry_limit(MemoizationKind::Pure) as u64;
        for n in 0..limit + 16 {
            shared.publish(0, key(n), RegisterWord(n));
        }
        assert_eq!(shared.len(0) as u64, limit, "full tables reject new keys between sweeps");
        assert_eq!(shared.get(0, &key(limit)), None);

        // The first sweep only advances the generation; afterwards only entries
        // observed since then survive the next sweep.
        assert!(!shared.force_sweep(0));
        assert_eq!(shared.get(0, &key(3)), Some(RegisterWord(3)));
        assert!(shared.force_sweep(0));
        assert_eq!(shared.len(0), 1);
        assert_eq!(shared.get(0, &key(3)), Some(RegisterWord(3)));
        shared.publish(0, key(limit), RegisterWord(limit));
        assert_eq!(shared.get(0, &key(limit)), Some(RegisterWord(limit)));
    }

    #[test]
    fn shared_memo_only_accepts_vm_independent_results() {
        assert!(is_vm_independent_memo_result(RegisterKind::Value, Value::from_num(2.5).word()));
        assert!(is_vm_independent_memo_result(RegisterKind::Value, Value::NULL.word()));
        assert!(is_vm_independent_memo_result(RegisterKind::Value, Value::from_bool(true).word()));
        assert!(is_vm_independent_memo_result(RegisterKind::Int32, RegisterWord(u64::MAX)));
        assert!(is_vm_independent_memo_result(RegisterKind::F64, RegisterWord(1.5f64.to_bits())));
        // Heap handles are meaningful only inside the VM that allocated them.
        assert!(!is_vm_independent_memo_result(RegisterKind::Value, Value::from_heap(3).word()));
        assert!(!is_vm_independent_memo_result(RegisterKind::RawPtr, RegisterWord(8)));
    }
}
