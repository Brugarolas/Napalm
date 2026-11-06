//! VmAdaptativeMutex: per-operation synchronization of values reachable from
//! `shared` collections.
//!
//! * Every heap handle owns a lock word (`TraceHeapEntry::sync`), so locks are
//!   per object, live in the process-wide handle table and are visible to every
//!   worker VM without any allocation.
//! * `SYNC_SHARED` marks objects reachable from a shared Papaya table. Marking is
//!   contagious and deep: when a bytecode operation touches a shared object, all
//!   of its heap operands and its result become shared together with everything
//!   they reach. Hence every object another thread can reach is marked before
//!   that thread can observe it, and unmarked objects are always thread-local.
//! * Shared objects publish `TraceHeapKind::Invalid` to the native tiers, so all
//!   native guards fail. Runtime steps of a trace that touch shared operands
//!   leave the trace. Shared state is therefore mutated only by the interpreter,
//!   which holds the operands' locks for exactly one bytecode operation.
//! * Acquisition spins adaptively for at most `SPIN_BUDGET` (1 ms). A concurrent
//!   worker then parks the *task*, not the OS thread: the operation has not run,
//!   its VM is set aside and the worker executes other jobs. Releasing the lock
//!   wakes the task, which re-executes the whole operation.
//! * Operands are locked in handle order and no lock outlives its operation, so
//!   acquisition cannot deadlock.
//! * Papaya tables (`SYNC_TABLE`) are contagious but never locked: their own
//!   operations are already lock-free and must stay parallel.

use super::*;
use std::time::{Duration, Instant};

pub(super) const SYNC_SHARED: u32 = 1 << 0;
pub(super) const SYNC_LOCKED: u32 = 1 << 1;
pub(super) const SYNC_WAITERS: u32 = 1 << 2;
pub(super) const SYNC_TABLE: u32 = 1 << 3;

/// Spin budget before a contended operation parks its task.
pub(super) const SPIN_BUDGET: Duration = Duration::from_millis(1);

pub(super) const OPERAND_A: u8 = 1 << 0;
pub(super) const OPERAND_B: u8 = 1 << 1;
pub(super) const OPERAND_C: u8 = 1 << 2;
/// Packed call registers (receiver/arguments of Invoke/CallPrimitive/...).
pub(super) const OPERAND_PACKED: u8 = 1 << 3;
/// a/b/c plus at most 17 packed call registers.
const MAX_OPERANDS: usize = 20;

/// Tasks parked after exhausting the spin budget (diagnostics and tests).
pub(crate) static PARKED_TASKS: AtomicU64 = AtomicU64::new(0);

/// Re-schedules one parked task. Runs on the releasing thread.
pub(crate) type LockWaker = Box<dyn FnOnce() + Send>;

fn waiters() -> &'static parking_lot::Mutex<FastMap<u32, Vec<LockWaker>>> {
    static WAITERS: OnceLock<parking_lot::Mutex<FastMap<u32, Vec<LockWaker>>>> = OnceLock::new();
    WAITERS.get_or_init(|| parking_lot::Mutex::new(FastMap::default()))
}

pub(crate) struct VmAdaptativeMutex;

impl VmAdaptativeMutex {
    #[inline]
    pub(super) fn try_lock(word: &AtomicU32) -> bool {
        let mut current = word.load(Ordering::Relaxed);
        loop {
            if current & SYNC_LOCKED != 0 {
                return false;
            }
            match word.compare_exchange_weak(
                current,
                current | SYNC_LOCKED,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    /// Exponential-backoff spin bounded by `SPIN_BUDGET`. Never sleeps: once the
    /// backoff saturates it only yields the core to a possibly preempted holder.
    pub(super) fn lock_spinning(word: &AtomicU32) -> bool {
        if Self::try_lock(word) {
            return true;
        }
        let start = Instant::now();
        let mut backoff = 1u32;
        loop {
            for _ in 0..backoff {
                std::hint::spin_loop();
            }
            if word.load(Ordering::Relaxed) & SYNC_LOCKED == 0 && Self::try_lock(word) {
                return true;
            }
            if backoff < 128 {
                backoff <<= 1;
                continue;
            }
            if start.elapsed() >= SPIN_BUDGET {
                return false;
            }
            std::thread::yield_now();
        }
    }

    pub(super) fn unlock(handle: u32, word: &AtomicU32) {
        let previous = word.fetch_and(!(SYNC_LOCKED | SYNC_WAITERS), Ordering::Release);
        if previous & SYNC_WAITERS != 0 {
            let parked = waiters().lock().remove(&handle);
            for wake in parked.into_iter().flatten() {
                wake();
            }
        }
    }

    /// Register a parked task. `Err` returns the waker when the lock was
    /// released meanwhile; the caller must retry instead of parking.
    ///
    /// The registry lock orders this against `unlock`: either the releaser
    /// cleared LOCKED before our `fetch_or` (we observe it and retry), or it
    /// observes WAITERS and drains the registry after our push.
    pub(crate) fn park(handle: u32, word: &AtomicU32, waker: LockWaker) -> Result<(), LockWaker> {
        let mut registry = waiters().lock();
        let previous = word.fetch_or(SYNC_WAITERS, Ordering::AcqRel);
        if previous & SYNC_LOCKED == 0 {
            return Err(waker);
        }
        registry.entry(handle).or_default().push(waker);
        Ok(())
    }
}

/// Operand registers whose objects a heap-touching opcode may read or write.
/// Pure register/arithmetic/control opcodes return 0 and pay nothing.
pub(super) fn shared_operand_mask(opcode: Opcode) -> u8 {
    use crate::napalm_bytecode::{operand_roles, OperandRole};
    use Opcode::*;
    let packed = match opcode {
        Invoke | InvokeSuper | CallPrimitive | CallClosure => OPERAND_PACKED,
        // Variable-length superinstructions carry packed call registers.
        _ if opcode.is_specialized() && opcode.fixed_word_len().is_none() => OPERAND_PACKED,
        _ => 0,
    };
    // Superinstructions fuse arbitrary heap work: lock every register operand
    // conservatively rather than risk an unlocked shared access.
    let touches_heap = packed != 0
        || opcode.is_specialized()
        || matches!(
            opcode,
            LoadField | StoreField | LoadCell | StoreCell
                | ListLen | ListGet | ListSet | ListAppend | ListInsert | ListRemoveAt
                | MapLen | MapGet | MapSet | MapRemove | MapLookupOrInsert
                | ListSpread | MapSpread | DestructureCheck | DestructureGet | DestructureRest
                | InvokeSpread | BindInterface | StringConcat
                | LoadProperty | StoreProperty | StorePropertyIfPresent
        );
    if !touches_heap {
        return 0;
    }
    let mut mask = packed;
    for (index, role) in operand_roles(opcode).iter().enumerate() {
        if matches!(role, OperandRole::General) {
            mask |= 1 << index;
        }
    }
    mask
}

impl Heap {
    /// Mark a ConcurrentMap/ConcurrentSet object: contagious but never locked.
    pub(super) fn mark_concurrent_table(&mut self, value: Value) {
        self.shared_mode = true;
        let Some(handle) = value.heap_handle() else { return };
        let Some(entry) = self.handle_table.existing_entry(handle) else { return };
        entry.sync.fetch_or(SYNC_SHARED | SYNC_TABLE, Ordering::AcqRel);
        if let Some(object) = self.object(value) {
            HandleTable::refresh_entry(entry, object.address(), &unsafe { managed_object(object) }.object);
        }
    }

    #[inline]
    pub(super) fn is_shared(&self, value: Value) -> bool {
        value
            .heap_handle()
            .and_then(|handle| self.handle_table.existing_entry(handle))
            .is_some_and(|entry| entry.sync.load(Ordering::Acquire) & SYNC_SHARED != 0)
    }

    /// Mark `root` and every mutable container it reaches as shared. Unmarked
    /// objects are thread-local by the contagion invariant, so the traversal
    /// never reads an object another thread may be mutating.
    pub(super) fn mark_shared_deep(&self, root: Value) {
        let Some(handle) = root.heap_handle() else { return };
        if self.handle_table.existing_entry(handle)
            .map_or(true, |entry| entry.sync.load(Ordering::Acquire) & SYNC_SHARED != 0)
        {
            return;
        }
        let mut stack = vec![handle];
        while let Some(handle) = stack.pop() {
            let Some(entry) = self.handle_table.existing_entry(handle) else { continue };
            if entry.sync.load(Ordering::Acquire) & SYNC_SHARED != 0 {
                continue;
            }
            let Some(object) = self.object(Value::from_heap(handle)) else { continue };
            let heap_object = &unsafe { managed_object(object) }.object;
            if !matches!(
                heap_object,
                HeapObject::Map(_)
                    | HeapObject::List(_)
                    | HeapObject::Instance { .. }
                    | HeapObject::Cell(_)
                    | HeapObject::Set { .. }
                    | HeapObject::Closure { .. }
            ) {
                // Strings and other immutable/VM-local objects need no lock.
                continue;
            }
            entry.sync.fetch_or(SYNC_SHARED, Ordering::AcqRel);
            // Republish the JIT view: native guards now fail for this object.
            HandleTable::refresh_entry(entry, object.address(), heap_object);
            collect_heap_object_handles(heap_object, &mut stack);
        }
    }
}

impl Interpreter<'_> {
    #[inline]
    unsafe fn shared_operand_words(
        &self,
        op: &PreparedOp,
        mask: u8,
        out: &mut [RegisterWord; MAX_OPERANDS],
    ) -> usize {
        let mut count = 0usize;
        if mask & OPERAND_A != 0 { out[count] = unsafe { self.reg(op.a) }; count += 1; }
        if mask & OPERAND_B != 0 { out[count] = unsafe { self.reg(op.b) }; count += 1; }
        if mask & OPERAND_C != 0 { out[count] = unsafe { self.reg(op.c) }; count += 1; }
        if mask & OPERAND_PACKED != 0 {
            if op.c <= 4 {
                for index in 0..op.c as usize {
                    out[count] = unsafe { self.reg(Self::packed_call_reg(op, index)) };
                    count += 1;
                }
            } else {
                let function = unsafe { &*self.hot_function };
                let start = op.args_offset as usize;
                let end = start + (op.c as usize).min(MAX_OPERANDS - count);
                for reg in function.call_args.get(start..end).unwrap_or(&[]) {
                    out[count] = unsafe { self.reg(reg.0) };
                    count += 1;
                }
            }
        }
        count
    }

    #[inline]
    fn sync_word(&self, word: RegisterWord) -> Option<(u32, u32)> {
        let handle = Value::from_word(word).heap_handle()?;
        let entry = self.heap.handle_table.existing_entry(handle)?;
        if entry.address.load(Ordering::Relaxed) == 0 {
            return None;
        }
        Some((handle, entry.sync.load(Ordering::Acquire)))
    }

    /// Whether the op at the current PC reads or writes a shared object.
    pub(super) unsafe fn current_op_touches_shared(&self) -> bool {
        if self.hot_frame.is_null() || self.hot_pc == NO_INDEX {
            return false;
        }
        let pc = self.hot_pc as usize;
        let function = unsafe { &*self.hot_function };
        let Some(&mask) = function.shared_operands.get(pc) else { return false };
        if mask == 0 {
            return false;
        }
        let op = &function.ops[pc];
        let mut words = [RegisterWord(0); MAX_OPERANDS];
        let count = unsafe { self.shared_operand_words(op, mask, &mut words) };
        words[..count].iter().any(|word| {
            self.sync_word(*word).is_some_and(|(_, sync)| sync & SYNC_SHARED != 0)
        })
    }

    /// Prologue of a heap-touching op in shared mode. Returns false when the
    /// op must not run now because its task parked on a contended lock.
    #[cold]
    #[inline(never)]
    pub(super) unsafe fn shared_enter_op(&mut self, op: *const PreparedOp, mask: u8) -> bool {
        let op = unsafe { &*op };
        let mut words = [RegisterWord(0); MAX_OPERANDS];
        let count = unsafe { self.shared_operand_words(op, mask, &mut words) };
        let mut handles = [0u32; MAX_OPERANDS];
        let mut lockable = 0usize;
        let mut shared = false;
        for word in &words[..count] {
            let Some((handle, sync)) = self.sync_word(*word) else { continue };
            if sync & SYNC_SHARED == 0 {
                continue;
            }
            shared = true;
            if sync & SYNC_TABLE == 0 {
                handles[lockable] = handle;
                lockable += 1;
            }
        }
        if !shared {
            return true;
        }
        // Contagion: the operation may link any operand into shared state.
        for word in &words[..count] {
            self.heap.mark_shared_deep(Value::from_word(*word));
        }
        // Operands that just became shared need a lock as well.
        for word in &words[..count] {
            if let Some((handle, sync)) = self.sync_word(*word) {
                if sync & SYNC_TABLE == 0 && sync & SYNC_SHARED != 0 && !handles[..lockable].contains(&handle) {
                    handles[lockable] = handle;
                    lockable += 1;
                }
            }
        }
        // Lock in handle order; an object used as several operands locks once.
        handles[..lockable].sort_unstable();
        let mut unique = 0usize;
        for index in 0..lockable {
            if unique == 0 || handles[unique - 1] != handles[index] {
                handles[unique] = handles[index];
                unique += 1;
            }
        }
        let handles = &handles[..unique];
        'acquire: loop {
            for index in 0..handles.len() {
                let entry = self.heap.handle_table.entry(handles[index]);
                if VmAdaptativeMutex::lock_spinning(&entry.sync) {
                    continue;
                }
                for &held in &handles[..index] {
                    VmAdaptativeMutex::unlock(held, &self.heap.handle_table.entry(held).sync);
                }
                if self.is_concurrent_worker {
                    self.park_on_shared_lock(op, handles[index]);
                    return false;
                }
                // The owner VM drives the event loop and cannot hand its thread
                // to another task. Publish a precise stack map so a collection
                // requested by the lock holder can proceed, then retry in order.
                self.mark_gc_point(op);
                self.heap.mutator.safepoint();
                continue 'acquire;
            }
            break;
        }
        self.held_locks.clear();
        self.held_locks.extend_from_slice(handles);
        self.shared_result = if mask & OPERAND_A != 0 { Some((op.a, self.hot_frame as usize)) } else { None };
        self.shared_op_active = true;
        true
    }

    /// Epilogue: publish the result as shared and release the operation's locks.
    #[cold]
    #[inline(never)]
    pub(super) unsafe fn shared_exit_op(&mut self) {
        self.shared_op_active = false;
        if let Some((reg, frame)) = self.shared_result.take() {
            if frame == self.hot_frame as usize && !self.hot_frame.is_null() {
                let value = Value::from_word(unsafe { self.reg(reg) });
                self.heap.mark_shared_deep(value);
            }
        }
        for &handle in &self.held_locks {
            VmAdaptativeMutex::unlock(handle, &self.heap.handle_table.entry(handle).sync);
        }
        self.held_locks.clear();
    }

    fn park_on_shared_lock(&mut self, op: &PreparedOp, handle: u32) {
        PARKED_TASKS.fetch_add(1, Ordering::Relaxed);
        // The PC is not advanced: on resume the whole operation runs again.
        self.mark_gc_point(op);
        self.blocked_on = Some(handle);
        self.run_state = RunState::Suspended;
    }

    /// Shared-mode interpreter quantum. Identical to the ordinary loop except
    /// for the per-op lock prologue/epilogue on heap-touching opcodes.
    #[inline(never)]
    pub(super) fn dispatch_shared_quantum(&mut self) {
        while self.run_state == RunState::Running {
            let pc = self.hot_pc as usize;
            let handler = unsafe { *self.hot_handlers.add(pc) };
            let op = unsafe { self.hot_ops.add(pc) };
            let function = unsafe { &*self.hot_function };
            let mask = unsafe { *function.shared_operands.get_unchecked(pc) };
            if mask == 0 {
                unsafe { handler(self, op) };
                continue;
            }
            if !unsafe { self.shared_enter_op(op, mask) } {
                break;
            }
            unsafe { handler(self, op) };
            if self.shared_op_active {
                unsafe { self.shared_exit_op() };
            }
        }
    }

    /// Handle the task is waiting for after `ExecutionResult::Suspended`.
    pub(crate) fn blocked_on_shared_lock(&self) -> Option<u32> {
        self.blocked_on
    }

    /// Register `waker` for the lock this task is blocked on. `Err` means the
    /// lock is already free and the task should be resumed immediately.
    pub(crate) fn park_blocked_task(&self, waker: LockWaker) -> Result<(), LockWaker> {
        let Some(handle) = self.blocked_on else { return Err(waker) };
        VmAdaptativeMutex::park(handle, &self.heap.handle_table.entry(handle).sync, waker)
    }

    /// Re-run a task parked by `park_on_shared_lock` from its blocked op.
    pub(crate) fn resume_blocked_task(&mut self) -> Result<ExecutionResult, RuntimeError> {
        let _gc_roots = self.gc_root_registration();
        if self.blocked_on.take().is_none() || self.run_state != RunState::Suspended {
            return Err(RuntimeError::new("resume_blocked_task requires a lock-parked VM"));
        }
        let current = self.current_fiber;
        self.fiber_mut(current).status = FiberStatus::Running;
        self.refresh_hot_frame();
        self.run_state = RunState::Running;
        self.result = Value::NULL;
        self.error = None;
        self.dispatch_loop();
        let result = self.finish_result();
        self.sync_persistent_roots();
        result
    }
}
