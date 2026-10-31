//! Isolated parallel execution backend for Napalm `concurrent`/`launch`.
//!
//! Worker threads never receive Napalm `Value`s from the caller. Arguments are
//! converted to immutable Arc-backed transfer snapshots before enqueueing and are
//! materialized into a worker-owned Interpreter/heap. Results take the reverse
//! path. Consequently no mutable managed object is ever shared between VMs.

use super::*;
use crate::napalm_builtin::{install_vm_builtins, BuiltinHost, BuiltinManifest};
use crate::napalm_tracing_jit::{SharedJitProfile, SharedJitRepository, TracingJitOptions};
use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

pub(crate) type ConcurrentJobId = u64;

/// Scalar Map keys supported by the isolated structured-copy boundary.
#[derive(Clone, Debug)]
pub(crate) enum ConcurrentKey {
    Null,
    Bool(bool),
    Num(u64),
    String(Arc<[u8]>),
}

/// Immutable transfer graph. Arc-backed children make repeated references and
/// queue handoff cheap while keeping the graph inaccessible to Napalm code.
#[derive(Clone, Debug)]
pub(crate) enum ConcurrentValue {
    Null,
    Bool(bool),
    Num(u64),
    String(Arc<[u8]>),
    List {
        id: u32,
        items: Arc<[ConcurrentValue]>,
    },
    Map {
        id: u32,
        entries: Arc<[(ConcurrentKey, ConcurrentValue)]>,
    },
}

#[derive(Debug)]
pub(crate) struct ConcurrentCompletion {
    pub id: ConcurrentJobId,
    pub result: Result<ConcurrentValue, String>,
}

#[derive(Debug)]
struct ConcurrentJob {
    id: ConcurrentJobId,
    function: FuncId,
    args: Vec<ConcurrentValue>,
    cancelled: Arc<AtomicBool>,
}

#[derive(Debug)]
struct WorkerSignal {
    lock: Mutex<()>,
    cv: Condvar,
    busy: AtomicBool,
}

pub(crate) struct ConcurrentRuntime {
    /// One ingress lane per worker. `submit()` hashes FuncId to a preferred lane;
    /// idle workers can still steal from every other lane.
    affinity_queues: Arc<Vec<Arc<Injector<ConcurrentJob>>>>,
    signals: Arc<Vec<Arc<WorkerSignal>>>,
    /// Shared JIT repository inherited from the owner VM. Its compiler thread is
    /// created lazily on the first portable trace; published machine code contains
    /// no worker-owned managed values, and each VM rebuilds its relocation table.
    shared_jit_repository: Arc<SharedJitRepository>,
    wake_cursor: AtomicUsize,
    shutdown: Arc<AtomicBool>,
    completion_rx: Receiver<ConcurrentCompletion>,
    cancellations: FastMap<ConcurrentJobId, Arc<AtomicBool>>,
    workers: Vec<JoinHandle<()>>,
    next_id: ConcurrentJobId,
}

impl ConcurrentRuntime {
    pub fn new(
        program: BytecodeProgram,
        manifest: BuiltinManifest,
        wake_tx: Sender<()>,
        jit_options: Option<TracingJitOptions>,
        shared_jit_profile: Arc<SharedJitProfile>,
        shared_jit_repository: Arc<SharedJitRepository>,
    ) -> Result<Self, RuntimeError> {
        let worker_count = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
            .max(1);
        let program = Arc::new(program);
        let manifest = Arc::new(manifest);
        let affinity_queues = Arc::new(
            (0..worker_count)
                .map(|_| Arc::new(Injector::new()))
                .collect::<Vec<_>>(),
        );
        let signals = Arc::new(
            (0..worker_count)
                .map(|_| {
                    Arc::new(WorkerSignal {
                        lock: Mutex::new(()),
                        cv: Condvar::new(),
                        busy: AtomicBool::new(false),
                    })
                })
                .collect::<Vec<_>>(),
        );
        let shutdown = Arc::new(AtomicBool::new(false));
        let (completion_tx, completion_rx) = mpsc::channel();

        let mut locals = Vec::with_capacity(worker_count);
        let mut raw_stealers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let local = Worker::new_fifo();
            raw_stealers.push(local.stealer());
            locals.push(local);
        }
        let stealers = Arc::new(raw_stealers);
        let mut workers = Vec::with_capacity(worker_count);

        for (index, local) in locals.into_iter().enumerate() {
            let worker_affinity_queues = affinity_queues.clone();
            let worker_stealers = stealers.clone();
            let worker_signals = signals.clone();
            let worker_shutdown = shutdown.clone();
            let worker_completion_tx = completion_tx.clone();
            let worker_wake_tx = wake_tx.clone();
            let worker_program = program.clone();
            let worker_manifest = manifest.clone();
            let worker_jit_options = jit_options.clone();
            let worker_shared_jit_profile = shared_jit_profile.clone();
            let worker_shared_jit_repository = shared_jit_repository.clone();
            let spawn = thread::Builder::new()
                .name(format!("napalm-worker-{index}"))
                .spawn(move || {
                    worker_main(
                        index,
                        local,
                        worker_affinity_queues,
                        worker_stealers,
                        worker_signals,
                        worker_shutdown,
                        worker_completion_tx,
                        worker_wake_tx,
                        worker_program,
                        worker_manifest,
                        worker_jit_options,
                        worker_shared_jit_profile,
                        worker_shared_jit_repository,
                    );
                });
            match spawn {
                Ok(handle) => workers.push(handle),
                Err(error) => {
                    // Do not detach already-started workers on partial pool
                    // construction failure.
                    shutdown.store(true, AtomicOrdering::Release);
                    for signal in signals.iter() {
                        let _guard = signal.lock.lock();
                        signal.cv.notify_all();
                    }
                    for handle in workers.drain(..) {
                        let _ = handle.join();
                    }
                    return Err(RuntimeError::new(format!(
                        "failed to create concurrent worker: {error}"
                    )));
                }
            }
        }

        Ok(Self {
            affinity_queues,
            signals,
            shared_jit_repository,
            wake_cursor: AtomicUsize::new(0),
            shutdown,
            completion_rx,
            cancellations: FastMap::default(),
            workers,
            next_id: 1,
        })
    }

    pub fn submit(
        &mut self,
        function: FuncId,
        args: Vec<ConcurrentValue>,
    ) -> ConcurrentJobId {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        self.cancellations.insert(id, cancelled.clone());

        let preferred = preferred_worker(function, self.affinity_queues.len());
        let target = self.submission_worker(preferred);
        let queue = &self.affinity_queues[target];
        let signal = &self.signals[target];

        {
            // Pair push and notification with this worker's sleeper lock so it
            // cannot decide to sleep after the notification has already passed.
            let _guard = signal.lock.lock();
            queue.push(ConcurrentJob {
                id,
                function,
                args,
                cancelled,
            });
            signal.cv.notify_one();
        }

        id
    }

    /// Preserve function affinity when its lane is genuinely idle, but fan a burst
    /// out directly across idle lanes instead of first serializing identical FuncIds
    /// behind one injector and relying on batch stealing to recover parallelism.
    fn submission_worker(&self, preferred: usize) -> usize {
        let count = self.signals.len();
        debug_assert!(count > 0);
        let preferred_idle = !self.signals[preferred].busy.load(AtomicOrdering::Acquire)
            && self.affinity_queues[preferred].is_empty();
        if preferred_idle {
            return preferred;
        }

        let start = self.wake_cursor.fetch_add(1, AtomicOrdering::Relaxed) % count;
        for offset in 0..count {
            let candidate = (start + offset) % count;
            if candidate == preferred {
                continue;
            }
            if !self.signals[candidate].busy.load(AtomicOrdering::Acquire)
                && self.affinity_queues[candidate].is_empty()
            {
                return candidate;
            }
        }
        preferred
    }


    #[inline]
    pub fn has_pending(&self) -> bool {
        !self.cancellations.is_empty()
    }

    pub fn cancel(&mut self, id: ConcurrentJobId) -> bool {
        let Some(cancelled) = self.cancellations.remove(&id) else {
            return false;
        };
        cancelled.store(true, AtomicOrdering::Release);
        true
    }

    pub fn try_completion(&mut self) -> Option<ConcurrentCompletion> {
        match self.completion_rx.try_recv() {
            Ok(completion) => {
                self.cancellations.remove(&completion.id);
                Some(completion)
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => None,
        }
    }

    /// Stop every worker and wait until all worker-owned VMs have been destroyed.
    /// This method is idempotent so the owning Interpreter can perform shutdown
    /// inside a GC-native section and Drop can remain a safe fallback.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, AtomicOrdering::Release);
        for signal in self.signals.iter() {
            let _guard = signal.lock.lock();
            signal.cv.notify_all();
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
        self.cancellations.clear();
    }
}

impl Drop for ConcurrentRuntime {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[allow(clippy::too_many_arguments)]
fn worker_main(
    index: usize,
    local: Worker<ConcurrentJob>,
    affinity_queues: Arc<Vec<Arc<Injector<ConcurrentJob>>>>,
    stealers: Arc<Vec<Stealer<ConcurrentJob>>>,
    signals: Arc<Vec<Arc<WorkerSignal>>>,
    shutdown: Arc<AtomicBool>,
    completion_tx: Sender<ConcurrentCompletion>,
    wake_tx: Sender<()>,
    program: Arc<BytecodeProgram>,
    manifest: Arc<BuiltinManifest>,
    jit_options: Option<TracingJitOptions>,
    shared_jit_profile: Arc<SharedJitProfile>,
    shared_jit_repository: Arc<SharedJitRepository>,
) {
    let signal = &signals[index];
    // Interpreter construction verifies and prepares the complete bytecode program,
    // attaches a GC mutator, allocates runtime metadata and initializes JIT state.
    // Keep that cost lazy so an N-core pool only constructs VMs for workers that
    // actually receive jobs.
    let mut vm = None;

    while !shutdown.load(AtomicOrdering::Acquire) {
        let Some(job) = take_job(index, &local, &affinity_queues, &stealers) else {
            signal.busy.store(false, AtomicOrdering::Release);
            let mut guard = signal.lock.lock();
            if shutdown.load(AtomicOrdering::Acquire) {
                break;
            }
            if local.is_empty() && affinity_queues.iter().all(|queue| queue.is_empty()) {
                signal.cv.wait(&mut guard);
            }
            continue;
        };

        signal.busy.store(true, AtomicOrdering::Release);
        if job.cancelled.load(AtomicOrdering::Acquire) {
            signal.busy.store(false, AtomicOrdering::Release);
            continue;
        }

        if vm.is_none() {
            vm = Some(build_worker_vm(
                program.as_ref(),
                manifest.as_ref(),
                jit_options.clone(),
                shared_jit_profile.clone(),
                shared_jit_repository.clone(),
                index as u32,
            ));
        }

        let (result, reset_vm) = match vm.as_mut().expect("worker VM must be initialized") {
            Ok(vm) => execute_job(vm, &job),
            Err(message) => (Err(message.clone()), true),
        };
        if reset_vm {
            // Drop the failed VM before constructing its replacement. Concurrent
            // workers suppress per-VM final full collections; the process-wide
            // collector will reclaim their unreachable objects normally.
            vm = Some(build_worker_vm(
                program.as_ref(),
                manifest.as_ref(),
                jit_options.clone(),
                shared_jit_profile.clone(),
                shared_jit_repository.clone(),
                index as u32,
            ));
        }
        signal.busy.store(false, AtomicOrdering::Release);
        if job.cancelled.load(AtomicOrdering::Acquire) {
            continue;
        }
        let _ = completion_tx.send(ConcurrentCompletion { id: job.id, result });
        let _ = wake_tx.send(());
    }
}

fn take_job(
    index: usize,
    local: &Worker<ConcurrentJob>,
    affinity_queues: &[Arc<Injector<ConcurrentJob>>],
    stealers: &[Stealer<ConcurrentJob>],
) -> Option<ConcurrentJob> {
    if let Some(job) = local.pop() {
        return Some(job);
    }

    // Prefer jobs whose FuncId hashes to this worker. Batch stealing keeps a hot
    // function on one worker long enough to benefit its private IC/JIT state.
    loop {
        match affinity_queues[index].steal_batch_and_pop(local) {
            Steal::Success(job) => return Some(job),
            Steal::Retry => continue,
            Steal::Empty => break,
        }
    }

    // No affinity work is available locally, so preserve load balancing by
    // stealing from other workers' ingress lanes. Rotate victims by worker index.
    for offset in 1..affinity_queues.len() {
        let victim = (index + offset) % affinity_queues.len();
        loop {
            match affinity_queues[victim].steal_batch_and_pop(local) {
                Steal::Success(job) => return Some(job),
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }
    }

    // Finally steal already-batched local work. This is the least affinity-friendly
    // path and therefore intentionally comes after all ingress lanes.
    for offset in 1..stealers.len() {
        let victim = (index + offset) % stealers.len();
        loop {
            match stealers[victim].steal() {
                Steal::Success(job) => return Some(job),
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }
    }
    None
}

#[inline]
fn preferred_worker(function: FuncId, worker_count: usize) -> usize {
    debug_assert!(worker_count > 0);
    // Multiplicative hashing spreads sequential FuncIds across power-of-two and
    // non-power-of-two worker counts while remaining deterministic.
    let mixed = function
        .0
        .wrapping_mul(0x9e37_79b1)
        ^ function.0.rotate_right(16);
    (mixed as usize) % worker_count
}

fn build_worker_vm<'a>(
    program: &'a BytecodeProgram,
    manifest: &BuiltinManifest,
    jit_options: Option<TracingJitOptions>,
    shared_jit_profile: Arc<SharedJitProfile>,
    shared_jit_repository: Arc<SharedJitRepository>,
    worker_id: u32,
) -> Result<Interpreter<'a>, String> {
    let host = Box::new(BuiltinHost::new(manifest.clone()));
    let mut vm = Interpreter::with_host(program, host)
        .map_err(|error| format!("failed to initialize concurrent VM: {}", error.message))?;
    install_vm_builtins(&mut vm, manifest);
    vm.set_jit_options(jit_options);
    vm.set_concurrent_worker(true)
        .map_err(|error| format!(
            "failed to initialize concurrent worker code bindings: {}",
            error.message
        ))?;
    vm.set_shared_jit_profile(shared_jit_profile);
    vm.set_shared_jit_repository(shared_jit_repository, worker_id);
    Ok(vm)
}

/// Execute one job in a worker-owned VM. The boolean indicates whether the VM
/// must be rebuilt before accepting another job (failed/suspended execution can
/// retain stack state, whereas a normal return leaves the main fiber idle).
fn execute_job(
    vm: &mut Interpreter<'_>,
    job: &ConcurrentJob,
) -> (Result<ConcurrentValue, String>, bool) {
    let (args, argument_roots) = match vm.materialize_concurrent_arguments(&job.args) {
        Ok(values) => values,
        Err(error) => return (Err(error.message), false),
    };

    let execution = vm.run_function(job.function, &args);
    for root in argument_roots {
        vm.release_root(root);
    }
    let execution = match execution {
        Ok(execution) => execution,
        Err(error) => return (Err(error.message), true),
    };
    let value = match execution {
        ExecutionResult::Returned(value) => value,
        ExecutionResult::Suspended(_) => {
            return (
                Err("concurrent worker suspended without a runnable Promise/event".to_owned()),
                true,
            );
        }
    };
    match vm.snapshot_concurrent_value(value) {
        Ok(value) => (Ok(value), false),
        Err(error) => (Err(error.message), false),
    }
}

#[cfg(test)]
mod phase1_tests {
    use super::*;

    #[test]
    fn function_affinity_is_stable() {
        for workers in 1..=32 {
            for function in 0..256u32 {
                let id = FuncId(function);
                let a = preferred_worker(id, workers);
                let b = preferred_worker(id, workers);
                assert_eq!(a, b);
                assert!(a < workers);
            }
        }
    }

    #[test]
    fn function_affinity_uses_multiple_workers() {
        let workers = 8;
        let mut seen = [false; 8];
        for function in 0..256u32 {
            seen[preferred_worker(FuncId(function), workers)] = true;
        }
        assert!(seen.into_iter().all(|used| used));
    }
}
