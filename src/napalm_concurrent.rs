//! Isolated parallel execution backend for Napalm `concurrent`/`launch`.
//!
//! Worker threads never receive Napalm `Value`s from the caller. Arguments are
//! converted to immutable Arc-backed transfer snapshots before enqueueing and are
//! materialized into a worker-owned Interpreter/heap. Results take the reverse
//! path. Explicit shared concurrent collections retain their globally rooted handle.
//! Their table operations are synchronized by Papaya; payload mutation is not.

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
    /// Explicitly shared table. Queue ownership roots this handle.
    Shared(Arc<SharedTransfer>),
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

pub(crate) struct SharedTransfer {
    pub value: Value,
    pub(super) _registration: GcRootRegistration,
}
impl std::fmt::Debug for SharedTransfer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.debug_tuple("SharedTransfer").field(&self.value).finish() }
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
    captures: Vec<ConcurrentValue>,
    shared_globals: Vec<(usize, usize, ConcurrentValue)>,
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
        shared_memo: Option<Arc<SharedMemoTables>>,
        shared_shapes: Arc<MapShapeTable>,
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
            let worker_shared_memo = shared_memo.clone();
            let worker_shared_shapes = shared_shapes.clone();
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
                        worker_shared_memo,
                        worker_shared_shapes,
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
        captures: Vec<ConcurrentValue>,
        shared_globals: Vec<(usize, usize, ConcurrentValue)>,
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
                captures,
                shared_globals,
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
    shared_memo: Option<Arc<SharedMemoTables>>,
    shared_shapes: Arc<MapShapeTable>,
) {
    let signal = &signals[index];
    // Interpreter construction verifies and prepares the complete bytecode program,
    // attaches a GC mutator, allocates runtime metadata and initializes JIT state.
    // Keep that cost lazy so an N-core pool only constructs VMs for workers that
    // actually receive jobs.
    //
    // Task scheduler: a job runs in one VM of `idle`. A job blocked on a
    // VmAdaptativeMutex keeps its VM (its frames live there) in `parked` while
    // this OS thread continues with other work; the releasing thread pushes the
    // task token onto `ready` and signals this worker, which resumes the task
    // on this same thread before taking new jobs. Several VMs on one thread are
    // GC-safe: a rendezvous parks all same-thread mutators together.
    let mut idle: Vec<Interpreter<'_>> = Vec::new();
    let mut parked: FastMap<u64, ParkedTask<'_>> = FastMap::default();
    let ready: Arc<parking_lot::Mutex<std::collections::VecDeque<u64>>> = Arc::default();
    let mut next_token = 0u64;

    while !shutdown.load(AtomicOrdering::Acquire) {
        let resumable = ready.lock().pop_front();
        let step = if let Some(token) = resumable {
            let Some(task) = parked.remove(&token) else { continue };
            signal.busy.store(true, AtomicOrdering::Release);
            resume_task(task)
        } else {
            let Some(job) = take_job(index, &local, &affinity_queues, &stealers) else {
                signal.busy.store(false, AtomicOrdering::Release);
                // Check-and-wait under `signal.lock`, which every producer
                // (submit, lock wakers) takes before notifying. The lock is only
                // ever taken inside the native section: holding it across the
                // safepoint that enters native would deadlock a collection
                // against a Running producer blocked on this mutex.
                let wait = || {
                    let mut guard = signal.lock.lock();
                    if !shutdown.load(AtomicOrdering::Acquire)
                        && local.is_empty()
                        && affinity_queues.iter().all(|queue| queue.is_empty())
                        && ready.lock().is_empty()
                    {
                        signal.cv.wait(&mut guard);
                    }
                };
                // An idle worker must not block another worker's collection.
                // Entering native marks every VM of this thread (idle and
                // parked) native; their persistent roots are published.
                let peer = idle.first_mut().or_else(|| parked.values_mut().next().map(|task| &mut task.vm));
                if let Some(vm) = peer {
                    unsafe { vm.heap.mutator.with_native(wait); }
                } else {
                    wait();
                }
                continue;
            };

            signal.busy.store(true, AtomicOrdering::Release);
            if job.cancelled.load(AtomicOrdering::Acquire) {
                signal.busy.store(false, AtomicOrdering::Release);
                continue;
            }
            let vm = match idle.pop() {
                Some(vm) => Ok(vm),
                None => build_worker_vm(
                    program.as_ref(),
                    manifest.as_ref(),
                    jit_options.clone(),
                    shared_jit_profile.clone(),
                    shared_jit_repository.clone(),
                    shared_memo.clone(),
                    shared_shapes.clone(),
                    index as u32,
                ),
            };
            match vm {
                Ok(vm) => start_job(vm, job),
                Err(message) => JobStep::Finished { job, result: Err(message), vm: None },
            }
        };

        match step {
            JobStep::Parked(task) => {
                let token = next_token;
                next_token = next_token.wrapping_add(1);
                let wake_ready = ready.clone();
                let wake_signal = signal.clone();
                let waker: LockWaker = Box::new(move || {
                    wake_ready.lock().push_back(token);
                    let _guard = wake_signal.lock.lock();
                    wake_signal.cv.notify_one();
                });
                if task.vm.park_blocked_task(waker).is_err() {
                    // Released between the spin and the registration: retry now.
                    ready.lock().push_back(token);
                }
                parked.insert(token, task);
            }
            JobStep::Finished { job, result, vm } => {
                // A VM that failed or suspended unexpectedly may retain stack
                // state; drop it. Concurrent workers suppress per-VM final full
                // collections, and the replacement keeps the shared memo domain.
                if let Some(vm) = vm {
                    idle.push(vm);
                }
                signal.busy.store(false, AtomicOrdering::Release);
                if job.cancelled.load(AtomicOrdering::Acquire) {
                    continue;
                }
                let _ = completion_tx.send(ConcurrentCompletion { id: job.id, result });
                let _ = wake_tx.send(());
            }
        }
    }
}

/// A job blocked on a VmAdaptativeMutex. Its frames live in `vm`, so it resumes
/// on the worker thread that owns that VM.
struct ParkedTask<'a> {
    vm: Interpreter<'a>,
    job: ConcurrentJob,
    roots: Vec<u32>,
}

enum JobStep<'a> {
    Finished {
        job: ConcurrentJob,
        result: Result<ConcurrentValue, String>,
        /// `Some` when the VM is clean and can run the next job.
        vm: Option<Interpreter<'a>>,
    },
    Parked(ParkedTask<'a>),
}

fn start_job<'a>(mut vm: Interpreter<'a>, job: ConcurrentJob) -> JobStep<'a> {
    let (args, mut roots) = match vm.materialize_concurrent_arguments(&job.args) {
        Ok(values) => values,
        Err(error) => return JobStep::Finished { job, result: Err(error.message), vm: Some(vm) },
    };
    for (module, slot, snapshot) in &job.shared_globals {
        match vm.materialize_concurrent_value(snapshot) {
            Ok(value) => vm.modules[*module].variables[*slot] = value,
            Err(error) => {
                for root in roots { vm.release_root(root); }
                return JobStep::Finished { job, result: Err(error.message), vm: None };
            }
        }
    }
    let (captures, capture_roots) = match vm.materialize_concurrent_arguments(&job.captures) {
        Ok(values) => values,
        Err(error) => {
            for root in roots { vm.release_root(root); }
            return JobStep::Finished { job, result: Err(error.message), vm: None };
        }
    };
    roots.extend(capture_roots);
    let execution = vm.run_function_with_captures(job.function, &args, &captures);
    finish_step(vm, job, roots, execution)
}

fn resume_task(task: ParkedTask<'_>) -> JobStep<'_> {
    let ParkedTask { mut vm, job, roots } = task;
    let execution = vm.resume_blocked_task();
    finish_step(vm, job, roots, execution)
}

fn finish_step<'a>(
    mut vm: Interpreter<'a>,
    job: ConcurrentJob,
    roots: Vec<u32>,
    execution: Result<ExecutionResult, RuntimeError>,
) -> JobStep<'a> {
    if matches!(execution, Ok(ExecutionResult::Suspended(_))) && vm.blocked_on_shared_lock().is_some() {
        // Argument/capture roots stay registered while the task is parked.
        return JobStep::Parked(ParkedTask { vm, job, roots });
    }
    for root in roots {
        vm.release_root(root);
    }
    let value = match execution {
        Ok(ExecutionResult::Returned(value)) => value,
        Ok(ExecutionResult::Suspended(_)) => {
            return JobStep::Finished {
                job,
                result: Err("concurrent worker suspended without a runnable Promise/event".to_owned()),
                vm: None,
            };
        }
        Err(error) => return JobStep::Finished { job, result: Err(error.message), vm: None },
    };
    let result = vm.snapshot_concurrent_value(value).map_err(|error| error.message);
    JobStep::Finished { job, result, vm: Some(vm) }
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
    shared_memo: Option<Arc<SharedMemoTables>>,
    shared_shapes: Arc<MapShapeTable>,
    worker_id: u32,
) -> Result<Interpreter<'a>, String> {
    let host = Box::new(BuiltinHost::new(manifest.clone()));
    // One hidden-class table per program: objects shaped by any VM can be read
    // by every other VM that reaches them through a shared collection.
    let mut vm = Interpreter::with_host_and_shapes(program, host, Some(shared_shapes))
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
    // Joined only after the stock manifest is sealed: results are shared on
    // the premise that every VM in the domain dispatches builtins identically.
    vm.set_shared_memo(shared_memo);
    Ok(vm)
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
