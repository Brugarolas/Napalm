//! Isolated parallel execution backend for Napalm `concurrent`/`launch`.
//!
//! Worker threads never receive Napalm `Value`s from the caller. Arguments are
//! converted to immutable Arc-backed transfer snapshots before enqueueing and are
//! materialized into a worker-owned Interpreter/heap. Results take the reverse
//! path. Consequently no mutable managed object is ever shared between VMs.

use super::*;
use crate::napalm_builtin::{install_vm_builtins, BuiltinHost, BuiltinManifest};
use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
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
}

pub(crate) struct ConcurrentRuntime {
    injector: Arc<Injector<ConcurrentJob>>,
    signal: Arc<WorkerSignal>,
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
    ) -> Result<Self, RuntimeError> {
        let worker_count = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
            .max(1);
        let program = Arc::new(program);
        let manifest = Arc::new(manifest);
        let injector: Arc<Injector<ConcurrentJob>> = Arc::new(Injector::new());
        let signal = Arc::new(WorkerSignal {
            lock: Mutex::new(()),
            cv: Condvar::new(),
        });
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
            let worker_injector = injector.clone();
            let worker_stealers = stealers.clone();
            let worker_signal = signal.clone();
            let worker_shutdown = shutdown.clone();
            let worker_completion_tx = completion_tx.clone();
            let worker_wake_tx = wake_tx.clone();
            let worker_program = program.clone();
            let worker_manifest = manifest.clone();
            let spawn = thread::Builder::new()
                .name(format!("napalm-worker-{index}"))
                .spawn(move || {
                    worker_main(
                        index,
                        local,
                        worker_injector,
                        worker_stealers,
                        worker_signal,
                        worker_shutdown,
                        worker_completion_tx,
                        worker_wake_tx,
                        worker_program,
                        worker_manifest,
                    );
                });
            match spawn {
                Ok(handle) => workers.push(handle),
                Err(error) => {
                    // Do not detach already-started workers on partial pool
                    // construction failure.
                    shutdown.store(true, AtomicOrdering::Release);
                    signal.cv.notify_all();
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
            injector,
            signal,
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

        // Pair the queue push with the sleeper lock so a worker cannot observe an
        // empty queue and go to sleep after the notification has already passed.
        let _guard = self.signal.lock.lock();
        self.injector.push(ConcurrentJob {
            id,
            function,
            args,
            cancelled,
        });
        self.signal.cv.notify_one();
        id
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
}

impl Drop for ConcurrentRuntime {
    fn drop(&mut self) {
        self.shutdown.store(true, AtomicOrdering::Release);
        self.signal.cv.notify_all();
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn worker_main(
    index: usize,
    local: Worker<ConcurrentJob>,
    injector: Arc<Injector<ConcurrentJob>>,
    stealers: Arc<Vec<Stealer<ConcurrentJob>>>,
    signal: Arc<WorkerSignal>,
    shutdown: Arc<AtomicBool>,
    completion_tx: Sender<ConcurrentCompletion>,
    wake_tx: Sender<()>,
    program: Arc<BytecodeProgram>,
    manifest: Arc<BuiltinManifest>,
) {
    let mut vm = build_worker_vm(program.as_ref(), manifest.as_ref());

    while !shutdown.load(AtomicOrdering::Acquire) {
        let Some(job) = take_job(index, &local, &injector, &stealers) else {
            let mut guard = signal.lock.lock();
            if shutdown.load(AtomicOrdering::Acquire) {
                break;
            }
            if injector.is_empty() && local.is_empty() {
                signal.cv.wait(&mut guard);
            }
            continue;
        };

        if job.cancelled.load(AtomicOrdering::Acquire) {
            continue;
        }

        let (result, reset_vm) = match &mut vm {
            Ok(vm) => execute_job(vm, &job),
            Err(message) => (Err(message.clone()), true),
        };
        if reset_vm {
            vm = build_worker_vm(program.as_ref(), manifest.as_ref());
        }
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
    injector: &Injector<ConcurrentJob>,
    stealers: &[Stealer<ConcurrentJob>],
) -> Option<ConcurrentJob> {
    if let Some(job) = local.pop() {
        return Some(job);
    }

    loop {
        match injector.steal_batch_and_pop(local) {
            Steal::Success(job) => return Some(job),
            Steal::Retry => continue,
            Steal::Empty => break,
        }
    }

    // Rotate the victim order by worker index. This avoids every idle worker
    // hammering stealer zero first under bursty launch workloads.
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

fn build_worker_vm<'a>(
    program: &'a BytecodeProgram,
    manifest: &BuiltinManifest,
) -> Result<Interpreter<'a>, String> {
    let host = Box::new(BuiltinHost::new(manifest.clone()));
    let mut vm = Interpreter::with_host(program, host)
        .map_err(|error| format!("failed to initialize concurrent VM: {}", error.message))?;
    install_vm_builtins(&mut vm, manifest);
    vm.set_concurrent_worker(true);
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
