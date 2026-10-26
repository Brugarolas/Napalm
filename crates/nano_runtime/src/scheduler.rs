//! Work stealing applies to unstarted Tasks, never to live native Rust stacks.
use crate::{
    event_loop::{EventLoop, Vow, VowState},
    fiber::Fiber,
    function::{Function, INLINE_ALIGN},
};
use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use std::{
    cell::RefCell,
    collections::HashMap,
    fmt,
    marker::PhantomData,
    rc::Rc,
    sync::{
        Arc, Condvar, Mutex as OsMutex, Weak,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

pub const TASK_INLINE_BYTES: usize = 192;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
fn id() -> u64 {
    let mut next = NEXT_ID.load(Ordering::Relaxed);
    loop {
        let successor = next.checked_add(1).expect("task ID space exhausted");
        match NEXT_ID.compare_exchange_weak(next, successor, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return next,
            Err(actual) => next = actual,
        }
    }
}
fn lock<T>(m: &OsMutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Clone, Debug)]
pub struct Config {
    pub workers: usize,
    pub stack_size: usize,
    /// Maximum accepted unfinished tasks; prevents unbounded admission.
    pub max_tasks: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            workers: thread::available_parallelism().map_or(1, usize::from),
            stack_size: 256 * 1024,
            max_tasks: 65_536,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpawnError {
    Closed,
    Full,
    CaptureTooLarge,
}
impl fmt::Display for SpawnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SpawnError {}

struct TaskControl {
    id: u64,
    completion: Vow,
    // 0 = no request, 1 = cancellation, 2 = interruption.
    request: AtomicU8,
    parked: OsMutex<Option<Weak<ParkToken>>>,
}
/// Read-only completion handle; cancellation is a request, not early settlement.
#[derive(Clone)]
pub struct TaskHandle {
    inner: Arc<TaskControl>,
}
impl fmt::Debug for TaskHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskHandle")
            .field("id", &self.id())
            .field("state", &self.state())
            .finish()
    }
}
impl TaskHandle {
    pub fn id(&self) -> u64 {
        self.inner.id
    }
    pub fn state(&self) -> VowState {
        self.inner.completion.state()
    }
    pub fn wait(&self) -> VowState {
        assert!(
            Thread::current_task().is_none_or(|t| t.id() != self.id()),
            "task cannot wait for itself"
        );
        self.inner.completion.wait()
    }
    pub fn wait_timeout(&self, duration: Duration) -> VowState {
        self.inner.completion.wait_timeout(duration)
    }
    pub fn on_settle<F: FnOnce() + Send + Unpin + 'static>(
        &self,
        f: F,
    ) -> crate::event_loop::VowSubscription {
        self.inner.completion.on_settle(f)
    }
    pub fn cancel(&self) -> bool {
        self.request(1)
    }
    pub fn interrupt(&self) -> bool {
        self.request(2)
    }
    fn request(&self, request: u8) -> bool {
        if self.state() != VowState::Pending {
            return false;
        }
        if self
            .inner
            .request
            .compare_exchange(0, request, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        let token = lock(&self.inner.parked).as_ref().and_then(Weak::upgrade);
        if let Some(token) = token {
            token.wake();
        }
        true
    }
    fn requested(&self) -> u8 {
        self.inner.request.load(Ordering::Acquire)
    }
}

/// The callable and captures are stored in Function's inline buffer. Oversized
/// captures are rejected rather than silently allocating a boxed callable.
pub struct Task {
    callable: Function<(), (), TASK_INLINE_BYTES>,
    handle: TaskHandle,
    managed: bool,
}
impl Task {
    pub fn new<F: FnOnce() + Send + Unpin + 'static>(f: F) -> Result<Self, SpawnError> {
        if std::mem::size_of::<Option<F>>() > TASK_INLINE_BYTES
            || std::mem::align_of::<Option<F>>() > INLINE_ALIGN
        {
            return Err(SpawnError::CaptureTooLarge);
        }
        let mut f = Some(f);
        Ok(Self {
            managed: false,
            callable: Function::new(move || {
                if let Some(f) = f.take() {
                    f();
                }
            }),
            handle: TaskHandle {
                inner: Arc::new(TaskControl {
                    id: id(),
                    completion: Vow::new(),
                    request: AtomicU8::new(0),
                    parked: OsMutex::new(None),
                }),
            },
        })
    }
    pub fn handle(&self) -> TaskHandle {
        self.handle.clone()
    }
    pub fn is_inline(&self) -> bool {
        self.callable.is_inline()
    }
}
impl Drop for Task {
    fn drop(&mut self) {
        if !self.managed {
            self.handle.inner.completion.cancel();
        }
    }
}

struct Mailbox {
    notifications: Injector<Arc<ParkToken>>,
    epoch: OsMutex<u64>,
    wake: Condvar,
}
impl Mailbox {
    fn new() -> Self {
        Self {
            notifications: Injector::new(),
            epoch: OsMutex::new(0),
            wake: Condvar::new(),
        }
    }
    fn signal(&self) {
        let mut epoch = lock(&self.epoch);
        *epoch = epoch.wrapping_add(1);
        self.wake.notify_one();
    }
}
/// One-shot wake permit. Publishing before suspension is safe: the permit is
/// sticky and the owning worker cannot consume its mailbox while running us.
pub(crate) struct ParkToken {
    task_id: u64,
    ready: AtomicBool,
    mailbox: Weak<Mailbox>,
}
impl ParkToken {
    pub(crate) fn wake(self: &Arc<Self>) {
        if !self.ready.swap(true, Ordering::AcqRel) {
            if let Some(mailbox) = self.mailbox.upgrade() {
                mailbox.notifications.push(self.clone());
                mailbox.signal();
            }
        }
    }
}
/// Cloneable, one-shot permit for custom fiber synchronization.
#[derive(Clone)]
pub struct FiberWaker {
    token: Arc<ParkToken>,
}
impl FiberWaker {
    /// May be called on any OS thread, before or after the corresponding park.
    pub fn wake(&self) {
        self.token.wake();
    }
}

struct Registry {
    closed: bool,
    tasks: HashMap<u64, TaskHandle>,
}
struct Shared {
    incoming: Injector<Task>,
    stealers: Vec<Stealer<Task>>,
    mailboxes: Vec<Arc<Mailbox>>,
    registry: OsMutex<Registry>,
    config: Config,
    event_loop: EventLoop,
}
impl Shared {
    fn submit(&self, mut task: Task) -> Result<TaskHandle, SpawnError> {
        let mut registry = lock(&self.registry);
        if registry.closed {
            return Err(SpawnError::Closed);
        }
        if registry.tasks.len() >= self.config.max_tasks {
            return Err(SpawnError::Full);
        }
        let handle = task.handle();
        task.managed = true;
        registry.tasks.insert(handle.id(), handle.clone());
        // Admission and shutdown are serialized, including queue publication.
        self.incoming.push(task);
        drop(registry);
        for mailbox in &self.mailboxes {
            mailbox.signal();
        }
        Ok(handle)
    }
    fn close(&self) {
        let tasks = {
            let mut registry = lock(&self.registry);
            registry.closed = true;
            registry.tasks.values().cloned().collect::<Vec<_>>()
        };
        for task in tasks {
            task.cancel();
        }
        for mailbox in &self.mailboxes {
            mailbox.signal();
        }
    }
    fn done(&self, task: &TaskHandle, panicked: bool) {
        match task.requested() {
            1 if !panicked => {
                task.inner.completion.cancel();
            }
            2 => {
                task.inner.completion.interrupt();
            }
            _ if panicked => {
                task.inner.completion.interrupt();
            }
            _ => {
                task.inner.completion.resolve();
            }
        }
        let empty = {
            let mut registry = lock(&self.registry);
            registry.tasks.remove(&task.id());
            registry.tasks.is_empty()
        };
        if empty {
            for mailbox in &self.mailboxes {
                mailbox.signal();
            }
        }
    }
}

/// Owning runtime. Workers default to the process's available logical CPUs.
/// One additional serial thread drives engine microtasks and timers.
pub struct TaskScheduler {
    shared: Arc<Shared>,
    joins: OsMutex<Vec<JoinHandle<()>>>,
}
pub type ThreadPool = TaskScheduler;
impl TaskScheduler {
    pub fn new() -> std::io::Result<Self> {
        Self::with_config(Config::default())
    }
    pub fn with_config(config: Config) -> std::io::Result<Self> {
        if config.workers == 0 || config.stack_size < 16 * 1024 || config.max_tasks == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid scheduler configuration",
            ));
        }
        let queues: Vec<_> = (0..config.workers).map(|_| Worker::new_fifo()).collect();
        let shared = Arc::new(Shared {
            incoming: Injector::new(),
            stealers: queues.iter().map(Worker::stealer).collect(),
            mailboxes: (0..config.workers)
                .map(|_| Arc::new(Mailbox::new()))
                .collect(),
            registry: OsMutex::new(Registry {
                closed: false,
                tasks: HashMap::new(),
            }),
            config,
            event_loop: EventLoop::new(),
        });
        let mut joins = Vec::new();
        for (index, queue) in queues.into_iter().enumerate() {
            let worker_shared = shared.clone();
            match thread::Builder::new()
                .name(format!("nano-worker-{index}"))
                .spawn(move || worker_loop(index, queue, worker_shared))
            {
                Ok(join) => joins.push(join),
                Err(error) => {
                    shared.close();
                    for join in joins {
                        let _ = join.join();
                    }
                    return Err(error);
                }
            }
        }
        let events = shared.event_loop.clone();
        let started = events.next_tick(|| {});
        match thread::Builder::new()
            .name("nano-engine-events".into())
            .spawn(move || {
                let _scope = RuntimeScope::enter();
                let _ = events.run();
            }) {
            Ok(join) => joins.push(join),
            Err(error) => {
                shared.close();
                for join in joins {
                    let _ = join.join();
                }
                return Err(error);
            }
        }
        // run() clears stop_requested at startup; this handshake prevents a
        // shutdown racing before the event thread has entered run().
        started.wait();
        Ok(Self {
            shared,
            joins: OsMutex::new(joins),
        })
    }
    pub fn spawn<F: FnOnce() + Send + Unpin + 'static>(
        &self,
        f: F,
    ) -> Result<TaskHandle, SpawnError> {
        self.submit(Task::new(f)?)
    }
    pub fn submit(&self, task: Task) -> Result<TaskHandle, SpawnError> {
        self.shared.submit(task)
    }
    pub fn worker_count(&self) -> usize {
        self.shared.config.workers
    }
    pub fn pending_tasks(&self) -> usize {
        lock(&self.shared.registry).tasks.len()
    }
    pub fn event_loop(&self) -> EventLoop {
        self.shared.event_loop.clone()
    }
    /// Cooperative cancellation; running code must reach a checkpoint.
    pub fn request_shutdown(&self) {
        self.shared.close();
    }
    /// Cancel, drain stacks and join all threads. Must be called externally.
    pub fn shutdown(&self) -> std::io::Result<()> {
        if in_runtime() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "shutdown must run outside scheduler tasks",
            ));
        }
        let mut joins = lock(&self.joins);
        if joins
            .iter()
            .any(|join| join.thread().id() == thread::current().id())
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "shutdown cannot join its own runtime thread",
            ));
        }
        self.shared.close();
        // Stop the engine after workers have unwound their tasks.
        let event_join = joins.pop();
        let mut failed = false;
        for join in joins.drain(..) {
            failed |= join.join().is_err();
        }
        self.shared.event_loop.stop();
        if let Some(join) = event_join {
            failed |= join.join().is_err();
        }
        if failed {
            Err(std::io::Error::other("runtime thread panicked"))
        } else {
            Ok(())
        }
    }
}
impl Drop for TaskScheduler {
    fn drop(&mut self) {
        self.shared.close();
        self.shared.event_loop.stop();
        // Drop never joins: the owner may itself be dropped on a worker.
        // Use shutdown() externally when synchronous teardown is required.
    }
}

/// Snapshot of the currently executing worker. Native Fiber ownership stays in
/// worker_loop; current_fiber identifies it without exposing aliasing pointers.
#[derive(Clone)]
pub struct Thread {
    pub index: usize,
    pub current_fiber: u64,
    pub current_task: TaskHandle,
    shared: Arc<Shared>,
    parked: Option<Arc<ParkToken>>,
    _affine: PhantomData<Rc<()>>,
}
thread_local! { static IN_RUNTIME: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
pub(crate) fn in_runtime() -> bool {
    IN_RUNTIME.with(|flag| flag.get())
}
struct RuntimeScope;
impl RuntimeScope {
    fn enter() -> Self {
        IN_RUNTIME.with(|flag| flag.set(true));
        Self
    }
}
impl Drop for RuntimeScope {
    fn drop(&mut self) {
        IN_RUNTIME.with(|flag| flag.set(false));
    }
}
thread_local! { static CURRENT: RefCell<Option<Thread>> = const { RefCell::new(None) }; }
#[derive(Debug)]
struct Cancellation;
impl Thread {
    pub fn current() -> Option<Self> {
        CURRENT.with(|slot| {
            slot.borrow()
                .as_ref()
                .filter(|t| Fiber::current_task_id() == Some(t.current_fiber))
                .cloned()
        })
    }
    pub fn current_task() -> Option<TaskHandle> {
        CURRENT.with(|slot| {
            slot.borrow()
                .as_ref()
                .filter(|t| Fiber::current_task_id() == Some(t.current_fiber))
                .map(|t| t.current_task.clone())
        })
    }
    pub fn is_current() -> bool {
        CURRENT.with(|slot| {
            slot.borrow()
                .as_ref()
                .is_some_and(|t| Fiber::current_task_id() == Some(t.current_fiber))
        })
    }
    pub fn scheduler_event_loop(&self) -> EventLoop {
        self.shared.event_loop.clone()
    }
    pub fn spawn<F: FnOnce() + Send + Unpin + 'static>(f: F) -> Result<TaskHandle, SpawnError> {
        let current = Self::current().expect("Thread::spawn requires a scheduler fiber");
        current.shared.submit(Task::new(f)?)
    }
    pub fn enqueue(task: Task) -> Result<TaskHandle, SpawnError> {
        Self::current()
            .expect("enqueue requires a scheduler fiber")
            .shared
            .submit(task)
    }
    /// Register this waker with your event source before calling park().
    pub fn prepare_park() -> FiberWaker {
        FiberWaker {
            token: Self::token(),
        }
    }
    /// Suspend until notified or cancelled. Unlike the internal mutex protocol,
    /// this public primitive performs cancellation checkpoints automatically.
    pub fn park(waker: &FiberWaker) {
        Self::checkpoint();
        let current = Self::current().expect("park requires a scheduler fiber");
        assert_eq!(
            waker.token.task_id,
            current.current_task.id(),
            "waker belongs to another task"
        );
        Self::park_token(&waker.token);
        Self::checkpoint();
    }

    pub fn cancellation_requested() -> bool {
        CURRENT.with(|slot| {
            slot.borrow().as_ref().is_some_and(|t| {
                Fiber::current_task_id() == Some(t.current_fiber) && t.current_task.requested() != 0
            })
        })
    }
    /// Cooperatively cancel by unwinding the current stack, preserving RAII.
    pub fn checkpoint() {
        if Self::cancellation_requested() && !thread::panicking() {
            std::panic::resume_unwind(Box::new(Cancellation));
        }
    }
    pub fn yield_now() {
        Self::checkpoint();
        assert!(Self::is_current(), "yield requires a scheduler fiber");
        Fiber::yield_now().expect("scheduler fiber context");
        Self::checkpoint();
    }
    pub(crate) fn token() -> Arc<ParkToken> {
        let current = Self::current().expect("operation requires a scheduler fiber");
        Arc::new(ParkToken {
            task_id: current.current_task.id(),
            ready: AtomicBool::new(false),
            mailbox: Arc::downgrade(&current.shared.mailboxes[current.index]),
        })
    }
    /// Internal parking primitive deliberately does not unwind on cancellation;
    /// synchronization objects first remove their wait records / relinquish grants.
    pub(crate) fn park_token(token: &Arc<ParkToken>) {
        let task = Self::current_task().expect("park requires a scheduler fiber");
        *lock(&task.inner.parked) = Some(Arc::downgrade(token));
        if task.requested() != 0 {
            token.wake();
        }
        if !token.ready.load(Ordering::Acquire) {
            CURRENT.with(|slot| slot.borrow_mut().as_mut().unwrap().parked = Some(token.clone()));
            Fiber::yield_now().expect("scheduler fiber context");
        }
        *lock(&task.inner.parked) = None;
    }
    pub fn wait(vow: &Vow) -> VowState {
        Self::checkpoint();
        if !vow.is_pending() {
            return vow.state();
        }
        let token = Self::token();
        let weak = Arc::downgrade(&token);
        let subscription = vow.on_settle(move || {
            if let Some(token) = weak.upgrade() {
                token.wake();
            }
        });
        Self::park_token(&token);
        drop(subscription);
        Self::checkpoint();
        vow.state()
    }
    pub fn sleep(duration: Duration) {
        let current = Self::current().expect("sleep requires a scheduler fiber");
        let vow = current.shared.event_loop.set_timeout(duration, || {});
        struct CancelTimer(Vow);
        impl Drop for CancelTimer {
            fn drop(&mut self) {
                self.0.cancel();
            }
        }
        let timer = CancelTimer(vow);
        Self::wait(&timer.0);
    }
}

struct Live {
    fiber: Fiber,
    task: TaskHandle,
    parked: Option<Arc<ParkToken>>,
}
fn steal_task(local: &Worker<Task>, shared: &Shared, index: usize) -> Option<Task> {
    if let Some(task) = local.pop() {
        return Some(task);
    }
    loop {
        let mut result = shared.incoming.steal_batch_and_pop(local);
        for offset in 1..shared.stealers.len() {
            if result.is_success() {
                break;
            }
            result = result
                .or_else(|| shared.stealers[(index + offset) % shared.stealers.len()].steal());
        }
        match result {
            Steal::Success(task) => return Some(task),
            Steal::Empty => return None,
            Steal::Retry => std::hint::spin_loop(),
        }
    }
}
fn worker_loop(index: usize, unstarted: Worker<Task>, shared: Arc<Shared>) {
    let _scope = RuntimeScope::enter();
    // These native Fiber queues are intentionally never exposed as Stealers.
    let ready = Worker::<Live>::new_fifo();
    // Slot indirection permits O(1) wake lookup while retaining the requested
    // blocked deque. Empty slots are compacted amortized, never scanned per wake.
    let blocked = Worker::<Rc<RefCell<Option<Live>>>>::new_fifo();
    let mut parked = HashMap::<u64, Rc<RefCell<Option<Live>>>>::new();
    let mailbox = &shared.mailboxes[index];
    let mut prefer_new = false;
    loop {
        let epoch = *lock(&mailbox.epoch);
        loop {
            match mailbox.notifications.steal() {
                Steal::Success(token) => {
                    let matched = parked.get(&token.task_id).is_some_and(|slot| {
                        slot.borrow().as_ref().is_some_and(|live| {
                            live.parked.as_ref().is_some_and(|p| Arc::ptr_eq(p, &token))
                        })
                    });
                    if matched {
                        let slot = parked.remove(&token.task_id).unwrap();
                        let mut live = slot.borrow_mut().take().unwrap();
                        live.parked = None;
                        ready.push(live);
                    }
                }
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }
        if blocked.len() > parked.len().saturating_mul(2).saturating_add(64) {
            for _ in 0..blocked.len() {
                let slot = blocked.pop().unwrap();
                if slot.borrow().is_some() {
                    blocked.push(slot);
                }
            }
        }
        // Alternate new and resumed work to avoid starvation in either queue.
        let task = if prefer_new || ready.is_empty() {
            steal_task(&unstarted, &shared, index)
        } else {
            None
        };
        if let Some(mut task) = task {
            let handle = task.handle();
            if handle.requested() != 0 {
                drop(task);
                shared.done(&handle, false);
            } else {
                let fiber = Fiber::new(shared.config.stack_size, move || {
                    Thread::checkpoint();
                    task.callable.call0();
                });
                match fiber {
                    Ok(mut fiber) => {
                        fiber.attach_task(handle.clone());
                        ready.push(Live {
                            fiber,
                            task: handle,
                            parked: None,
                        });
                    }
                    Err(_) => shared.done(&handle, true),
                }
            }
        }
        prefer_new = !prefer_new;
        if let Some(mut live) = ready.pop() {
            CURRENT.with(|slot| {
                *slot.borrow_mut() = Some(Thread {
                    index,
                    current_fiber: live.task.id(),
                    current_task: live.task.clone(),
                    shared: shared.clone(),
                    parked: None,
                    _affine: PhantomData,
                })
            });
            let result = live.fiber.resume();
            let context = CURRENT.with(|slot| slot.borrow_mut().take().unwrap());
            if live.fiber.is_finished() || result.is_err() {
                let panic = live.fiber.take_panic();
                let failed =
                    result.is_err() || panic.as_ref().is_some_and(|p| !p.is::<Cancellation>());
                let task = live.task.clone();
                // Stack, locals and entry captures are dropped before completion.
                let dropped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    drop(live);
                    drop(panic);
                }));
                shared.done(&task, failed || dropped.is_err());
            } else if let Some(token) = context.parked {
                if token.ready.load(Ordering::Acquire) {
                    ready.push(live);
                } else {
                    live.fiber.mark_blocked();
                    live.parked = Some(token);
                    let task_id = live.task.id();
                    let slot = Rc::new(RefCell::new(Some(live)));
                    parked.insert(task_id, slot.clone());
                    blocked.push(slot);
                }
            } else {
                ready.push(live);
            }
            continue;
        }
        if !unstarted.is_empty() || !shared.incoming.is_empty() {
            continue;
        }
        let registry = lock(&shared.registry);
        if registry.closed && registry.tasks.is_empty() {
            break;
        }
        drop(registry);
        let mut observed = lock(&mailbox.epoch);
        while *observed == epoch {
            observed = mailbox
                .wake
                .wait(observed)
                .unwrap_or_else(|e| e.into_inner());
        }
    }
}
