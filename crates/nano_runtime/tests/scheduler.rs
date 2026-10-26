use nano_runtime::{Config, Mutex, Task, TaskScheduler, Thread, Vow, VowState};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;
fn pool(workers: usize) -> TaskScheduler {
    TaskScheduler::with_config(Config {
        workers,
        ..Config::default()
    })
    .unwrap()
}
fn finished(task: &nano_runtime::TaskHandle, expected: VowState) {
    assert_eq!(task.wait_timeout(Duration::from_secs(10)), expected);
}
#[test]
fn many_tasks_and_affinity() {
    let pool = pool(4);
    let total = Arc::new(AtomicUsize::new(0));
    let mut tasks = Vec::new();
    for _ in 0..1000 {
        let total = total.clone();
        tasks.push(
            pool.spawn(move || {
                let owner = std::thread::current().id();
                for _ in 0..8 {
                    Thread::yield_now();
                    assert_eq!(owner, std::thread::current().id());
                }
                total.fetch_add(1, Ordering::Relaxed);
            })
            .unwrap(),
        );
    }
    for task in tasks {
        finished(&task, VowState::Resolved);
    }
    assert_eq!(total.load(Ordering::Relaxed), 1000);
    pool.shutdown().unwrap();
}
#[test]
fn mutex_blocks_fiber_not_sole_worker() {
    let pool = pool(1);
    let value = Arc::new(Mutex::new(0));
    let mut tasks = Vec::new();
    for _ in 0..100 {
        let value = value.clone();
        tasks.push(
            pool.spawn(move || {
                for _ in 0..10 {
                    let mut guard = value.lock().unwrap();
                    *guard += 1;
                    Thread::yield_now();
                }
            })
            .unwrap(),
        );
    }
    for task in tasks {
        finished(&task, VowState::Resolved);
    }
    let check = pool
        .spawn(move || assert_eq!(*value.lock().unwrap(), 1000))
        .unwrap();
    finished(&check, VowState::Resolved);
    pool.shutdown().unwrap();
}
#[test]
fn waiters_observe_all_vow_outcomes() {
    let pool = pool(1);
    for outcome in [
        VowState::Resolved,
        VowState::Cancelled,
        VowState::Interrupted,
    ] {
        let vow = Vow::new();
        let wait = vow.clone();
        let task = pool
            .spawn(move || assert_eq!(wait.wait(), outcome))
            .unwrap();
        match outcome {
            VowState::Resolved => {
                vow.resolve();
            }
            VowState::Cancelled => {
                vow.cancel();
            }
            _ => {
                vow.interrupt();
            }
        }
        finished(&task, VowState::Resolved);
    }
    pool.shutdown().unwrap();
}
#[test]
fn cancellation_unwinds_and_wait_does_not_resolve_early() {
    struct CountDrop(Arc<AtomicUsize>);
    impl Drop for CountDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Release);
        }
    }
    let pool = pool(1);
    let entered = Vow::new();
    let notify = entered.clone();
    let dropped = Arc::new(AtomicUsize::new(0));
    let count = dropped.clone();
    let task = pool
        .spawn(move || {
            let _guard = CountDrop(count);
            notify.resolve();
            Vow::new().wait();
        })
        .unwrap();
    assert_eq!(
        entered.wait_timeout(Duration::from_secs(10)),
        VowState::Resolved
    );
    assert!(task.cancel());
    finished(&task, VowState::Cancelled);
    assert_eq!(dropped.load(Ordering::Acquire), 1);
    pool.shutdown().unwrap();
}
#[test]
fn cancelled_mutex_waiter_does_not_lose_ownership() {
    let pool = pool(2);
    let mutex = Arc::new(Mutex::new(0));
    let release = Vow::new();
    let held = Vow::new();
    let (m, r, h) = (mutex.clone(), release.clone(), held.clone());
    let owner = pool
        .spawn(move || {
            let _g = m.lock().unwrap();
            h.resolve();
            r.wait();
        })
        .unwrap();
    assert_eq!(
        held.wait_timeout(Duration::from_secs(10)),
        VowState::Resolved
    );
    let m = mutex.clone();
    let waiting = pool
        .spawn(move || {
            let _guard = m.lock().unwrap();
        })
        .unwrap();
    waiting.cancel();
    release.resolve();
    finished(&waiting, VowState::Cancelled);
    finished(&owner, VowState::Resolved);
    let check = pool
        .spawn(move || {
            *mutex.lock().unwrap_or_else(|e| e.into_inner()) += 1;
        })
        .unwrap();
    finished(&check, VowState::Resolved);
    pool.shutdown().unwrap();
}
#[test]
fn panic_poison_and_worker_survives() {
    let pool = pool(1);
    let m = Arc::new(Mutex::new(0));
    let copy = m.clone();
    let task = pool
        .spawn(move || {
            let _guard = copy.lock().unwrap();
            panic!("test panic");
        })
        .unwrap();
    finished(&task, VowState::Interrupted);
    let check = pool
        .spawn(move || {
            assert!(m.is_poisoned());
            *m.lock().unwrap_or_else(|e| e.into_inner()) = 5;
        })
        .unwrap();
    finished(&check, VowState::Resolved);
    pool.shutdown().unwrap();
}
#[test]
fn child_wait_and_timer_on_one_worker() {
    let pool = pool(1);
    let task = pool
        .spawn(|| {
            let child = Thread::spawn(|| Thread::sleep(Duration::from_millis(5))).unwrap();
            assert_eq!(child.wait(), VowState::Resolved);
        })
        .unwrap();
    finished(&task, VowState::Resolved);
    pool.shutdown().unwrap();
}
#[test]
fn event_panic_isolated_and_timer_continues() {
    let pool = pool(1);
    let events = pool.event_loop();
    let bad = events.next_tick(|| panic!("engine panic"));
    assert_eq!(
        bad.wait_timeout(Duration::from_secs(10)),
        VowState::Interrupted
    );
    let good = events.set_timeout(Duration::from_millis(2), || {});
    assert_eq!(
        good.wait_timeout(Duration::from_secs(10)),
        VowState::Resolved
    );
    pool.shutdown().unwrap();
}
#[test]
fn inline_limit_and_unsubmitted_task_cancellation() {
    let task = Task::new(|| {}).unwrap();
    assert!(task.is_inline());
    let handle = task.handle();
    drop(task);
    assert_eq!(handle.state(), VowState::Cancelled);
    let big = [1u8; 2048];
    assert!(
        Task::new(move || {
            std::hint::black_box(big);
        })
        .is_err()
    );
}
#[test]
fn shutdown_wakes_blocked_tasks_and_rejects_admission() {
    let pool = pool(2);
    let tasks: Vec<_> = (0..100)
        .map(|_| {
            pool.spawn(|| {
                Vow::new().wait();
            })
            .unwrap()
        })
        .collect();
    pool.shutdown().unwrap();
    for task in tasks {
        assert_eq!(task.state(), VowState::Cancelled);
    }
    assert!(pool.spawn(|| {}).is_err());
    pool.shutdown().unwrap();
}
#[test]
fn wake_before_park_and_multiple_waiters_stress() {
    let pool = pool(4);
    for _ in 0..100 {
        let vow = Vow::new();
        let mut tasks = Vec::new();
        for _ in 0..16 {
            let copy = vow.clone();
            tasks.push(
                pool.spawn(move || {
                    assert_eq!(copy.wait(), VowState::Resolved);
                })
                .unwrap(),
            );
        }
        vow.resolve();
        for task in tasks {
            finished(&task, VowState::Resolved);
        }
    }
    pool.shutdown().unwrap();
}

#[test]
fn explicit_waker_supports_early_and_cross_thread_notification() {
    let pool = pool(1);
    let (send, receive) = std::sync::mpsc::channel();
    let task = pool
        .spawn(move || {
            let early = Thread::prepare_park();
            early.wake();
            Thread::park(&early);
            let later = Thread::prepare_park();
            send.send(later.clone()).unwrap();
            Thread::park(&later);
        })
        .unwrap();
    receive
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .wake();
    finished(&task, VowState::Resolved);
    pool.shutdown().unwrap();
}
#[test]
fn interrupt_and_bounded_admission() {
    let pool = TaskScheduler::with_config(Config {
        workers: 1,
        max_tasks: 1,
        ..Config::default()
    })
    .unwrap();
    let task = pool
        .spawn(|| {
            Vow::new().wait();
        })
        .unwrap();
    assert!(matches!(
        pool.spawn(|| {}),
        Err(nano_runtime::SpawnError::Full)
    ));
    assert!(task.interrupt());
    finished(&task, VowState::Interrupted);
    pool.shutdown().unwrap();
}
#[test]
fn event_shutdown_settles_pending_timers_and_rejects_new_work() {
    let pool = pool(1);
    let events = pool.event_loop();
    let pending = events.set_timeout(Duration::from_secs(3600), || {});
    pool.shutdown().unwrap();
    assert_eq!(pending.state(), VowState::Cancelled);
    assert_eq!(events.next_tick(|| {}).state(), VowState::Cancelled);
}
#[test]
fn fiber_locals_cannot_be_reentrantly_borrowed_or_suspended() {
    let mut fiber = nano_runtime::fiber::Fiber::new(64 * 1024, || {
        nano_runtime::fiber::Fiber::with_current_locals(|locals| {
            locals.set("value", 1usize);
            assert!(nano_runtime::fiber::Fiber::with_current_locals(|_| {}).is_err());
            assert!(nano_runtime::fiber::Fiber::yield_now().is_err());
        })
        .unwrap();
        nano_runtime::fiber::Fiber::yield_now().unwrap();
    })
    .unwrap();
    fiber.resume().unwrap();
    fiber.resume().unwrap();
}
