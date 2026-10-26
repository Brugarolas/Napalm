use nano_runtime::{Mutex, TaskScheduler, Thread, VowState};
use std::{sync::Arc, time::Duration};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scheduler = TaskScheduler::new()?;
    let counter = Arc::new(Mutex::new(0usize));
    let mut handles = Vec::new();
    for _ in 0..128 {
        let counter = counter.clone();
        handles.push(scheduler.spawn(move || {
            for _ in 0..100 {
                let mut value = counter.lock().unwrap();
                *value += 1;
                // This intentionally yields while holding the guard to exercise
                // fiber parking; other tasks never block the OS worker here.
                Thread::yield_now();
            }
            Thread::sleep(Duration::from_millis(1));
        })?);
    }
    for task in handles {
        assert_eq!(task.wait(), VowState::Resolved);
    }
    let result = scheduler.spawn(move || println!("counter = {}", *counter.lock().unwrap()))?;
    assert_eq!(result.wait(), VowState::Resolved);
    let tick = scheduler
        .event_loop()
        .next_tick(|| println!("engine microtask"));
    assert_eq!(tick.wait(), VowState::Resolved);
    scheduler.shutdown()?;
    Ok(())
}
