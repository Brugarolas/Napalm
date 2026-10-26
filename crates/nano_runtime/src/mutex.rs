//! FIFO fiber mutex. OS mutexes only protect short internal metadata operations.
use crate::scheduler::{ParkToken, Thread};
use std::{
    cell::UnsafeCell,
    collections::VecDeque,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::{Arc, LockResult, Mutex as OsMutex, PoisonError, TryLockError, TryLockResult},
};

struct Waiter {
    owner: u64,
    token: Arc<ParkToken>,
}
struct State {
    owner: Option<u64>,
    waiters: VecDeque<Waiter>,
    poisoned: bool,
}
pub struct Mutex<T: ?Sized> {
    state: OsMutex<State>,
    value: UnsafeCell<T>,
}
// SAFETY: the ownership grant is unique and transferred under state's OS mutex.
// Access to T exists only through a non-Send guard for the granted task.
unsafe impl<T: ?Sized + Send> Send for Mutex<T> {}
unsafe impl<T: ?Sized + Send> Sync for Mutex<T> {}
impl<T> Mutex<T> {
    pub fn new(value: T) -> Self {
        Self {
            state: OsMutex::new(State {
                owner: None,
                waiters: VecDeque::new(),
                poisoned: false,
            }),
            value: UnsafeCell::new(value),
        }
    }
    pub fn into_inner(self) -> LockResult<T> {
        let poisoned = self
            .state
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
            .poisoned;
        let value = self.value.into_inner();
        if poisoned {
            Err(PoisonError::new(value))
        } else {
            Ok(value)
        }
    }
}
impl<T: Default> Default for Mutex<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}
impl<T: ?Sized> Mutex<T> {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
    fn guard(&self, owner: u64, poisoned: bool) -> LockResult<MutexGuard<'_, T>> {
        let guard = MutexGuard {
            mutex: self,
            owner,
            panicking: std::thread::panicking(),
            _affine: PhantomData,
        };
        if poisoned {
            Err(PoisonError::new(guard))
        } else {
            Ok(guard)
        }
    }
    /// Requires a scheduler task. Recursive acquisition panics instead of
    /// allowing two mutable references or silently deadlocking the same fiber.
    pub fn lock(&self) -> LockResult<MutexGuard<'_, T>> {
        Thread::checkpoint();
        let owner = Thread::current_task()
            .expect("fiber Mutex requires a scheduler task")
            .id();
        let token = Thread::token();
        {
            let mut state = self.state();
            assert_ne!(
                state.owner,
                Some(owner),
                "recursive fiber Mutex acquisition"
            );
            if state.owner.is_none() {
                state.owner = Some(owner);
                return self.guard(owner, state.poisoned);
            }
            state.waiters.push_back(Waiter {
                owner,
                token: token.clone(),
            });
        }
        Thread::park_token(&token);
        let mut state = self.state();
        if Thread::cancellation_requested() && !std::thread::panicking() {
            let next = if state.owner == Some(owner) {
                Self::release(&mut state)
            } else {
                state.waiters.retain(|waiter| waiter.owner != owner);
                None
            };
            drop(state);
            if let Some(next) = next {
                next.wake();
            }
            Thread::checkpoint();
            unreachable!("cancellation checkpoint must unwind");
        }
        assert_eq!(state.owner, Some(owner), "wake without ownership grant");
        self.guard(owner, state.poisoned)
    }
    pub fn try_lock(&self) -> TryLockResult<MutexGuard<'_, T>> {
        Thread::checkpoint();
        let owner = Thread::current_task()
            .expect("fiber Mutex requires a scheduler task")
            .id();
        let mut state = self.state();
        if state.owner.is_some() {
            return Err(TryLockError::WouldBlock);
        }
        state.owner = Some(owner);
        self.guard(owner, state.poisoned)
            .map_err(TryLockError::Poisoned)
    }
    fn release(state: &mut State) -> Option<Arc<ParkToken>> {
        let next = state.waiters.pop_front();
        state.owner = next.as_ref().map(|waiter| waiter.owner);
        next.map(|waiter| waiter.token)
    }
    pub fn is_poisoned(&self) -> bool {
        self.state().poisoned
    }
    pub fn clear_poison(&self) {
        self.state().poisoned = false;
    }
    pub fn get_mut(&mut self) -> LockResult<&mut T> {
        let poisoned = self
            .state
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .poisoned;
        let value = self.value.get_mut();
        if poisoned {
            Err(PoisonError::new(value))
        } else {
            Ok(value)
        }
    }
}

#[must_use = "dropping the guard releases the mutex"]
pub struct MutexGuard<'a, T: ?Sized> {
    mutex: &'a Mutex<T>,
    owner: u64,
    panicking: bool,
    _affine: PhantomData<Rc<()>>,
}
impl<T: ?Sized> Deref for MutexGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: this guard holds the unique ownership grant.
        unsafe { &*self.mutex.value.get() }
    }
}
impl<T: ?Sized> DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: the grant is exclusive and &mut self prevents aliasing guards.
        unsafe { &mut *self.mutex.value.get() }
    }
}
impl<T: ?Sized> Drop for MutexGuard<'_, T> {
    fn drop(&mut self) {
        let mut state = self.mutex.state();
        debug_assert_eq!(state.owner, Some(self.owner));
        state.poisoned |= !self.panicking && std::thread::panicking();
        let next = Mutex::<T>::release(&mut state);
        drop(state);
        if let Some(next) = next {
            next.wake();
        }
    }
}
