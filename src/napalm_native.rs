//! Native Rust decorators: `@memoize`, `@once`, `@memoizeExpiring(ttl)`,
//! `@delay(ms)`, `@debounce(ms, options)` and `@throttle(ms, options)`.
//!
//! Applying a decorator returns an ordinary bytecode closure with exactly the
//! arity of the decorated callable (`install_hir_builtins` builds one wrapper per
//! arity). The closure captures one GC-managed `NativeDecorator` sidecar and runs
//! three small native steps:
//!
//! * `state.__enter(args)` answers cache hits, schedules timing decorators and
//!   returns `state` itself for a cache miss;
//! * `state.__run(args)` calls the original callable through the normal bytecode
//!   ABI (no nested interpreter loop, no blocking);
//! * `state.__store(result, args)` records a successful result. A synchronous
//!   exception unwinds past `__store`, so failures are never cached.
//!
//! All durations are milliseconds. Timing decorators always return Promises and
//! run the original as a cooperative task; calls coalesced into one invocation
//! share that invocation's Promise. Member decorators (methods, getters and
//! setters receive the explicit receiver as argument 0) keep once/debounce/
//! throttle state per receiver.

use super::*;

/// Wrapper closures exist for every arity the call ABI supports.
pub(super) const MAX_DECORATED_ARITY: usize = ARGUMENT_REGISTER_COUNT;
/// Largest delay accepted by timing decorators (the JavaScript timer limit).
const MAX_TIMER_MS: f64 = 2_147_483_647.0;
/// Expired `@memoizeExpiring` entries are swept once the table doubles.
const MIN_EXPIRY_SWEEP: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EdgeConfig {
    pub wait: Duration,
    pub leading: bool,
    pub trailing: bool,
    pub max_wait: Option<Duration>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DecoratorConfig {
    Memoize,
    Once,
    MemoizeExpiring(Duration),
    Delay(Duration),
    Debounce(EdgeConfig),
    Throttle(EdgeConfig),
}

impl DecoratorConfig {
    fn label(self) -> &'static str {
        match self {
            Self::Memoize => "memoize",
            Self::Once => "once",
            Self::MemoizeExpiring(_) => "memoizeExpiring",
            Self::Delay(_) => "delay",
            Self::Debounce(_) => "debounce",
            Self::Throttle(_) => "throttle",
        }
    }
}

/// GC-managed decorator sidecar. `target` is `None` for a configured factory
/// result such as `debounce(100)` that has not been applied yet.
#[derive(Debug)]
pub(super) struct NativeDecorator {
    config: DecoratorConfig,
    target: Option<DecoratedTarget>,
}

#[derive(Debug)]
struct DecoratedTarget {
    original: Value,
    function: u32,
    member: bool,
    state: TargetState,
}

#[derive(Debug)]
enum TargetState {
    Cache {
        entries: FastMap<Box<[MapKey]>, CacheEntry>,
        sweep_at: usize,
    },
    Delay {
        queued: FastMap<u64, QueuedCall>,
        next_token: u64,
    },
    Edge {
        slots: FastMap<u64, EdgeSlot>,
        partitions: FastMap<MapKey, u64>,
        next_token: u64,
    },
}

#[derive(Debug)]
struct CacheEntry {
    value: Value,
    inserted: Instant,
}

#[derive(Debug)]
struct QueuedCall {
    args: Box<[Value]>,
    promise: Value,
}

#[derive(Debug)]
struct EdgeSlot {
    partition: MapKey,
    machine: EdgeMachine,
    pending_args: Box<[Value]>,
    /// Promise shared by every call coalesced into the next trailing invocation.
    pending_promise: Value,
    /// Promise of the latest actual invocation; reused by suppressed calls when
    /// `trailing` is disabled.
    latest: Value,
    /// Armed timer and its deadline; stale fired events are ignored.
    timer: Option<(EventId, Instant)>,
}

impl TargetState {
    fn new(config: DecoratorConfig) -> Self {
        match config {
            DecoratorConfig::Memoize | DecoratorConfig::Once | DecoratorConfig::MemoizeExpiring(_) => {
                Self::Cache { entries: FastMap::default(), sweep_at: MIN_EXPIRY_SWEEP }
            }
            DecoratorConfig::Delay(_) => Self::Delay { queued: FastMap::default(), next_token: 0 },
            DecoratorConfig::Debounce(_) | DecoratorConfig::Throttle(_) => Self::Edge {
                slots: FastMap::default(),
                partitions: FastMap::default(),
                next_token: 0,
            },
        }
    }
}

fn map_key_handle(key: &MapKey) -> Option<u32> {
    match key {
        MapKey::Object(handle) => Some(*handle),
        _ => None,
    }
}

impl NativeDecorator {
    /// Visit every managed reference owned by this sidecar: the original
    /// callable, cached results, object identities used as cache keys and the
    /// arguments/Promises of queued invocations.
    pub(super) fn visit_references(&self, mut value: impl FnMut(Value), mut handle: impl FnMut(u32)) {
        let Some(target) = &self.target else { return; };
        value(target.original);
        match &target.state {
            TargetState::Cache { entries, .. } => {
                for (key, entry) in entries.iter() {
                    for part in key.iter() {
                        if let Some(object) = map_key_handle(part) { handle(object); }
                    }
                    value(entry.value);
                }
            }
            TargetState::Delay { queued, .. } => {
                for call in queued.values() {
                    for &arg in call.args.iter() { value(arg); }
                    value(call.promise);
                }
            }
            TargetState::Edge { slots, .. } => {
                for slot in slots.values() {
                    if let Some(object) = map_key_handle(&slot.partition) { handle(object); }
                    for &arg in slot.pending_args.iter() { value(arg); }
                    value(slot.pending_promise);
                    value(slot.latest);
                }
            }
        }
    }

    /// A factory result such as `debounce(100)` that can still be applied.
    pub(super) fn is_configured(&self) -> bool {
        self.target.is_none()
    }

    pub(super) fn external_bytes(&self) -> usize {
        let Some(target) = &self.target else { return 0; };
        match &target.state {
            TargetState::Cache { entries, .. } => {
                entries.capacity() * std::mem::size_of::<(Box<[MapKey]>, CacheEntry)>()
                    + entries.keys().map(|key| {
                        key.len() * std::mem::size_of::<MapKey>()
                            + key.iter().map(|part| match part {
                                MapKey::String(text) => text.as_ref().len(),
                                _ => 0,
                            }).sum::<usize>()
                    }).sum::<usize>()
            }
            TargetState::Delay { queued, .. } => {
                queued.capacity() * std::mem::size_of::<(u64, QueuedCall)>()
                    + queued.values().map(|call| call.args.len() * std::mem::size_of::<Value>()).sum::<usize>()
            }
            TargetState::Edge { slots, partitions, .. } => {
                slots.capacity() * std::mem::size_of::<(u64, EdgeSlot)>()
                    + partitions.capacity() * std::mem::size_of::<(MapKey, u64)>()
                    + slots.values().map(|slot| slot.pending_args.len() * std::mem::size_of::<Value>()).sum::<usize>()
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Pure debounce/throttle state machine. It never reads the clock; callers pass
// the monotonic `now`, which keeps maxWait/boundary behavior unit-testable.
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EdgeCall {
    /// Invoke the original immediately (leading edge or maxWait-forced leading).
    InvokeNow,
    /// Join the pending trailing invocation.
    Pending,
    /// Suppressed with `trailing: false`; reuse the latest invocation Promise.
    ReuseLatest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EdgeTimer {
    /// Run the pending trailing invocation now.
    Flush,
    /// Nothing to run; `is_active` tells whether the burst ended.
    Idle,
    /// Fired before the (moved) deadline; re-arm for `next_deadline`.
    Wait,
}

#[derive(Clone, Debug)]
pub(super) struct EdgeMachine {
    config: EdgeConfig,
    active: bool,
    has_pending: bool,
    quiet_deadline: Option<Instant>,
    max_deadline: Option<Instant>,
}

impl EdgeMachine {
    pub(super) fn new(config: EdgeConfig) -> Self {
        Self { config, active: false, has_pending: false, quiet_deadline: None, max_deadline: None }
    }

    #[inline]
    pub(super) fn is_active(&self) -> bool {
        self.active
    }

    #[inline]
    pub(super) fn has_pending(&self) -> bool {
        self.has_pending
    }

    pub(super) fn on_call(&mut self, now: Instant) -> EdgeCall {
        let config = self.config;
        self.quiet_deadline = Some(now + config.wait);
        if !self.active {
            self.active = true;
            self.max_deadline = config.max_wait.map(|max| now + max);
            if config.leading {
                return EdgeCall::InvokeNow;
            }
            self.has_pending = true;
            return EdgeCall::Pending;
        }
        if config.trailing {
            self.has_pending = true;
            return EdgeCall::Pending;
        }
        if self.max_deadline.is_some_and(|max| now >= max) {
            self.max_deadline = config.max_wait.map(|max| now + max);
            return EdgeCall::InvokeNow;
        }
        EdgeCall::ReuseLatest
    }

    /// The next instant at which `on_timer` has work to do.
    pub(super) fn next_deadline(&self) -> Option<Instant> {
        if !self.active {
            return None;
        }
        let quiet = self.quiet_deadline?;
        Some(match (self.has_pending, self.max_deadline) {
            (true, Some(max)) => quiet.min(max),
            _ => quiet,
        })
    }

    pub(super) fn on_timer(&mut self, now: Instant) -> EdgeTimer {
        if !self.active {
            return EdgeTimer::Idle;
        }
        if self.quiet_deadline.is_some_and(|quiet| now >= quiet) {
            let flush = self.has_pending;
            self.active = false;
            self.has_pending = false;
            self.quiet_deadline = None;
            self.max_deadline = None;
            return if flush { EdgeTimer::Flush } else { EdgeTimer::Idle };
        }
        if self.has_pending && self.max_deadline.is_some_and(|max| now >= max) {
            self.has_pending = false;
            self.max_deadline = self.config.max_wait.map(|max| now + max);
            return EdgeTimer::Flush;
        }
        EdgeTimer::Wait
    }
}

/// Milliseconds -> Duration for timer-backed decorators.
pub(super) fn timer_duration_from_ms(ms: f64) -> Option<Duration> {
    (ms.is_finite() && (0.0..=MAX_TIMER_MS).contains(&ms)).then(|| Duration::from_secs_f64(ms / 1000.0))
}

/// Milliseconds -> Duration for cache TTLs (not limited by the timer range).
pub(super) fn ttl_from_ms(ms: f64) -> Option<Duration> {
    if !ms.is_finite() || ms < 0.0 {
        return None;
    }
    Duration::try_from_secs_f64(ms / 1000.0).ok()
}

pub(super) fn edge_config(
    label: &str,
    wait: Duration,
    leading: bool,
    trailing: bool,
    max_wait: Option<Duration>,
    throttle: bool,
) -> Result<EdgeConfig, String> {
    if !leading && !trailing {
        return Err(format!(
            "@{label} requires leading or trailing to be true; with both false it would never invoke"
        ));
    }
    let max_wait = if throttle { Some(wait) } else { max_wait.map(|max| max.max(wait)) };
    Ok(EdgeConfig { wait, leading, trailing, max_wait })
}

fn string_key(name: &str) -> MapKey {
    MapKey::String(MapStringKey::new(Arc::from(name.as_bytes())))
}

fn decorator_target_mut(heap: &mut Heap, state: Value) -> Result<&mut DecoratedTarget, RuntimeError> {
    match heap.get_mut(state) {
        Some(HeapObject::NativeDecorator(decorator)) => decorator
            .target
            .as_mut()
            .ok_or_else(|| RuntimeError::new("native decorator has not been applied")),
        _ => Err(RuntimeError::new("native decorator state is invalid")),
    }
}

fn decorator_target(heap: &Heap, state: Value) -> Result<(DecoratorConfig, &DecoratedTarget), RuntimeError> {
    match heap.get(state) {
        Some(HeapObject::NativeDecorator(decorator)) => decorator
            .target
            .as_ref()
            .map(|target| (decorator.config, target))
            .ok_or_else(|| RuntimeError::new("native decorator has not been applied")),
        _ => Err(RuntimeError::new("native decorator state is invalid")),
    }
}

fn edge_slot_mut(heap: &mut Heap, state: Value, token: u64) -> Option<&mut EdgeSlot> {
    match &mut decorator_target_mut(heap, state).ok()?.state {
        TargetState::Edge { slots, .. } => slots.get_mut(&token),
        _ => None,
    }
}

impl Interpreter<'_> {
    pub fn bind_native_decorator_wrappers(&mut self, wrappers: &[FuncId]) {
        self.native_decorator_wrappers = wrappers
            .iter()
            .map(|function| {
                *self.prepared.function_by_id.get(function)
                    .expect("native decorator wrapper must exist in the compiled program")
            })
            .collect();
    }

    fn finish_native_decorator(&mut self, op: &PreparedOp, result: Result<Value, RuntimeError>) {
        match result {
            Ok(value) => unsafe {
                self.set_reg(op.a, value.word());
                self.advance(op);
            },
            Err(error) => self.fail_at(op, error.message),
        }
    }

    /// `memoize(value, context)`, `delay(ms)`, `debounce(ms, options)`, ...
    pub(super) fn native_decorator_builtin_call(&mut self, op: &PreparedOp, builtin: BuiltinId, args: &[RegisterWord]) {
        self.mark_gc_point(op);
        self.maybe_collect();
        let result = self.decorator_builtin_call(builtin, args);
        self.finish_native_decorator(op, result);
    }

    /// `configured.call(value, context)` for a factory result.
    pub(super) fn native_decorator_apply(&mut self, op: &PreparedOp, decorator: Value, args: &[RegisterWord]) {
        self.mark_gc_point(op);
        self.maybe_collect();
        let arg = |index: usize| args.get(index).copied().map(Value::from_word).unwrap_or(Value::NULL);
        let config = match self.heap.get(decorator) {
            Some(HeapObject::NativeDecorator(state)) if state.target.is_none() => Some(state.config),
            _ => None,
        };
        let result = match config {
            Some(config) => self.apply_native_decorator(config, arg(0), arg(1)),
            None => Err(RuntimeError::new("an applied native decorator cannot be applied again")),
        };
        self.finish_native_decorator(op, result);
    }

    pub(super) fn native_decorator_enter(&mut self, op: &PreparedOp, state: Value, args: &[RegisterWord]) {
        self.mark_gc_point(op);
        self.maybe_collect();
        let result = self.decorator_enter(state, args);
        self.finish_native_decorator(op, result);
    }

    pub(super) fn native_decorator_run(&mut self, op: &PreparedOp, state: Value, args: &[RegisterWord]) {
        let (original, function) = match decorator_target(&self.heap, state) {
            Ok((_, target)) => (target.original, target.function),
            Err(error) => { self.fail_at(op, error.message); return; }
        };
        self.enter_bytecode_call(op, function, None, args, Some(original));
    }

    pub(super) fn native_decorator_store(&mut self, op: &PreparedOp, state: Value, args: &[RegisterWord]) {
        let result = self.decorator_store(state, args);
        self.finish_native_decorator(op, result);
    }

    /// `__store` variant used at the argument-register limit: the arguments
    /// arrive as one list.
    pub(super) fn native_decorator_store_list(&mut self, op: &PreparedOp, state: Value, result: Value, list: Value) {
        let mut words = vec![result.word()];
        match self.heap.get(list) {
            Some(HeapObject::List(values)) => words.extend(values.iter().map(|value| value.word())),
            _ => { self.fail_at(op, "native decorator store is missing its arguments"); return; }
        }
        let result = self.decorator_store(state, &words);
        self.finish_native_decorator(op, result);
    }
    fn decorator_builtin_call(&mut self, builtin: BuiltinId, args: &[RegisterWord]) -> Result<Value, RuntimeError> {
        let arg = |index: usize| args.get(index).copied().map(Value::from_word).unwrap_or(Value::NULL);
        let label = builtin.name();
        if matches!(builtin, BuiltinId::Memoize | BuiltinId::Once) {
            if args.len() != 2 {
                return Err(RuntimeError::new(format!(
                    "@{label} takes no arguments; write @{label} without parentheses"
                )));
            }
            let config = if builtin == BuiltinId::Memoize { DecoratorConfig::Memoize } else { DecoratorConfig::Once };
            return self.apply_native_decorator(config, arg(0), arg(1));
        }
        let Some(ms) = arg(0).as_num() else {
            return Err(RuntimeError::new(format!(
                "@{label} requires a duration in milliseconds, for example @{label}(100)"
            )));
        };
        let config = match builtin {
            BuiltinId::MemoizeExpiring => {
                if args.len() != 1 {
                    return Err(RuntimeError::new("@memoizeExpiring takes exactly one TTL argument in milliseconds"));
                }
                let ttl = ttl_from_ms(ms).ok_or_else(|| RuntimeError::new(
                    "@memoizeExpiring TTL must be a finite, non-negative number of milliseconds",
                ))?;
                DecoratorConfig::MemoizeExpiring(ttl)
            }
            BuiltinId::Delay => {
                if args.len() != 1 {
                    return Err(RuntimeError::new("@delay takes exactly one delay argument in milliseconds"));
                }
                DecoratorConfig::Delay(Self::decorator_timer_duration(label, "delay", ms)?)
            }
            BuiltinId::Debounce | BuiltinId::Throttle => {
                let throttle = builtin == BuiltinId::Throttle;
                let wait = Self::decorator_timer_duration(label, "wait", ms)?;
                let options = (args.len() > 1).then(|| arg(1));
                let edge = self.decorator_edge_options(label, throttle, wait, options)?;
                if throttle { DecoratorConfig::Throttle(edge) } else { DecoratorConfig::Debounce(edge) }
            }
            _ => return Err(RuntimeError::new("unknown native decorator")),
        };
        Ok(self.heap.alloc(HeapObject::NativeDecorator(Box::new(NativeDecorator { config, target: None }))))
    }

    fn decorator_timer_duration(label: &str, what: &str, ms: f64) -> Result<Duration, RuntimeError> {
        timer_duration_from_ms(ms).ok_or_else(|| RuntimeError::new(format!(
            "@{label} {what} must be a finite number of milliseconds between 0 and {MAX_TIMER_MS}"
        )))
    }

    fn decorator_edge_options(
        &self,
        label: &str,
        throttle: bool,
        wait: Duration,
        options: Option<Value>,
    ) -> Result<EdgeConfig, RuntimeError> {
        let (mut leading, mut trailing, mut max_wait) = (throttle, true, None);
        if let Some(options) = options.filter(|value| !value.is_null()) {
            let Some(HeapObject::Map(map)) = self.heap.get(options) else {
                return Err(RuntimeError::new(format!(
                    "@{label} options must be an object such as {{leading: true, trailing: true}}"
                )));
            };
            let mut recognized = 0usize;
            for name in ["leading", "trailing", "maxWait"] {
                let Some(value) = map.get(&self.heap.map_shapes, &string_key(name)) else { continue; };
                recognized += 1;
                if name == "maxWait" {
                    if throttle {
                        return Err(RuntimeError::new("@throttle does not accept maxWait; its wait already bounds every burst"));
                    }
                    let ms = value.as_num().ok_or_else(|| RuntimeError::new(
                        "@debounce maxWait must be a number of milliseconds",
                    ))?;
                    max_wait = Some(Self::decorator_timer_duration(label, "maxWait", ms)?);
                } else {
                    let enabled = value.as_bool().ok_or_else(|| RuntimeError::new(format!(
                        "@{label} option {name} must be a boolean"
                    )))?;
                    if name == "leading" { leading = enabled; } else { trailing = enabled; }
                }
            }
            if recognized != map.len() {
                return Err(RuntimeError::new(if throttle {
                    "@throttle options only accept leading and trailing"
                } else {
                    "@debounce options only accept leading, trailing and maxWait"
                }));
            }
        }
        edge_config(label, wait, leading, trailing, max_wait, throttle).map_err(RuntimeError::new)
    }

    fn decorator_target_kind(&self, label: &str, context: Value) -> Result<String, RuntimeError> {
        if context.is_null() {
            return Ok("function".to_owned());
        }
        let kind = self.heap.map_get_encoded(context, &string_key("kind")).map_err(|_| {
            RuntimeError::new(format!("@{label} decorator context must be an object"))
        })?;
        match self.heap.get(kind) {
            Some(HeapObject::String(text)) => Ok(String::from_utf8_lossy(text.as_ref()).into_owned()),
            _ => Err(RuntimeError::new(format!("@{label} decorator context has no kind"))),
        }
    }

    fn apply_native_decorator(&mut self, config: DecoratorConfig, value: Value, context: Value) -> Result<Value, RuntimeError> {
        let label = config.label();
        let kind = self.decorator_target_kind(label, context)?;
        let member = match kind.as_str() {
            "function" => false,
            "method" | "getter" | "setter" => true,
            other => {
                return Err(RuntimeError::new(format!(
                    "@{label} cannot decorate a {other}; native decorators support functions, methods, getters and setters"
                )));
            }
        };
        let function = match self.heap.get(value) {
            Some(HeapObject::Closure { function_index, .. }) => *function_index,
            _ => return Err(RuntimeError::new(format!("@{label} target must be a function"))),
        };
        let (arity, is_concurrent) = {
            let prepared = &self.prepared.functions[function as usize];
            (prepared.arity as usize, prepared.is_concurrent)
        };
        if is_concurrent {
            return Err(RuntimeError::new(format!("@{label} cannot decorate a concurrent function")));
        }
        if arity > MAX_DECORATED_ARITY || (member && arity == 0) {
            return Err(RuntimeError::new(format!(
                "@{label} supports at most {} parameters",
                MAX_DECORATED_ARITY - usize::from(member)
            )));
        }
        let wrapper = *self.native_decorator_wrappers.get(arity)
            .ok_or_else(|| RuntimeError::new("native decorator wrappers are not installed"))?;
        let decorator = self.heap.alloc(HeapObject::NativeDecorator(Box::new(NativeDecorator {
            config,
            target: Some(DecoratedTarget { original: value, function, member, state: TargetState::new(config) }),
        })));
        let root = self.add_root(decorator);
        let reference_indices = self.prepared.functions[wrapper as usize].capture_reference_indices.clone();
        let closure = self.heap.alloc(HeapObject::Closure {
            function_index: wrapper,
            captures: vec![decorator.word()].into_boxed_slice(),
            reference_indices,
        });
        self.release_root(root);
        Ok(closure)
    }

    fn decorator_cache_key(&self, config: DecoratorConfig, member: bool, args: &[RegisterWord]) -> Result<Box<[MapKey]>, RuntimeError> {
        let args = match config {
            DecoratorConfig::Once if member => &args[..1.min(args.len())],
            DecoratorConfig::Once => &[][..],
            _ => args,
        };
        args.iter().map(|&word| self.heap.map_key(Value::from_word(word))).collect()
    }

    fn refresh_decorator_bytes(&mut self, state: Value) {
        let bytes = match self.heap.get(state) {
            Some(HeapObject::NativeDecorator(decorator)) => decorator.external_bytes(),
            _ => return,
        };
        self.heap.set_external_bytes(state, bytes);
    }

    fn decorator_enter(&mut self, state: Value, args: &[RegisterWord]) -> Result<Value, RuntimeError> {
        let (config, member) = {
            let (config, target) = decorator_target(&self.heap, state)?;
            (config, target.member)
        };
        match config {
            DecoratorConfig::Memoize | DecoratorConfig::Once | DecoratorConfig::MemoizeExpiring(_) => {
                let key = self.decorator_cache_key(config, member, args)?;
                let now = Instant::now();
                let target = decorator_target_mut(&mut self.heap, state)?;
                let TargetState::Cache { entries, .. } = &mut target.state else {
                    return Err(RuntimeError::new("native decorator cache state is invalid"));
                };
                let Some(entry) = entries.get(&key) else { return Ok(state); };
                if let DecoratorConfig::MemoizeExpiring(ttl) = config {
                    if now.saturating_duration_since(entry.inserted) >= ttl {
                        entries.remove(&key);
                        self.refresh_decorator_bytes(state);
                        return Ok(state);
                    }
                }
                Ok(entry.value)
            }
            DecoratorConfig::Delay(wait) => self.decorator_delay_call(state, wait, args),
            DecoratorConfig::Debounce(_) | DecoratorConfig::Throttle(_) => self.decorator_edge_call(state, member, args),
        }
    }

    fn decorator_store(&mut self, state: Value, args: &[RegisterWord]) -> Result<Value, RuntimeError> {
        let Some((&result, call_args)) = args.split_first() else {
            return Err(RuntimeError::new("native decorator store is missing its result"));
        };
        let value = Value::from_word(result);
        let (config, member) = {
            let (config, target) = decorator_target(&self.heap, state)?;
            (config, target.member)
        };
        let key = self.decorator_cache_key(config, member, call_args)?;
        let now = Instant::now();
        self.heap.remember_owner(state);
        let target = decorator_target_mut(&mut self.heap, state)?;
        let TargetState::Cache { entries, sweep_at } = &mut target.state else {
            return Err(RuntimeError::new("native decorator cache state is invalid"));
        };
        let stored = match config {
            // A re-entrant @once call that completed first keeps the first result.
            DecoratorConfig::Once => entries.entry(key).or_insert(CacheEntry { value, inserted: now }).value,
            _ => {
                entries.insert(key, CacheEntry { value, inserted: now });
                value
            }
        };
        if let DecoratorConfig::MemoizeExpiring(ttl) = config {
            if entries.len() >= *sweep_at {
                entries.retain(|_, entry| now.saturating_duration_since(entry.inserted) < ttl);
                *sweep_at = (entries.len() * 2).max(MIN_EXPIRY_SWEEP);
            }
        }
        self.refresh_decorator_bytes(state);
        Ok(stored)
    }

    fn decorator_promise_pending(&self, promise: Value) -> bool {
        !promise.is_null()
            && self.promise_cell(promise).ok().is_some_and(|cell| {
                matches!(unsafe { cell.get() }, PromiseState::Pending { .. })
            })
    }

    fn decorator_delay_call(&mut self, state: Value, wait: Duration, args: &[RegisterWord]) -> Result<Value, RuntimeError> {
        let args: Box<[Value]> = args.iter().map(|&word| Value::from_word(word)).collect();
        let promise = self.allocate_pending_promise();
        let promise_root = self.add_root(promise);
        self.heap.remember_owner(state);
        let token = {
            let target = decorator_target_mut(&mut self.heap, state)?;
            let TargetState::Delay { queued, next_token } = &mut target.state else {
                return Err(RuntimeError::new("native decorator delay state is invalid"));
            };
            let token = *next_token;
            *next_token += 1;
            queued.insert(token, QueuedCall { args, promise });
            token
        };
        self.refresh_decorator_bytes(state);
        self.release_root(promise_root);
        if self.event_loop.set_timeout(wait, RuntimeEvent::NativeDecoratorFire { state, token }).is_none() {
            if let Ok(TargetState::Delay { queued, .. }) = decorator_target_mut(&mut self.heap, state).map(|target| &mut target.state) {
                queued.remove(&token);
            }
            return Err(RuntimeError::new("@delay duration exceeds the platform timer range"));
        }
        Ok(promise)
    }

    /// Start one invocation of the original as a cooperative task. The original
    /// never runs inline, so every timing-decorated call yields a Promise.
    fn decorator_spawn(&mut self, state: Value, args: &[Value]) -> Result<Value, RuntimeError> {
        let (original, function) = {
            let (_, target) = decorator_target(&self.heap, state)?;
            (target.original, target.function)
        };
        let words: Vec<RegisterWord> = args.iter().map(|value| value.word()).collect();
        self.spawn_async_task(function, None, &words, Some(original), None)
    }

    fn decorator_edge_call(&mut self, state: Value, member: bool, args: &[RegisterWord]) -> Result<Value, RuntimeError> {
        let args: Box<[Value]> = args.iter().map(|&word| Value::from_word(word)).collect();
        let partition = match args.first() {
            Some(&receiver) if member => self.heap.map_key(receiver)?,
            _ => MapKey::Null,
        };
        let config = match decorator_target(&self.heap, state)?.0 {
            DecoratorConfig::Debounce(edge) | DecoratorConfig::Throttle(edge) => edge,
            _ => return Err(RuntimeError::new("native decorator edge state is invalid")),
        };
        let now = Instant::now();
        self.heap.remember_owner(state);
        let (token, outcome, latest, pending_promise) = {
            let target = decorator_target_mut(&mut self.heap, state)?;
            let TargetState::Edge { slots, partitions, next_token } = &mut target.state else {
                return Err(RuntimeError::new("native decorator edge state is invalid"));
            };
            let token = *partitions.entry(partition.clone()).or_insert_with(|| {
                let token = *next_token;
                *next_token += 1;
                token
            });
            let slot = slots.entry(token).or_insert_with(|| EdgeSlot {
                partition,
                machine: EdgeMachine::new(config),
                pending_args: Box::default(),
                pending_promise: Value::NULL,
                latest: Value::NULL,
                timer: None,
            });
            let outcome = slot.machine.on_call(now);
            if outcome == EdgeCall::Pending {
                slot.pending_args = args.clone();
            }
            (token, outcome, slot.latest, slot.pending_promise)
        };
        let result = match outcome {
            EdgeCall::InvokeNow => {
                let promise = self.decorator_spawn(state, &args)?;
                self.heap.remember_owner(state);
                if let Some(slot) = edge_slot_mut(&mut self.heap, state, token) {
                    slot.latest = promise;
                }
                promise
            }
            EdgeCall::Pending if self.decorator_promise_pending(pending_promise) => pending_promise,
            EdgeCall::Pending => {
                let promise = self.allocate_pending_promise();
                self.heap.remember_owner(state);
                if let Some(slot) = edge_slot_mut(&mut self.heap, state, token) {
                    slot.pending_promise = promise;
                }
                promise
            }
            EdgeCall::ReuseLatest => latest,
        };
        let root = self.add_root(result);
        self.arm_decorator_timer(state, token, now);
        self.refresh_decorator_bytes(state);
        self.release_root(root);
        Ok(result)
    }

    /// Arm the slot timer for the machine's next deadline, replacing an armed
    /// timer that would fire too late.
    fn arm_decorator_timer(&mut self, state: Value, token: u64, now: Instant) {
        let Some(slot) = edge_slot_mut(&mut self.heap, state, token) else { return; };
        let Some(deadline) = slot.machine.next_deadline() else { return; };
        if let Some((id, armed)) = slot.timer {
            if armed <= deadline {
                return;
            }
            let _ = self.event_loop.cancel(id);
        }
        let delay = deadline.saturating_duration_since(now);
        slot.timer = self
            .event_loop
            .set_timeout(delay, RuntimeEvent::NativeDecoratorFire { state, token })
            .map(|id| (id, deadline));
    }

    pub(super) fn fire_native_decorator(&mut self, state: Value, token: u64, event: EventId) {
        // The fired event no longer roots `state`; keep it alive across tasks.
        let state_root = self.add_root(state);
        let result = match decorator_target(&self.heap, state).map(|(config, _)| config) {
            Ok(DecoratorConfig::Delay(_)) => self.fire_decorator_delay(state, token),
            Ok(DecoratorConfig::Debounce(_) | DecoratorConfig::Throttle(_)) => self.fire_decorator_edge(state, token, event),
            Ok(_) => Ok(()),
            Err(error) => Err(error),
        };
        self.refresh_decorator_bytes(state);
        self.release_root(state_root);
        if let Err(error) = result {
            self.fail(error);
        }
    }

    fn fire_decorator_delay(&mut self, state: Value, token: u64) -> Result<(), RuntimeError> {
        let (args, promise) = {
            let target = decorator_target_mut(&mut self.heap, state)?;
            let TargetState::Delay { queued, .. } = &mut target.state else { return Ok(()); };
            let Some(call) = queued.get(&token) else { return Ok(()); };
            (call.args.clone(), call.promise)
        };
        // The queued entry keeps args and promise reachable while the task starts.
        let result = if self.decorator_promise_pending(promise) {
            self.decorator_spawn(state, &args).and_then(|task| self.resolve_async_result(promise, task))
        } else {
            Ok(())
        };
        if let Ok(TargetState::Delay { queued, .. }) = decorator_target_mut(&mut self.heap, state).map(|target| &mut target.state) {
            queued.remove(&token);
        }
        result
    }

    fn fire_decorator_edge(&mut self, state: Value, token: u64, event: EventId) -> Result<(), RuntimeError> {
        let now = Instant::now();
        let (outcome, args, promise) = {
            let Some(slot) = edge_slot_mut(&mut self.heap, state, token) else { return Ok(()); };
            if slot.timer.map(|(id, _)| id) != Some(event) {
                return Ok(());
            }
            slot.timer = None;
            let outcome = slot.machine.on_timer(now);
            (outcome, slot.pending_args.clone(), slot.pending_promise)
        };
        let mut result = Ok(());
        if outcome == EdgeTimer::Flush && self.decorator_promise_pending(promise) {
            // The slot still owns args and promise until the task is created.
            result = self.decorator_spawn(state, &args).and_then(|task| {
                self.heap.remember_owner(state);
                if let Some(slot) = edge_slot_mut(&mut self.heap, state, token) {
                    slot.latest = task;
                }
                self.resolve_async_result(promise, task)
            });
        }
        if outcome == EdgeTimer::Flush {
            if let Some(slot) = edge_slot_mut(&mut self.heap, state, token) {
                slot.pending_args = Box::default();
                slot.pending_promise = Value::NULL;
            }
        }
        let finished = edge_slot_mut(&mut self.heap, state, token).is_some_and(|slot| !slot.machine.is_active());
        if finished {
            // Drop the burst state so receivers and results are not retained.
            if let Ok(TargetState::Edge { slots, partitions, .. }) =
                decorator_target_mut(&mut self.heap, state).map(|target| &mut target.state)
            {
                if let Some(slot) = slots.remove(&token) {
                    partitions.remove(&slot.partition);
                }
            }
        } else {
            self.arm_decorator_timer(state, token, now);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    fn debounce(wait: u64, leading: bool, trailing: bool, max_wait: Option<u64>) -> EdgeMachine {
        EdgeMachine::new(edge_config("debounce", ms(wait), leading, trailing, max_wait.map(ms), false).unwrap())
    }

    fn throttle(wait: u64, leading: bool, trailing: bool) -> EdgeMachine {
        EdgeMachine::new(edge_config("throttle", ms(wait), leading, trailing, None, true).unwrap())
    }

    /// Drive the machine: calls at the given offsets, timers fired exactly at
    /// each deadline. Returns (offset, label) for every invocation.
    fn simulate(machine: &mut EdgeMachine, calls: &[u64]) -> Vec<(u64, &'static str)> {
        let start = Instant::now();
        let at = |offset: u64| start + ms(offset);
        let mut invocations = Vec::new();
        let mut calls = calls.iter().copied().peekable();
        loop {
            let deadline = machine.next_deadline().map(|deadline| deadline.duration_since(start).as_millis() as u64);
            match (calls.peek().copied(), deadline) {
                (Some(call), deadline) if deadline.is_none_or(|deadline| call < deadline) => {
                    calls.next();
                    if machine.on_call(at(call)) == EdgeCall::InvokeNow {
                        invocations.push((call, "leading"));
                    }
                }
                (_, Some(deadline)) => {
                    if machine.on_timer(at(deadline)) == EdgeTimer::Flush {
                        invocations.push((deadline, "trailing"));
                    }
                }
                (Some(_), None) => unreachable!("guard accepts calls without a deadline"),
                (None, None) => return invocations,
            }
        }
    }

    #[test]
    fn debounce_trailing_runs_once_after_quiet_period() {
        let mut machine = debounce(30, false, true, None);
        assert_eq!(simulate(&mut machine, &[0, 10, 20]), vec![(50, "trailing")]);
        assert!(!machine.is_active() && !machine.has_pending());
    }

    #[test]
    fn debounce_leading_and_trailing_skip_duplicate_for_lone_call() {
        let mut machine = debounce(30, true, true, None);
        assert_eq!(simulate(&mut machine, &[0]), vec![(0, "leading")]);
        let mut machine = debounce(30, true, true, None);
        assert_eq!(simulate(&mut machine, &[0, 10, 20]), vec![(0, "leading"), (50, "trailing")]);
    }

    #[test]
    fn debounce_leading_only_suppresses_burst_and_restarts_after_quiet() {
        let mut machine = debounce(30, true, false, None);
        assert_eq!(simulate(&mut machine, &[0, 10, 20, 49, 80]), vec![(0, "leading"), (80, "leading")]);
        let start = Instant::now();
        let mut machine = debounce(30, true, false, None);
        assert_eq!(machine.on_call(start), EdgeCall::InvokeNow);
        assert_eq!(machine.on_call(start + ms(5)), EdgeCall::ReuseLatest);
    }

    #[test]
    fn debounce_max_wait_forces_trailing_invocations_during_continuous_calls() {
        let mut machine = debounce(30, false, true, Some(50));
        let calls: Vec<u64> = (0..=12).map(|index| index * 10).collect();
        assert_eq!(
            simulate(&mut machine, &calls),
            vec![(50, "trailing"), (100, "trailing"), (150, "trailing")]
        );
    }

    #[test]
    fn debounce_max_wait_boundary_is_inclusive_and_clamped_to_wait() {
        let start = Instant::now();
        let mut machine = debounce(30, false, true, Some(50));
        assert_eq!(machine.on_call(start), EdgeCall::Pending);
        assert_eq!(machine.on_call(start + ms(25)), EdgeCall::Pending);
        assert_eq!(machine.next_deadline(), Some(start + ms(50)));
        assert_eq!(machine.on_timer(start + ms(49)), EdgeTimer::Wait);
        assert_eq!(machine.on_timer(start + ms(50)), EdgeTimer::Flush);
        assert!(machine.is_active());
        assert_eq!(machine.next_deadline(), Some(start + ms(55)));
        assert_eq!(machine.on_timer(start + ms(55)), EdgeTimer::Idle);
        assert!(!machine.is_active());

        let config = edge_config("debounce", ms(30), false, true, Some(ms(10)), false).unwrap();
        assert_eq!(config.max_wait, Some(ms(30)));
    }

    #[test]
    fn debounce_leading_only_with_max_wait_reinvokes_at_boundary() {
        let mut machine = debounce(30, true, false, Some(40));
        let calls: Vec<u64> = (0..=10).map(|index| index * 10).collect();
        assert_eq!(simulate(&mut machine, &calls), vec![(0, "leading"), (40, "leading"), (80, "leading")]);
    }

    #[test]
    fn throttle_default_runs_leading_then_latest_trailing() {
        let mut machine = throttle(30, true, true);
        assert_eq!(simulate(&mut machine, &[0, 1, 2]), vec![(0, "leading"), (30, "trailing")]);
    }

    #[test]
    fn throttle_continuous_calls_invoke_once_per_window() {
        let mut machine = throttle(30, true, true);
        let calls: Vec<u64> = (0..=10).map(|index| index * 10).collect();
        assert_eq!(
            simulate(&mut machine, &calls),
            vec![(0, "leading"), (30, "trailing"), (60, "trailing"), (90, "trailing"), (120, "trailing")]
        );
        let mut machine = throttle(30, false, true);
        assert_eq!(simulate(&mut machine, &[0, 10]), vec![(30, "trailing")]);
    }

    #[test]
    fn edge_config_rejects_disabled_edges_and_converts_milliseconds() {
        assert!(edge_config("debounce", ms(10), false, false, None, false).is_err());
        assert!(edge_config("throttle", ms(10), false, false, None, true).is_err());
        assert_eq!(edge_config("throttle", ms(10), true, false, None, true).unwrap().max_wait, Some(ms(10)));
        assert_eq!(timer_duration_from_ms(1000.0), Some(ms(1000)));
        assert_eq!(timer_duration_from_ms(-1.0), None);
        assert_eq!(timer_duration_from_ms(f64::INFINITY), None);
        assert_eq!(timer_duration_from_ms(MAX_TIMER_MS + 1.0), None);
        assert_eq!(ttl_from_ms(50_000_000.0), Some(ms(50_000_000)));
        assert_eq!(ttl_from_ms(f64::NAN), None);
    }
}
