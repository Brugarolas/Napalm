//! Bounded, replayable inlining, including nested calls.
//!
//! No callee frame is ever needed on a side exit: every guard of every inlined
//! frame (however deep) resumes at the *outermost* original call, before any
//! callee effect. Field writes (to any object) and captured-cell writes are
//! therefore buffered for the whole inlined call tree and emitted only after the
//! outermost callee returns, behind every guard (`flush_inline_effects`).
//!
//! Inlined frames are abstract-interpreted straight-line bytecode. Nested calls
//! are inlined recursively when their target can be resolved from a
//! recording-time observation: methods on observed receivers (class-guarded),
//! direct functions, and closures (identity/capture guarded). Invoke may guard
//! up to four observed classes sharing the same target.
use super::*;
use crate::napalm_bytecode::{IncomingSource, function_kind_uses_this};

/// Maximum inlined frame depth below the outermost call site.
const MAX_INLINE_DEPTH: usize = 3;
/// Instruction budget for one outermost inlined call (all frames).
const MAX_INLINE_STEPS: usize = 96;
/// Inlined frames per trace.
const MAX_INLINED_FRAMES_PER_TRACE: usize = 16;

/// A buffered `object.field = value` write.
struct PendingFieldStore {
    object: TraceValueId,
    /// Identity base of `object` (through class/null/type guards), used for alias tests.
    base: TraceValueId,
    /// Recording-time word of `object`, when observed: lets a later read of a
    /// different SSA object speculate on observed (non-)aliasing.
    word: Option<RegisterWord>,
    slot: u32,
    value: TraceValueId,
}

/// State shared by every frame of one outermost inlined call.
struct InlineEffects {
    exit: SideExitId,
    /// Bytecode PC of the outermost call: every guard replays it.
    replay_pc: BytecodePc,
    stores: Vec<PendingFieldStore>,
    /// Buffered captured-cell writes: (cell, value).
    cells: Vec<(TraceValueId, TraceValueId)>,
    /// Recording-time word of each buffered cell (parallel to `cells`).
    cell_words: Vec<Option<RegisterWord>>,
    /// Identity bases whose exact class (hence field layout and field types)
    /// is guarded inside this inlined region.
    class_guarded: Vec<TraceValueId>,
    /// Objects allocated by the trace itself (NewClassInstance): identity base
    /// and exact receiver class. They have no recording-time word, so field
    /// types/genericity come from the class instead of an observed object.
    fresh: Vec<(TraceValueId, u64)>,
    /// Interpreter instructions the recording run executes for this call tree.
    steps: usize,
    frames: usize,
    devirtualized: usize,
    /// Functions of the caller and every active inlined frame (no recursion).
    active: Vec<FuncId>,
}

impl InlineEffects {
    fn fresh_class(&self, base: TraceValueId) -> Option<u64> {
        self.fresh.iter().find(|(known, _)| *known == base).map(|(_, class)| *class)
    }
}

/// One abstract inlined frame.
struct InlineFrame<'p> {
    target: FuncId,
    callee: &'p BytecodeFunction,
    registers: Vec<Option<TraceValueId>>,
    /// Recording-time words of registers holding values observed before the
    /// outermost call (parameters, receiver, captures and their copies).
    words: Vec<Option<RegisterWord>>,
    args: Vec<TraceValueId>,
    arg_words: Vec<Option<RegisterWord>>,
    receiver: Option<TraceValueId>,
    receiver_word: Option<RegisterWord>,
}

fn inlinable(callee: &BytecodeFunction, closure: bool) -> bool {
    !(callee.is_async
        || callee.is_concurrent
        || (!closure && callee.capture_count != 0)
        || (closure
            && (function_kind_uses_this(callee.kind)
                || callee.frame.captures.len() > callee.capture_count as usize))
        || !callee.chunk.exception_handlers.is_empty()
        || callee.memoization.enabled()
        || callee.frame.return_kind != RegisterKind::Value)
}

/// Follow value-preserving guards back to the value they check, so two SSA
/// names of one guarded object compare as the same identity.
fn identity_base(builder: &crate::napalm_trace_ir::TraceBuilder, mut value: TraceValueId) -> TraceValueId {
    for _ in 0..16 {
        let trace = builder.trace();
        let canonical = trace.canonical_value(value);
        let Some(info) = trace.value(canonical) else { return canonical; };
        let TraceValueOrigin::InstResult { node } = info.origin else { return canonical; };
        match trace.nodes.get(node).map(|node| &node.kind) {
            Some(TraceInst::GuardClass { value: inner, .. })
            | Some(TraceInst::GuardNotNull { value: inner, .. })
            | Some(TraceInst::GuardType { value: inner, .. }) => value = *inner,
            _ => return canonical,
        }
    }
    trace_canonical(builder, value)
}

fn trace_canonical(builder: &crate::napalm_trace_ir::TraceBuilder, value: TraceValueId) -> TraceValueId {
    builder.trace().canonical_value(value)
}

/// Guard that two tagged references name different objects. Valid only while
/// every effect is still buffered: a failure replays the outermost call.
fn guard_distinct(
    builder: &mut crate::napalm_trace_ir::TraceBuilder,
    lhs: TraceValueId,
    rhs: TraceValueId,
    exit: SideExitId,
) {
    let same = builder.emit1(TraceInst::RefEq { lhs, rhs }, TraceType::Bool, TraceRep::Bool);
    builder.emit0(TraceInst::GuardSwitch {
        tag: same,
        observed: crate::napalm_trace_ir::TraceSwitchGuard::Case(0),
        exit,
    });
}

impl TraceRecorder<'_> {
    /// Inline `closure.call(args)` (an ordinary Invoke resolving to the stock
    /// Fn.call) whose closure currently runs `target`.
    pub(super) fn inline_closure_call<R: TraceRuntime>(
        &self, runtime: &R, program: &BytecodeProgram, op: &RecordedOp,
        builder: &mut crate::napalm_trace_ir::TraceBuilder,
    ) -> Result<ExpectedStep, RecordingAbort> {
        if op.opcode != Opcode::Invoke || op.c as usize != op.args.len() {
            return Err(RecordingAbort::UnsupportedCall);
        }
        let closure_word = runtime
            .trace_read_slot(FrameSlot::General(Reg(op.b)))
            .map_err(|_| RecordingAbort::UnsupportedCall)?;
        let target = runtime
            .trace_invoke_closure_target(op.location.function, op.x, closure_word)
            .ok_or(RecordingAbort::UnsupportedCall)?;
        self.inline_call_impl(runtime, program, op, builder, Some((target, closure_word)), None)
    }

    pub(super) fn inline_call<R: TraceRuntime>(
        &self, runtime: &R, program: &BytecodeProgram, op: &RecordedOp,
        builder: &mut crate::napalm_trace_ir::TraceBuilder,
    ) -> Result<ExpectedStep, RecordingAbort> {
        self.inline_call_impl(runtime, program, op, builder, None, None)
    }

    /// Inline a constructor (`op` is a synthetic CallDirect whose first
    /// argument register holds an instance the trace just allocated, of exact
    /// receiver class `class`). The instance needs no class guard and its field
    /// contracts come from the class.
    pub(super) fn inline_constructor_call<R: TraceRuntime>(
        &self, runtime: &R, program: &BytecodeProgram, op: &RecordedOp,
        builder: &mut crate::napalm_trace_ir::TraceBuilder,
        class: u64,
    ) -> Result<ExpectedStep, RecordingAbort> {
        self.inline_call_impl(runtime, program, op, builder, None, Some(class))
    }

    fn inline_call_impl<R: TraceRuntime>(
        &self, runtime: &R, program: &BytecodeProgram, op: &RecordedOp,
        builder: &mut crate::napalm_trace_ir::TraceBuilder,
        closure: Option<(FuncId, RegisterWord)>,
        fresh_receiver: Option<u64>,
    ) -> Result<ExpectedStep, RecordingAbort> {
        // Reactive dependency tracking is intentionally implemented in the baseline
        // field handlers. Bounded replayable inlining would otherwise turn a
        // LoadField into a native HeapLoadField and make the dependency invisible.
        if runtime.trace_reactive_tracking_active() {
            return Err(RecordingAbort::UnsupportedCall);
        }
        if builder.trace().stats.inlined_calls >= MAX_INLINED_FRAMES_PER_TRACE {
            return Err(RecordingAbort::UnsupportedCall);
        }

        let (target, class) = if let Some((target, _)) = closure {
            (target, None)
        } else if op.opcode == Opcode::Invoke {
            let receiver = runtime.trace_read_slot(FrameSlot::General(Reg(op.b)))
                .map_err(|_| RecordingAbort::UnsupportedCall)?;
            let (target, class) = runtime.trace_inline_target(op.location.function, op.x, receiver)
                .ok_or(RecordingAbort::UnsupportedCall)?;
            (target, Some(class))
        } else { (FuncId(op.x), None) };
        let callee = program.function(target).ok_or(RecordingAbort::UnsupportedCall)?;
        if !inlinable(callee, closure.is_some()) || target == op.location.function {
            if jit_events_enabled() {
                eprintln!(
                    "[napalm-jit] inline {target:?} not inlinable (async: {}, concurrent: {}, captures: {}/{}, handlers: {}, memoized: {}, self: {})",
                    callee.is_async,
                    callee.is_concurrent,
                    callee.capture_count,
                    callee.frame.captures.len(),
                    !callee.chunk.exception_handlers.is_empty(),
                    callee.memoization.enabled(),
                    target == op.location.function
                );
            }
            return Err(RecordingAbort::UnsupportedCall);
        }
        let uses_this = closure.is_none() && function_kind_uses_this(callee.kind);
        let receiver_reg = if class.is_some() { Some(Reg(op.b)) }
            else if uses_this { op.args.first().copied() } else { None };
        let arguments = if class.is_none() && uses_this { op.args.get(1..).ok_or(RecordingAbort::UnsupportedCall)? } else { &op.args };
        if arguments.len() != callee.arity as usize { return Err(RecordingAbort::UnsupportedCall); }
        let mut args = Vec::with_capacity(arguments.len());
        let mut arg_words = Vec::with_capacity(arguments.len());
        for reg in arguments {
            args.push(self.read_gpr(builder, reg.0, TraceType::Any, TraceRep::Tagged)?);
            arg_words.push(runtime.trace_read_slot(FrameSlot::General(*reg)).ok());
        }
        let receiver = receiver_reg
            .map(|reg| self.read_gpr(builder, reg.0, TraceType::Any, TraceRep::Tagged))
            .transpose()?;
        // A trace-allocated receiver has no meaningful recording-time word: the
        // register still holds whatever preceded the allocation.
        let receiver_word = if fresh_receiver.is_some() {
            None
        } else {
            receiver_reg
                .map(|reg| runtime.trace_read_slot(FrameSlot::General(reg)).map_err(|_| RecordingAbort::UnsupportedCall))
                .transpose()?
        };
        if fresh_receiver.is_some() && (receiver.is_none() || class.is_some() || closure.is_some()) {
            return Err(RecordingAbort::UnsupportedCall);
        }

        let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
        builder.emit0(TraceInst::GuardMethodEpoch { expected: runtime.trace_method_epoch(), exit });
        let mut fx = InlineEffects {
            exit,
            replay_pc: op.location.pc,
            stores: Vec::new(),
            cells: Vec::new(),
            cell_words: Vec::new(),
            class_guarded: Vec::new(),
            fresh: Vec::new(),
            steps: 0,
            frames: 0,
            devirtualized: class.is_some() as usize,
            active: vec![op.location.function, target],
        };
        if let (Some(fresh_class), Some(value)) = (fresh_receiver, receiver) {
            // Exact class by construction: no guard needed.
            let base = identity_base(builder, value);
            fx.class_guarded.push(base);
            fx.fresh.push((base, fresh_class));
        }
        let receiver = match (&class, receiver) {
            (Some(classes), Some(value)) => {
                let guarded = builder.emit1(
                    TraceInst::GuardClass { value, classes: classes.clone(), exit },
                    TraceType::Any,
                    TraceRep::Tagged,
                );
                fx.class_guarded.push(identity_base(builder, guarded));
                Some(guarded)
            }
            (Some(_), None) => return Err(RecordingAbort::UnsupportedCall),
            (None, receiver) => receiver,
        };
        let mut frame = InlineFrame {
            target,
            callee,
            registers: vec![None; 256],
            words: vec![None; 256],
            args,
            arg_words,
            receiver,
            receiver_word,
        };
        if let Some((_, closure_word)) = closure {
            let closure_value = self.read_gpr(builder, op.b, TraceType::Any, TraceRep::Tagged)?;
            self.bind_captures(runtime, builder, &mut frame, closure_value, Some(closure_word), exit)?;
        }
        Self::bind_parameters(&mut frame)?;

        let result = self.inline_frame(runtime, program, builder, &mut fx, frame)?;
        self.flush_inline_effects(builder, &fx);
        self.write_gpr(builder, op.a, result);
        builder.write_slot(FrameSlot::Return(ReturnReg(0)), result);
        builder.trace_mut().stats.inlined_calls += fx.frames;
        builder.trace_mut().stats.devirtualized_calls += fx.devirtualized;
        Ok(ExpectedStep::Inlined {
            next: TraceLocation::new(op.location.function, op.next_pc),
            callee: target,
            steps: fx.steps,
        })
    }

    /// Install parameters exactly like the frame builder does.
    fn bind_parameters(frame: &mut InlineFrame<'_>) -> Result<(), RecordingAbort> {
        for binding in &frame.callee.frame.parameters {
            if binding.kind != RegisterKind::Value {
                if jit_events_enabled() {
                    eprintln!("[napalm-jit] inline {:?}: parameter {:?} has kind {:?}", frame.target, binding.destination, binding.kind);
                }
                return Err(RecordingAbort::UnsupportedCall);
            }
            let destination = binding.destination.0 as usize;
            frame.registers[destination] = match binding.source {
                IncomingSource::This => frame.receiver,
                IncomingSource::Argument(arg) => frame.args.get(arg.0 as usize).copied(),
            };
            frame.words[destination] = match binding.source {
                IncomingSource::This => frame.receiver_word,
                IncomingSource::Argument(arg) => frame.arg_words.get(arg.0 as usize).copied().flatten(),
            };
        }
        Ok(())
    }

    /// Fix the closure's function identity and capture layout, then install its
    /// captures as callee registers like the frame builder does.
    fn bind_captures<R: TraceRuntime>(
        &self,
        runtime: &R,
        builder: &mut crate::napalm_trace_ir::TraceBuilder,
        frame: &mut InlineFrame<'_>,
        closure: TraceValueId,
        closure_word: Option<RegisterWord>,
        exit: SideExitId,
    ) -> Result<(), RecordingAbort> {
        let function_index = runtime
            .trace_prepared_function_index(frame.target)
            .ok_or(RecordingAbort::UnsupportedCall)?;
        let capture_count = frame.callee.capture_count;
        builder.emit0(TraceInst::GuardClosure { value: closure, function_index, capture_count, exit });
        for binding in &frame.callee.frame.captures {
            if binding.kind != RegisterKind::Value || binding.capture_index >= capture_count {
                return Err(RecordingAbort::UnsupportedCall);
            }
            let destination = binding.destination.0 as usize;
            frame.registers[destination] = Some(builder.emit1(
                TraceInst::ClosureCapture { closure, index: binding.capture_index, exit },
                TraceType::Any,
                TraceRep::Tagged,
            ));
            frame.words[destination] = closure_word
                .and_then(|word| runtime.trace_closure_capture_word(word, binding.capture_index));
        }
        Ok(())
    }

    /// Emit every buffered effect of one outermost inlined call. All guards that
    /// can fail (heap kinds) precede the first store; the class guards that fix
    /// field layouts were emitted when the stores were buffered, so the stores'
    /// own slot checks cannot fail. No runtime boundary occurs inside the inlined
    /// region, so the validated heap entries stay valid for the stores.
    fn flush_inline_effects(&self, builder: &mut crate::napalm_trace_ir::TraceBuilder, fx: &InlineEffects) {
        let exit = fx.exit;
        let mut objects: Vec<(TraceValueId, TraceValueId)> = Vec::new();
        for store in &fx.stores {
            if !objects.iter().any(|(base, _)| *base == store.base) {
                let entry = self.guard_heap_object(builder, store.object, TraceHeapObjectKind::Instance, exit);
                objects.push((store.base, entry));
            }
        }
        let mut cells: Vec<(TraceValueId, TraceValueId)> = Vec::new();
        for &(cell, _) in &fx.cells {
            if !cells.iter().any(|(known, _)| *known == cell) {
                let entry = self.guard_heap_object(builder, cell, TraceHeapObjectKind::Cell, exit);
                cells.push((cell, entry));
            }
        }
        // Program order keeps last-write-wins even if two bases alias.
        for store in &fx.stores {
            let object = objects.iter().find(|(base, _)| *base == store.base).map(|(_, entry)| *entry).unwrap();
            builder.emit0(TraceInst::HeapStoreField { object, value: store.value, slot: store.slot, exit });
        }
        for (position, &(cell, value)) in fx.cells.iter().enumerate() {
            if fx.cells[position + 1..].iter().any(|(later, _)| *later == cell) {
                continue; // only the last write to one SSA cell is observable
            }
            let cell = cells.iter().find(|(known, _)| *known == cell).map(|(_, entry)| *entry).unwrap();
            builder.emit0(TraceInst::HeapStoreCell { cell, value, exit });
        }
    }

    /// Ensure the exact class of `object` (observed as `word`) is guarded, so
    /// its field slots and field types are fixed. Returns false if the class is
    /// unknown.
    fn guard_object_class<R: TraceRuntime>(
        runtime: &R,
        builder: &mut crate::napalm_trace_ir::TraceBuilder,
        fx: &mut InlineEffects,
        object: TraceValueId,
        word: Option<RegisterWord>,
    ) -> bool {
        let base = identity_base(builder, object);
        if fx.class_guarded.contains(&base) {
            return true;
        }
        let Some(class) = word.and_then(|word| runtime.trace_receiver_class(word)) else {
            return false;
        };
        builder.emit1(
            TraceInst::GuardClass { value: object, classes: vec![class], exit: fx.exit },
            TraceType::Any,
            TraceRep::Tagged,
        );
        fx.class_guarded.push(base);
        true
    }

    /// Abstract-interpret one inlined frame and return its result value.
    fn inline_frame<R: TraceRuntime>(
        &self,
        runtime: &R,
        program: &BytecodeProgram,
        builder: &mut crate::napalm_trace_ir::TraceBuilder,
        fx: &mut InlineEffects,
        mut frame: InlineFrame<'_>,
    ) -> Result<TraceValueId, RecordingAbort> {
        let exit = fx.exit;
        let target = frame.target;
        let callee = frame.callee;
        fx.frames += 1;
        let mut pc = BytecodePc(0);
        loop {
            fx.steps += 1;
            if fx.steps > MAX_INLINE_STEPS {
                return Err(RecordingAbort::UnsupportedCall);
            }
            let inst = decode_at(program, TraceLocation::new(target, pc)).map_err(|_| RecordingAbort::UnsupportedCall)?;
            let opcode = inst.opcode;
            let reject = move |reg: u8, why: &str| {
                if jit_events_enabled() {
                    eprintln!("[napalm-jit] inline {target:?} {opcode:?}: r{reg} {why}");
                }
                RecordingAbort::UnsupportedCall
            };
            let get = |frame: &InlineFrame<'_>, reg: u8| frame.registers[reg as usize].ok_or_else(|| reject(reg, "undefined"));
            let scalar = |frame: &InlineFrame<'_>, builder: &crate::napalm_trace_ir::TraceBuilder, reg: u8, rep: TraceRep| {
                let v = frame.registers[reg as usize].ok_or_else(|| reject(reg, "undefined"))?;
                match builder.trace().value(v).map(|v| v.rep) {
                    Some(actual) if actual == rep => Ok(v),
                    actual => Err(reject(reg, &format!("is {actual:?}, expected {rep:?}"))),
                }
            };
            use Opcode::*;
            // Copies keep the recording-time word of their source.
            let module_slot = if inst.opcode == LoadModuleVar {
                runtime.trace_module_slot_address(inst.location)
            } else {
                None
            };
            let word = match inst.opcode {
                Move | BindInterface | ImplementInterfaceOrType => frame.words[inst.b as usize],
                MoveFromArg => frame.arg_words.get(inst.b as usize).copied().flatten(),
                MoveFromSpecial if inst.b == 0 => frame.receiver_word,
                // Module variable arrays are fixed for the interpreter's lifetime
                // and nothing runs concurrently while recording: observe the slot.
                LoadModuleVar => module_slot.map(|address| unsafe { *(address as *const RegisterWord) }),
                _ => None,
            };
            let value = match inst.opcode {
                Nop | OsrPoint => { pc = inst.next_pc; continue; }
                Move => get(&frame, inst.b)?,
                BindInterface if !matches!(
                    callee.chunk.type_guards.get(inst.x as usize).map(|guard| &guard.expected),
                    Some(Type::Top | Type::Null | Type::Bool | Type::Num)
                ) => {
                    // Structural/heap contracts are persistent; a proven operand
                    // needs no binding, anything else would install state. Pure
                    // class checks (`Instance<C>`) are replayed by a class guard.
                    let descriptor = callee.chunk.type_guards.get(inst.x as usize)
                        .ok_or(RecordingAbort::UnsupportedCall)?;
                    let value = scalar(&frame, builder, inst.b, TraceRep::Tagged)?;
                    let observed = frame.words[inst.b as usize];
                    if builder.proven_type(value) == Some(&descriptor.expected) {
                        value
                    } else if let Some(word) = observed
                        && runtime.trace_value_satisfies_pure_type(word, &descriptor.expected)
                        && let Some(class) = runtime.trace_receiver_class(word)
                    {
                        let guarded = builder.emit1(
                            TraceInst::GuardClass { value, classes: vec![class], exit },
                            TraceType::Any,
                            TraceRep::Tagged,
                        );
                        builder.set_proven_type(guarded, descriptor.expected.clone());
                        let base = identity_base(builder, guarded);
                        if !fx.class_guarded.contains(&base) {
                            fx.class_guarded.push(base);
                        }
                        guarded
                    } else {
                        if jit_events_enabled() {
                            eprintln!(
                                "[napalm-jit] inline {target:?} stops at BindInterface {:?} (operand proven {:?})",
                                descriptor.expected,
                                builder.proven_type(value)
                            );
                        }
                        return Err(RecordingAbort::UnsupportedCall);
                    }
                }
                BindInterface => {
                    let descriptor = callee.chunk.type_guards.get(inst.x as usize)
                        .ok_or(RecordingAbort::UnsupportedCall)?;
                    if !matches!(descriptor.expected, Type::Top | Type::Null | Type::Bool | Type::Num) {
                        return Err(RecordingAbort::UnsupportedCall);
                    }
                    // Primitive checks are replayable; a failure replays the outermost call.
                    let value = scalar(&frame, builder, inst.b, TraceRep::Tagged)?;
                    self.guard_store_value(builder, value, &descriptor.expected, fx.replay_pc)
                        .ok_or(RecordingAbort::UnsupportedCall)?
                }
                // Structural bind in its specialized encoding: a no-op only when
                // the operand already carries exactly that contract (anything
                // else would install persistent state inside the inlined body).
                ImplementInterfaceOrType => {
                    let Some(crate::napalm_bytecode::SpecializedDescriptor::Contract { contract, .. }) =
                        callee.chunk.specializations.get(inst.x as usize)
                    else { return Err(RecordingAbort::UnsupportedCall); };
                    let value = scalar(&frame, builder, inst.b, TraceRep::Tagged)?;
                    if builder.proven_type(value) != Some(contract) {
                        if jit_events_enabled() {
                            eprintln!(
                                "[napalm-jit] inline {target:?} stops at ImplementInterfaceOrType {contract:?} (operand proven {:?})",
                                builder.proven_type(value)
                            );
                        }
                        return Err(RecordingAbort::UnsupportedCall);
                    }
                    value
                }
                LoadNull => builder.constant(TraceConstant::Null),
                LoadFalse | LoadTrue => builder.constant(TraceConstant::Bool(inst.opcode == LoadTrue)),
                LoadI8 => builder.constant(TraceConstant::Int32(inst.b as i8 as i32)),
                LoadI32 => builder.constant(TraceConstant::Int32(inst.x as i32)),
                LoadConst => {
                    let c = match callee.chunk.constants.get(inst.x as usize) {
                        Some(Constant::Null) => TraceConstant::Null,
                        Some(Constant::Bool(v)) => TraceConstant::Bool(*v),
                        Some(Constant::Num(v)) => TraceConstant::NumBits(v.to_bits()),
                        Some(Constant::Int32(v)) => TraceConstant::Int32(*v),
                        // Program strings are persistent VM roots: embed the
                        // stable tagged handle, exactly like a root LoadConst.
                        Some(Constant::String(id)) => {
                            let word = runtime
                                .trace_materialize_constant(TraceConstant::String(*id))
                                .map_err(|_| RecordingAbort::UnsupportedCall)?;
                            TraceConstant::StableTagged(word.0)
                        }
                        _ => return Err(RecordingAbort::UnsupportedCall),
                    };
                    builder.constant(c)
                }
                LoadProperty => {
                    // Structural read of a parameter-derived map. Shape/PIC
                    // misses exit before any effect and replay the outermost call.
                    let object = scalar(&frame, builder, inst.b, TraceRep::Tagged)?;
                    let key = scalar(&frame, builder, inst.c, TraceRep::Tagged)?;
                    let object_word = frame.words[inst.b as usize].ok_or(RecordingAbort::UnsupportedCall)?;
                    let layout = StructuralLayoutId(inst.x);
                    let slot = PropertySlotId(inst.y as u16);
                    let access = runtime
                        .trace_structural_map_access(object_word, layout, slot)
                        .ok_or(RecordingAbort::UnsupportedCall)?;
                    let map = self.guard_heap_object(builder, object, TraceHeapObjectKind::Map, exit);
                    let field_type = program
                        .metadata
                        .structural_layout(layout)
                        .and_then(|layout| layout.property(slot))
                        .map(|property| property.ty.clone())
                        .unwrap_or(Type::Top);
                    let (trace_ty, trace_rep) = Self::semantic_trace_layout(&field_type);
                    let current = TraceShapeSlot { shape: access.shape, slot: access.slot as u32 };
                    let mut entries = vec![current];
                    for entry in runtime.trace_structural_map_pic(inst.location, layout, slot) {
                        if entries.len() == 4 { break; }
                        if !entries.iter().any(|known| known.shape == entry.shape) {
                            entries.push(entry);
                        }
                    }
                    let kind = if entries.len() > 1 {
                        TraceInst::HeapMapGetPic { map, key, expected_key: access.key_bytes, entries, exit }
                    } else {
                        TraceInst::HeapMapGetShaped {
                            map, key, expected_key: access.key_bytes,
                            shape: current.shape, slot: current.slot, exit,
                        }
                    };
                    let result = builder.emit1(kind, trace_ty, trace_rep);
                    if !matches!(field_type, Type::Top) {
                        builder.set_proven_type(result, field_type);
                    }
                    result
                }
                LoadModuleVar => {
                    // The value is reloaded natively; anything resolved from the
                    // observed word (e.g. a nested closure call) is guarded.
                    let address = module_slot.ok_or(RecordingAbort::UnsupportedCall)?;
                    builder.emit1(
                        TraceInst::LoadAbsoluteValue { address: address as u64 },
                        TraceType::Any,
                        TraceRep::Tagged,
                    )
                }
                MoveFromArg => *frame.args.get(inst.b as usize).ok_or(RecordingAbort::UnsupportedCall)?,
                MoveFromSpecial if inst.b == 0 => frame.receiver.ok_or(RecordingAbort::UnsupportedCall)?,
                StoreField => {
                    // Any object's field, outside reactive VMs (field writes notify
                    // subscribers in the baseline handler). The object's exact class
                    // is guarded so the slot and its contract are fixed; the contract
                    // check runs here, before any store.
                    if runtime.trace_reactive_enabled() {
                        return Err(RecordingAbort::UnsupportedCall);
                    }
                    let object = scalar(&frame, builder, inst.a, TraceRep::Tagged)?;
                    let object_word = frame.words[inst.a as usize];
                    if !Self::guard_object_class(runtime, builder, fx, object, object_word) {
                        return Err(RecordingAbort::UnsupportedCall);
                    }
                    let slot = runtime.trace_field_slot(inst.location, inst.x);
                    let fresh = fx.fresh_class(identity_base(builder, object));
                    // SetFieldClass whose value statically satisfies the field.
                    let proven = inst.y & crate::napalm_bytecode::FIELD_CLASS_UNCHECKED_STORE != 0;
                    // Generic-parameter fields are checked against per-instance
                    // arguments by the baseline handler; keep the call.
                    let generic = !proven && match (object_word, fresh) {
                        (Some(word), _) => runtime.trace_field_is_generic(word, slot),
                        (None, Some(class)) => runtime.trace_class_field_is_generic(class, slot),
                        (None, None) => false,
                    };
                    if generic {
                        return Err(RecordingAbort::UnsupportedCall);
                    }
                    let expected = match (object_word, fresh) {
                        _ if proven => Type::Top,
                        (Some(word), _) => runtime.trace_field_type(word, slot).unwrap_or(Type::Top),
                        // A fresh instance's field contract must be known exactly.
                        (None, Some(class)) => runtime.trace_class_field_type(class, slot)
                            .ok_or(RecordingAbort::UnsupportedCall)?,
                        (None, None) => Type::Top,
                    };
                    let value = scalar(&frame, builder, inst.b, TraceRep::Tagged)?;
                    let value = if matches!(expected, Type::Top) {
                        value
                    } else {
                        self.guard_store_value(builder, value, &expected, fx.replay_pc)
                            .ok_or(RecordingAbort::UnsupportedCall)?
                    };
                    let base = identity_base(builder, object);
                    fx.stores.push(PendingFieldStore { object, base, word: object_word, slot, value });
                    pc = inst.next_pc;
                    continue;
                }
                LoadField => {
                    let object = scalar(&frame, builder, inst.b, TraceRep::Tagged)?;
                    let slot = runtime.trace_field_slot(inst.location, inst.x);
                    let base = identity_base(builder, object);
                    let object_word = frame.words[inst.b as usize];
                    // Walk buffered writes to this slot newest-first. A write to the
                    // same identity is forwarded. A write to another SSA object may
                    // alias it: speculate on what recording observed (equal words ->
                    // guard equal and forward; otherwise guard distinct and keep
                    // looking). Effects are buffered, so guards replay the call.
                    let mut own = None;
                    let mut proven: Vec<TraceValueId> = Vec::new();
                    for index in (0..fx.stores.len()).rev() {
                        let (other, other_base, other_word) =
                            (fx.stores[index].object, fx.stores[index].base, fx.stores[index].word);
                        if fx.stores[index].slot != slot {
                            continue;
                        }
                        if other_base == base {
                            own = Some(index);
                            break;
                        }
                        if object_word.is_some() && object_word == other_word {
                            builder.emit0(TraceInst::GuardEq { lhs: object, rhs: other, exit });
                            own = Some(index);
                            break;
                        }
                        if !proven.contains(&other_base) {
                            proven.push(other_base);
                            guard_distinct(builder, object, other, exit);
                        }
                    }
                    match own {
                        Some(index) => fx.stores[index].value,
                        // A trace-allocated instance with no buffered write to
                        // this slot still holds its initial null. The baseline
                        // traps on a null read of a non-nullable field: keep the
                        // call then, otherwise the read folds to null.
                        None if fx.fresh_class(base).is_some() => {
                            let class = fx.fresh_class(base).unwrap();
                            let field_type = runtime.trace_class_field_type(class, slot)
                                .ok_or(RecordingAbort::UnsupportedCall)?;
                            if !field_type.accepts(&Type::Null) {
                                return Err(RecordingAbort::UnsupportedCall);
                            }
                            builder.constant(TraceConstant::Null)
                        }
                        None => {
                            // Field types are only known for exactly class-guarded objects.
                            let field_type = if fx.class_guarded.contains(&base) {
                                frame.words[inst.b as usize]
                                    .and_then(|word| runtime.trace_field_type(word, slot))
                                    .or_else(|| fx.fresh_class(base)
                                        .and_then(|class| runtime.trace_class_field_type(class, slot)))
                                    .unwrap_or(Type::Top)
                            } else {
                                Type::Top
                            };
                            let (trace_ty, trace_rep) = Self::semantic_trace_layout(&field_type);
                            let result = builder.emit1(
                                TraceInst::HeapLoadField { object, slot, exit },
                                trace_ty,
                                trace_rep,
                            );
                            if !matches!(&field_type, Type::Top) {
                                builder.set_proven_type(result, field_type);
                            }
                            result
                        }
                    }
                }
                LoadCell => {
                    let cell = scalar(&frame, builder, inst.b, TraceRep::Tagged)?;
                    let cell_word = frame.words[inst.b as usize];
                    // Newest-first, like field reads: same SSA cell -> forward;
                    // observed same Cell -> guard equal and forward; otherwise
                    // guard distinct and keep looking.
                    let mut own = None;
                    let mut proven: Vec<TraceValueId> = Vec::new();
                    for index in (0..fx.cells.len()).rev() {
                        let stored = fx.cells[index].0;
                        if stored == cell {
                            own = Some(index);
                            break;
                        }
                        if cell_word.is_some() && cell_word == fx.cell_words[index] {
                            builder.emit0(TraceInst::GuardEq { lhs: cell, rhs: stored, exit });
                            own = Some(index);
                            break;
                        }
                        if !proven.contains(&stored) {
                            proven.push(stored);
                            guard_distinct(builder, cell, stored, exit);
                        }
                    }
                    match own {
                        Some(index) => fx.cells[index].1,
                        None => builder.emit1(TraceInst::HeapLoadCell { cell, exit }, TraceType::Any, TraceRep::Tagged),
                    }
                }
                StoreCell => {
                    let cell = scalar(&frame, builder, inst.a, TraceRep::Tagged)?;
                    let value = scalar(&frame, builder, inst.b, TraceRep::Tagged)?;
                    fx.cells.push((cell, value));
                    fx.cell_words.push(frame.words[inst.a as usize]);
                    pc = inst.next_pc;
                    continue;
                }
                ListLen | MapLen | StringLen => {
                    let object = scalar(&frame, builder, inst.b, TraceRep::Tagged)?;
                    let kind = match inst.opcode {
                        ListLen => TraceInst::HeapListLen { list: object, exit },
                        MapLen => TraceInst::HeapMapLen { map: object, exit },
                        _ => TraceInst::HeapStringLen { string: object, exit },
                    };
                    builder.emit1(kind, TraceType::Int32, TraceRep::Int32)
                }
                ListGet => {
                    let list = scalar(&frame, builder, inst.b, TraceRep::Tagged)?;
                    let index = scalar(&frame, builder, inst.c, TraceRep::Int32)?;
                    let element_type = match builder.proven_type(list) {
                        Some(Type::List(Some(element))) => (**element).clone(),
                        _ => Type::Top,
                    };
                    let (trace_ty, trace_rep) = Self::semantic_trace_layout(&element_type);
                    let result = builder.emit1(
                        TraceInst::HeapListGet { list, index, exit },
                        trace_ty,
                        trace_rep,
                    );
                    if !matches!(&element_type, Type::Top) {
                        builder.set_proven_type(result, element_type);
                    }
                    result
                }
                GuardBounds => {
                    let index = scalar(&frame, builder, inst.a, TraceRep::Int32)?;
                    let length = scalar(&frame, builder, inst.b, TraceRep::Int32)?;
                    builder.emit0(TraceInst::GuardBounds { index, length, exit });
                    pc = inst.next_pc;
                    continue;
                }
                NumToF64 | GuardF64 => {
                    let value = scalar(&frame, builder, inst.b, TraceRep::Tagged)?;
                    builder.emit1(TraceInst::GuardF64 { value, exit }, TraceType::F64, TraceRep::F64)
                }
                NumToInt32 | GuardInt32 => {
                    let value = scalar(&frame, builder, inst.b, TraceRep::Tagged)?;
                    builder.emit1(TraceInst::GuardInt32 { value, exit }, TraceType::Int32, TraceRep::Int32)
                }
                Int32ToNum => {
                    let value = scalar(&frame, builder, inst.b, TraceRep::Int32)?;
                    builder.emit1(TraceInst::BoxInt32 { value }, TraceType::Num, TraceRep::Tagged)
                }
                F64ToNum => {
                    let value = scalar(&frame, builder, inst.b, TraceRep::F64)?;
                    builder.emit1(TraceInst::BoxF64 { value }, TraceType::Num, TraceRep::Tagged)
                }
                INeg | IBitNot => {
                    let value = scalar(&frame, builder, inst.b, TraceRep::Int32)?;
                    let unary = if inst.opcode == INeg { IntUnaryOp::Neg } else { IntUnaryOp::BitNot };
                    builder.emit1(TraceInst::IntUnary { op: unary, value }, TraceType::Int32, TraceRep::Int32)
                }
                IAdd | ISub | IMul | IBitAnd | IBitOr | IBitXor | IShl | IShrU | ISar => {
                    let lhs = scalar(&frame, builder, inst.b, TraceRep::Int32)?;
                    let rhs = scalar(&frame, builder, inst.c, TraceRep::Int32)?;
                    let bin = match inst.opcode {
                        IAdd => IntBinOp::Add, ISub => IntBinOp::Sub, IMul => IntBinOp::Mul,
                        IBitAnd => IntBinOp::BitAnd, IBitOr => IntBinOp::BitOr, IBitXor => IntBinOp::BitXor,
                        IShl => IntBinOp::Shl, IShrU => IntBinOp::ShrU, _ => IntBinOp::Sar,
                    };
                    builder.emit1(TraceInst::IntBinary { op: bin, lhs, rhs }, TraceType::Int32, TraceRep::Int32)
                }
                FNeg => {
                    let value = scalar(&frame, builder, inst.b, TraceRep::F64)?;
                    builder.emit1(TraceInst::F64Unary { op: F64UnaryOp::Neg, value }, TraceType::F64, TraceRep::F64)
                }
                ICmpEq | ICmpNe | ICmpLt | ICmpLe | ICmpGt | ICmpGe
                | FCmpEq | FCmpNe | FCmpLt | FCmpLe | FCmpGt | FCmpGe => {
                    let integer = matches!(inst.opcode, ICmpEq | ICmpNe | ICmpLt | ICmpLe | ICmpGt | ICmpGe);
                    let rep = if integer { TraceRep::Int32 } else { TraceRep::F64 };
                    let lhs = scalar(&frame, builder, inst.b, rep)?;
                    let rhs = scalar(&frame, builder, inst.c, rep)?;
                    let cmp = match inst.opcode {
                        ICmpEq | FCmpEq => CmpKind::Eq, ICmpNe | FCmpNe => CmpKind::Ne,
                        ICmpLt | FCmpLt => CmpKind::Lt, ICmpLe | FCmpLe => CmpKind::Le,
                        ICmpGt | FCmpGt => CmpKind::Gt, _ => CmpKind::Ge,
                    };
                    let kind = if integer { TraceInst::IntCompare { op: cmp, lhs, rhs } }
                        else { TraceInst::F64Compare { op: cmp, lhs, rhs } };
                    let condition = builder.emit1(kind, TraceType::Bool, TraceRep::Bool);
                    let if_true = builder.constant(TraceConstant::Bool(true));
                    let if_false = builder.constant(TraceConstant::Bool(false));
                    builder.emit1(TraceInst::Select { condition, if_true, if_false }, TraceType::Bool, TraceRep::Tagged)
                }
                FAdd | FSub | FMul | FDiv | FRem => {
                    let lhs = scalar(&frame, builder, inst.b, TraceRep::F64)?;
                    let rhs = scalar(&frame, builder, inst.c, TraceRep::F64)?;
                    let bin = match inst.opcode { FAdd => F64BinOp::Add, FSub => F64BinOp::Sub, FMul => F64BinOp::Mul, FDiv => F64BinOp::Div, _ => F64BinOp::Rem };
                    builder.emit1(TraceInst::F64Binary { op: bin, lhs, rhs }, TraceType::F64, TraceRep::F64)
                }
                Invoke if runtime.trace_standard_numeric_method(target, inst.x) => {
                    self.inline_numeric_invoke(program, callee, &inst, &frame.registers, builder, exit)?
                }
                Invoke | CallDirect => self.inline_nested_call(runtime, program, builder, fx, &frame, &inst)?,
                Return => return scalar(&frame, builder, inst.a, TraceRep::Tagged),
                _ => {
                    if jit_events_enabled() {
                        eprintln!(
                            "[napalm-jit] inline {target:?} stops at {:?} (pc {})",
                            inst.opcode, inst.location.pc.0
                        );
                    }
                    return Err(RecordingAbort::UnsupportedCall);
                }
            };
            frame.registers[inst.a as usize] = Some(value);
            frame.words[inst.a as usize] = word;
            pc = inst.next_pc;
        }
    }

    /// Inline a call made from inside an inlined frame: a closure call or method
    /// Invoke whose subject was observed before the outermost call, or a direct
    /// function call. Its guards replay the outermost call as well.
    fn inline_nested_call<R: TraceRuntime>(
        &self,
        runtime: &R,
        program: &BytecodeProgram,
        builder: &mut crate::napalm_trace_ir::TraceBuilder,
        fx: &mut InlineEffects,
        caller: &InlineFrame<'_>,
        inst: &RecordedOp,
    ) -> Result<TraceValueId, RecordingAbort> {
        let exit = fx.exit;
        // `active` holds the outermost caller plus every inlined frame.
        if fx.active.len() > MAX_INLINE_DEPTH + 1
            || builder.trace().stats.inlined_calls + fx.frames >= MAX_INLINED_FRAMES_PER_TRACE
        {
            return Err(RecordingAbort::UnsupportedCall);
        }
        let register = |reg: u8| {
            caller.registers[reg as usize].ok_or_else(|| {
                if jit_events_enabled() {
                    eprintln!("[napalm-jit] nested call at {:?}:{}: r{reg} undefined", caller.target, inst.location.pc.0);
                }
                RecordingAbort::UnsupportedCall
            })
        };
        let mut args = Vec::with_capacity(inst.args.len());
        let mut arg_words = Vec::with_capacity(inst.args.len());
        for reg in &inst.args {
            args.push(register(reg.0)?);
            arg_words.push(caller.words[reg.0 as usize]);
        }

        enum Shape { Closure(TraceValueId, RegisterWord), Method, Function }
        let (target, shape, receiver, receiver_word) = match inst.opcode {
            Opcode::Invoke => {
                if inst.c as usize != inst.args.len() {
                    return Err(RecordingAbort::UnsupportedCall);
                }
                let subject = register(inst.b)?;
                let Some(subject_word) = caller.words[inst.b as usize] else {
                    if jit_events_enabled() {
                        eprintln!("[napalm-jit] nested call at {:?}:{} has no observed subject", caller.target, inst.location.pc.0);
                    }
                    return Err(RecordingAbort::UnsupportedCall);
                };
                if let Some(target) = runtime.trace_invoke_closure_target(caller.target, inst.x, subject_word) {
                    (target, Shape::Closure(subject, subject_word), None, None)
                } else {
                    let Some((target, classes)) = runtime.trace_inline_target(caller.target, inst.x, subject_word) else {
                        if jit_events_enabled() {
                            eprintln!(
                                "[napalm-jit] nested call at {:?}:{} has no inlinable target (subject {:?})",
                                caller.target, inst.location.pc.0, runtime.trace_heap_object_kind(subject_word)
                            );
                        }
                        return Err(RecordingAbort::UnsupportedCall);
                    };
                    if classes.is_empty() {
                        return Err(RecordingAbort::UnsupportedCall);
                    }
                    builder.emit0(TraceInst::GuardMethodEpoch { expected: runtime.trace_method_epoch(), exit });
                    let guarded = builder.emit1(
                        TraceInst::GuardClass { value: subject, classes, exit },
                        TraceType::Any,
                        TraceRep::Tagged,
                    );
                    let base = identity_base(builder, guarded);
                    if !fx.class_guarded.contains(&base) {
                        fx.class_guarded.push(base);
                    }
                    fx.devirtualized += 1;
                    (target, Shape::Method, Some(guarded), Some(subject_word))
                }
            }
            Opcode::CallDirect => (FuncId(inst.x), Shape::Function, None, None),
            _ => return Err(RecordingAbort::UnsupportedCall),
        };
        let callee = program.function(target).ok_or(RecordingAbort::UnsupportedCall)?;
        if fx.active.contains(&target) || !inlinable(callee, matches!(shape, Shape::Closure(..))) {
            if jit_events_enabled() {
                eprintln!(
                    "[napalm-jit] nested call at {:?}:{} to {target:?} not inlinable (recursive: {}, async: {}, captures: {}, handlers: {}, memoized: {})",
                    caller.target,
                    inst.location.pc.0,
                    fx.active.contains(&target),
                    callee.is_async,
                    callee.capture_count,
                    !callee.chunk.exception_handlers.is_empty(),
                    callee.memoization.enabled()
                );
            }
            return Err(RecordingAbort::UnsupportedCall);
        }
        // Direct calls to `this`-using functions pass the receiver as argument 0.
        let (receiver, receiver_word, args, arg_words) = match shape {
            Shape::Function if function_kind_uses_this(callee.kind) => {
                if args.is_empty() {
                    return Err(RecordingAbort::UnsupportedCall);
                }
                (Some(args[0]), arg_words[0], args[1..].to_vec(), arg_words[1..].to_vec())
            }
            _ => (receiver, receiver_word, args, arg_words),
        };
        if args.len() != callee.arity as usize {
            return Err(RecordingAbort::UnsupportedCall);
        }
        let mut frame = InlineFrame {
            target,
            callee,
            registers: vec![None; 256],
            words: vec![None; 256],
            args,
            arg_words,
            receiver,
            receiver_word,
        };
        if let Shape::Closure(closure, closure_word) = shape {
            builder.emit0(TraceInst::GuardMethodEpoch { expected: runtime.trace_method_epoch(), exit });
            self.bind_captures(runtime, builder, &mut frame, closure, Some(closure_word), exit)?;
        }
        Self::bind_parameters(&mut frame)?;
        fx.active.push(target);
        let result = self.inline_frame(runtime, program, builder, fx, frame);
        fx.active.pop();
        result
    }

    fn inline_numeric_invoke(
        &self, program: &BytecodeProgram, function: &BytecodeFunction, op: &RecordedOp,
        registers: &[Option<TraceValueId>], builder: &mut crate::napalm_trace_ir::TraceBuilder,
        exit: SideExitId,
    ) -> Result<TraceValueId, RecordingAbort> {
        self.numeric_invoke(program, function, op, registers, builder, exit, false)
    }

    /// Shared by the inliner and diamond if-conversion. With `cse`, each node
    /// reuses an identical node already in the trace (an earlier guard on the
    /// same value dominates this point of a linear trace), so an arm that
    /// recomputes the condition's arithmetic adds nothing.
    #[allow(clippy::too_many_arguments)]
    fn numeric_invoke(
        &self, program: &BytecodeProgram, function: &BytecodeFunction, op: &RecordedOp,
        registers: &[Option<TraceValueId>], builder: &mut crate::napalm_trace_ir::TraceBuilder,
        exit: SideExitId, cse: bool,
    ) -> Result<TraceValueId, RecordingAbort> {
        let emit = |builder: &mut crate::napalm_trace_ir::TraceBuilder, kind: TraceInst, ty: TraceType, rep: TraceRep| {
            if cse { cse_emit1(builder, kind, ty, rep) } else { builder.emit1(kind, ty, rep) }
        };
        let cache = function.chunk.inline_caches.get(op.x as usize).ok_or(RecordingAbort::UnsupportedCall)?;
        let signature = program.metadata.method(cache.method).ok_or(RecordingAbort::UnsupportedCall)?;
        if signature.kind != MethodKind::Method || signature.arity != 1 || op.args.len() != 1 {
            return Err(RecordingAbort::UnsupportedCall);
        }
        let get = |reg: u8| {
            registers[reg as usize].filter(|&value| builder.trace().value(value)
                .is_some_and(|v| v.rep == TraceRep::Tagged)).ok_or(RecordingAbort::UnsupportedCall)
        };
        let (left, right) = (get(op.b)?, get(op.args[0].0)?);
        match program.metadata.string(signature.name) {
            Some(name @ ("+" | "-" | "*" | "/" | "%")) => {
                let op = match name {
                    "+" => F64BinOp::Add, "-" => F64BinOp::Sub, "*" => F64BinOp::Mul,
                    "/" => F64BinOp::Div, _ => F64BinOp::Rem,
                };
                let lhs = emit(builder, TraceInst::GuardF64 { value: left, exit }, TraceType::F64, TraceRep::F64);
                let rhs = emit(builder, TraceInst::GuardF64 { value: right, exit }, TraceType::F64, TraceRep::F64);
                let result = emit(builder, TraceInst::F64Binary { op, lhs, rhs }, TraceType::F64, TraceRep::F64);
                Ok(emit(builder, TraceInst::BoxF64 { value: result }, TraceType::Num, TraceRep::Tagged))
            }
            Some(name @ ("&" | "|" | "^" | "<<" | ">>")) => {
                let op = match name {
                    "&" => IntBinOp::BitAnd, "|" => IntBinOp::BitOr, "^" => IntBinOp::BitXor,
                    "<<" => IntBinOp::Shl, _ => IntBinOp::ShrU,
                };
                let lhs = emit(builder, TraceInst::GuardUInt32 { value: left, exit }, TraceType::UInt32, TraceRep::UInt32);
                let rhs = emit(builder, TraceInst::GuardUInt32 { value: right, exit }, TraceType::UInt32, TraceRep::UInt32);
                let result = emit(builder, TraceInst::IntBinary { op, lhs, rhs }, TraceType::UInt32, TraceRep::UInt32);
                Ok(emit(builder, TraceInst::BoxUInt32 { value: result }, TraceType::Num, TraceRep::Tagged))
            }
            _ => Err(RecordingAbort::UnsupportedCall),
        }
    }

    /// If-conversion of a short diamond (`c ? a : b`, `if/else`) or one-armed
    /// `if (c) { ... }`: both arms are evaluated and merged with Select.
    ///
    /// Arms are found on the control-flow graph, not by position: each
    /// successor of the branch is followed through unconditional jumps, and the
    /// join is the first instruction both straight-line paths reach. The
    /// bytecode optimizer lays cold arms out of line (the `else` of
    /// `c ? 0 : 1` after the function's return, jumping back), which a
    /// "fallthrough arm, jump, second arm, join" scan would never match.
    pub(super) fn inline_numeric_diamond<R: TraceRuntime>(
        &self, runtime: &R, program: &BytecodeProgram, function: &BytecodeFunction,
        branch: &RecordedOp, condition: TraceValueId,
        builder: &mut crate::napalm_trace_ir::TraceBuilder,
    ) -> Result<ExpectedStep, RecordingAbort> {
        const MAX_ARM_OPS: usize = 8;
        const MAX_WALK: usize = 3 * MAX_ARM_OPS;
        let alternate = relative_target(branch.next_pc, branch.x);
        // Straight-line path from `start`: through arm-eligible ops and
        // unconditional jumps; the first other instruction (which may be the
        // join itself) ends it.
        let walk = |start: BytecodePc| {
            let mut path: Vec<(BytecodePc, RecordedOp)> = Vec::new();
            let mut pc = start;
            while path.len() < MAX_WALK && !path.iter().any(|(seen, _)| *seen == pc) {
                let Ok(op) = decode_at(program, TraceLocation::new(function.id, pc)) else { break };
                let next = match op.opcode {
                    Opcode::Jump => Some(relative_target(op.next_pc, op.x)),
                    Opcode::Move | Opcode::LoadConst | Opcode::LoadNull | Opcode::LoadFalse
                    | Opcode::LoadTrue | Opcode::BindInterface | Opcode::Invoke => Some(op.next_pc),
                    _ => None,
                };
                path.push((pc, op));
                match next {
                    Some(next) => pc = next,
                    None => break,
                }
            }
            path
        };
        let first = walk(branch.next_pc);
        let second = walk(alternate);
        let (second_len, join, first_len) = second
            .iter()
            .enumerate()
            .find_map(|(index, (pc, _))| {
                first.iter().position(|(other, _)| other == pc).map(|position| (index, *pc, position))
            })
            .ok_or(RecordingAbort::UnsupportedCall)?;
        // Forward joins only: a merge at or above the branch is a loop edge.
        if join.0 <= branch.location.pc.0 { return Err(RecordingAbort::UnsupportedCall); }        let arm = |path: Vec<(BytecodePc, RecordedOp)>, len: usize| -> Vec<RecordedOp> {
            path.into_iter().take(len).map(|(_, op)| op).filter(|op| op.opcode != Opcode::Jump).collect()
        };
        let arms = [arm(first, first_len), arm(second, second_len)];
        if arms.iter().all(Vec::is_empty) || arms.iter().any(|arm| arm.len() > MAX_ARM_OPS) {
            return Err(RecordingAbort::UnsupportedCall);
        }
        // A one-armed `if` pays for its arm on every iteration once converted,
        // and a Select on a loop-carried value lengthens the recurrence. That
        // only wins when the branch is unpredictable, typically when it tests
        // data loaded in this iteration (`if (a[i] < lo) lo = a[i]`), not a
        // counter or accumulator (`if (acc > limit) ...`, rarely taken).
        if arms.iter().any(Vec::is_empty) && !condition_reads_heap(builder.trace(), condition) {
            return Err(RecordingAbort::UnsupportedCall);
        }

        let mut registers = [None; 256];
        for op in arms.iter().flatten() {
            if !matches!(op.opcode, Opcode::Move | Opcode::LoadConst | Opcode::LoadNull
                | Opcode::LoadFalse | Opcode::LoadTrue | Opcode::BindInterface | Opcode::Invoke)
            {
                return Err(RecordingAbort::UnsupportedCall);
            }
            if op.opcode == Opcode::Invoke && !runtime.trace_standard_numeric_method(function.id, op.x)
                && runtime.trace_standard_list_method(function.id, op.x) != Some(TraceListMethod::Subscript)
            {
                return Err(RecordingAbort::UnsupportedCall);
            }
            let roles = crate::napalm_bytecode::operand_roles(op.opcode);
            for (reg, role) in [op.a, op.b, op.c].into_iter().zip(roles) {
                if role == crate::napalm_bytecode::OperandRole::General && registers[reg as usize].is_none() {
                    registers[reg as usize] = Some(self.read_gpr(builder, reg, TraceType::Any, TraceRep::Tagged)?);
                }
            }
            for &reg in &op.args {
                if registers[reg.0 as usize].is_none() {
                    registers[reg.0 as usize] = Some(self.read_gpr(builder, reg.0, TraceType::Any, TraceRep::Tagged)?);
                }
            }
        }

        // Both arms are replayable scalar work. Guards precede the merge and
        // resume at the branch, so an untaken arm can never introduce an effect.
        let exit = builder.add_side_exit(ExitReason::GuardValue, branch.location.pc);
        builder.emit0(TraceInst::GuardMethodEpoch { expected: runtime.trace_method_epoch(), exit });
        let mut results = [registers, registers];
        for (arm, state) in arms.iter().zip(&mut results) {
            for op in arm {
                let source = || state[op.b as usize].ok_or(RecordingAbort::UnsupportedCall);
                let value = match op.opcode {
                    Opcode::Move => source()?,
                    Opcode::LoadNull => builder.constant(TraceConstant::Null),
                    Opcode::LoadFalse | Opcode::LoadTrue => builder.constant(TraceConstant::Bool(op.opcode == Opcode::LoadTrue)),
                    Opcode::LoadConst => {
                        let constant = match function.chunk.constants.get(op.x as usize) {
                            Some(Constant::Null) => TraceConstant::Null,
                            Some(Constant::Bool(value)) => TraceConstant::Bool(*value),
                            Some(Constant::Num(value)) => TraceConstant::NumBits(
                                if value.is_nan() { CANONICAL_NAN } else { value.to_bits() }),
                            _ => return Err(RecordingAbort::UnsupportedCall),
                        };
                        builder.constant(constant)
                    }
                    Opcode::BindInterface => {
                        let descriptor = function.chunk.type_guards.get(op.x as usize)
                            .ok_or(RecordingAbort::UnsupportedCall)?;
                        if !matches!(descriptor.expected, Type::Top | Type::Num | Type::Bool | Type::Null) {
                            return Err(RecordingAbort::UnsupportedCall);
                        }
                        self.guard_store_value(builder, source()?, &descriptor.expected, branch.location.pc)
                            .ok_or(RecordingAbort::UnsupportedCall)?
                    }
                    Opcode::Invoke if runtime.trace_standard_numeric_method(function.id, op.x) => {
                        self.numeric_invoke(program, function, op, state, builder, exit, true)?
                    }
                    Opcode::Invoke => repeated_list_read(builder, op, state)?,
                    _ => return Err(RecordingAbort::UnsupportedCall),
                };
                state[op.a as usize] = Some(value);
            }
        }
        let live = function.chunk.live_registers_at(join);
        for reg in 0..256 {
            if live.is_some_and(|live| !live.general_is_live(Reg(reg as u8))) { continue; }
            let (Some(left), Some(right)) = (results[0][reg], results[1][reg]) else { continue };
            if results[0][reg] == registers[reg] && results[1][reg] == registers[reg] { continue; }
            let value = if left == right { left } else {
                let lhs = builder.trace().value(left).unwrap();
                let rhs = builder.trace().value(right).unwrap();
                if lhs.rep != rhs.rep { return Err(RecordingAbort::UnsupportedCall); }
                let rep = lhs.rep;
                let ty = if lhs.ty == rhs.ty { lhs.ty } else { TraceType::Any };
                let (if_true, if_false) = if branch.opcode == Opcode::JumpIfFalse { (left, right) } else { (right, left) };
                if ty == TraceType::Num && rep == TraceRep::Tagged {
                    let if_true = builder.emit1(TraceInst::GuardF64 { value: if_true, exit }, TraceType::F64, TraceRep::F64);
                    let if_false = builder.emit1(TraceInst::GuardF64 { value: if_false, exit }, TraceType::F64, TraceRep::F64);
                    let value = builder.emit1(TraceInst::Select { condition, if_true, if_false }, TraceType::F64, TraceRep::F64);
                    builder.emit1(TraceInst::BoxF64 { value }, TraceType::Num, TraceRep::Tagged)
                } else {
                    builder.emit1(TraceInst::Select { condition, if_true, if_false }, ty, rep)
                }
            };
            self.write_gpr(builder, reg as u8, value);
        }
        Ok(ExpectedStep::Region {
            next: TraceLocation::new(function.id, join),
            // Either path, jumps included, plus the branch itself.
            max_steps: first_len + second_len + 3,
        })
    }
}

/// Numeric guards of the same value are interchangeable (the first one
/// dominates every later point of a linear trace), but the recorder emits a
/// fresh guard per use. Map every such guard result to the first one so CSE
/// sees through the duplicates.
fn guard_representatives(trace: &crate::napalm_trace_ir::Trace) -> HashMap<TraceValueId, TraceValueId> {
    let mut first = HashMap::<(u8, TraceValueId), TraceValueId>::default();
    let mut representatives = HashMap::default();
    for node in &trace.nodes {
        let (kind, value) = match node.kind {
            TraceInst::GuardF64 { value, .. } => (0u8, value),
            TraceInst::GuardInt32 { value, .. } => (1, value),
            TraceInst::GuardUInt32 { value, .. } => (2, value),
            _ => continue,
        };
        let Some(&result) = node.results.first() else { continue };
        let result = trace.canonical_value(result);
        let representative = *first.entry((kind, trace.canonical_value(value))).or_insert(result);
        representatives.insert(result, representative);
    }
    representatives
}

fn representative(
    trace: &crate::napalm_trace_ir::Trace, representatives: &HashMap<TraceValueId, TraceValueId>, value: TraceValueId,
) -> TraceValueId {
    let value = trace.canonical_value(value);
    representatives.get(&value).copied().unwrap_or(value)
}

/// Kind with representative operands and (for the guards CSE may reuse) no exit.
fn cse_key(
    trace: &crate::napalm_trace_ir::Trace, representatives: &HashMap<TraceValueId, TraceValueId>, kind: &TraceInst,
) -> TraceInst {
    let mut key = kind.clone();
    key.visit_operands(|value| *value = representative(trace, representatives, *value));
    if matches!(key, TraceInst::GuardF64 { .. } | TraceInst::GuardInt32 { .. } | TraceInst::GuardUInt32 { .. }) {
        key.set_exit(SideExitId(u32::MAX));
    }
    key
}

/// Bounded backward search window for recording-time CSE.
const CSE_WINDOW: usize = 512;

/// Representative result of a node identical to `kind` (pure, or a numeric
/// guard of the same value: an earlier guard dominates every later point).
fn cse_lookup(builder: &crate::napalm_trace_ir::TraceBuilder, kind: &TraceInst) -> Option<TraceValueId> {
    let pure = kind.is_speculatable()
        || matches!(kind, TraceInst::GuardF64 { .. } | TraceInst::GuardInt32 { .. } | TraceInst::GuardUInt32 { .. });
    if !pure {
        return None;
    }
    let trace = builder.trace();
    let representatives = guard_representatives(trace);
    let key = cse_key(trace, &representatives, kind);
    trace.nodes.iter().rev().take(CSE_WINDOW)
        .find(|node| node.results.len() == 1 && cse_key(trace, &representatives, &node.kind) == key)
        .map(|node| representative(trace, &representatives, node.results[0]))
}

fn cse_emit1(
    builder: &mut crate::napalm_trace_ir::TraceBuilder, kind: TraceInst, ty: TraceType, rep: TraceRep,
) -> TraceValueId {
    match cse_lookup(builder, &kind) {
        Some(existing) => existing,
        None => builder.emit1(kind, ty, rep),
    }
}

/// `list[i]` inside an if-converted arm, accepted only as an exact repeat of a
/// list read this trace already performed: same List value, same Int32 index,
/// and only effect-free nodes since then (no store, call, runtime or GC). That
/// read succeeded, nothing can have changed the element, so the untaken arm can
/// never fail or observe anything new; it simply reuses the value. Any other
/// read could fail only on the untaken path (`if (i < n) x = a[i]`) and keeps
/// the branch.
fn repeated_list_read(
    builder: &crate::napalm_trace_ir::TraceBuilder, op: &RecordedOp, state: &[Option<TraceValueId>],
) -> Result<TraceValueId, RecordingAbort> {
    if op.c != 1 || op.args.len() != 1 {
        return Err(RecordingAbort::UnsupportedCall);
    }
    let trace = builder.trace();
    let representatives = guard_representatives(trace);
    let rep = |value: TraceValueId| representative(trace, &representatives, value);
    let list = state[op.b as usize].map(rep).ok_or(RecordingAbort::UnsupportedCall)?;
    let index = state[op.args[0].0 as usize].map(rep).ok_or(RecordingAbort::UnsupportedCall)?;
    let effect_free = |kind: &TraceInst| {
        kind.is_speculatable() || kind.is_entry_guard() || kind.is_heap_read()
            || matches!(kind, TraceInst::Nop | TraceInst::LoadFrameSlot { .. } | TraceInst::CommitFrameSlot { .. }
                | TraceInst::F64ToInt32 { .. } | TraceInst::IntBinaryChecked { .. })
    };
    // Int32 views of the arm's index: results of a GuardInt32 of it.
    let int_index = trace.nodes.iter().find_map(|node| match node.kind {
        TraceInst::GuardInt32 { value, .. } if rep(value) == index => node.results.first().map(|&result| rep(result)),
        _ => None,
    });
    let Some(int_index) = int_index else { return Err(RecordingAbort::UnsupportedCall) };
    for node in trace.nodes.iter().rev().take(CSE_WINDOW) {
        if let TraceInst::HeapListGet { list: read_list, index: read_index, .. } = node.kind {
            if rep(read_list) == list && rep(read_index) == int_index && node.results.len() == 1 {
                return Ok(node.results[0]);
            }
        }
        // Everything after the matching read (scanned first) must be effect-free.
        if !effect_free(&node.kind) {
            return Err(RecordingAbort::UnsupportedCall);
        }
    }
    Err(RecordingAbort::UnsupportedCall)
}

/// Whether `condition` depends, within a few producers, on a heap read
/// recorded in the trace (List/Map/String loads), as opposed to only loop
/// phis, entry values and arithmetic on them.
fn condition_reads_heap(trace: &crate::napalm_trace_ir::Trace, condition: TraceValueId) -> bool {
    let mut pending = vec![trace.canonical_value(condition)];
    let mut seen = HashSet::<TraceValueId>::default();
    while let Some(value) = pending.pop() {
        if !seen.insert(value) || seen.len() > 32 {
            continue;
        }
        let Some(TraceValueOrigin::InstResult { node }) = trace.value(value).map(|info| info.origin.clone()) else {
            continue;
        };
        let Some(node) = trace.nodes.get(node) else { continue };
        if node.kind.is_heap_read() {
            return true;
        }
        let mut operands = node.kind.clone();
        operands.visit_operands(|operand| pending.push(trace.canonical_value(*operand)));
    }
    false
}
