//! Bounded, replayable leaf inlining. No callee frame is needed on a side exit:
//! every check resumes at the caller's original call, before any callee effects.
//! Invoke may guard up to four observed classes sharing the same target. Reads
//! and scalar computations are replayable. Receiver field stores are buffered
//! and emitted after every guard (see `flush_inlined_stores`); other stores,
//! allocations and nested Napalm calls still require the synchronized fallback.
use super::*;
use crate::napalm_bytecode::{IncomingSource, function_kind_uses_this};

impl TraceRecorder<'_> {
    pub(super) fn inline_call<R: TraceRuntime>(
        &self, runtime: &R, program: &BytecodeProgram, op: &RecordedOp,
        builder: &mut crate::napalm_trace_ir::TraceBuilder,
    ) -> Result<ExpectedStep, RecordingAbort> {
        // Reactive dependency tracking is intentionally implemented in the baseline
        // field handlers. Bounded replayable inlining would otherwise turn a
        // LoadField into a native HeapLoadField and make the dependency invisible.
        // Ordinary execution, including reactive programs outside a collector,
        // keeps the normal inlining path.
        if runtime.trace_reactive_tracking_active() {
            return Err(RecordingAbort::UnsupportedCall);
        }

        let (target, class) = if op.opcode == Opcode::Invoke {
            let receiver = runtime.trace_read_slot(FrameSlot::General(Reg(op.b)))
                .map_err(|_| RecordingAbort::UnsupportedCall)?;
            let (target, class) = runtime.trace_inline_target(op.location.function, op.x, receiver)
                .ok_or(RecordingAbort::UnsupportedCall)?;
            (target, Some(class))
        } else { (FuncId(op.x), None) };
        let callee = program.function(target).ok_or(RecordingAbort::UnsupportedCall)?;
        if callee.is_async || callee.capture_count != 0 || !callee.chunk.exception_handlers.is_empty()
            || callee.memoization.enabled()
            || callee.frame.return_kind != RegisterKind::Value || target == op.location.function
            || builder.trace().stats.inlined_calls >= 8 {
            return Err(RecordingAbort::UnsupportedCall);
        }
        let uses_this = function_kind_uses_this(callee.kind);
        let receiver_reg = if class.is_some() { Some(Reg(op.b)) }
            else if uses_this { op.args.first().copied() } else { None };
        let arguments = if class.is_none() && uses_this { op.args.get(1..).ok_or(RecordingAbort::UnsupportedCall)? } else { &op.args };
        if arguments.len() != callee.arity as usize { return Err(RecordingAbort::UnsupportedCall); }
        let mut args = Vec::new();
        for reg in arguments { args.push(self.read_gpr(builder, reg.0, TraceType::Any, TraceRep::Tagged)?); }
        let receiver = receiver_reg
            .map(|reg| self.read_gpr(builder, reg.0, TraceType::Any, TraceRep::Tagged))
            .transpose()?;
        let receiver_word = receiver_reg
            .map(|reg| {
                runtime
                    .trace_read_slot(FrameSlot::General(reg))
                    .map_err(|_| RecordingAbort::UnsupportedCall)
            })
            .transpose()?;
        let exit = builder.add_side_exit(ExitReason::GuardValue, op.location.pc);
        builder.emit0(TraceInst::GuardMethodEpoch { expected: runtime.trace_method_epoch(), exit });
        if let Some(classes) = &class {
            builder.emit1(TraceInst::GuardClass { value: receiver.ok_or(RecordingAbort::UnsupportedCall)?, classes: classes.clone(), exit }, TraceType::Any, TraceRep::Tagged);
        }
        let mut registers = vec![None; 256];
        // Recording-time words of callee registers that alias caller values
        // (parameters and their copies). Structural property loads need the
        // observed object to pick its hidden-shape slot.
        let mut words: Vec<Option<RegisterWord>> = vec![None; 256];
        for binding in &callee.frame.parameters {
            if binding.kind != RegisterKind::Value { return Err(RecordingAbort::UnsupportedCall); }
            let destination = binding.destination.0 as usize;
            registers[destination] = match binding.source {
                IncomingSource::This => receiver,
                IncomingSource::Argument(arg) => args.get(arg.0 as usize).copied(),
            };
            words[destination] = match binding.source {
                IncomingSource::This => receiver_word,
                IncomingSource::Argument(arg) => arguments
                    .get(arg.0 as usize)
                    .and_then(|reg| runtime.trace_read_slot(FrameSlot::General(*reg)).ok()),
            };
        }
        // Receiver field stores are buffered and emitted only at the inlined
        // Return, after every callee guard: an exit always replays the caller's
        // call, so no exit may follow an effect. See `flush_inlined_stores`.
        let mut pending_stores: Vec<(TraceValueId, u32, TraceValueId)> = Vec::new();
        let mut pc = BytecodePc(0);
        for count in 1..=32 {
            let inst = decode_at(program, TraceLocation::new(target, pc)).map_err(|_| RecordingAbort::UnsupportedCall)?;
            let get = |reg: u8| registers[reg as usize].ok_or(RecordingAbort::UnsupportedCall);
            let scalar = |reg: u8, rep: TraceRep| {
                let v = get(reg)?;
                if builder.trace().value(v).is_some_and(|v| v.rep == rep) { Ok(v) } else { Err(RecordingAbort::UnsupportedCall) }
            };
            use Opcode::*;
            // Only copies keep a known recording-time word.
            let word = match inst.opcode {
                Move | BindInterface => words[inst.b as usize],
                _ => None,
            };
            let value = match inst.opcode {
                Nop | OsrPoint => { pc = inst.next_pc; continue; }
                Move => get(inst.b)?,
                BindInterface if !matches!(
                    callee.chunk.type_guards.get(inst.x as usize).map(|guard| &guard.expected),
                    Some(Type::Top | Type::Null | Type::Bool | Type::Num)
                ) => {
                    // Structural/heap contracts are persistent and rebinding the
                    // exact installed contract is a no-op (`bind_contract`). SSA
                    // values carry an Interface/Record/List<T> type only once that
                    // contract is installed, so a proven operand needs no binding.
                    // Anything else would install state: not replayable.
                    let descriptor = callee.chunk.type_guards.get(inst.x as usize)
                        .ok_or(RecordingAbort::UnsupportedCall)?;
                    let value = scalar(inst.b, TraceRep::Tagged)?;
                    if builder.proven_type(value) != Some(&descriptor.expected) {
                        return Err(RecordingAbort::UnsupportedCall);
                    }
                    value
                }
                BindInterface => {
                    let descriptor = callee.chunk.type_guards.get(inst.x as usize)
                        .ok_or(RecordingAbort::UnsupportedCall)?;
                    if !matches!(descriptor.expected, Type::Top | Type::Null | Type::Bool | Type::Num) {
                        return Err(RecordingAbort::UnsupportedCall);
                    }
                    // Primitive checks are replayable; heap contracts are not.
                    // A failed guard must replay the caller, not a callee PC.
                    self.guard_store_value(builder, scalar(inst.b, TraceRep::Tagged)?,
                        &descriptor.expected, op.location.pc)
                        .ok_or(RecordingAbort::UnsupportedCall)?
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
                    // misses exit before any effect and replay the caller's call.
                    let object = scalar(inst.b, TraceRep::Tagged)?;
                    let key = scalar(inst.c, TraceRep::Tagged)?;
                    let object_word = words[inst.b as usize].ok_or(RecordingAbort::UnsupportedCall)?;
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
                MoveFromArg => *args.get(inst.b as usize).ok_or(RecordingAbort::UnsupportedCall)?,
                MoveFromSpecial if inst.b == 0 => receiver.ok_or(RecordingAbort::UnsupportedCall)?,
                LoadField if !pending_stores.is_empty() => {
                    // Forward a buffered store to the same receiver slot. Any
                    // other object may alias the receiver, so give up.
                    let object = scalar(inst.b, TraceRep::Tagged)?;
                    if Some(object) != receiver {
                        return Err(RecordingAbort::UnsupportedCall);
                    }
                    let slot = runtime.trace_field_slot(inst.location, inst.x);
                    match pending_stores.iter().rev().find(|(_, stored, _)| *stored == slot) {
                        Some(&(_, _, value)) => value,
                        None => {
                            let field_type = receiver_word
                                .and_then(|word| runtime.trace_field_type(word, slot))
                                .unwrap_or(Type::Top);
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
                StoreField => {
                    // Only the receiver's own fields, outside reactive VMs (field
                    // writes notify subscribers in the baseline handler).
                    let object = scalar(inst.a, TraceRep::Tagged)?;
                    if runtime.trace_reactive_enabled() || Some(object) != receiver {
                        return Err(RecordingAbort::UnsupportedCall);
                    }
                    let slot = runtime.trace_field_slot(inst.location, inst.x);
                    let expected = receiver_word
                        .and_then(|word| runtime.trace_field_type(word, slot))
                        .unwrap_or(Type::Top);
                    let value = scalar(inst.b, TraceRep::Tagged)?;
                    // The field contract is checked here, before any store.
                    let value = if matches!(expected, Type::Top) {
                        value
                    } else {
                        self.guard_store_value(builder, value, &expected, op.location.pc)
                            .ok_or(RecordingAbort::UnsupportedCall)?
                    };
                    pending_stores.push((object, slot, value));
                    pc = inst.next_pc;
                    continue;
                }
                LoadField => {
                    let object = scalar(inst.b, TraceRep::Tagged)?;
                    let slot = runtime.trace_field_slot(inst.location, inst.x);
                    let field_type = if Some(object) == receiver {
                        receiver_word
                            .and_then(|word| runtime.trace_field_type(word, slot))
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
                LoadCell => {
                    let cell = scalar(inst.b, TraceRep::Tagged)?;
                    builder.emit1(TraceInst::HeapLoadCell { cell, exit }, TraceType::Any, TraceRep::Tagged)
                }
                ListLen | MapLen | StringLen => {
                    let object = scalar(inst.b, TraceRep::Tagged)?;
                    let kind = match inst.opcode {
                        ListLen => TraceInst::HeapListLen { list: object, exit },
                        MapLen => TraceInst::HeapMapLen { map: object, exit },
                        _ => TraceInst::HeapStringLen { string: object, exit },
                    };
                    builder.emit1(kind, TraceType::Int32, TraceRep::Int32)
                }
                ListGet => {
                    let list = scalar(inst.b, TraceRep::Tagged)?;
                    let index = scalar(inst.c, TraceRep::Int32)?;
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
                    let index = scalar(inst.a, TraceRep::Int32)?;
                    let length = scalar(inst.b, TraceRep::Int32)?;
                    builder.emit0(TraceInst::GuardBounds { index, length, exit });
                    pc = inst.next_pc;
                    continue;
                }
                NumToF64 | GuardF64 => {
                    let value = scalar(inst.b, TraceRep::Tagged)?;
                    builder.emit1(TraceInst::GuardF64 { value, exit }, TraceType::F64, TraceRep::F64)
                }
                NumToInt32 | GuardInt32 => {
                    let value = scalar(inst.b, TraceRep::Tagged)?;
                    builder.emit1(TraceInst::GuardInt32 { value, exit }, TraceType::Int32, TraceRep::Int32)
                }
                Int32ToNum => {
                    let value = scalar(inst.b, TraceRep::Int32)?;
                    builder.emit1(TraceInst::BoxInt32 { value }, TraceType::Num, TraceRep::Tagged)
                }
                F64ToNum => {
                    let value = scalar(inst.b, TraceRep::F64)?;
                    builder.emit1(TraceInst::BoxF64 { value }, TraceType::Num, TraceRep::Tagged)
                }
                INeg | IBitNot => {
                    let value = scalar(inst.b, TraceRep::Int32)?;
                    let unary = if inst.opcode == INeg { IntUnaryOp::Neg } else { IntUnaryOp::BitNot };
                    builder.emit1(TraceInst::IntUnary { op: unary, value }, TraceType::Int32, TraceRep::Int32)
                }
                IAdd | ISub | IMul | IBitAnd | IBitOr | IBitXor | IShl | IShrU | ISar => {
                    let (lhs, rhs) = (scalar(inst.b, TraceRep::Int32)?, scalar(inst.c, TraceRep::Int32)?);
                    let bin = match inst.opcode {
                        IAdd => IntBinOp::Add, ISub => IntBinOp::Sub, IMul => IntBinOp::Mul,
                        IBitAnd => IntBinOp::BitAnd, IBitOr => IntBinOp::BitOr, IBitXor => IntBinOp::BitXor,
                        IShl => IntBinOp::Shl, IShrU => IntBinOp::ShrU, _ => IntBinOp::Sar,
                    };
                    builder.emit1(TraceInst::IntBinary { op: bin, lhs, rhs }, TraceType::Int32, TraceRep::Int32)
                }
                FNeg => {
                    let value = scalar(inst.b, TraceRep::F64)?;
                    builder.emit1(TraceInst::F64Unary { op: F64UnaryOp::Neg, value }, TraceType::F64, TraceRep::F64)
                }
                ICmpEq | ICmpNe | ICmpLt | ICmpLe | ICmpGt | ICmpGe
                | FCmpEq | FCmpNe | FCmpLt | FCmpLe | FCmpGt | FCmpGe => {
                    let integer = matches!(inst.opcode, ICmpEq | ICmpNe | ICmpLt | ICmpLe | ICmpGt | ICmpGe);
                    let rep = if integer { TraceRep::Int32 } else { TraceRep::F64 };
                    let (lhs, rhs) = (scalar(inst.b, rep)?, scalar(inst.c, rep)?);
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
                    let (lhs, rhs) = (scalar(inst.b, TraceRep::F64)?, scalar(inst.c, TraceRep::F64)?);
                    let bin = match inst.opcode { FAdd => F64BinOp::Add, FSub => F64BinOp::Sub, FMul => F64BinOp::Mul, FDiv => F64BinOp::Div, _ => F64BinOp::Rem };
                    builder.emit1(TraceInst::F64Binary { op: bin, lhs, rhs }, TraceType::F64, TraceRep::F64)
                }
                Invoke if runtime.trace_standard_numeric_method(target, inst.x) => {
                    self.inline_numeric_invoke(program, callee, &inst, &registers, builder, exit)?
                }
                Return => {
                    let result = scalar(inst.a, TraceRep::Tagged)?;
                    if !pending_stores.is_empty() {
                        self.flush_inlined_stores(
                            runtime, builder, &pending_stores, receiver, receiver_word,
                            class.is_some(), exit,
                        )?;
                    }
                    self.write_gpr(builder, op.a, result);
                    builder.write_slot(FrameSlot::Return(ReturnReg(0)), result);
                    builder.trace_mut().stats.inlined_calls += 1;
                    builder.trace_mut().stats.devirtualized_calls += class.is_some() as usize;
                    return Ok(ExpectedStep::Inlined { next: TraceLocation::new(op.location.function, op.next_pc), callee: target, steps: count });
                }
                _ => return Err(RecordingAbort::UnsupportedCall),
            };
            registers[inst.a as usize] = Some(value);
            words[inst.a as usize] = word;
            pc = inst.next_pc;
        }
        Err(RecordingAbort::UnsupportedCall)
    }

    /// Emit the buffered receiver stores of an inlined callee. Every guard that
    /// can fail, including the receiver's exact-class guard (which fixes the
    /// field layout, so the stores' slot checks cannot fail), precedes the first
    /// store. No runtime boundary occurs inside the inlined region, so the heap
    /// entry validated here stays valid for the stores that follow.
    fn flush_inlined_stores<R: TraceRuntime>(
        &self,
        runtime: &R,
        builder: &mut crate::napalm_trace_ir::TraceBuilder,
        stores: &[(TraceValueId, u32, TraceValueId)],
        receiver: Option<TraceValueId>,
        receiver_word: Option<RegisterWord>,
        class_guarded: bool,
        exit: SideExitId,
    ) -> Result<(), RecordingAbort> {
        let receiver = receiver.ok_or(RecordingAbort::UnsupportedCall)?;
        if !class_guarded {
            let class = receiver_word
                .and_then(|word| runtime.trace_receiver_class(word))
                .ok_or(RecordingAbort::UnsupportedCall)?;
            builder.emit1(
                TraceInst::GuardClass { value: receiver, classes: vec![class], exit },
                TraceType::Any,
                TraceRep::Tagged,
            );
        }
        let object = self.guard_heap_object(builder, receiver, TraceHeapObjectKind::Instance, exit);
        for &(_, slot, value) in stores {
            builder.emit0(TraceInst::HeapStoreField { object, value, slot, exit });
        }
        Ok(())
    }

    fn inline_numeric_invoke(
        &self, program: &BytecodeProgram, function: &BytecodeFunction, op: &RecordedOp,
        registers: &[Option<TraceValueId>], builder: &mut crate::napalm_trace_ir::TraceBuilder,
        exit: SideExitId,
    ) -> Result<TraceValueId, RecordingAbort> {
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
                let lhs = builder.emit1(TraceInst::GuardF64 { value: left, exit }, TraceType::F64, TraceRep::F64);
                let rhs = builder.emit1(TraceInst::GuardF64 { value: right, exit }, TraceType::F64, TraceRep::F64);
                let result = builder.emit1(TraceInst::F64Binary { op, lhs, rhs }, TraceType::F64, TraceRep::F64);
                Ok(builder.emit1(TraceInst::BoxF64 { value: result }, TraceType::Num, TraceRep::Tagged))
            }
            Some(name @ ("&" | "|" | "^" | "<<" | ">>")) => {
                let op = match name {
                    "&" => IntBinOp::BitAnd, "|" => IntBinOp::BitOr, "^" => IntBinOp::BitXor,
                    "<<" => IntBinOp::Shl, _ => IntBinOp::ShrU,
                };
                let lhs = builder.emit1(TraceInst::GuardUInt32 { value: left, exit }, TraceType::UInt32, TraceRep::UInt32);
                let rhs = builder.emit1(TraceInst::GuardUInt32 { value: right, exit }, TraceType::UInt32, TraceRep::UInt32);
                let result = builder.emit1(TraceInst::IntBinary { op, lhs, rhs }, TraceType::UInt32, TraceRep::UInt32);
                Ok(builder.emit1(TraceInst::BoxUInt32 { value: result }, TraceType::Num, TraceRep::Tagged))
            }
            _ => Err(RecordingAbort::UnsupportedCall),
        }
    }

    pub(super) fn inline_numeric_diamond<R: TraceRuntime>(
        &self, runtime: &R, program: &BytecodeProgram, function: &BytecodeFunction,
        branch: &RecordedOp, condition: TraceValueId,
        builder: &mut crate::napalm_trace_ir::TraceBuilder,
    ) -> Result<ExpectedStep, RecordingAbort> {
        let alternate = relative_target(branch.next_pc, branch.x);
        if alternate.0 <= branch.next_pc.0 { return Err(RecordingAbort::UnsupportedCall); }
        let mut arms = [Vec::new(), Vec::new()];
        let mut pc = branch.next_pc;
        let mut join = None;
        for _ in 0..8 {
            if pc.0 >= alternate.0 { break; }
            let op = decode_at(program, TraceLocation::new(function.id, pc))
                .map_err(|_| RecordingAbort::UnsupportedCall)?;
            if op.opcode == Opcode::Jump {
                join = Some(relative_target(op.next_pc, op.x));
                break;
            }
            pc = op.next_pc;
            arms[0].push(op);
        }
        let join = join.filter(|pc| pc.0 >= alternate.0).ok_or(RecordingAbort::UnsupportedCall)?;
        pc = alternate;
        for _ in 0..8 {
            if pc == join { break; }
            if pc.0 > join.0 { return Err(RecordingAbort::UnsupportedCall); }
            let op = decode_at(program, TraceLocation::new(function.id, pc))
                .map_err(|_| RecordingAbort::UnsupportedCall)?;
            if op.opcode == Opcode::Jump {
                pc = relative_target(op.next_pc, op.x);
                break;
            }
            pc = op.next_pc;
            arms[1].push(op);
        }
        if pc != join { return Err(RecordingAbort::UnsupportedCall); }

        let mut registers = [None; 256];
        for op in arms.iter().flatten() {
            if !matches!(op.opcode, Opcode::Move | Opcode::LoadConst | Opcode::LoadNull
                | Opcode::LoadFalse | Opcode::LoadTrue | Opcode::BindInterface | Opcode::Invoke)
            {
                return Err(RecordingAbort::UnsupportedCall);
            }
            if op.opcode == Opcode::Invoke && !runtime.trace_standard_numeric_method(function.id, op.x) {
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
                    Opcode::Invoke => self.inline_numeric_invoke(program, function, op, state, builder, exit)?,
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
            max_steps: arms[0].len() + arms[1].len() + 3,
        })
    }
}
