//! Whole-function Cranelift tier for hot synchronous bytecode functions.
//!
//! This tier intentionally executes against the interpreter's existing frame and
//! register stack. GC-capable helpers therefore reuse the bytecode safepoint maps,
//! and direct calls still flow through `enter_direct_bytecode_call`, preserving
//! auto-memoization, async semantics and the normal frame ABI.

use super::*;
use cranelift::codegen::ir::{BlockArg, UserFuncName};
use cranelift::codegen::settings::{self, Configurable};
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::jit::{JITBuilder, JITModule};
use cranelift::module::{default_libcall_names, Linkage, Module};
use cranelift::prelude::*;
use std::ffi::c_void;
use std::mem::offset_of;

use crate::napalm_tracing_jit::{TRACE_HEAP_KIND_MASK, TRACE_HEAP_METADATA_SHIFT};

const MIN_WHOLE_FUNCTION_HOTNESS: u32 = 128;
const MAX_WHOLE_FUNCTION_OPS: usize = 4096;
const MAX_NATIVE_CALL_DEPTH: u32 = 256;
const WFJIT_BAILOUT: u64 = TAG_BASE;

pub(super) type WholeFunctionEntry = unsafe extern "C" fn(*mut c_void, *mut RegisterWord) -> u64;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct WholeFunctionJitStats {
    pub hot_calls: u64,
    pub compile_attempts: u64,
    pub compiled_functions: u64,
    pub rejected_functions: u64,
    pub native_entries: u64,
    pub native_direct_calls: u64,
    pub baseline_direct_calls: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FunctionState {
    Cold,
    Compiled,
    Rejected,
}

struct FunctionSlot {
    hotness: u32,
    state: FunctionState,
    entry: Option<WholeFunctionEntry>,
}

impl Default for FunctionSlot {
    fn default() -> Self {
        Self { hotness: 0, state: FunctionState::Cold, entry: None }
    }
}

pub(super) struct WholeFunctionJit {
    module: JITModule,
    context: cranelift::codegen::Context,
    function_context: FunctionBuilderContext,
    runtime_op: cranelift::module::FuncId,
    direct_call: cranelift::module::FuncId,
    small_map_literal: cranelift::module::FuncId,
    slots: Vec<FunctionSlot>,
    threshold: u32,
    active_depth: u32,
    stats: WholeFunctionJitStats,
}

impl WholeFunctionJit {
    pub(super) fn new(function_count: usize, entry_threshold: u32) -> Result<Self, String> {
        let mut flags = settings::builder();
        flags.set("opt_level", "speed").map_err(|e| e.to_string())?;
        flags.set("enable_alias_analysis", "true").map_err(|e| e.to_string())?;
        flags.set("use_colocated_libcalls", "false").map_err(|e| e.to_string())?;
        flags.set("is_pic", if cfg!(target_arch = "x86_64") { "true" } else { "false" })
            .map_err(|e| e.to_string())?;
        let isa = cranelift::native::builder()
            .map_err(|e| format!("Cranelift does not support this host: {e}"))?
            .finish(settings::Flags::new(flags))
            .map_err(|e| format!("cannot build whole-function ISA: {e}"))?;

        let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
        builder.symbol("napalm_wfjit_runtime_op", napalm_wfjit_runtime_op as *const u8);
        builder.symbol("napalm_wfjit_direct_call", napalm_wfjit_direct_call as *const u8);
        builder.symbol("napalm_wfjit_small_map_literal", napalm_wfjit_small_map_literal as *const u8);
        let mut module = JITModule::new(builder);
        if module.target_config().pointer_type() != types::I64 {
            return Err("Napalm whole-function JIT requires a 64-bit host".to_owned());
        }
        let pointer = module.target_config().pointer_type();

        let mut helper_sig = module.make_signature();
        helper_sig.params.push(AbiParam::new(pointer));
        helper_sig.params.push(AbiParam::new(types::I32));
        helper_sig.returns.push(AbiParam::new(pointer));
        let runtime_id = module
            .declare_function("napalm_wfjit_runtime_op", Linkage::Import, &helper_sig)
            .map_err(|e| format!("cannot declare whole-function runtime helper: {e}"))?;
        let direct_id = module
            .declare_function("napalm_wfjit_direct_call", Linkage::Import, &helper_sig)
            .map_err(|e| format!("cannot declare whole-function direct-call helper: {e}"))?;
        let small_map_id = module
            .declare_function("napalm_wfjit_small_map_literal", Linkage::Import, &helper_sig)
            .map_err(|e| format!("cannot declare whole-function small-map helper: {e}"))?;

        let context = module.make_context();
        let function_context = FunctionBuilderContext::new();
        let result = Self {
            module,
            context,
            function_context,
            runtime_op: runtime_id,
            direct_call: direct_id,
            small_map_literal: small_map_id,
            slots: (0..function_count).map(|_| FunctionSlot::default()).collect(),
            threshold: if entry_threshold == 0 { u32::MAX } else { entry_threshold.saturating_mul(2).max(MIN_WHOLE_FUNCTION_HOTNESS) },
            active_depth: 0,
            stats: WholeFunctionJitStats::default(),
        };
        Ok(result)
    }

    #[inline]
    pub(super) fn stats(&self) -> WholeFunctionJitStats { self.stats }

    #[inline]
    fn entry(&self, function: u32) -> Option<WholeFunctionEntry> {
        self.slots.get(function as usize).and_then(|slot| slot.entry)
    }

    #[inline]
    fn note_call(&mut self, function: u32) -> bool {
        self.stats.hot_calls = self.stats.hot_calls.saturating_add(1);
        let Some(slot) = self.slots.get_mut(function as usize) else { return false; };
        if slot.state != FunctionState::Cold { return false; }
        slot.hotness = slot.hotness.saturating_add(1);
        slot.hotness >= self.threshold
    }

    fn reject(&mut self, function: u32) {
        if let Some(slot) = self.slots.get_mut(function as usize) {
            slot.state = FunctionState::Rejected;
            slot.entry = None;
        }
        self.stats.rejected_functions = self.stats.rejected_functions.saturating_add(1);
    }

    fn compile(
        &mut self,
        function_index: u32,
        function: &PreparedFunction,
        bytecode: &BytecodeFunction,
        constants: &[RegisterWord],
        expected_method_epoch: u64,
    ) -> Result<(), String> {
        self.stats.compile_attempts = self.stats.compile_attempts.saturating_add(1);
        if function.memoization.enabled()
            || function.is_async
            || function.is_concurrent
            || function.uses_this
            || function.capture_count != 0
            || !function.exception_handlers.is_empty()
            || function.return_kind == RegisterKind::RawPtr
            || function.ops.is_empty()
            || function.ops.len() > MAX_WHOLE_FUNCTION_OPS
        {
            self.reject(function_index);
            return Ok(());
        }

        for (index, op) in function.ops.iter().enumerate() {
            let opcode = bytecode.chunk.code
                .get(op.raw_pc as usize)
                .and_then(|word| word.opcode())
                .ok_or_else(|| format!("invalid opcode at prepared op {index}"))?;
            let unsupported_invoke = opcode == Opcode::Invoke
                && native_numeric_quick(function, index).is_none()
                && native_map_property_pic(function, index).is_none();
            if !eligible_opcode(opcode)
                || unsupported_invoke
                || (matches!(opcode, Opcode::Jump | Opcode::JumpIfFalse | Opcode::JumpIfTrue)
                    && op.x as usize <= index)
            {
                // Backedge-heavy functions belong to the tracing tier. Keeping
                // whole-function compilation acyclic also preserves cooperative
                // GC polling without adding a second stack-map scheme.
                self.reject(function_index);
                return Ok(());
            }
        }

        let pointer = self.module.target_config().pointer_type();
        let mut signature = self.module.make_signature();
        signature.params.push(AbiParam::new(pointer));
        signature.params.push(AbiParam::new(pointer));
        signature.returns.push(AbiParam::new(types::I64));
        let func_id = self.module.declare_anonymous_function(&signature)
            .map_err(|e| format!("cannot declare whole function: {e}"))?;

        self.context.clear();
        self.context.func.signature = signature;
        self.context.func.name = UserFuncName::user(1, func_id.as_u32());
        self.function_context = FunctionBuilderContext::new();

        let runtime_module_id = self.runtime_op;
        let direct_module_id = self.direct_call;
        let small_map_module_id = self.small_map_literal;

        {
            let target_config = self.module.target_config();
            let mut b = FunctionBuilder::new(&mut self.context.func, &mut self.function_context);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            let abort = b.create_block();
            let mut blocks = Vec::with_capacity(function.ops.len());
            for _ in &function.ops {
                let block = b.create_block();
                b.append_block_param(block, pointer); // vm
                b.append_block_param(block, pointer); // current frame GPR base
                blocks.push(block);
            }

            b.switch_to_block(entry);
            let vm = b.block_params(entry)[0];
            let regs = b.block_params(entry)[1];
            b.ins().jump(blocks[0], &[BlockArg::Value(vm), BlockArg::Value(regs)]);
            b.seal_block(entry);

            let runtime_ref = self.module.declare_func_in_func(runtime_module_id, b.func);
            let direct_ref = self.module.declare_func_in_func(direct_module_id, b.func);
            let small_map_ref = self.module.declare_func_in_func(small_map_module_id, b.func);

            for index in 0..function.ops.len() {
                b.switch_to_block(blocks[index]);
                let vm = b.block_params(blocks[index])[0];
                let mut regs = b.block_params(blocks[index])[1];
                let prepared_op = function.ops[index];
                let op = function.original_ops[index];
                let opcode = bytecode.chunk.code[op.raw_pc as usize].opcode().unwrap();

                macro_rules! next {
                    () => {{
                        let target = op.next as usize;
                        if target >= blocks.len() { b.ins().jump(abort, &[]); }
                        else { b.ins().jump(blocks[target], &[BlockArg::Value(vm), BlockArg::Value(regs)]); }
                    }};
                }

                match opcode {
                    Opcode::Nop => next!(),
                    Opcode::Move => {
                        let v = load_reg(&mut b, regs, op.b);
                        store_reg(&mut b, regs, op.a, v);
                        next!();
                    }
                    Opcode::Swap => {
                        let a = load_reg(&mut b, regs, op.a);
                        let c = load_reg(&mut b, regs, op.b);
                        store_reg(&mut b, regs, op.a, c);
                        store_reg(&mut b, regs, op.b, a);
                        next!();
                    }
                    Opcode::LoadNull => { store_imm(&mut b, regs, op.a, TAG_NULL); next!(); }
                    Opcode::LoadFalse => { store_imm(&mut b, regs, op.a, TAG_FALSE); next!(); }
                    Opcode::LoadTrue => { store_imm(&mut b, regs, op.a, TAG_TRUE); next!(); }
                    Opcode::LoadI8 => { store_imm(&mut b, regs, op.a, op.b as i8 as i32 as u32 as u64); next!(); }
                    Opcode::LoadI32 => { store_imm(&mut b, regs, op.a, op.x as u32 as u64); next!(); }
                    Opcode::LoadConst => {
                        let value = constants.get(op.x as usize)
                            .ok_or_else(|| format!("constant {} out of bounds", op.x))?.0;
                        store_imm(&mut b, regs, op.a, value);
                        next!();
                    }
                    Opcode::Jump => jump_to(&mut b, &blocks, prepared_op.x, vm, regs, abort),
                    Opcode::JumpIfFalse | Opcode::JumpIfTrue => {
                        let value = load_reg(&mut b, regs, op.a);
                        let shifted = b.ins().iadd_imm(value, -(TAG_NULL as i64));
                        let falsey = b.ins().icmp_imm_u(IntCC::UnsignedLessThanOrEqual, shifted, 1);
                        let target = prepared_op.x as usize;
                        let fall = op.next as usize;
                        if target >= blocks.len() || fall >= blocks.len() {
                            b.ins().jump(abort, &[]);
                        } else if opcode == Opcode::JumpIfFalse {
                            b.ins().brif(falsey, blocks[target], &[BlockArg::Value(vm), BlockArg::Value(regs)], blocks[fall], &[BlockArg::Value(vm), BlockArg::Value(regs)]);
                        } else {
                            b.ins().brif(falsey, blocks[fall], &[BlockArg::Value(vm), BlockArg::Value(regs)], blocks[target], &[BlockArg::Value(vm), BlockArg::Value(regs)]);
                        }
                    }
                    Opcode::Return => {
                        let value = load_reg(&mut b, regs, op.a);
                        b.ins().return_(&[value]);
                    }
                    Opcode::INeg | Opcode::IBitNot => {
                        let x = load_i32(&mut b, regs, op.b);
                        let r = if opcode == Opcode::INeg { b.ins().ineg(x) } else { b.ins().bnot(x) };
                        store_i32(&mut b, regs, op.a, r);
                        next!();
                    }
                    Opcode::IAdd | Opcode::ISub | Opcode::IMul | Opcode::IBitAnd | Opcode::IBitOr
                    | Opcode::IBitXor | Opcode::IShl | Opcode::IShrU | Opcode::ISar => {
                        let lhs = load_i32(&mut b, regs, op.b);
                        let rhs = load_i32(&mut b, regs, op.c);
                        let r = match opcode {
                            Opcode::IAdd => b.ins().iadd(lhs, rhs),
                            Opcode::ISub => b.ins().isub(lhs, rhs),
                            Opcode::IMul => b.ins().imul(lhs, rhs),
                            Opcode::IBitAnd => b.ins().band(lhs, rhs),
                            Opcode::IBitOr => b.ins().bor(lhs, rhs),
                            Opcode::IBitXor => b.ins().bxor(lhs, rhs),
                            Opcode::IShl | Opcode::IShrU | Opcode::ISar => {
                                let mask = b.ins().iconst(types::I32, 31);
                                let amount = b.ins().band(rhs, mask);
                                match opcode {
                                    Opcode::IShl => b.ins().ishl(lhs, amount),
                                    Opcode::IShrU => b.ins().ushr(lhs, amount),
                                    _ => b.ins().sshr(lhs, amount),
                                }
                            }
                            _ => unreachable!(),
                        };
                        store_i32(&mut b, regs, op.a, r);
                        next!();
                    }
                    Opcode::FNeg | Opcode::FAdd | Opcode::FSub | Opcode::FMul | Opcode::FDiv => {
                        let lhs = load_f64(&mut b, regs, op.b);
                        let r = if opcode == Opcode::FNeg {
                            b.ins().fneg(lhs)
                        } else {
                            let rhs = load_f64(&mut b, regs, op.c);
                            match opcode {
                                Opcode::FAdd => b.ins().fadd(lhs, rhs),
                                Opcode::FSub => b.ins().fsub(lhs, rhs),
                                Opcode::FMul => b.ins().fmul(lhs, rhs),
                                Opcode::FDiv => b.ins().fdiv(lhs, rhs),
                                _ => unreachable!(),
                            }
                        };
                        store_f64(&mut b, regs, op.a, r);
                        next!();
                    }
                    Opcode::ICmpEq | Opcode::ICmpNe | Opcode::ICmpLt | Opcode::ICmpLe
                    | Opcode::ICmpGt | Opcode::ICmpGe => {
                        let lhs = load_i32(&mut b, regs, op.b);
                        let rhs = load_i32(&mut b, regs, op.c);
                        let cc = match opcode {
                            Opcode::ICmpEq => IntCC::Equal, Opcode::ICmpNe => IntCC::NotEqual,
                            Opcode::ICmpLt => IntCC::SignedLessThan, Opcode::ICmpLe => IntCC::SignedLessThanOrEqual,
                            Opcode::ICmpGt => IntCC::SignedGreaterThan, _ => IntCC::SignedGreaterThanOrEqual,
                        };
                        let cond = b.ins().icmp(cc, lhs, rhs);
                        store_tagged_bool(&mut b, regs, op.a, cond);
                        next!();
                    }
                    Opcode::FCmpEq | Opcode::FCmpNe | Opcode::FCmpLt | Opcode::FCmpLe
                    | Opcode::FCmpGt | Opcode::FCmpGe => {
                        let lhs = load_f64(&mut b, regs, op.b);
                        let rhs = load_f64(&mut b, regs, op.c);
                        let cc = match opcode {
                            Opcode::FCmpEq => FloatCC::Equal, Opcode::FCmpNe => FloatCC::NotEqual,
                            Opcode::FCmpLt => FloatCC::LessThan, Opcode::FCmpLe => FloatCC::LessThanOrEqual,
                            Opcode::FCmpGt => FloatCC::GreaterThan, _ => FloatCC::GreaterThanOrEqual,
                        };
                        let cond = b.ins().fcmp(cc, lhs, rhs);
                        store_tagged_bool(&mut b, regs, op.a, cond);
                        next!();
                    }
                    Opcode::RefEq => {
                        let lhs = load_reg(&mut b, regs, op.b);
                        let rhs = load_reg(&mut b, regs, op.c);
                        let cond = b.ins().icmp(IntCC::Equal, lhs, rhs);
                        store_tagged_bool(&mut b, regs, op.a, cond);
                        next!();
                    }
                    Opcode::AllocMap
                        if prepared_op.c > 0
                            && prepared_op.c <= 4
                            && prepared_op.next > index as u32 + 1 =>
                    {
                        regs = call_runtime_helper(&mut b, small_map_ref, vm, index as u32, abort);
                        let target = prepared_op.next as usize;
                        if target >= blocks.len() { b.ins().jump(abort, &[]); }
                        else { b.ins().jump(blocks[target], &[BlockArg::Value(vm), BlockArg::Value(regs)]); }
                    }
                    Opcode::CallDirect => {
                        regs = call_runtime_helper(&mut b, direct_ref, vm, index as u32, abort);
                        let target = op.next as usize;
                        if target >= blocks.len() { b.ins().jump(abort, &[]); }
                        else { b.ins().jump(blocks[target], &[BlockArg::Value(vm), BlockArg::Value(regs)]); }
                    }
                    Opcode::Invoke if native_numeric_quick(function, index).is_some() => {
                        let quick = native_numeric_quick(function, index).unwrap();
                        regs = emit_numeric_quick_fast_path(
                            &mut b,
                            runtime_ref,
                            vm,
                            regs,
                            index as u32,
                            op.a,
                            op.b,
                            prepared_op.y as u8,
                            quick,
                            expected_method_epoch,
                            abort,
                        );
                        let target = op.next as usize;
                        if target >= blocks.len() { b.ins().jump(abort, &[]); }
                        else { b.ins().jump(blocks[target], &[BlockArg::Value(vm), BlockArg::Value(regs)]); }
                    }
                    Opcode::Invoke if native_map_property_pic(function, index).is_some() => {
                        let pic = native_map_property_pic(function, index).unwrap();
                        regs = emit_map_property_fast_path(
                            &mut b,
                            runtime_ref,
                            vm,
                            regs,
                            index as u32,
                            op.a,
                            op.b,
                            &pic,
                            expected_method_epoch,
                            abort,
                        );
                        let target = op.next as usize;
                        if target >= blocks.len() { b.ins().jump(abort, &[]); }
                        else { b.ins().jump(blocks[target], &[BlockArg::Value(vm), BlockArg::Value(regs)]); }
                    }
                    _ => {
                        regs = call_runtime_helper(&mut b, runtime_ref, vm, index as u32, abort);
                        let target = op.next as usize;
                        if target >= blocks.len() { b.ins().jump(abort, &[]); }
                        else { b.ins().jump(blocks[target], &[BlockArg::Value(vm), BlockArg::Value(regs)]); }
                    }
                }
            }

            b.switch_to_block(abort);
            let bailout = b.ins().iconst(types::I64, WFJIT_BAILOUT as i64);
            b.ins().return_(&[bailout]);
            b.seal_all_blocks();
            b.finalize(target_config);
        }

        self.module.define_function(func_id, &mut self.context)
            .map_err(|e| format!("cannot define whole function {:?}: {e}", function.id))?;
        self.module.finalize_definitions()
            .map_err(|e| format!("cannot finalize whole function {:?}: {e}", function.id))?;
        let code = self.module.get_finalized_function(func_id);
        let entry: WholeFunctionEntry = unsafe { std::mem::transmute(code) };
        let slot = &mut self.slots[function_index as usize];
        slot.state = FunctionState::Compiled;
        slot.entry = Some(entry);
        self.stats.compiled_functions = self.stats.compiled_functions.saturating_add(1);
        Ok(())
    }
}



#[inline]
fn native_numeric_quick(function: &PreparedFunction, index: usize) -> Option<NumericQuickOp> {
    let flags = function.ops.get(index)?.flags;
    if (flags & QUICK_GENERIC_DISABLED) != 0 {
        return None;
    }
    let quick = NumericQuickOp::from_flag(flags)?;
    // Bitwise Number operators require full ToUint32 coercion. Keep those in the
    // already-fast specialized interpreter handler until we add range feedback.
    if matches!(
        quick,
        NumericQuickOp::BitNot
            | NumericQuickOp::Rem
            | NumericQuickOp::BitAnd
            | NumericQuickOp::BitOr
            | NumericQuickOp::BitXor
            | NumericQuickOp::Shl
            | NumericQuickOp::ShrU
    ) {
        return None;
    }
    Some(quick)
}

fn emit_numeric_quick_fast_path(
    b: &mut FunctionBuilder<'_>,
    runtime_ref: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    regs: cranelift::prelude::Value,
    op_index: u32,
    destination: u8,
    receiver: u8,
    rhs_reg: u8,
    quick: NumericQuickOp,
    expected_method_epoch: u64,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    let fallback = b.create_block();
    let receiver_ok = b.create_block();
    let calculate = b.create_block();
    let done = b.create_block();
    b.append_block_param(done, b.func.dfg.value_type(regs));

    let epoch = b.ins().load(
        types::I64,
        MemFlagsData::trusted(),
        vm,
        offset_of!(Interpreter<'static>, method_epoch) as i32,
    );
    let epoch_ok = b.ins().icmp_imm(IntCC::Equal, epoch, expected_method_epoch as i64);
    b.ins().brif(epoch_ok, receiver_ok, &[], fallback, &[]);

    b.switch_to_block(receiver_ok);
    let lhs_word = load_reg(b, regs, receiver);
    let tag_mask = b.ins().iconst(types::I64, TAG_MASK as i64);
    let tag_base = b.ins().iconst(types::I64, TAG_BASE as i64);
    let lhs_tag = b.ins().band(lhs_word, tag_mask);
    let lhs_num = b.ins().icmp(IntCC::NotEqual, lhs_tag, tag_base);

    let binary = !matches!(quick, NumericQuickOp::Neg);
    if binary {
        let rhs_check = b.create_block();
        b.ins().brif(lhs_num, rhs_check, &[], fallback, &[]);
        b.seal_block(receiver_ok);
        b.switch_to_block(rhs_check);
        let rhs_word = load_reg(b, regs, rhs_reg);
        let rhs_tag = b.ins().band(rhs_word, tag_mask);
        let rhs_num = b.ins().icmp(IntCC::NotEqual, rhs_tag, tag_base);
        b.ins().brif(rhs_num, calculate, &[], fallback, &[]);
        b.seal_block(rhs_check);
    } else {
        b.ins().brif(lhs_num, calculate, &[], fallback, &[]);
        b.seal_block(receiver_ok);
    }

    b.switch_to_block(calculate);
    let lhs = b.ins().bitcast(types::F64, MemFlagsData::new(), lhs_word);
    let rhs = if binary {
        let word = load_reg(b, regs, rhs_reg);
        Some(b.ins().bitcast(types::F64, MemFlagsData::new(), word))
    } else {
        None
    };

    if quick.is_comparison() {
        let rhs = rhs.expect("comparison is binary");
        let cc = match quick {
            NumericQuickOp::Lt => FloatCC::LessThan,
            NumericQuickOp::Le => FloatCC::LessThanOrEqual,
            NumericQuickOp::Gt => FloatCC::GreaterThan,
            NumericQuickOp::Ge => FloatCC::GreaterThanOrEqual,
            NumericQuickOp::Eq => FloatCC::Equal,
            NumericQuickOp::Ne => FloatCC::NotEqual,
            _ => unreachable!(),
        };
        let condition = b.ins().fcmp(cc, lhs, rhs);
        store_tagged_bool(b, regs, destination, condition);
    } else {
        let result = match quick {
            NumericQuickOp::Neg => b.ins().fneg(lhs),
            NumericQuickOp::Add => b.ins().fadd(lhs, rhs.expect("add is binary")),
            NumericQuickOp::Sub => b.ins().fsub(lhs, rhs.expect("sub is binary")),
            NumericQuickOp::Mul => b.ins().fmul(lhs, rhs.expect("mul is binary")),
            NumericQuickOp::Div => b.ins().fdiv(lhs, rhs.expect("div is binary")),
            _ => unreachable!("remainder/bitwise quick ops are filtered before lowering"),
        };
        // Value::from_num canonicalizes every NaN. Preserve that exact observable
        // representation rather than letting backend-specific NaN payloads leak.
        let is_nan = b.ins().fcmp(FloatCC::Unordered, result, result);
        let bits = b.ins().bitcast(types::I64, MemFlagsData::new(), result);
        let canonical_nan = b.ins().iconst(types::I64, CANONICAL_NAN as i64);
        let word = b.ins().select(is_nan, canonical_nan, bits);
        store_reg(b, regs, destination, word);
    }
    b.ins().jump(done, &[BlockArg::Value(regs)]);
    b.seal_block(calculate);

    b.switch_to_block(fallback);
    let fallback_regs = call_runtime_helper(b, runtime_ref, vm, op_index, abort);
    b.ins().jump(done, &[BlockArg::Value(fallback_regs)]);
    b.seal_block(fallback);

    b.switch_to_block(done);
    b.seal_block(done);
    b.block_params(done)[0]
}

#[derive(Clone, Debug)]
struct NativeMapPropertyPic {
    entries: smallvec::SmallVec<[MapIcEntry; 4]>,
}

#[inline]
fn native_map_property_pic(function: &PreparedFunction, index: usize) -> Option<NativeMapPropertyPic> {
    let site = *function.quick_invokes.get(index)?;
    if !matches!(site.target, CallTarget::NativeBuiltin(NativeBuiltin::MapPropertyGet(_))) {
        return None;
    }
    let cache = &function.quick_map_properties.get(index)?.cache;
    if cache.len == 0 || cache.megamorphic {
        return None;
    }
    let mut entries = smallvec::SmallVec::<[MapIcEntry; 4]>::new();
    entries.extend_from_slice(&cache.entries[..cache.len as usize]);
    Some(NativeMapPropertyPic { entries })
}

/// Emit the equivalent of V8's common monomorphic property-load shape: guard the
/// current method epoch, resolve the stable Napalm handle sidecar, guard Map+shape,
/// then load the dense slot directly. Any failed speculation executes the ordinary
/// runtime Invoke and rejoins with the possibly relocated frame-register base.
fn emit_map_property_fast_path(
    b: &mut FunctionBuilder<'_>,
    runtime_ref: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    regs: cranelift::prelude::Value,
    op_index: u32,
    destination: u8,
    receiver: u8,
    pic: &NativeMapPropertyPic,
    expected_method_epoch: u64,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    debug_assert!(!pic.entries.is_empty());

    let fallback = b.create_block();
    let tag_ok = b.create_block();
    let live_ok = b.create_block();
    let kind_ok = b.create_block();
    let done = b.create_block();
    b.append_block_param(done, b.func.dfg.value_type(regs));

    // A method rebind invalidates the meaning of an Invoke site even if the Map
    // shape itself is unchanged. Keep compiled code valid by falling back rather
    // than destroying code that may currently be active on the native stack.
    let epoch = b.ins().load(
        types::I64,
        MemFlagsData::trusted(),
        vm,
        offset_of!(Interpreter<'static>, method_epoch) as i32,
    );
    let epoch_ok = b.ins().icmp_imm(IntCC::Equal, epoch, expected_method_epoch as i64);
    b.ins().brif(epoch_ok, tag_ok, &[], fallback, &[]);

    b.switch_to_block(tag_ok);
    let tagged = load_reg(b, regs, receiver);
    let heap_mask = b.ins().iconst(types::I64, TAG_HEAP_MASK as i64);
    let heap_tag = b.ins().iconst(types::I64, TAG_HEAP as i64);
    let masked = b.ins().band(tagged, heap_mask);
    let is_heap = b.ins().icmp(IntCC::Equal, masked, heap_tag);
    let heap_entry = b.create_block();
    b.ins().brif(is_heap, heap_entry, &[], fallback, &[]);
    b.seal_block(tag_ok);

    b.switch_to_block(heap_entry);
    // Heap handles are compact u32 IDs in the low bits of the tagged Value.
    let handle32 = b.ins().ireduce(types::I32, tagged);
    let handle = b.ins().uextend(types::I64, handle32);
    let chunk_index = b.ins().ushr_imm_u(handle, TRACE_HEAP_CHUNK_BITS as i64);
    let entry_index_mask = b.ins().iconst(types::I64, TRACE_HEAP_CHUNK_MASK as i64);
    let entry_index = b.ins().band(handle, entry_index_mask);
    let chunks = b.ins().iconst(types::I64, handles().jit_chunks() as usize as i64);
    let chunk_offset = b.ins().ishl_imm(chunk_index, 3);
    let chunk_slot = b.ins().iadd(chunks, chunk_offset);
    let chunk = b.ins().load(types::I64, MemFlagsData::trusted(), chunk_slot, 0);

    let entry_offset = if std::mem::size_of::<TraceHeapEntry>() == 48 {
        let x32 = b.ins().ishl_imm(entry_index, 5);
        let x16 = b.ins().ishl_imm(entry_index, 4);
        b.ins().iadd(x32, x16)
    } else {
        let stride = b.ins().iconst(types::I64, std::mem::size_of::<TraceHeapEntry>() as i64);
        b.ins().imul(entry_index, stride)
    };
    let entry = b.ins().iadd(chunk, entry_offset);
    let address = b.ins().load(
        types::I64,
        MemFlagsData::trusted(),
        entry,
        offset_of!(TraceHeapEntry, address) as i32,
    );
    let live = b.ins().icmp_imm(IntCC::NotEqual, address, 0);
    b.ins().brif(live, live_ok, &[], fallback, &[]);
    b.seal_block(heap_entry);

    b.switch_to_block(live_ok);
    let metadata = b.ins().load(
        types::I32,
        MemFlagsData::trusted(),
        entry,
        offset_of!(TraceHeapEntry, kind) as i32,
    );
    let kind_mask = b.ins().iconst(types::I32, TRACE_HEAP_KIND_MASK as i64);
    let kind = b.ins().band(metadata, kind_mask);
    let is_map = b.ins().icmp_imm(IntCC::Equal, kind, TraceHeapKind::Map as i64);
    b.ins().brif(is_map, kind_ok, &[], fallback, &[]);
    b.seal_block(live_ok);

    b.switch_to_block(kind_ok);
    let shape = b.ins().ushr_imm_u(metadata, TRACE_HEAP_METADATA_SHIFT as i64);
    let data = b.ins().load(
        types::I64,
        MemFlagsData::trusted(),
        entry,
        offset_of!(TraceHeapEntry, data) as i32,
    );

    let mut check_block = kind_ok;
    for (entry_index, cached) in pic.entries.iter().enumerate() {
        if entry_index != 0 {
            b.switch_to_block(check_block);
        }
        let hit = b.create_block();
        let miss = if entry_index + 1 == pic.entries.len() {
            fallback
        } else {
            b.create_block()
        };
        let matches_shape = b.ins().icmp_imm(IntCC::Equal, shape, cached.shape as i64);
        b.ins().brif(matches_shape, hit, &[], miss, &[]);
        b.seal_block(check_block);

        b.switch_to_block(hit);
        let value = b.ins().load(
            types::I64,
            MemFlagsData::trusted(),
            data,
            cached.slot as i32 * std::mem::size_of::<RegisterWord>() as i32,
        );
        store_reg(b, regs, destination, value);
        b.ins().jump(done, &[BlockArg::Value(regs)]);
        b.seal_block(hit);

        check_block = miss;
    }

    b.switch_to_block(fallback);
    let fallback_regs = call_runtime_helper(b, runtime_ref, vm, op_index, abort);
    b.ins().jump(done, &[BlockArg::Value(fallback_regs)]);
    b.seal_block(fallback);

    b.switch_to_block(done);
    b.seal_block(done);
    b.block_params(done)[0]
}

#[inline]
fn eligible_opcode(opcode: Opcode) -> bool {
    !matches!(opcode,
        Opcode::Trap | Opcode::Switch | Opcode::Return2 | Opcode::Bailout
        | Opcode::GuardType | Opcode::GuardClass | Opcode::GuardBounds
        | Opcode::GuardNotNull | Opcode::GuardInt32 | Opcode::GuardF64
        | Opcode::InvokeSuper | Opcode::CallClosure | Opcode::Launch
        | Opcode::CallPrimitive | Opcode::CallForeign
        | Opcode::CreateClass | Opcode::BindMethod | Opcode::BindForeignMethod
        | Opcode::AllocForeign | Opcode::MakeClosure
        | Opcode::ImportModule | Opcode::ImportVariable
        | Opcode::MakeFiber | Opcode::FiberCall | Opcode::FiberTry
        | Opcode::FiberTransfer | Opcode::FiberTransferError | Opcode::FiberYield
        | Opcode::FiberSuspend | Opcode::FiberAbort | Opcode::Await | Opcode::Catch
        | Opcode::InvokeSpread
    )
}

#[inline]
fn load_reg(b: &mut FunctionBuilder<'_>, regs: cranelift::prelude::Value, reg: u8) -> cranelift::prelude::Value {
    b.ins().load(types::I64, MemFlagsData::trusted(), regs, reg as i32 * 8)
}
#[inline]
fn store_reg(b: &mut FunctionBuilder<'_>, regs: cranelift::prelude::Value, reg: u8, value: cranelift::prelude::Value) {
    b.ins().store(MemFlagsData::trusted(), value, regs, reg as i32 * 8);
}
#[inline]
fn store_imm(b: &mut FunctionBuilder<'_>, regs: cranelift::prelude::Value, reg: u8, value: u64) {
    let value = b.ins().iconst(types::I64, value as i64);
    store_reg(b, regs, reg, value);
}
#[inline]
fn load_i32(b: &mut FunctionBuilder<'_>, regs: cranelift::prelude::Value, reg: u8) -> cranelift::prelude::Value {
    let word = load_reg(b, regs, reg);
    b.ins().ireduce(types::I32, word)
}
#[inline]
fn store_i32(b: &mut FunctionBuilder<'_>, regs: cranelift::prelude::Value, reg: u8, value: cranelift::prelude::Value) {
    let wide = b.ins().uextend(types::I64, value);
    store_reg(b, regs, reg, wide);
}
#[inline]
fn load_f64(b: &mut FunctionBuilder<'_>, regs: cranelift::prelude::Value, reg: u8) -> cranelift::prelude::Value {
    let word = load_reg(b, regs, reg);
    b.ins().bitcast(types::F64, MemFlagsData::new(), word)
}
#[inline]
fn store_f64(b: &mut FunctionBuilder<'_>, regs: cranelift::prelude::Value, reg: u8, value: cranelift::prelude::Value) {
    let word = b.ins().bitcast(types::I64, MemFlagsData::new(), value);
    store_reg(b, regs, reg, word);
}
#[inline]
fn store_tagged_bool(
    b: &mut FunctionBuilder<'_>,
    regs: cranelift::prelude::Value,
    reg: u8,
    condition: cranelift::prelude::Value,
) {
    let bit = b.ins().uextend(types::I64, condition);
    let tagged = b.ins().iadd_imm(bit, TAG_FALSE as i64);
    store_reg(b, regs, reg, tagged);
}

fn jump_to(
    b: &mut FunctionBuilder<'_>,
    blocks: &[cranelift::prelude::Block],
    target: u32,
    vm: cranelift::prelude::Value,
    regs: cranelift::prelude::Value,
    abort: cranelift::prelude::Block,
) {
    if let Some(&block) = blocks.get(target as usize) { b.ins().jump(block, &[BlockArg::Value(vm), BlockArg::Value(regs)]); }
    else { b.ins().jump(abort, &[]); }
}

fn call_runtime_helper(
    b: &mut FunctionBuilder<'_>,
    helper: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    op: u32,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    let op = b.ins().iconst(types::I32, op as i64);
    let call = b.ins().call(helper, &[vm, op]);
    let regs = b.inst_results(call)[0];
    let failed = b.ins().icmp_imm(IntCC::Equal, regs, 0);
    let ok = b.create_block();
    b.append_block_param(ok, b.func.dfg.value_type(regs));
    b.ins().brif(failed, abort, &[], ok, &[BlockArg::Value(regs)]);
    b.seal_block(ok);
    b.switch_to_block(ok);
    b.block_params(ok)[0]
}

/// Compile a target after it has become hot and execute it immediately if a
/// native entry is available. Auto-memoized targets are never compiled: their
/// normal call path remains authoritative.
pub(super) unsafe fn try_direct_call(
    vm: &mut Interpreter<'_>,
    op: &PreparedOp,
    sources: &[RegisterWord],
) -> bool {
    let target = op.x;
    let Some(jit_ptr) = vm.whole_function_jit.as_mut().map(|jit| &mut **jit as *mut WholeFunctionJit) else {
        return false;
    };
    if vm.prepared.functions[target as usize].memoization.enabled() {
        return false;
    }
    if unsafe { (*jit_ptr).active_depth >= MAX_NATIVE_CALL_DEPTH } {
        return false;
    }

    if unsafe { (*jit_ptr).note_call(target) } {
        let prepared = vm.prepared.functions.as_ptr().add(target as usize);
        let bytecode = &vm.program.functions[target as usize] as *const BytecodeFunction;
        let constants = &vm.resolved_constants[target as usize] as *const Vec<RegisterWord>;
        if let Err(error) = unsafe {
            (*jit_ptr).compile(target, &*prepared, &*bytecode, &*constants, vm.method_epoch)
        } {
            unsafe { (*jit_ptr).reject(target); }
            if std::env::var_os("NAPALM_JIT_EVENTS").is_some() {
                eprintln!("[napalm-wfjit] rejected {:?}: {error}", (&*prepared).id);
            }
        } else if std::env::var_os("NAPALM_JIT_EVENTS").is_some() {
            if unsafe { (*jit_ptr).entry(target) }.is_some() {
                eprintln!("[napalm-wfjit] compiled {:?}", unsafe { (&*prepared).id });
            }
        }
    }

    let Some(entry) = (unsafe { (*jit_ptr).entry(target) }) else { return false; };
    unsafe { execute_native_direct_call(vm, *op, sources, target, entry) };
    true
}

unsafe fn execute_native_direct_call(
    vm: &mut Interpreter<'_>,
    op: PreparedOp,
    sources: &[RegisterWord],
    target: u32,
    entry: WholeFunctionEntry,
) {
    let caller_fiber = vm.current_fiber;
    let caller_depth = vm.fiber(caller_fiber).frames.len();
    vm.enter_direct_bytecode_call(&op, target, sources, None);
    if vm.run_state != RunState::Running { return; }

    // A memo hit or async call completes without pushing a frame. This is kept
    // here even though compiled targets themselves are never memoized, because
    // it makes the call helper safe for future cross-target specialization.
    if vm.current_fiber != caller_fiber || vm.fiber(caller_fiber).frames.len() == caller_depth {
        return;
    }
    if vm.hot_function_index != target { return; }

    {
        let jit = vm.whole_function_jit.as_mut().unwrap();
        jit.stats.native_entries = jit.stats.native_entries.saturating_add(1);
        jit.active_depth = jit.active_depth.saturating_add(1);
    }
    let result = unsafe { entry(vm as *mut _ as *mut c_void, vm.hot_registers) };
    if let Some(jit) = vm.whole_function_jit.as_mut() {
        jit.active_depth = jit.active_depth.saturating_sub(1);
    }
    if result == WFJIT_BAILOUT {
        // The native body deliberately handed control back to the interpreter at
        // the current bytecode PC. Leave the callee frame intact; the outer
        // token-threaded loop resumes it normally.
        return;
    }
    if vm.run_state == RunState::Running
        && vm.current_fiber == caller_fiber
        && vm.fiber(caller_fiber).frames.len() == caller_depth + 1
        && vm.hot_function_index == target
    {
        vm.return_from_frame(RegisterWord(result), None);
    }
}

unsafe extern "C" fn napalm_wfjit_direct_call(vm: *mut c_void, op_index: u32) -> *mut RegisterWord {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running || vm.hot_frame.is_null() { return std::ptr::null_mut(); }
    let caller_fiber = vm.current_fiber;
    let caller_depth = vm.fiber(caller_fiber).frames.len();
    let caller_function = vm.hot_function_index;
    let op = unsafe { *vm.current_prepared().ops.get_unchecked(op_index as usize) };
    let mut words = [RegisterWord(0); 17];
    let count = if op.c <= 4 {
        // Small-call register IDs are packed directly in PreparedOp::args_offset.
        // Treating args_offset as a side-table offset here corrupts recursive
        // arguments (for example createTree(d)). Reuse the interpreter ABI.
        let mut small = [RegisterWord(0); 4];
        let count = unsafe { vm.gather_call_words_small(&op, &mut small) };
        words[..count].copy_from_slice(&small[..count]);
        count
    } else {
        unsafe { vm.gather_call_words_large(&op, &mut words) }
    };
    let target = op.x;
    unsafe { vm.set_pc(op_index); }

    // Always use the ordinary call entry. This is the memoization firewall: if a
    // future native caller reaches an auto-memoized target, probe/miss bookkeeping
    // and return caching still happen exactly once in the existing runtime path.
    vm.enter_direct_bytecode_call(&op, target, &words[..count], None);
    if vm.run_state != RunState::Running { return std::ptr::null_mut(); }
    if vm.current_fiber != caller_fiber { return std::ptr::null_mut(); }
    if vm.fiber(caller_fiber).frames.len() == caller_depth {
        return vm.hot_registers;
    }

    let native = vm.whole_function_jit.as_ref().and_then(|jit| {
        (!vm.prepared.functions[target as usize].memoization.enabled()
            && jit.active_depth < MAX_NATIVE_CALL_DEPTH)
            .then(|| jit.entry(target))
            .flatten()
    });
    if let Some(entry) = native {
        if let Some(jit) = vm.whole_function_jit.as_mut() {
            jit.stats.native_direct_calls = jit.stats.native_direct_calls.saturating_add(1);
            jit.stats.native_entries = jit.stats.native_entries.saturating_add(1);
            jit.active_depth = jit.active_depth.saturating_add(1);
        }
        let result = unsafe { entry(vm as *mut _ as *mut c_void, vm.hot_registers) };
        if let Some(jit) = vm.whole_function_jit.as_mut() {
            jit.active_depth = jit.active_depth.saturating_sub(1);
        }
        if result == WFJIT_BAILOUT {
            // Finish this callee in baseline so the native caller can resume with
            // the same direct-call ABI and a freshly relocated register base.
            while vm.run_state == RunState::Running
                && vm.current_fiber == caller_fiber
                && vm.fiber(caller_fiber).frames.len() > caller_depth
            {
                let pc = vm.hot_pc as usize;
                let handler = unsafe { *vm.hot_handlers.add(pc) };
                let prepared_op = unsafe { vm.hot_ops.add(pc) };
                unsafe { handler(vm, prepared_op); }
            }
        } else if vm.run_state == RunState::Running
            && vm.current_fiber == caller_fiber
            && vm.fiber(caller_fiber).frames.len() == caller_depth + 1
            && vm.hot_function_index == target
        {
            vm.return_from_frame(RegisterWord(result), None);
        }
    } else {
        if let Some(jit) = vm.whole_function_jit.as_mut() {
            jit.stats.baseline_direct_calls = jit.stats.baseline_direct_calls.saturating_add(1);
        }
        while vm.run_state == RunState::Running
            && vm.current_fiber == caller_fiber
            && vm.fiber(caller_fiber).frames.len() > caller_depth
        {
            let pc = vm.hot_pc as usize;
            let handler = unsafe { *vm.hot_handlers.add(pc) };
            let prepared_op = unsafe { vm.hot_ops.add(pc) };
            unsafe { handler(vm, prepared_op); }
        }
    }

    if vm.run_state == RunState::Running
        && vm.current_fiber == caller_fiber
        && vm.fiber(caller_fiber).frames.len() == caller_depth
        && vm.hot_function_index == caller_function
    {
        vm.hot_registers
    } else {
        std::ptr::null_mut()
    }
}

unsafe extern "C" fn napalm_wfjit_small_map_literal(
    vm: *mut c_void,
    op_index: u32,
) -> *mut RegisterWord {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running || vm.hot_frame.is_null() {
        return std::ptr::null_mut();
    }
    let caller_fiber = vm.current_fiber;
    let caller_depth = vm.fiber(caller_fiber).frames.len();
    let caller_function = vm.hot_function_index;
    unsafe { vm.set_pc(op_index); }

    let prepared = vm.current_prepared();
    let op = unsafe { *prepared.ops.get_unchecked(op_index as usize) };
    let handler = prepared.jit_entry_handlers[op_index as usize]
        .unwrap_or(prepared.handlers[op_index as usize]);
    unsafe { handler(vm, &op); }

    if vm.run_state == RunState::Running
        && vm.current_fiber == caller_fiber
        && vm.fiber(caller_fiber).frames.len() == caller_depth
        && vm.hot_function_index == caller_function
    {
        vm.hot_registers
    } else {
        std::ptr::null_mut()
    }
}

unsafe extern "C" fn napalm_wfjit_runtime_op(vm: *mut c_void, op_index: u32) -> *mut RegisterWord {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running || vm.hot_frame.is_null() { return std::ptr::null_mut(); }
    let caller_fiber = vm.current_fiber;
    let caller_depth = vm.fiber(caller_fiber).frames.len();
    let caller_function = vm.hot_function_index;
    unsafe { vm.set_pc(op_index); }

    let raw = unsafe { *vm.current_prepared().original_ops.get_unchecked(op_index as usize) };
    let opcode = vm.program.functions[caller_function as usize].chunk.code[raw.raw_pc as usize]
        .opcode().unwrap();

    if opcode == Opcode::Invoke {
        let prepared = vm.current_prepared();
        let op = unsafe { *prepared.ops.get_unchecked(op_index as usize) };
        let current = prepared.jit_entry_handlers[op_index as usize]
            .unwrap_or(prepared.handlers[op_index as usize]);
        let numeric_quick = (op.flags & QUICK_GENERIC_DISABLED) == 0
            && NumericQuickOp::from_flag(op.flags).is_some();
        if numeric_quick || is_unfused_monomorphic_invoke_handler(current) {
            unsafe { current(vm, &op); }
        } else {
            // Unsupported/polymorphic Invoke stays authoritative in baseline. The
            // native caller sees a null helper result, returns WFJIT_BAILOUT, and
            // resumes this exact PC without executing the call twice.
            return std::ptr::null_mut();
        }
    } else {
        let mut op = raw;
        let prepared_op = unsafe { *vm.current_prepared().ops.get_unchecked(op_index as usize) };
        if matches!(opcode, Opcode::Switch | Opcode::LoadModuleVar | Opcode::StoreModuleVar
            | Opcode::LoadField | Opcode::StoreField | Opcode::MakeClosure | Opcode::CreateClass
            | Opcode::AllocInstance | Opcode::AllocForeign) {
            op.x = prepared_op.x;
            op.y = prepared_op.y;
        }
        if matches!(opcode, Opcode::MapGet | Opcode::MapSet | Opcode::LoadProperty
            | Opcode::StoreProperty | Opcode::StorePropertyIfPresent) {
            op.cache = prepared_op.cache;
        }

        let (handler, op_to_run) = if opcode == Opcode::AllocMap
            && prepared_op.c > 0 && prepared_op.c <= 4 && prepared_op.next > op_index + 1
        {
            let handler = vm.current_prepared().jit_entry_handlers[op_index as usize]
                .unwrap_or(vm.current_prepared().handlers[op_index as usize]);
            (handler, prepared_op)
        } else {
            (handler_for(opcode), op)
        };
        unsafe { handler(vm, &op_to_run); }
    }

    if vm.run_state == RunState::Running
        && vm.current_fiber == caller_fiber
        && vm.fiber(caller_fiber).frames.len() == caller_depth
        && vm.hot_function_index == caller_function
    {
        vm.hot_registers
    } else {
        std::ptr::null_mut()
    }
}
