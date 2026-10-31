//! Whole-function Cranelift tier for hot synchronous bytecode functions.
//!
//! Top-level entries execute against the interpreter's existing frame/register
//! stack. Eligible non-memoized self-recursion uses a lightweight native shadow
//! frame instead of constructing another Napalm `Frame`; the VM root provider
//! scans those native registers with the ordinary bytecode `ReferenceMap`.
//! Calls that cannot use that ABI still flow through `enter_direct_bytecode_call`,
//! preserving auto-memoization, async semantics and the normal frame ABI.

use super::*;
use cranelift::codegen::ir::{BlockArg, StackSlotData, StackSlotKind, UserFuncName};
use cranelift::codegen::settings::{self, Configurable};
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::jit::{JITBuilder, JITModule};
use cranelift::module::{default_libcall_names, Linkage, Module};
use cranelift::prelude::{AbiParam, FloatCC, InstBuilder, IntCC, MemFlagsData, types};
use std::ffi::c_void;
use std::mem::offset_of;

use crate::napalm_tracing_jit::{TRACE_HEAP_KIND_MASK, TRACE_HEAP_METADATA_SHIFT};

const MIN_WHOLE_FUNCTION_HOTNESS: u32 = 128;
const MAX_WHOLE_FUNCTION_OPS: usize = 4096;
const MAX_NATIVE_CALL_DEPTH: u32 = 256;
const WFJIT_BAILOUT: u64 = TAG_BASE;

pub(super) type WholeFunctionEntry = unsafe extern "C" fn(*mut c_void, *mut RegisterWord) -> u64;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WholeFunctionJitStats {
    pub hot_calls: u64,
    pub compile_attempts: u64,
    pub compiled_functions: u64,
    pub rejected_functions: u64,
    pub profile_deferrals: u64,
    pub native_entries: u64,
    pub native_direct_calls: u64,
    /// Self-recursive calls executed through the lightweight GC-visible native ABI.
    pub native_recursive_calls: u64,
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
    profile_retries: u8,
    state: FunctionState,
    entry: Option<WholeFunctionEntry>,
}

impl Default for FunctionSlot {
    fn default() -> Self {
        Self { hotness: 0, profile_retries: 0, state: FunctionState::Cold, entry: None }
    }
}

pub(super) struct WholeFunctionJit {
    module: JITModule,
    context: cranelift::codegen::Context,
    function_context: FunctionBuilderContext,
    runtime_op: cranelift::module::FuncId,
    direct_call: cranelift::module::FuncId,
    small_map_literal: cranelift::module::FuncId,
    native_self_call: cranelift::module::FuncId,
    small_map_literal_regs: cranelift::module::FuncId,
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
        builder.symbol("napalm_wfjit_native_self_call", napalm_wfjit_native_self_call as *const u8);
        builder.symbol(
            "napalm_wfjit_small_map_literal_regs",
            napalm_wfjit_small_map_literal_regs as *const u8,
        );
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

        // Lightweight native-recursion helpers receive the owning function/op
        // identity plus the explicit register base. They never rely on hot-frame
        // register pointers, which is what lets nested calls avoid a Napalm Frame.
        let mut native_call_sig = module.make_signature();
        native_call_sig.params.push(AbiParam::new(pointer));
        native_call_sig.params.push(AbiParam::new(types::I32));
        native_call_sig.params.push(AbiParam::new(types::I32));
        native_call_sig.params.push(AbiParam::new(pointer));
        native_call_sig.returns.push(AbiParam::new(types::I64));
        let native_self_call_id = module
            .declare_function("napalm_wfjit_native_self_call", Linkage::Import, &native_call_sig)
            .map_err(|e| format!("cannot declare native self-call helper: {e}"))?;
        let small_map_regs_id = module
            .declare_function(
                "napalm_wfjit_small_map_literal_regs",
                Linkage::Import,
                &native_call_sig,
            )
            .map_err(|e| format!("cannot declare register-based small-map helper: {e}"))?;

        let context = module.make_context();
        let function_context = FunctionBuilderContext::new();
        let result = Self {
            module,
            context,
            function_context,
            runtime_op: runtime_id,
            direct_call: direct_id,
            small_map_literal: small_map_id,
            native_self_call: native_self_call_id,
            small_map_literal_regs: small_map_regs_id,
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
        identity_sites: &[Option<NativeIdentitySite>],
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

        let reachable = reachable_ops(function, bytecode);
        for (index, op) in function.ops.iter().enumerate() {
            if !reachable[index] { continue; }
            let opcode = bytecode.chunk.code
                .get(op.raw_pc as usize)
                .and_then(|word| word.opcode())
                .ok_or_else(|| format!("invalid opcode at prepared op {index}"))?;
            let unsupported_invoke = opcode == Opcode::Invoke
                && native_numeric_quick(function, index).is_none()
                && native_map_property_pic(function, index).is_none()
                && identity_sites[index].is_none();
            // A recursive descent can become hot before it has returned even
            // once, leaving the unwind path unprofiled. Retry after more calls
            // instead of permanently rejecting a function on its first visit.
            if unsupported_invoke && self.slots[function_index as usize].profile_retries < 4 {
                let slot = &mut self.slots[function_index as usize];
                slot.hotness = 0;
                slot.profile_retries += 1;
                self.stats.profile_deferrals += 1;
                return Ok(());
            }
            if !eligible_opcode(opcode)
                || unsupported_invoke
                || (matches!(opcode, Opcode::Jump | Opcode::JumpIfFalse | Opcode::JumpIfTrue)
                    && op.x as usize <= index)
            {
                if std::env::var_os("NAPALM_JIT_EVENTS").is_some() {
                    eprintln!("[napalm-wfjit] rejected {:?} at {index}: {opcode:?} (flags={}, target={:?})", function.id, op.flags, function.quick_invokes.get(index).map(|site| site.target));
                }
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

        let native_recursive_safe =
            native_recursive_body_safe(function_index, function, bytecode, identity_sites);
        if std::env::var_os("NAPALM_JIT_EVENTS").is_some() {
            eprintln!("[napalm-wfjit] {:?}: native recursion={native_recursive_safe}, registers={}", function.id, function.frame.general_register_count);
        }
        let runtime_module_id = self.runtime_op;
        let direct_module_id = self.direct_call;
        let small_map_module_id = self.small_map_literal;
        let native_self_module_id = self.native_self_call;
        let small_map_regs_module_id = self.small_map_literal_regs;

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
            let _small_map_ref = self.module.declare_func_in_func(small_map_module_id, b.func);
            let native_self_ref = self.module.declare_func_in_func(native_self_module_id, b.func);
            let self_ref = self.module.declare_func_in_func(func_id, b.func);
            let small_map_regs_ref =
                self.module.declare_func_in_func(small_map_regs_module_id, b.func);

            for index in 0..function.ops.len() {
                b.switch_to_block(blocks[index]);
                if !reachable[index] {
                    b.ins().jump(abort, &[]);
                    continue;
                }
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
                    Opcode::LoadModuleVar => {
                        // VM preparation resolves module slots to stable native
                        // storage. Read each time so assignments remain visible.
                        let address = unsafe { module_slot_ptr(&prepared_op) } as usize;
                        let slot = b.ins().iconst(types::I64, address as i64);
                        let value = b.ins().load(types::I64, MemFlagsData::trusted(), slot, 0);
                        store_reg(&mut b, regs, op.a, value);
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
                        let value = call_native_value_helper(
                            &mut b,
                            small_map_regs_ref,
                            vm,
                            function_index,
                            index as u32,
                            regs,
                            abort,
                        );
                        store_reg(&mut b, regs, op.a, value);
                        let target = prepared_op.next as usize;
                        if target >= blocks.len() { b.ins().jump(abort, &[]); }
                        else { b.ins().jump(blocks[target], &[BlockArg::Value(vm), BlockArg::Value(regs)]); }
                    }
                    Opcode::CallDirect
                        if native_recursive_safe && prepared_op.x == function_index =>
                    {
                        let value = emit_native_self_call(
                            &mut b, self_ref, vm, regs, function_index,
                            function, index, abort,
                        );
                        store_reg(&mut b, regs, op.a, value);
                        let target = op.next as usize;
                        if target >= blocks.len() { b.ins().jump(abort, &[]); }
                        else { b.ins().jump(blocks[target], &[BlockArg::Value(vm), BlockArg::Value(regs)]); }
                    }
                    Opcode::CallDirect => {
                        regs = call_runtime_helper(&mut b, direct_ref, vm, regs, index as u32, abort);
                        let target = op.next as usize;
                        if target >= blocks.len() { b.ins().jump(abort, &[]); }
                        else { b.ins().jump(blocks[target], &[BlockArg::Value(vm), BlockArg::Value(regs)]); }
                    }
                    Opcode::Invoke if identity_sites[index].is_some() => {
                        regs = emit_identity_fast_path(
                            &mut b, runtime_ref, vm, regs, index as u32, &op,
                            identity_sites[index].as_ref().unwrap(), expected_method_epoch, abort,
                        );
                        next!();
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
                        regs = call_runtime_helper(&mut b, runtime_ref, vm, regs, index as u32, abort);
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



struct NativeIdentitySite {
    classes: smallvec::SmallVec<[ReceiverClass; 4]>,
    negate: bool,
}

/// Snapshot only implementations the host explicitly permits us to inline.
/// A polymorphic site (e.g. Map | Null) can still compare in native code.
fn native_identity_sites(vm: &Interpreter<'_>, function: u32) -> Vec<Option<NativeIdentitySite>> {
    let bytecode = &vm.program.functions[function as usize];
    vm.prepared.functions[function as usize].original_ops.iter().map(|op| {
        if bytecode.chunk.code[op.raw_pc as usize].opcode() != Some(Opcode::Invoke) || op.c != 1 {
            return None;
        }
        let descriptor = bytecode.chunk.inline_caches.get(op.x as usize)?;
        let signature = vm.program.metadata.method(descriptor.method)?;
        let name = vm.program.metadata.string(signature.name)?;
        if name != "==" && name != "!=" { return None; }
        let cache = vm.inline_caches[function as usize].get(op.x as usize)?;
        let mut classes = smallvec::SmallVec::new();
        for entry in &cache.entries[..cache.len as usize] {
            if let CallTarget::Primitive(primitive) = entry.target {
                if vm.host.standard_identity_method(primitive, name, signature.arity) {
                    classes.push(entry.class);
                }
            }
        }
        (!classes.is_empty()).then_some(NativeIdentitySite { classes, negate: name == "!=" })
    }).collect()
}

fn emit_identity_fast_path(
    b: &mut FunctionBuilder<'_>,
    runtime: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    regs: cranelift::prelude::Value,
    index: u32,
    op: &PreparedOp,
    site: &NativeIdentitySite,
    expected_epoch: u64,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    let classify = b.create_block();
    let heap = b.create_block();
    let resolve = b.create_block();
    let calculate = b.create_block();
    let fallback = b.create_block();
    let done = b.create_block();
    b.append_block_param(done, types::I64);
    let epoch = b.ins().load(types::I64, MemFlagsData::trusted(), vm,
        offset_of!(Interpreter<'static>, method_epoch) as i32);
    let valid = b.ins().icmp_imm(IntCC::Equal, epoch, expected_epoch as i64);
    b.ins().brif(valid, classify, &[], fallback, &[]);

    b.switch_to_block(classify);
    let lhs = load_reg(b, regs, op.b);
    let mut immediate = b.ins().iconst(types::I8, 0);
    for class in &site.classes {
        let tags: &[u64] = if class.0 == CLASS_KIND_BUILTIN | BuiltinClass::Null as u64 {
            &[TAG_NULL]
        } else if class.0 == CLASS_KIND_BUILTIN | BuiltinClass::Bool as u64 {
            &[TAG_FALSE, TAG_TRUE]
        } else { &[] };
        for &tag in tags {
            let matches = b.ins().icmp_imm(IntCC::Equal, lhs, tag as i64);
            immediate = b.ins().bor(immediate, matches);
        }
    }
    b.ins().brif(immediate, calculate, &[], heap, &[]);
    b.switch_to_block(heap);
    let mask = b.ins().iconst(types::I64, TAG_HEAP_MASK as i64);
    let tag = b.ins().band(lhs, mask);
    let is_heap = b.ins().icmp_imm(IntCC::Equal, tag, TAG_HEAP as i64);
    b.ins().brif(is_heap, resolve, &[], fallback, &[]);

    b.switch_to_block(resolve);
    let handle32 = b.ins().ireduce(types::I32, lhs);
    let handle = b.ins().uextend(types::I64, handle32);
    let chunk_index = b.ins().ushr_imm_u(handle, TRACE_HEAP_CHUNK_BITS as i64);
    let slot_mask = b.ins().iconst(types::I64, TRACE_HEAP_CHUNK_MASK as i64);
    let slot = b.ins().band(handle, slot_mask);
    let chunks = b.ins().iconst(types::I64, handles().jit_chunks() as usize as i64);
    let offset = b.ins().ishl_imm(chunk_index, 3);
    let chunk_slot = b.ins().iadd(chunks, offset);
    let chunk = b.ins().load(types::I64, MemFlagsData::trusted(), chunk_slot, 0);
    let offset = b.ins().imul_imm(slot, std::mem::size_of::<TraceHeapEntry>() as i64);
    let entry = b.ins().iadd(chunk, offset);
    let class = b.ins().load(types::I64, MemFlagsData::trusted(), entry,
        offset_of!(TraceHeapEntry, receiver_class) as i32);
    let mut accepted = b.ins().iconst(types::I8, 0);
    for expected in &site.classes {
        let matches = b.ins().icmp_imm(IntCC::Equal, class, expected.0 as i64);
        accepted = b.ins().bor(accepted, matches);
    }
    b.ins().brif(accepted, calculate, &[], fallback, &[]);

    b.switch_to_block(calculate);
    let rhs = load_reg(b, regs, op.args_offset as u8);
    let cc = if site.negate { IntCC::NotEqual } else { IntCC::Equal };
    let result = b.ins().icmp(cc, lhs, rhs);
    store_tagged_bool(b, regs, op.a, result);
    b.ins().jump(done, &[BlockArg::Value(regs)]);

    b.switch_to_block(fallback);
    let fallback_regs = call_runtime_helper(b, runtime, vm, regs, index, abort);
    b.ins().jump(done, &[BlockArg::Value(fallback_regs)]);
    b.switch_to_block(done);
    b.block_params(done)[0]
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
    let fallback_regs = call_runtime_helper(b, runtime_ref, vm, regs, op_index, abort);
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
    let fallback_regs = call_runtime_helper(b, runtime_ref, vm, regs, op_index, abort);
    b.ins().jump(done, &[BlockArg::Value(fallback_regs)]);
    b.seal_block(fallback);

    b.switch_to_block(done);
    b.seal_block(done);
    b.block_params(done)[0]
}


/// Whether a compiled function may recurse through the lightweight native ABI.
///
/// A bailout from a nested native call replays the outermost self-call in the
/// ordinary interpreter frame. Restrict this path to replay-safe operations so a
/// speculative guard failure cannot duplicate an externally visible store or
/// host effect. Small object-literal allocation is allowed because an abandoned
/// object is unreachable and therefore semantically invisible.
fn reachable_ops(function: &PreparedFunction, bytecode: &BytecodeFunction) -> Vec<bool> {
    let mut seen = vec![false; function.ops.len()];
    let mut work = vec![0usize];
    while let Some(index) = work.pop() {
        if index >= seen.len() || seen[index] { continue; }
        seen[index] = true;
        let prepared = function.ops[index];
        let original = function.original_ops[index];
        match bytecode.chunk.code[original.raw_pc as usize].opcode() {
            Some(Opcode::Return | Opcode::Return2 | Opcode::Trap) => {},
            Some(Opcode::Jump) => work.push(prepared.x as usize),
            Some(Opcode::JumpIfFalse | Opcode::JumpIfTrue) => {
                work.push(prepared.x as usize);
                work.push(original.next as usize);
            }
            Some(Opcode::AllocMap) if prepared.c > 0 && prepared.c <= 4
                && prepared.next > index as u32 + 1 => work.push(prepared.next as usize),
            _ => work.push(original.next as usize),
        }
    }
    seen
}

fn native_recursive_body_safe(
    function_index: u32,
    function: &PreparedFunction,
    bytecode: &BytecodeFunction,
    identity_sites: &[Option<NativeIdentitySite>],
) -> bool {
    if function.uses_this
        || function.capture_count != 0
        || function.memoization.enabled()
        || function.is_async
        || function.is_concurrent
        || !function.exception_handlers.is_empty()
    {
        if std::env::var_os("NAPALM_JIT_EVENTS").is_some() {
            eprintln!("[napalm-wfjit] native recursion banks: args={}, returns={}, this={}, captures={}", function.needs_argument_bank, function.needs_return_bank, function.uses_this, function.capture_count);
        }
        return false;
    }

    let mut seen = vec![false; function.ops.len()];
    let mut work = vec![0usize];
    while let Some(index) = work.pop() {
        if index >= function.ops.len() || seen[index] {
            continue;
        }
        seen[index] = true;

        let prepared = function.ops[index];
        let op = function.original_ops[index];
        let Some(opcode) = bytecode
            .chunk
            .code
            .get(op.raw_pc as usize)
            .and_then(|word| word.opcode())
        else {
            return false;
        };

        let replay_safe = match opcode {
            Opcode::Nop
            | Opcode::Move
            | Opcode::Swap
            | Opcode::LoadNull
            | Opcode::LoadFalse
            | Opcode::LoadTrue
            | Opcode::LoadI8
            | Opcode::LoadI32
            | Opcode::LoadConst
            | Opcode::LoadModuleVar
            | Opcode::Jump
            | Opcode::JumpIfFalse
            | Opcode::JumpIfTrue
            | Opcode::Return
            | Opcode::INeg
            | Opcode::IBitNot
            | Opcode::IAdd
            | Opcode::ISub
            | Opcode::IMul
            | Opcode::IBitAnd
            | Opcode::IBitOr
            | Opcode::IBitXor
            | Opcode::IShl
            | Opcode::IShrU
            | Opcode::ISar
            | Opcode::FNeg
            | Opcode::FAdd
            | Opcode::FSub
            | Opcode::FMul
            | Opcode::FDiv
            | Opcode::ICmpEq
            | Opcode::ICmpNe
            | Opcode::ICmpLt
            | Opcode::ICmpLe
            | Opcode::ICmpGt
            | Opcode::ICmpGe
            | Opcode::FCmpEq
            | Opcode::FCmpNe
            | Opcode::FCmpLt
            | Opcode::FCmpLe
            | Opcode::FCmpGt
            | Opcode::FCmpGe
            | Opcode::RefEq => true,
            Opcode::CallDirect => prepared.x == function_index,
            Opcode::Invoke => {
                native_numeric_quick(function, index).is_some()
                    || native_map_property_pic(function, index).is_some()
                    || identity_sites[index].is_some()
            }
            Opcode::AllocMap => {
                prepared.c > 0
                    && prepared.c <= 4
                    && prepared.next > index as u32 + 1
            }
            _ => false,
        };
        if !replay_safe {
            if std::env::var_os("NAPALM_JIT_EVENTS").is_some() { eprintln!("[napalm-wfjit] native recursion unsafe {index}: {opcode:?}"); }
            return false;
        }

        if let Some(&root_map) = function.gc_map_by_op.get(index) {
            if root_map != NO_INDEX {
                let Some(_map) = function.gc_maps.get(root_map as usize) else {
                    return false;
                };
                // These maps conservatively include interpreter banks. None of
                // the reachable instructions above reads them: all operands
                // live in GPRs. Inherited VM objects (module, fiber, etc.) are
                // still rooted by the outer interpreter frame. Do not reject a
                // native body merely because its baseline map has bank bits.
            }
        }

        match opcode {
            Opcode::Return => {}
            Opcode::Jump => {
                let target = prepared.x as usize;
                if target >= function.ops.len() {
                    return false;
                }
                work.push(target);
            }
            Opcode::JumpIfFalse | Opcode::JumpIfTrue => {
                let target = prepared.x as usize;
                let fallthrough = op.next as usize;
                if target >= function.ops.len() || fallthrough >= function.ops.len() {
                    return false;
                }
                work.push(target);
                work.push(fallthrough);
            }
            Opcode::AllocMap
                if prepared.c > 0
                    && prepared.c <= 4
                    && prepared.next > index as u32 + 1 =>
            {
                let target = prepared.next as usize;
                if target >= function.ops.len() {
                    return false;
                }
                work.push(target);
            }
            _ => {
                let target = op.next as usize;
                if target >= function.ops.len() {
                    return false;
                }
                work.push(target);
            }
        }
    }
    true
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
    current_regs: cranelift::prelude::Value,
    op: u32,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    // The legacy runtime helpers execute against Interpreter::hot_registers.
    // A native shadow-frame callee has a different register base, so crossing
    // such a helper would silently operate on its outer interpreter frame.
    // Bail the complete native recursion chain instead; the top-level Napalm
    // frame will replay the call through the authoritative baseline path.
    let hot_regs = b.ins().load(
        types::I64,
        MemFlagsData::trusted(),
        vm,
        offset_of!(Interpreter<'static>, hot_registers) as i32,
    );
    let is_top_frame = b.ins().icmp(IntCC::Equal, current_regs, hot_regs);
    let runtime = b.create_block();
    b.ins().brif(is_top_frame, runtime, &[], abort, &[]);
    b.switch_to_block(runtime);
    b.seal_block(runtime);

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

/// Call a helper that operates on an explicit GPR base and returns one Napalm
/// register word. `WFJIT_BAILOUT` is reserved and propagates directly to the
/// enclosing whole-function bailout block.
fn emit_native_self_call(
    b: &mut FunctionBuilder<'_>,
    callee: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    regs: cranelift::prelude::Value,
    function_index: u32,
    function: &PreparedFunction,
    index: usize,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    let op = function.ops[index];
    let map = function.gc_map_by_op[index];
    let flags = MemFlagsData::trusted();
    let top = b.create_block();
    let nested = b.create_block();
    let ready = b.create_block();
    let hot_regs = b.ins().load(types::I64, flags, vm,
        offset_of!(Interpreter<'static>, hot_registers) as i32);
    let is_top = b.ins().icmp(IntCC::Equal, regs, hot_regs);
    let root_map = b.ins().iconst(types::I32, map as i64);
    b.ins().brif(is_top, top, &[], nested, &[]);
    b.switch_to_block(top);
    let frame = b.ins().load(types::I64, flags, vm,
        offset_of!(Interpreter<'static>, hot_frame) as i32);
    b.ins().store(flags, root_map, frame, offset_of!(Frame, root_map) as i32);
    let pc = b.ins().iconst(types::I32, index as i64);
    b.ins().store(flags, pc, vm, offset_of!(Interpreter<'static>, hot_pc) as i32);
    b.ins().jump(ready, &[]);
    b.switch_to_block(nested);
    let parent = b.ins().load(types::I64, flags, vm,
        offset_of!(Interpreter<'static>, native_shadow_top) as i32);
    b.ins().store(flags, root_map, parent, offset_of!(NativeShadowFrame, root_map) as i32);
    b.ins().jump(ready, &[]);
    b.switch_to_block(ready);

    // Option<Box<T>> has the nullable-pointer ABI. A native entry always owns
    // a live JIT; its box and compiled code cannot be replaced while active.
    let jit = b.ins().load(types::I64, flags, vm,
        offset_of!(Interpreter<'static>, whole_function_jit) as i32);
    let depth = b.ins().load(types::I32, flags, jit, offset_of!(WholeFunctionJit, active_depth) as i32);
    let allowed = b.ins().icmp_imm(IntCC::UnsignedLessThan, depth, MAX_NATIVE_CALL_DEPTH as i64);
    let call_block = b.create_block();
    b.ins().brif(allowed, call_block, &[], abort, &[]);
    b.switch_to_block(call_block);

    // Size storage for this function's actual register count, rather than
    // paying for a 256-register Rust helper frame at every recursive call.
    let count = function.frame.general_register_count;
    let storage = b.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot, u32::from(count.max(1)) * 8, 3));
    let child_regs = b.ins().stack_addr(types::I64, storage, 0);
    for argument in 0..function.arity.max(op.c) as usize {
        let destination = function.entry_arg_dsts[argument];
        if destination == u16::MAX { continue; }
        let value = if argument < op.c as usize {
            let source = unsafe { wfjit_call_source_reg(function, &op, argument) }.unwrap();
            load_reg(b, regs, source)
        } else {
            b.ins().iconst(types::I64, TAG_NULL as i64)
        };
        store_reg(b, child_regs, destination as u8, value);
    }

    let shadow_slot = b.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot, std::mem::size_of::<NativeShadowFrame>() as u32, 3));
    let shadow = b.ins().stack_addr(types::I64, shadow_slot, 0);
    let previous = b.ins().load(types::I64, flags, vm,
        offset_of!(Interpreter<'static>, native_shadow_top) as i32);
    b.ins().store(flags, previous, shadow, offset_of!(NativeShadowFrame, previous) as i32);
    b.ins().store(flags, child_regs, shadow, offset_of!(NativeShadowFrame, registers) as i32);
    let function_id = b.ins().iconst(types::I32, function_index as i64);
    b.ins().store(flags, function_id, shadow, offset_of!(NativeShadowFrame, function) as i32);
    let entry_map = b.ins().iconst(types::I32, function.entry_gc_map as i64);
    b.ins().store(flags, entry_map, shadow, offset_of!(NativeShadowFrame, root_map) as i32);
    let register_count = b.ins().iconst(types::I16, count as i64);
    b.ins().store(flags, register_count, shadow, offset_of!(NativeShadowFrame, reg_count) as i32);
    b.ins().store(flags, shadow, vm, offset_of!(Interpreter<'static>, native_shadow_top) as i32);
    let child_depth = b.ins().iadd_imm(depth, 1);
    b.ins().store(flags, child_depth, jit, offset_of!(WholeFunctionJit, active_depth) as i32);
    for counter in [offset_of!(WholeFunctionJitStats, native_entries),
        offset_of!(WholeFunctionJitStats, native_recursive_calls)] {
        let offset = (offset_of!(WholeFunctionJit, stats) + counter) as i32;
        let old = b.ins().load(types::I64, flags, jit, offset);
        let next = b.ins().iadd_imm(old, 1);
        b.ins().store(flags, next, jit, offset);
    }

    let call = b.ins().call(callee, &[vm, child_regs]);
    let value = b.inst_results(call)[0];
    b.ins().store(flags, previous, vm, offset_of!(Interpreter<'static>, native_shadow_top) as i32);
    b.ins().store(flags, depth, jit, offset_of!(WholeFunctionJit, active_depth) as i32);
    let failed = b.ins().icmp_imm(IntCC::Equal, value, WFJIT_BAILOUT as i64);
    let done = b.create_block();
    b.ins().brif(failed, abort, &[], done, &[]);
    b.switch_to_block(done);
    value
}

fn call_native_value_helper(
    b: &mut FunctionBuilder<'_>,
    helper: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    function_index: u32,
    op_index: u32,
    regs: cranelift::prelude::Value,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    let function = b.ins().iconst(types::I32, function_index as i64);
    let op = b.ins().iconst(types::I32, op_index as i64);
    let call = b.ins().call(helper, &[vm, function, op, regs]);
    let value = b.inst_results(call)[0];
    let failed = b.ins().icmp_imm(IntCC::Equal, value, WFJIT_BAILOUT as i64);
    let ok = b.create_block();
    b.append_block_param(ok, types::I64);
    b.ins().brif(failed, abort, &[], ok, &[BlockArg::Value(value)]);
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
        let identity_sites = native_identity_sites(vm, target);
        let prepared = vm.prepared.functions.as_ptr().add(target as usize);
        let bytecode = &vm.program.functions[target as usize] as *const BytecodeFunction;
        let constants = &vm.resolved_constants[target as usize] as *const Vec<RegisterWord>;
        if let Err(error) = unsafe {
            (*jit_ptr).compile(target, &*prepared, &*bytecode, &*constants, vm.method_epoch, &identity_sites)
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


#[inline(always)]
unsafe fn wfjit_set_native_root_map(
    vm: &mut Interpreter<'_>,
    function_index: u32,
    op_index: u32,
    registers: *mut RegisterWord,
) -> bool {
    let Some(function) = vm.prepared.functions.get(function_index as usize) else {
        return false;
    };
    let Some(&root_map) = function.gc_map_by_op.get(op_index as usize) else {
        return false;
    };
    if root_map == NO_INDEX {
        return false;
    }

    if registers == vm.hot_registers {
        if vm.hot_frame.is_null() || vm.hot_function_index != function_index {
            return false;
        }
        // Keep the ordinary Fiber frame precise while the top-level native body
        // is active. hot_pc is also the exact replay point if a nested native
        // speculation bails out.
        unsafe { (*vm.hot_frame).root_map = root_map; }
        vm.hot_pc = op_index;
        return true;
    }

    let shadow = vm.native_shadow_top;
    if shadow.is_null() {
        return false;
    }
    let frame = unsafe { &mut *shadow };
    if frame.registers != registers || frame.function != function_index {
        return false;
    }
    frame.root_map = root_map;
    true
}

#[inline(always)]
unsafe fn wfjit_call_source_reg(
    function: &PreparedFunction,
    op: &PreparedOp,
    index: usize,
) -> Option<u8> {
    if index >= op.c as usize {
        return None;
    }
    if op.c <= 4 {
        Some(((op.args_offset >> (index * 8)) & 0xff) as u8)
    } else {
        function
            .call_args
            .get(op.args_offset as usize + index)
            .map(|reg| reg.0)
    }
}

fn allocate_prepared_small_map_literal_from_regs<const N: usize>(
    vm: &mut Interpreter<'_>,
    function_index: u32,
    op: &PreparedOp,
    registers: *mut RegisterWord,
) -> Result<Value, RuntimeError> {
    debug_assert_eq!(op.c as usize, N);
    let function_ptr = unsafe {
        vm.prepared
            .functions
            .as_mut_ptr()
            .add(function_index as usize)
    };
    let offset = op.args_offset as usize;
    let mut pairs = [(Value::NULL, Value::NULL); N];
    let args = unsafe { &(*function_ptr).call_args };

    for (index, pair) in pairs.iter_mut().enumerate() {
        let key_reg = unsafe { args.get_unchecked(offset + index * 2).0 };
        let value_reg = unsafe { args.get_unchecked(offset + index * 2 + 1).0 };
        *pair = (
            Value::from_word(unsafe { *registers.add(key_reg as usize) }),
            Value::from_word(unsafe { *registers.add(value_reg as usize) }),
        );
    }

    let site_index = op.cache as usize;
    if site_index >= unsafe { (*function_ptr).small_map_literal_sites.len() } {
        return Err(RuntimeError::new("invalid whole-function small Map literal site"));
    }
    let site = unsafe {
        (*function_ptr)
            .small_map_literal_sites
            .as_mut_ptr()
            .add(site_index)
    };
    if unsafe { !(*site).initialized } {
        let stable = pairs.iter().all(|(key, _)| {
            vm.strings
                .iter()
                .any(|candidate| candidate.raw_bits() == key.raw_bits())
        });
        unsafe { (*site).stable_program_keys = stable; }
    }

    vm.heap.alloc_map_from_site(pairs, unsafe { &mut *site })
}

unsafe extern "C" fn napalm_wfjit_small_map_literal_regs(
    vm: *mut c_void,
    function_index: u32,
    op_index: u32,
    registers: *mut RegisterWord,
) -> u64 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running
        || registers.is_null()
        || !unsafe { wfjit_set_native_root_map(vm, function_index, op_index, registers) }
    {
        return WFJIT_BAILOUT;
    }

    // Heap allocation polls GC after protecting the copied operands. These are
    // stable handles, so a second poll before copying them is unnecessary.

    let Some(function) = vm.prepared.functions.get(function_index as usize) else {
        return WFJIT_BAILOUT;
    };
    let Some(&op) = function.ops.get(op_index as usize) else {
        return WFJIT_BAILOUT;
    };

    let result = match op.c {
        1 => allocate_prepared_small_map_literal_from_regs::<1>(
            vm, function_index, &op, registers,
        ),
        2 => allocate_prepared_small_map_literal_from_regs::<2>(
            vm, function_index, &op, registers,
        ),
        3 => allocate_prepared_small_map_literal_from_regs::<3>(
            vm, function_index, &op, registers,
        ),
        4 => allocate_prepared_small_map_literal_from_regs::<4>(
            vm, function_index, &op, registers,
        ),
        _ => return WFJIT_BAILOUT,
    };

    result.map_or(WFJIT_BAILOUT, Value::raw_bits)
}

/// Lightweight self-recursive native call.
///
/// No Napalm Frame/FrameBanks/register-stack entry is created. The callee's GPRs
/// live on this Rust/native stack frame and a `NativeShadowFrame` exposes exactly
/// the live managed registers to MMTk. Auto-memoized functions can never reach
/// this helper because whole-function compilation rejects them, and the helper
/// repeats that guard defensively.
unsafe extern "C" fn napalm_wfjit_native_self_call(
    vm: *mut c_void,
    function_index: u32,
    op_index: u32,
    caller_registers: *mut RegisterWord,
) -> u64 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running || caller_registers.is_null() {
        return WFJIT_BAILOUT;
    }
    if !unsafe {
        wfjit_set_native_root_map(vm, function_index, op_index, caller_registers)
    } {
        return WFJIT_BAILOUT;
    }

    let (op, arity, reg_count, entry_gc_map, entry_arg_dsts, entry) = {
        let Some(function) = vm.prepared.functions.get(function_index as usize) else {
            return WFJIT_BAILOUT;
        };
        if function.memoization.enabled()
            || function.uses_this
            || function.capture_count != 0
            || function.is_async
            || function.is_concurrent
            || !function.exception_handlers.is_empty()
        {
            return WFJIT_BAILOUT;
        }
        let Some(&op) = function.ops.get(op_index as usize) else {
            return WFJIT_BAILOUT;
        };
        if op.x != function_index || op.c as usize > ARGUMENT_REGISTER_COUNT {
            return WFJIT_BAILOUT;
        }
        let Some(entry) = vm
            .whole_function_jit
            .as_ref()
            .and_then(|jit| {
                (jit.active_depth < MAX_NATIVE_CALL_DEPTH)
                    .then(|| jit.entry(function_index))
                    .flatten()
            })
        else {
            return WFJIT_BAILOUT;
        };
        (
            op,
            function.arity as usize,
            function.frame.general_register_count as usize,
            function.entry_gc_map,
            function.entry_arg_dsts,
            entry,
        )
    };

    if reg_count > GENERAL_REGISTER_COUNT || entry_gc_map == NO_INDEX {
        return WFJIT_BAILOUT;
    }

    // Gather source values before constructing the callee storage. No GC-capable
    // operation occurs here; the caller map above protects every live reference.
    let mut arguments = [RegisterWord(0); ARGUMENT_REGISTER_COUNT];
    let count = op.c as usize;
    {
        let function = unsafe {
            vm.prepared
                .functions
                .get_unchecked(function_index as usize)
        };
        for index in 0..count {
            let Some(source) = (unsafe { wfjit_call_source_reg(function, &op, index) }) else {
                return WFJIT_BAILOUT;
            };
            arguments[index] = unsafe { *caller_registers.add(source as usize) };
        }
    }

    let mut storage =
        std::mem::MaybeUninit::<[RegisterWord; GENERAL_REGISTER_COUNT]>::uninit();
    let callee_registers = storage.as_mut_ptr().cast::<RegisterWord>();

    for index in 0..count {
        let destination = entry_arg_dsts[index];
        if destination != u16::MAX {
            unsafe {
                callee_registers
                    .add(destination as usize)
                    .write(arguments[index]);
            }
        }
    }
    let mut missing = count.min(arity);
    while missing < arity {
        let destination = entry_arg_dsts[missing];
        if destination != u16::MAX {
            unsafe {
                callee_registers
                    .add(destination as usize)
                    .write(Value::NULL.word());
            }
        }
        missing += 1;
    }

    let previous = vm.native_shadow_top;
    let mut shadow = NativeShadowFrame {
        previous,
        registers: callee_registers,
        function: function_index,
        root_map: entry_gc_map,
        reg_count: reg_count as u16,
        _padding: 0,
    };
    vm.native_shadow_top = &mut shadow;

    if let Some(jit) = vm.whole_function_jit.as_mut() {
        jit.active_depth = jit.active_depth.saturating_add(1);
        jit.stats.native_recursive_calls =
            jit.stats.native_recursive_calls.saturating_add(1);
        jit.stats.native_entries = jit.stats.native_entries.saturating_add(1);
    }

    let result = unsafe { entry(vm as *mut _ as *mut c_void, callee_registers) };

    if let Some(jit) = vm.whole_function_jit.as_mut() {
        jit.active_depth = jit.active_depth.saturating_sub(1);
    }
    debug_assert_eq!(
        vm.native_shadow_top,
        &mut shadow as *mut NativeShadowFrame,
        "whole-function native shadow-frame chain corrupted"
    );
    vm.native_shadow_top = previous;
    result
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
