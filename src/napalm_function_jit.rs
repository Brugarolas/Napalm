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

use crate::napalm_gc::FastAllocationInfo as GcFastAllocationInfo;
use crate::napalm_tracing_jit::{
    TRACE_HEAP_DATA_RELATIVE_MASK, TRACE_HEAP_KIND_MASK, TRACE_HEAP_METADATA_SHIFT,
};

const MIN_WHOLE_FUNCTION_HOTNESS: u32 = 64;
const MAX_WHOLE_FUNCTION_OPS: usize = 8192;
const MAX_NATIVE_CALL_DEPTH: u32 = 512;
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
    /// Bounded self-call sites expanded in the generated function body.
    pub inlined_recursive_sites: u64,
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
    direct_method: cranelift::module::FuncId,
    profiled_method: cranelift::module::FuncId,
    small_map_literal: cranelift::module::FuncId,
    native_self_call: cranelift::module::FuncId,
    small_map_literal_regs: cranelift::module::FuncId,
    profiled_small_map: cranelift::module::FuncId,
    profiled_small_map_commit: cranelift::module::FuncId,
    specialized_factory: cranelift::module::FuncId,
    bind_factory_contract: cranelift::module::FuncId,
    reserved_typed_array: cranelift::module::FuncId,
    class_factory_probe: cranelift::module::FuncId,
    class_factory_commit: cranelift::module::FuncId,
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
        builder.symbol("napalm_wfjit_direct_method", napalm_wfjit_direct_method as *const u8);
        builder.symbol("napalm_wfjit_profiled_method", napalm_wfjit_profiled_method as *const u8);
        builder.symbol("napalm_wfjit_small_map_literal", napalm_wfjit_small_map_literal as *const u8);
        builder.symbol("napalm_wfjit_native_self_call", napalm_wfjit_native_self_call as *const u8);
        builder.symbol(
            "napalm_wfjit_small_map_literal_regs",
            napalm_wfjit_small_map_literal_regs as *const u8,
        );
        builder.symbol(
            "napalm_wfjit_profiled_small_map",
            napalm_wfjit_profiled_small_map as *const u8,
        );
        builder.symbol(
            "napalm_wfjit_profiled_small_map_commit",
            napalm_wfjit_profiled_small_map_commit as *const u8,
        );
        builder.symbol(
            "napalm_wfjit_specialized_factory",
            napalm_wfjit_specialized_factory as *const u8,
        );
        builder.symbol(
            "napalm_wfjit_bind_factory_contract",
            napalm_wfjit_bind_factory_contract as *const u8,
        );
        builder.symbol(
            "napalm_wfjit_commit_reserved_typed_array",
            napalm_wfjit_commit_reserved_typed_array as *const u8,
        );
        builder.symbol("napalm_wfjit_class_factory_probe", napalm_wfjit_class_factory_probe as *const u8);
        builder.symbol("napalm_wfjit_class_factory_commit", napalm_wfjit_class_factory_commit as *const u8);
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
        let direct_method_id = module
            .declare_function("napalm_wfjit_direct_method", Linkage::Import, &helper_sig)
            .map_err(|e| format!("cannot declare whole-function direct method helper: {e}"))?;
        let mut profiled_method_sig = module.make_signature();
        profiled_method_sig.params.push(AbiParam::new(pointer)); // vm
        profiled_method_sig.params.push(AbiParam::new(types::I32)); // op index
        profiled_method_sig.params.push(AbiParam::new(types::I32)); // prepared target
        profiled_method_sig.returns.push(AbiParam::new(pointer));
        let profiled_method_id = module
            .declare_function("napalm_wfjit_profiled_method", Linkage::Import, &profiled_method_sig)
            .map_err(|e| format!("cannot declare whole-function profiled method helper: {e}"))?;
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

        let specialized_factory_id = module
            .declare_function(
                "napalm_wfjit_specialized_factory",
                Linkage::Import,
                &native_call_sig,
            )
            .map_err(|e| format!("cannot declare native specialized factory: {e}"))?;

        let mut bind_contract_sig = module.make_signature();
        bind_contract_sig.params.push(AbiParam::new(pointer));
        bind_contract_sig.params.push(AbiParam::new(types::I32));
        bind_contract_sig.params.push(AbiParam::new(types::I32));
        bind_contract_sig.params.push(AbiParam::new(types::I64));
        bind_contract_sig.returns.push(AbiParam::new(types::I64));
        let bind_factory_contract_id = module
            .declare_function(
                "napalm_wfjit_bind_factory_contract",
                Linkage::Import,
                &bind_contract_sig,
            )
            .map_err(|e| format!("cannot declare specialized contract binder: {e}"))?;

        let mut profiled_map_sig = module.make_signature();
        profiled_map_sig.params.push(AbiParam::new(pointer)); // vm
        profiled_map_sig.params.push(AbiParam::new(types::I32)); // shape
        profiled_map_sig.params.push(AbiParam::new(pointer)); // initialized InlineMapValues
        profiled_map_sig.returns.push(AbiParam::new(types::I64));
        let profiled_small_map_id = module
            .declare_function(
                "napalm_wfjit_profiled_small_map",
                Linkage::Import,
                &profiled_map_sig,
            )
            .map_err(|e| format!("cannot declare profiled small-map helper: {e}"))?;

        let mut profiled_map_commit_sig = module.make_signature();
        profiled_map_commit_sig.params.push(AbiParam::new(pointer)); // vm
        profiled_map_commit_sig.params.push(AbiParam::new(types::I64)); // reserved object address
        profiled_map_commit_sig.params.push(AbiParam::new(types::I32)); // shape
        profiled_map_commit_sig.params.push(AbiParam::new(pointer)); // initialized InlineMapValues
        profiled_map_commit_sig.returns.push(AbiParam::new(types::I64));
        let profiled_small_map_commit_id = module
            .declare_function(
                "napalm_wfjit_profiled_small_map_commit",
                Linkage::Import,
                &profiled_map_commit_sig,
            )
            .map_err(|e| format!("cannot declare profiled small-map commit helper: {e}"))?;

        let mut typed_array_commit_sig = module.make_signature();
        typed_array_commit_sig.params.push(AbiParam::new(pointer)); // interpreter
        typed_array_commit_sig.params.push(AbiParam::new(types::I32)); // function
        typed_array_commit_sig.params.push(AbiParam::new(types::I32)); // op
        typed_array_commit_sig.params.push(AbiParam::new(types::I64)); // reserved object address
        typed_array_commit_sig.returns.push(AbiParam::new(types::I64));
        let reserved_typed_array_id = module
            .declare_function(
                "napalm_wfjit_commit_reserved_typed_array",
                Linkage::Import,
                &typed_array_commit_sig,
            )
            .map_err(|e| format!("cannot declare typed array Immix commit helper: {e}"))?;

        let mut probe_sig = module.make_signature();
        probe_sig.params.extend([
            AbiParam::new(pointer), AbiParam::new(types::I32),
            AbiParam::new(types::I32), AbiParam::new(pointer),
        ]);
        probe_sig.returns.push(AbiParam::new(types::I32));
        let class_factory_probe_id = module
            .declare_function("napalm_wfjit_class_factory_probe", Linkage::Import, &probe_sig)
            .map_err(|e| format!("cannot declare native class probe: {e}"))?;

        let mut commit_sig = module.make_signature();
        commit_sig.params.extend([
            AbiParam::new(pointer), AbiParam::new(types::I32),
            AbiParam::new(types::I32), AbiParam::new(types::I64),
            AbiParam::new(pointer),
        ]);
        commit_sig.returns.push(AbiParam::new(pointer));
        let class_factory_commit_id = module
            .declare_function("napalm_wfjit_class_factory_commit", Linkage::Import, &commit_sig)
            .map_err(|e| format!("cannot declare native class commit: {e}"))?;

        let context = module.make_context();
        let function_context = FunctionBuilderContext::new();
        let result = Self {
            module,
            context,
            function_context,
            runtime_op: runtime_id,
            direct_call: direct_id,
            direct_method: direct_method_id,
            profiled_method: profiled_method_id,
            small_map_literal: small_map_id,
            native_self_call: native_self_call_id,
            small_map_literal_regs: small_map_regs_id,
            profiled_small_map: profiled_small_map_id,
            profiled_small_map_commit: profiled_small_map_commit_id,
            specialized_factory: specialized_factory_id,
            bind_factory_contract: bind_factory_contract_id,
            reserved_typed_array: reserved_typed_array_id,
            class_factory_probe: class_factory_probe_id,
            class_factory_commit: class_factory_commit_id,
            slots: (0..function_count).map(|_| FunctionSlot::default()).collect(),
            // Direct-call tiering runs before the callee's entry hook. Matching
            // its threshold avoids compiling an entry trace just before a whole
            // function supersedes it; rejected/deferred functions can still trace.
            threshold: if entry_threshold == 0 { u32::MAX } else { entry_threshold.max(MIN_WHOLE_FUNCTION_HOTNESS) },
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
        small_map_fast_alloc: Option<GcFastAllocationInfo>,
        native_direct_method_class_slots: &crate::napalm_collections::HashMap<ClassId, u32>,
        native_direct_method_targets: &crate::napalm_collections::HashMap<(ReceiverClass, MethodSymbol), CallTarget>,
        native_plain_store_classes: &crate::napalm_collections::HashMap<usize, u64>,
        native_shape_read_pics: &crate::napalm_collections::HashMap<usize, NativeMapPropertyPic>,
        native_shape_write_pics: &crate::napalm_collections::HashMap<usize, NativeMapPropertyPic>,
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
                .get(function.raw_pcs[index] as usize)
                .and_then(|word| word.opcode())
                .ok_or_else(|| format!("invalid opcode at prepared op {index}"))?;
            let unsupported_invoke = opcode == Opcode::Invoke
                && native_numeric_quick(function, index).is_none()
                && native_map_property_pic(function, index).is_none()
                && identity_sites[index].is_none()
                // Runtime-profiled InvokeMethodDirect (promoted by the VM).
                && profiled_direct_invoke_site(function, index).is_none();
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
        // In the replay-safe subset, AllocMap is the only operation that can
        // collect. Failed nested guards unwind before entering a runtime helper.
        let recursive_needs_roots = !native_recursive_safe || reachable.iter().enumerate().any(|(i, &live)| {
            live && bytecode.chunk.code[function.raw_pcs[i] as usize].opcode()
                .is_some_and(|opcode| matches!(opcode,
                    Opcode::AllocMap | Opcode::NewObjectEmpty
                    | Opcode::NewObjectShape | Opcode::NewObjectWithProperties
                    | Opcode::NewInterfaceOrTypeObject | Opcode::DefineArrayWithGenerics
                ))
        });
        if std::env::var_os("NAPALM_JIT_EVENTS").is_some() {
            eprintln!(
                "[napalm-wfjit] {:?}: native recursion={native_recursive_safe}, tlab={}, registers={}",
                function.id,
                small_map_fast_alloc.is_some(),
                function.frame.general_register_count,
            );
        }
        let runtime_module_id = self.runtime_op;
        let direct_module_id = self.direct_call;
        let direct_method_module_id = self.direct_method;
        let profiled_method_module_id = self.profiled_method;
        let small_map_module_id = self.small_map_literal;
        let native_self_module_id = self.native_self_call;
        let small_map_regs_module_id = self.small_map_literal_regs;
        let profiled_small_map_module_id = self.profiled_small_map;
        let profiled_small_map_commit_module_id = self.profiled_small_map_commit;
        let specialized_factory_module_id = self.specialized_factory;
        let bind_factory_contract_module_id = self.bind_factory_contract;
        let reserved_typed_array_module_id = self.reserved_typed_array;
        let class_factory_probe_module_id = self.class_factory_probe;
        let class_factory_commit_module_id = self.class_factory_commit;

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
            let direct_method_ref = self.module.declare_func_in_func(direct_method_module_id, b.func);
            let profiled_method_ref = self.module.declare_func_in_func(profiled_method_module_id, b.func);
            let _small_map_ref = self.module.declare_func_in_func(small_map_module_id, b.func);
            let _native_self_ref = self.module.declare_func_in_func(native_self_module_id, b.func);
            let self_ref = self.module.declare_func_in_func(func_id, b.func);
            let small_map_regs_ref =
                self.module.declare_func_in_func(small_map_regs_module_id, b.func);
            let profiled_small_map_ref =
                self.module.declare_func_in_func(profiled_small_map_module_id, b.func);
            let profiled_small_map_commit_ref =
                self.module.declare_func_in_func(profiled_small_map_commit_module_id, b.func);
            let specialized_factory_ref =
                self.module.declare_func_in_func(specialized_factory_module_id, b.func);
            let bind_factory_contract_ref =
                self.module.declare_func_in_func(bind_factory_contract_module_id, b.func);
            let reserved_typed_array_ref =
                self.module.declare_func_in_func(reserved_typed_array_module_id, b.func);
            let class_factory_probe_ref =
                self.module.declare_func_in_func(class_factory_probe_module_id, b.func);
            let class_factory_commit_ref =
                self.module.declare_func_in_func(class_factory_commit_module_id, b.func);

            // Expand one recursion level for small replay-safe bodies. Child
            // registers and leaf branches become visible to the optimizer.
            // Separate return/bailout edges preserve shadow-frame unwinding.
            let mut regions = vec![(blocks, abort, None)];
            let mut inline_calls = vec![None; function.ops.len()];
            let self_calls: Vec<_> = function.ops.iter().enumerate().filter_map(|(i, op)| {
                (reachable[i] && op.x == function_index
                    && bytecode.chunk.code[function.raw_pcs[i] as usize].opcode()
                        == Some(Opcode::CallDirect)).then_some(i)
            }).collect();
            let live_ops = reachable.iter().filter(|&&live| live).count();
            if native_recursive_safe && self_calls.len() <= 4
                && live_ops * (self_calls.len() + 1) <= 192
            {
                for index in self_calls {
                    let child_blocks: Vec<_> = function.ops.iter().map(|_| {
                        let block = b.create_block();
                        b.append_block_param(block, pointer);
                        b.append_block_param(block, pointer);
                        block
                    }).collect();
                    let resume = b.create_block();
                    b.append_block_param(resume, types::I64);
                    inline_calls[index] = Some((child_blocks[0], resume));
                    regions.push((child_blocks, b.create_block(), Some(resume)));
                    self.stats.inlined_recursive_sites += 1;
                }
            }

            for (region_index, (blocks, abort, return_to)) in regions.iter().enumerate() {
            let abort = *abort;
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
                let opcode = bytecode.chunk.code[function.raw_pcs[index] as usize].opcode().unwrap();

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
                        if let Some(target) = return_to {
                            b.ins().jump(*target, &[BlockArg::Value(value)]);
                        } else {
                            b.ins().return_(&[value]);
                        }
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
                    Opcode::SetFieldClass => {
                        regs = if let Some(&class_key) = native_plain_store_classes.get(&index) {
                            emit_native_class_field_store(
                                &mut b, runtime_ref, vm, regs, index as u32,
                                &prepared_op, class_key, abort,
                            )
                        } else {
                            call_runtime_helper(&mut b, runtime_ref, vm, regs, index as u32, abort)
                        };
                        next!();
                    }
                    Opcode::GetFieldShape => {
                        regs = if let Some(pic) = native_shape_read_pics.get(&index) {
                            emit_map_property_fast_path(
                                &mut b, runtime_ref, vm, regs, index as u32,
                                op.a, op.b, pic, expected_method_epoch, abort,
                            )
                        } else {
                            call_runtime_helper(&mut b, runtime_ref, vm, regs, index as u32, abort)
                        };
                        next!();
                    }
                    Opcode::SetFieldShape => {
                        regs = if let Some(pic) = native_shape_write_pics.get(&index) {
                            emit_native_shape_property_store(
                                &mut b, runtime_ref, vm, regs, index as u32,
                                &prepared_op, pic, abort,
                            )
                        } else {
                            call_runtime_helper(&mut b, runtime_ref, vm, regs, index as u32, abort)
                        };
                        next!();
                    }
                    Opcode::GetFieldClass => {
                        regs = emit_native_class_field_load(
                            &mut b, runtime_ref, vm, regs, index as u32,
                            &prepared_op, abort,
                        );
                        next!();
                    }
                    Opcode::AllocMap
                        if prepared_op.c > 0
                            && prepared_op.c <= 4
                            && prepared_op.next > index as u32 + 1 =>
                    {
                        let value = if native_recursive_safe {
                            if let Some(site) = native_profiled_small_map_site(function, bytecode, index) {
                                emit_profiled_small_map_alloc(
                                    &mut b, profiled_small_map_ref, profiled_small_map_commit_ref,
                                    bind_factory_contract_ref, vm, regs, function,
                                    function_index, index, &site, small_map_fast_alloc, abort,
                                )
                            } else {
                                call_native_value_helper(&mut b, small_map_regs_ref, vm,
                                    function_index, index as u32, regs, abort)
                            }
                        } else {
                            call_native_value_helper(&mut b, small_map_regs_ref, vm,
                                function_index, index as u32, regs, abort)
                        };
                        store_reg(&mut b, regs, op.a, value);
                        let target = prepared_op.next as usize;
                        if target >= blocks.len() { b.ins().jump(abort, &[]); }
                        else { b.ins().jump(blocks[target], &[BlockArg::Value(vm), BlockArg::Value(regs)]); }
                    }
                    Opcode::NewObjectEmpty | Opcode::NewObjectShape
                    | Opcode::NewObjectWithProperties | Opcode::NewInterfaceOrTypeObject => {
                        // Specialized bytecodes must not make an otherwise
                        // compilable function fall back to the interpreter.
                        let value = if native_recursive_safe {
                            if let Some(site) = native_profiled_small_map_site(function, bytecode, index) {
                                emit_profiled_small_map_alloc(
                                    &mut b, profiled_small_map_ref, profiled_small_map_commit_ref,
                                    bind_factory_contract_ref, vm, regs, function,
                                    function_index, index, &site, small_map_fast_alloc, abort,
                                )
                            } else {
                                call_native_value_helper(&mut b, specialized_factory_ref, vm,
                                    function_index, index as u32, regs, abort)
                            }
                        } else {
                            call_native_value_helper(&mut b, specialized_factory_ref, vm,
                                function_index, index as u32, regs, abort)
                        };
                        store_reg(&mut b, regs, op.a, value);
                        let target = op.next as usize;
                        if target >= blocks.len() { b.ins().jump(abort, &[]); }
                        else { b.ins().jump(blocks[target], &[BlockArg::Value(vm), BlockArg::Value(regs)]); }
                    }
                    Opcode::NewClassInstance => {
                        // The constructor can replace the frame register bank.
                        // Carry the refreshed base through the successor edge.
                        regs = emit_native_class_instance_alloc(
                            &mut b, runtime_ref, class_factory_probe_ref,
                            class_factory_commit_ref, vm, regs,
                            function, function_index, index, small_map_fast_alloc, abort,
                        );
                        let target = op.next as usize;
                        if target >= blocks.len() { b.ins().jump(abort, &[]); }
                        else { b.ins().jump(blocks[target], &[BlockArg::Value(vm), BlockArg::Value(regs)]); }
                    }
                    Opcode::DefineArrayWithGenerics => {
                        let value = emit_native_typed_array_alloc(
                            &mut b, specialized_factory_ref, reserved_typed_array_ref,
                            vm, regs, function, bytecode, function_index, index,
                            small_map_fast_alloc, abort,
                        );
                        store_reg(&mut b, regs, op.a, value);
                        let target = op.next as usize;
                        if target >= blocks.len() { b.ins().jump(abort, &[]); }
                        else { b.ins().jump(blocks[target], &[BlockArg::Value(vm), BlockArg::Value(regs)]); }
                    }
                    Opcode::InvokeMethodDirect => {
                        let fast_target = bytecode.chunk.specializations.get(prepared_op.x as usize)
                            .and_then(|descriptor| match descriptor {
                                crate::napalm_bytecode::SpecializedDescriptor::DirectMethod {
                                    owner, symbol, ..
                                } => Some((*owner, *symbol)),
                                _ => None,
                            });
                        let class_key = fast_target.and_then(|(owner, symbol)| {
                            let class_slot = native_direct_method_class_slots.get(&owner).copied()?;
                            let key = ReceiverClass(CLASS_KIND_INSTANCE | class_slot as u64);
                            (native_direct_method_targets.get(&(key, symbol)) == Some(&CallTarget::Bytecode(prepared_op.y)))
                                .then_some(key.0)
                        });
                        regs = if let Some(class_key) = class_key {
                            emit_native_direct_method_call(
                                &mut b, runtime_ref, direct_method_ref, vm, regs,
                                index as u32, op.b, class_key, expected_method_epoch, None, abort,
                            )
                        } else {
                            call_runtime_helper(&mut b, runtime_ref, vm, regs, index as u32, abort)
                        };
                        let target = op.next as usize;
                        if target >= blocks.len() { b.ins().jump(abort, &[]); }
                        else { b.ins().jump(blocks[target], &[BlockArg::Value(vm), BlockArg::Value(regs)]); }
                    }
                    Opcode::CallDirect
                        if native_recursive_safe && prepared_op.x == function_index =>
                    {
                        let value = emit_native_self_call(
                            &mut b, self_ref, vm, regs, function_index,
                            function, index, recursive_needs_roots,
                            if region_index == 0 { inline_calls[index] } else { None }, abort,
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
                    // Runtime-profiled InvokeMethodDirect: native class+epoch
                    // guard, then the direct bytecode call; a miss replays the
                    // generic Invoke through the runtime helper.
                    Opcode::Invoke if native_map_property_pic(function, index).is_none()
                        && profiled_direct_invoke_site(function, index).is_some() =>
                    {
                        let (class, target) = profiled_direct_invoke_site(function, index).unwrap();
                        regs = emit_native_direct_method_call(
                            &mut b, runtime_ref, profiled_method_ref, vm, regs,
                            index as u32, op.b, class.0, expected_method_epoch, Some(target), abort,
                        );
                        next!();
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
            if let Some(target) = return_to {
                b.ins().jump(*target, &[BlockArg::Value(bailout)]);
            } else {
                b.ins().return_(&[bailout]);
            }
            }
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
    let prepared = &vm.prepared.functions[function as usize];
    prepared.original_ops.iter().zip(&prepared.raw_pcs).map(|(op, &raw_pc)| {
        if bytecode.chunk.code[raw_pc as usize].opcode() != Some(Opcode::Invoke) || op.c != 1 {
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
    let encoded_data = b.ins().load(
        types::I64,
        MemFlagsData::trusted(),
        entry,
        offset_of!(TraceHeapEntry, data) as i32,
    );
    // A successful shaped-Map PIC can only target the fixed inline representation.
    // `data` is therefore a tagged owner-relative offset, never an external pointer.
    let offset_mask = b.ins().iconst(
        types::I64,
        !(TRACE_HEAP_DATA_RELATIVE_MASK as u64) as i64,
    );
    let relative_offset = b.ins().band(encoded_data, offset_mask);
    let data = b.ins().iadd(address, relative_offset);

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


/// Inline a shape-guarded write only for an existing, mutable and untyped
/// physical slot. All other cases replay the original typed-map setter, which
/// owns contract validation, reference write barriers and dictionary growth.
fn emit_native_shape_property_store(
    b: &mut FunctionBuilder<'_>,
    runtime_ref: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    regs: cranelift::prelude::Value,
    index: u32,
    op: &PreparedOp,
    pic: &NativeMapPropertyPic,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    let fallback = b.create_block();
    let map_tag_ok = b.create_block();
    let live_ok = b.create_block();
    let map_kind_ok = b.create_block();
    let done = b.create_block();
    b.append_block_param(done, types::I64);

    let reactive = b.ins().load(types::I8, MemFlagsData::trusted(), vm,
        offset_of!(Interpreter<'static>, reactive_enabled) as i32);
    let no_reactivity = b.ins().icmp_imm(IntCC::Equal, reactive, 0);
    let value = load_reg(b, regs, op.c);
    let val_tag = b.ins().band_imm(value, TAG_HEAP_MASK as i64);
    let reference = b.ins().icmp_imm(IntCC::Equal, val_tag, TAG_HEAP as i64);
    let immediate = b.ins().bxor_imm(reference, 1);
    let fast = b.ins().band(no_reactivity, immediate);
    b.ins().brif(fast, map_tag_ok, &[], fallback, &[]);

    b.switch_to_block(map_tag_ok);
    let map = load_reg(b, regs, op.a);
    let map_tag = b.ins().band_imm(map, TAG_HEAP_MASK as i64);
    let is_heap = b.ins().icmp_imm(IntCC::Equal, map_tag, TAG_HEAP as i64);
    let entry_block = b.create_block();
    b.ins().brif(is_heap, entry_block, &[], fallback, &[]);
    b.seal_block(map_tag_ok);

    b.switch_to_block(entry_block);
    let handle = b.ins().ireduce(types::I32, map);
    let handle = b.ins().uextend(types::I64, handle);
    let chunk_index = b.ins().ushr_imm_u(handle, TRACE_HEAP_CHUNK_BITS as i64);
    let slot_index = b.ins().band_imm(handle, TRACE_HEAP_CHUNK_MASK as i64);
    let chunks = b.ins().iconst(types::I64, handles().jit_chunks() as usize as i64);
    let chunk_offset = b.ins().ishl_imm(chunk_index, 3);
    let chunk_slot = b.ins().iadd(chunks, chunk_offset);
    let chunk = b.ins().load(types::I64, MemFlagsData::trusted(), chunk_slot, 0);
    let entry_offset = b.ins().imul_imm(slot_index, std::mem::size_of::<TraceHeapEntry>() as i64);
    let entry = b.ins().iadd(chunk, entry_offset);
    let address = b.ins().load(types::I64, MemFlagsData::trusted(), entry,
        offset_of!(TraceHeapEntry, address) as i32);
    let live = b.ins().icmp_imm(IntCC::NotEqual, address, 0);
    b.ins().brif(live, live_ok, &[], fallback, &[]);
    b.seal_block(entry_block);

    b.switch_to_block(live_ok);
    // Other persistent contracts (for example a generic Map<K,V>) may impose
    // additional checks not represented by the shape's slot contract.
    // They must always use the authoritative VM write path.
    let has_contract = b.ins().load(types::I32, MemFlagsData::trusted(), entry,
        offset_of!(TraceHeapEntry, has_contract) as i32);
    let contract_free = b.ins().icmp_imm(IntCC::Equal, has_contract, 0);
    let contract_ok = b.create_block();
    b.ins().brif(contract_free, contract_ok, &[], fallback, &[]);
    b.seal_block(live_ok);
    b.switch_to_block(contract_ok);
    let kind = b.ins().load(types::I32, MemFlagsData::trusted(), entry,
        offset_of!(TraceHeapEntry, kind) as i32);
    let object_kind = b.ins().band_imm(kind, TRACE_HEAP_KIND_MASK as i64);
    let is_map = b.ins().icmp_imm(IntCC::Equal, object_kind, TraceHeapKind::Map as i64);
    b.ins().brif(is_map, map_kind_ok, &[], fallback, &[]);
    b.seal_block(contract_ok);

    b.switch_to_block(map_kind_ok);
    let shape = b.ins().ushr_imm_u(kind, TRACE_HEAP_METADATA_SHIFT as i64);
    let encoded = b.ins().load(types::I64, MemFlagsData::trusted(), entry,
        offset_of!(TraceHeapEntry, data) as i32);
    let relative_bit = b.ins().band_imm(encoded, TRACE_HEAP_DATA_RELATIVE_MASK as i64);
    let relative = b.ins().icmp_imm(IntCC::NotEqual, relative_bit, 0);
    let relative_ok = b.create_block();
    b.ins().brif(relative, relative_ok, &[], fallback, &[]);
    b.seal_block(map_kind_ok);

    b.switch_to_block(relative_ok);
    let offset = b.ins().band_imm(encoded, !(TRACE_HEAP_DATA_RELATIVE_MASK as u64) as i64);
    let data = b.ins().iadd(address, offset);
    let mut check_block = relative_ok;
    for (i, cached) in pic.entries.iter().enumerate() {
        if i != 0 { b.switch_to_block(check_block); }
        let hit = b.create_block();
        let miss = if i + 1 == pic.entries.len() { fallback } else { b.create_block() };
        let matches = b.ins().icmp_imm(IntCC::Equal, shape, cached.shape as i64);
        b.ins().brif(matches, hit, &[], miss, &[]);
        b.seal_block(check_block);
        b.switch_to_block(hit);
        b.ins().store(MemFlagsData::trusted(), value, data,
            cached.slot as i32 * std::mem::size_of::<RegisterWord>() as i32);
        b.ins().jump(done, &[BlockArg::Value(regs)]);
        b.seal_block(hit);
        check_block = miss;
    }
    b.switch_to_block(fallback);
    let fallback_regs = call_runtime_helper(b, runtime_ref, vm, regs, index, abort);
    b.ins().jump(done, &[BlockArg::Value(fallback_regs)]);
    b.seal_block(fallback);

    b.switch_to_block(done);
    b.seal_block(done);
    b.block_params(done)[0]
}

/// Instance field-array base from a handle sidecar: inline fields publish a
/// tagged owner-relative offset (they move with the GC object), boxed fields
/// an absolute pointer (bit 0 clear). Branch-free decode against the live
/// owner `address` already loaded from the same entry.
fn emit_instance_field_base(
    b: &mut FunctionBuilder<'_>,
    entry: cranelift::prelude::Value,
    address: cranelift::prelude::Value,
) -> cranelift::prelude::Value {
    let encoded = b.ins().load(types::I64, MemFlagsData::trusted(), entry,
        offset_of!(TraceHeapEntry, data) as i32);
    let relative = b.ins().band_imm(encoded, TRACE_HEAP_DATA_RELATIVE_MASK as i64);
    let owner_mask = b.ins().ineg(relative);
    let base = b.ins().band(address, owner_mask);
    let offset = b.ins().band_imm(encoded, !(TRACE_HEAP_DATA_RELATIVE_MASK as u64) as i64);
    b.ins().iadd(base, offset)
}

/// Direct fixed-slot write for non-reference values in untyped class fields.
/// A reference store must run the MMTk post-barrier, so it is intentionally
/// handled by the exact baseline path. Guard failure never commits the store.
fn emit_native_class_field_store(
    b: &mut FunctionBuilder<'_>,
    runtime_ref: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    regs: cranelift::prelude::Value,
    index: u32,
    op: &PreparedOp,
    class_key: u64,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    let fallback = b.create_block();
    let tag_ok = b.create_block();
    let live_ok = b.create_block();
    let kind_ok = b.create_block();
    let class_ok = b.create_block();
    let store_ok = b.create_block();
    let done = b.create_block();
    b.append_block_param(done, types::I64);

    let reactive = b.ins().load(types::I8, MemFlagsData::trusted(), vm,
        offset_of!(Interpreter<'static>, reactive_enabled) as i32);
    let no_reactivity = b.ins().icmp_imm(IntCC::Equal, reactive, 0);
    let value = load_reg(b, regs, op.b);
    let value_tag = b.ins().band_imm(value, TAG_HEAP_MASK as i64);
    let is_reference = b.ins().icmp_imm(IntCC::Equal, value_tag, TAG_HEAP as i64);
    let is_immediate = b.ins().bxor_imm(is_reference, 1);
    let fast = b.ins().band(no_reactivity, is_immediate);
    b.ins().brif(fast, tag_ok, &[], fallback, &[]);

    b.switch_to_block(tag_ok);
    let receiver = load_reg(b, regs, op.a);
    let tag = b.ins().band_imm(receiver, TAG_HEAP_MASK as i64);
    let valid_tag = b.ins().icmp_imm(IntCC::Equal, tag, TAG_HEAP as i64);
    let entry_block = b.create_block();
    b.ins().brif(valid_tag, entry_block, &[], fallback, &[]);
    b.seal_block(tag_ok);

    b.switch_to_block(entry_block);
    let handle = b.ins().ireduce(types::I32, receiver);
    let handle = b.ins().uextend(types::I64, handle);
    let chunk_index = b.ins().ushr_imm_u(handle, TRACE_HEAP_CHUNK_BITS as i64);
    let entry_index = b.ins().band_imm(handle, TRACE_HEAP_CHUNK_MASK as i64);
    let chunks = b.ins().iconst(types::I64, handles().jit_chunks() as usize as i64);
    let chunk_offset = b.ins().ishl_imm(chunk_index, 3);
    let chunk_ptr_slot = b.ins().iadd(chunks, chunk_offset);
    let chunk = b.ins().load(types::I64, MemFlagsData::trusted(), chunk_ptr_slot, 0);
    let offset = b.ins().imul_imm(entry_index, std::mem::size_of::<TraceHeapEntry>() as i64);
    let entry = b.ins().iadd(chunk, offset);
    let address = b.ins().load(types::I64, MemFlagsData::trusted(), entry,
        offset_of!(TraceHeapEntry, address) as i32);
    let live = b.ins().icmp_imm(IntCC::NotEqual, address, 0);
    b.ins().brif(live, live_ok, &[], fallback, &[]);
    b.seal_block(entry_block);

    b.switch_to_block(live_ok);
    let kind = b.ins().load(types::I32, MemFlagsData::trusted(), entry,
        offset_of!(TraceHeapEntry, kind) as i32);
    let kind = b.ins().band_imm(kind, TRACE_HEAP_KIND_MASK as i64);
    let is_instance = b.ins().icmp_imm(IntCC::Equal, kind, TraceHeapKind::Instance as i64);
    b.ins().brif(is_instance, kind_ok, &[], fallback, &[]);
    b.seal_block(live_ok);

    b.switch_to_block(kind_ok);
    let class = b.ins().load(types::I64, MemFlagsData::trusted(), entry,
        offset_of!(TraceHeapEntry, receiver_class) as i32);
    let matches = b.ins().icmp_imm(IntCC::Equal, class, class_key as i64);
    b.ins().brif(matches, class_ok, &[], fallback, &[]);
    b.seal_block(kind_ok);

    b.switch_to_block(class_ok);
    let len = b.ins().load(types::I64, MemFlagsData::trusted(), entry,
        offset_of!(TraceHeapEntry, len) as i32);
    let in_bounds = b.ins().icmp_imm(IntCC::UnsignedGreaterThan, len, op.x as i64);
    b.ins().brif(in_bounds, store_ok, &[], fallback, &[]);
    b.seal_block(class_ok);

    b.switch_to_block(store_ok);
    let data = emit_instance_field_base(b, entry, address);
    b.ins().store(MemFlagsData::trusted(), value, data,
        op.x as i32 * std::mem::size_of::<RegisterWord>() as i32);
    b.ins().jump(done, &[BlockArg::Value(regs)]);
    b.seal_block(store_ok);

    b.switch_to_block(fallback);
    let next_regs = call_runtime_helper(b, runtime_ref, vm, regs, index, abort);
    b.ins().jump(done, &[BlockArg::Value(next_regs)]);
    b.seal_block(fallback);

    b.switch_to_block(done);
    b.seal_block(done);
    b.block_params(done)[0]
}

/// Guard method identity in native code, then enter the known bytecode
/// function without the dynamic Invoke lookup. The helper retains Napalm's
/// complete call ABI (memo, errors, async and native callee promotion).
fn emit_native_direct_method_call(
    b: &mut FunctionBuilder<'_>,
    runtime_ref: cranelift::codegen::ir::FuncRef,
    direct_method_ref: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    regs: cranelift::prelude::Value,
    index: u32,
    receiver: u8,
    receiver_class: u64,
    epoch: u64,
    // Some(target): runtime-profiled site; the helper takes the target as a
    // compile-time argument instead of reading it from the prepared op.
    profiled_target: Option<u32>,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    let fallback = b.create_block();
    let tag_ok = b.create_block();
    let live_ok = b.create_block();
    let class_ok = b.create_block();
    let done = b.create_block();
    b.append_block_param(done, types::I64);

    let actual_epoch = b.ins().load(types::I64, MemFlagsData::trusted(), vm,
        offset_of!(Interpreter<'static>, method_epoch) as i32);
    let epoch_match = b.ins().icmp_imm(IntCC::Equal, actual_epoch, epoch as i64);
    b.ins().brif(epoch_match, tag_ok, &[], fallback, &[]);

    b.switch_to_block(tag_ok);
    let tagged = load_reg(b, regs, receiver);
    let masked = b.ins().band_imm(tagged, TAG_HEAP_MASK as i64);
    let valid_tag = b.ins().icmp_imm(IntCC::Equal, masked, TAG_HEAP as i64);
    let entry_block = b.create_block();
    b.ins().brif(valid_tag, entry_block, &[], fallback, &[]);
    b.seal_block(tag_ok);

    b.switch_to_block(entry_block);
    let handle = b.ins().ireduce(types::I32, tagged);
    let handle = b.ins().uextend(types::I64, handle);
    let chunk_index = b.ins().ushr_imm_u(handle, TRACE_HEAP_CHUNK_BITS as i64);
    let entry_index = b.ins().band_imm(handle, TRACE_HEAP_CHUNK_MASK as i64);
    let chunks = b.ins().iconst(types::I64, handles().jit_chunks() as usize as i64);
    let chunk_offset = b.ins().ishl_imm(chunk_index, 3);
    let chunk_ptr_slot = b.ins().iadd(chunks, chunk_offset);
    let chunk = b.ins().load(types::I64, MemFlagsData::trusted(), chunk_ptr_slot, 0);
    let offset = b.ins().imul_imm(entry_index, std::mem::size_of::<TraceHeapEntry>() as i64);
    let entry = b.ins().iadd(chunk, offset);
    let address = b.ins().load(types::I64, MemFlagsData::trusted(), entry,
        offset_of!(TraceHeapEntry, address) as i32);
    let live = b.ins().icmp_imm(IntCC::NotEqual, address, 0);
    b.ins().brif(live, live_ok, &[], fallback, &[]);
    b.seal_block(entry_block);

    b.switch_to_block(live_ok);
    let class = b.ins().load(types::I64, MemFlagsData::trusted(), entry,
        offset_of!(TraceHeapEntry, receiver_class) as i32);
    let class_match = b.ins().icmp_imm(IntCC::Equal, class, receiver_class as i64);
    b.ins().brif(class_match, class_ok, &[], fallback, &[]);
    b.seal_block(live_ok);

    b.switch_to_block(class_ok);
    let current_regs = call_runtime_helper_with(b, direct_method_ref, vm, regs, index, profiled_target, abort);
    b.ins().jump(done, &[BlockArg::Value(current_regs)]);
    b.seal_block(class_ok);

    b.switch_to_block(fallback);
    let current_regs = call_runtime_helper(b, runtime_ref, vm, regs, index, abort);
    b.ins().jump(done, &[BlockArg::Value(current_regs)]);
    b.seal_block(fallback);

    b.switch_to_block(done);
    b.seal_block(done);
    b.block_params(done)[0]
}

/// Load a known-offset class field without entering the interpreter.
/// No GC safepoint occurs between reading the published sidecar pointer and
/// loading the field. Reactive tracking and uninitialized fields replay the
/// baseline handler, which owns their exact observable behavior.
fn emit_native_class_field_load(
    b: &mut FunctionBuilder<'_>,
    runtime_ref: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    regs: cranelift::prelude::Value,
    index: u32,
    op: &PreparedOp,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    let fallback = b.create_block();
    let tag_ok = b.create_block();
    let live_ok = b.create_block();
    let kind_ok = b.create_block();
    let bounds_ok = b.create_block();
    let done = b.create_block();
    b.append_block_param(done, b.func.dfg.value_type(regs));

    let tracking = b.ins().load(types::I8, MemFlagsData::trusted(), vm,
        offset_of!(Interpreter<'static>, reactive_tracking_active) as i32);
    let skip = b.ins().load(types::I8, MemFlagsData::trusted(), vm,
        offset_of!(Interpreter<'static>, reactive_skip_next_field_track) as i32);
    let skip_or_track = b.ins().bor(tracking, skip);
    let allow = b.ins().icmp_imm(IntCC::Equal, skip_or_track, 0);
    b.ins().brif(allow, tag_ok, &[], fallback, &[]);

    b.switch_to_block(tag_ok);
    let receiver = load_reg(b, regs, op.b);
    let tagged_mask = b.ins().band_imm(receiver, TAG_HEAP_MASK as i64);
    let valid_tag = b.ins().icmp_imm(IntCC::Equal, tagged_mask, TAG_HEAP as i64);
    let entry_block = b.create_block();
    b.ins().brif(valid_tag, entry_block, &[], fallback, &[]);
    b.seal_block(tag_ok);

    b.switch_to_block(entry_block);
    let handle32 = b.ins().ireduce(types::I32, receiver);
    let handle = b.ins().uextend(types::I64, handle32);
    let chunk_index = b.ins().ushr_imm_u(handle, TRACE_HEAP_CHUNK_BITS as i64);
    let entry_index = b.ins().band_imm(handle, TRACE_HEAP_CHUNK_MASK as i64);
    let chunks = b.ins().iconst(types::I64, handles().jit_chunks() as usize as i64);
    let chunk_offset = b.ins().ishl_imm(chunk_index, 3);
    let chunk_ptr_slot = b.ins().iadd(chunks, chunk_offset);
    let chunk = b.ins().load(types::I64, MemFlagsData::trusted(), chunk_ptr_slot, 0);
    let stride = std::mem::size_of::<TraceHeapEntry>() as i64;
    let entry_offset = b.ins().imul_imm(entry_index, stride);
    let entry = b.ins().iadd(chunk, entry_offset);
    let address = b.ins().load(types::I64, MemFlagsData::trusted(), entry,
        offset_of!(TraceHeapEntry, address) as i32);
    let live = b.ins().icmp_imm(IntCC::NotEqual, address, 0);
    b.ins().brif(live, live_ok, &[], fallback, &[]);
    b.seal_block(entry_block);

    b.switch_to_block(live_ok);
    let kind = b.ins().load(types::I32, MemFlagsData::trusted(), entry,
        offset_of!(TraceHeapEntry, kind) as i32);
    let kind = b.ins().band_imm(kind, TRACE_HEAP_KIND_MASK as i64);
    let is_instance = b.ins().icmp_imm(IntCC::Equal, kind, TraceHeapKind::Instance as i64);
    b.ins().brif(is_instance, kind_ok, &[], fallback, &[]);
    b.seal_block(live_ok);

    b.switch_to_block(kind_ok);
    let len = b.ins().load(types::I64, MemFlagsData::trusted(), entry,
        offset_of!(TraceHeapEntry, len) as i32);
    let valid_slot = b.ins().icmp_imm(IntCC::UnsignedGreaterThan, len, op.x as i64);
    b.ins().brif(valid_slot, bounds_ok, &[], fallback, &[]);
    b.seal_block(kind_ok);

    b.switch_to_block(bounds_ok);
    let data = emit_instance_field_base(b, entry, address);
    let value = b.ins().load(types::I64, MemFlagsData::trusted(), data,
        op.x as i32 * std::mem::size_of::<RegisterWord>() as i32);
    let result_ok = b.create_block();
    if op.y & crate::napalm_bytecode::FIELD_CLASS_NULLABLE != 0 {
        // Proven nullable field: null is an ordinary result, no runtime trap.
        b.ins().jump(result_ok, &[]);
    } else {
        let initialized = b.ins().icmp_imm(IntCC::NotEqual, value, TAG_NULL as i64);
        b.ins().brif(initialized, result_ok, &[], fallback, &[]);
    }
    b.seal_block(bounds_ok);

    b.switch_to_block(result_ok);
    store_reg(b, regs, op.a, value);
    b.ins().jump(done, &[BlockArg::Value(regs)]);
    b.seal_block(result_ok);

    b.switch_to_block(fallback);
    let next_regs = call_runtime_helper(b, runtime_ref, vm, regs, index, abort);
    b.ins().jump(done, &[BlockArg::Value(next_regs)]);
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
        match bytecode.chunk.code[function.raw_pcs[index] as usize].opcode() {
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
            .get(function.raw_pcs[index] as usize)
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
            Opcode::GetFieldClass => true,
            Opcode::Invoke => {
                native_numeric_quick(function, index).is_some()
                    || native_map_property_pic(function, index).is_some()
                    || identity_sites[index].is_some()
            }
            Opcode::NewObjectEmpty | Opcode::NewObjectWithProperties
            | Opcode::NewObjectShape | Opcode::NewInterfaceOrTypeObject => true,
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
    call_runtime_helper_with(b, helper, vm, current_regs, op, None, abort)
}

/// `call_runtime_helper` for helpers taking one extra compile-time u32
/// (e.g. the profiled direct-call target).
fn call_runtime_helper_with(
    b: &mut FunctionBuilder<'_>,
    helper: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    current_regs: cranelift::prelude::Value,
    op: u32,
    extra: Option<u32>,
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
    let call = match extra {
        Some(extra) => {
            let extra = b.ins().iconst(types::I32, extra as i64);
            b.ins().call(helper, &[vm, op, extra])
        }
        None => b.ins().call(helper, &[vm, op]),
    };
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
    needs_roots: bool,
    inline_region: Option<(cranelift::prelude::Block, cranelift::prelude::Block)>,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    let op = function.ops[index];
    let map = function.gc_map_by_op[index];
    let flags = MemFlagsData::trusted();
    let call_abort = if needs_roots { abort } else { b.create_block() };
    if needs_roots {
        emit_native_root_map_update(b, vm, regs, map, index as u32);
    }

    // Option<Box<T>> has the nullable-pointer ABI. A native entry always owns
    // a live JIT; its box and compiled code cannot be replaced while active.
    let jit = b.ins().load(types::I64, flags, vm,
        offset_of!(Interpreter<'static>, whole_function_jit) as i32);
    let depth = b.ins().load(types::I32, flags, jit, offset_of!(WholeFunctionJit, active_depth) as i32);
    let allowed = b.ins().icmp_imm(IntCC::UnsignedLessThan, depth, MAX_NATIVE_CALL_DEPTH as i64);
    let call_block = b.create_block();
    b.ins().brif(allowed, call_block, &[], call_abort, &[]);
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

    let previous = if needs_roots {
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
        Some(previous)
    } else {
        None
    };
    let child_depth = b.ins().iadd_imm(depth, 1);
    b.ins().store(flags, child_depth, jit, offset_of!(WholeFunctionJit, active_depth) as i32);
    for counter in [offset_of!(WholeFunctionJitStats, native_entries),
        offset_of!(WholeFunctionJitStats, native_recursive_calls)] {
        let offset = (offset_of!(WholeFunctionJit, stats) + counter) as i32;
        let old = b.ins().load(types::I64, flags, jit, offset);
        let next = b.ins().iadd_imm(old, 1);
        b.ins().store(flags, next, jit, offset);
    }

    let value = if let Some((entry, resume)) = inline_region {
        b.ins().jump(entry, &[BlockArg::Value(vm), BlockArg::Value(child_regs)]);
        b.switch_to_block(resume);
        b.block_params(resume)[0]
    } else {
        let call = b.ins().call(callee, &[vm, child_regs]);
        b.inst_results(call)[0]
    };
    if let Some(previous) = previous {
        b.ins().store(flags, previous, vm, offset_of!(Interpreter<'static>, native_shadow_top) as i32);
    }
    b.ins().store(flags, depth, jit, offset_of!(WholeFunctionJit, active_depth) as i32);
    let failed = b.ins().icmp_imm(IntCC::Equal, value, WFJIT_BAILOUT as i64);
    let done = b.create_block();
    b.ins().brif(failed, call_abort, &[], done, &[]);
    if !needs_roots {
        // A read-only recursion chain cannot collect. Publish the replay PC
        // only if it fails, and only when unwinding to the ordinary VM frame.
        b.switch_to_block(call_abort);
        let hot_regs = b.ins().load(types::I64, flags, vm,
            offset_of!(Interpreter<'static>, hot_registers) as i32);
        let is_top = b.ins().icmp(IntCC::Equal, regs, hot_regs);
        let top = b.create_block();
        b.ins().brif(is_top, top, &[], abort, &[]);
        b.switch_to_block(top);
        let frame = b.ins().load(types::I64, flags, vm,
            offset_of!(Interpreter<'static>, hot_frame) as i32);
        let root_map = b.ins().iconst(types::I32, map as i64);
        b.ins().store(flags, root_map, frame, offset_of!(Frame, root_map) as i32);
        let pc = b.ins().iconst(types::I32, index as i64);
        b.ins().store(flags, pc, vm, offset_of!(Interpreter<'static>, hot_pc) as i32);
        b.ins().jump(abort, &[]);
    }
    b.switch_to_block(done);
    value
}


#[derive(Clone, Copy)]
struct NativeProfiledSmallMapSite {
    shape: u32,
    value_count: u8,
    field_count: u8,
    key_regs: [u8; SMALL_MAP_INLINE_FIELDS],
    value_regs: [u8; SMALL_MAP_INLINE_FIELDS],
    key_words: [u64; SMALL_MAP_INLINE_FIELDS],
    slots: [u16; SMALL_MAP_INLINE_FIELDS],
    /// Legacy AllocMap+MapSet needs dynamic string-key guards.
    guard_keys: bool,
    /// ObjectShape and NewObjectEmpty create only null-initialized slots.
    null_fields: bool,
    /// A structural contract must be installed before the new object escapes.
    bind_contract: bool,
}

fn native_profiled_small_map_site(
    function: &PreparedFunction,
    bytecode: &BytecodeFunction,
    index: usize,
) -> Option<NativeProfiledSmallMapSite> {
    let op = *function.ops.get(index)?;
    if function.gc_map_by_op.get(index).copied()? == NO_INDEX {
        return None;
    }
    let opcode = bytecode.chunk.code.get(*function.raw_pcs.get(index)? as usize)?.opcode()?;
    let make = |shape: u32, field_count: u8, value_count: u8, slots: [u16; SMALL_MAP_INLINE_FIELDS],
                guard_keys: bool, null_fields: bool, bind_contract: bool| {
        NativeProfiledSmallMapSite {
            shape, field_count, value_count, slots, guard_keys, null_fields, bind_contract,
            key_regs: [0; SMALL_MAP_INLINE_FIELDS],
            value_regs: [0; SMALL_MAP_INLINE_FIELDS],
            key_words: [0; SMALL_MAP_INLINE_FIELDS],
        }
    };
    if opcode == Opcode::NewObjectEmpty {
        return Some(make(0, 0, 0, [0; SMALL_MAP_INLINE_FIELDS], false, true, false));
    }
    let site = function.small_map_literal_sites.get(op.cache as usize)?;
    if !site.initialized || site.disabled || !site.stable_program_keys
        || site.value_count as usize > SMALL_MAP_INLINE_FIELDS
        || site.field_count as usize > SMALL_MAP_INLINE_FIELDS
    {
        return None;
    }
    let (field_count, guard_keys, null_fields, bind_contract) = match opcode {
        Opcode::AllocMap if op.c > 0 && op.c <= 4 => (op.c, true, false, false),
        Opcode::NewObjectWithProperties if op.c > 0 && op.c <= 4 =>
            (op.c, false, false, false),
        Opcode::NewInterfaceOrTypeObject if op.c > 0 && op.c <= 4 =>
            (op.c, false, false, true),
        Opcode::NewObjectShape => (site.field_count, false, true, true),
        _ => return None,
    };
    if site.field_count != field_count {
        return None;
    }
    let needs_contract = bind_contract && matches!(
        bytecode.chunk.specializations.get(op.x as usize),
        Some(crate::napalm_bytecode::SpecializedDescriptor::ObjectShape {
            contract: Some(_), ..
        })
    );
    let mut result = make(site.shape, field_count, site.value_count,
        site.slots, guard_keys, null_fields, needs_contract);
    for i in 0..field_count as usize {
        if guard_keys {
            let offset = op.args_offset as usize;
            result.key_regs[i] = function.call_args.get(offset + i * 2)?.0;
            result.value_regs[i] = function.call_args.get(offset + i * 2 + 1)?.0;
            result.key_words[i] = Value::from_heap(site.key_handles[i]).raw_bits();
        } else if !null_fields {
            result.value_regs[i] = unsafe { wfjit_call_source_reg(function, &op, i) }?;
        }
        if result.slots[i] as usize >= site.value_count as usize {
            return None;
        }
    }
    Some(result)
}

fn emit_native_root_map_update(
    b: &mut FunctionBuilder<'_>,
    vm: cranelift::prelude::Value,
    regs: cranelift::prelude::Value,
    root_map: u32,
    op_index: u32,
) {
    let flags = MemFlagsData::trusted();
    let top = b.create_block();
    let nested = b.create_block();
    let ready = b.create_block();
    let hot_regs = b.ins().load(
        types::I64,
        flags,
        vm,
        offset_of!(Interpreter<'static>, hot_registers) as i32,
    );
    let is_top = b.ins().icmp(IntCC::Equal, regs, hot_regs);
    b.ins().brif(is_top, top, &[], nested, &[]);

    b.switch_to_block(top);
    let frame = b.ins().load(
        types::I64,
        flags,
        vm,
        offset_of!(Interpreter<'static>, hot_frame) as i32,
    );
    let root = b.ins().iconst(types::I32, root_map as i64);
    b.ins().store(flags, root, frame, offset_of!(Frame, root_map) as i32);
    let pc = b.ins().iconst(types::I32, op_index as i64);
    b.ins().store(flags, pc, vm, offset_of!(Interpreter<'static>, hot_pc) as i32);
    b.ins().jump(ready, &[]);

    b.switch_to_block(nested);
    let shadow = b.ins().load(
        types::I64,
        flags,
        vm,
        offset_of!(Interpreter<'static>, native_shadow_top) as i32,
    );
    let root = b.ins().iconst(types::I32, root_map as i64);
    b.ins().store(flags, root, shadow, offset_of!(NativeShadowFrame, root_map) as i32);
    b.ins().jump(ready, &[]);

    b.switch_to_block(ready);
    b.seal_block(top);
    b.seal_block(nested);
    b.seal_block(ready);
}

/// Native zero-capacity List<T> allocation. No managed external buffer is
/// needed, so the generated function can reserve a fixed Immix object in its
/// TLAB, then publish it through a non-safepoint GC helper. Dynamic capacities
/// that are not zero retain the ordinary specialized allocation helper.
/// Native fast allocation of a zero-field class instance. The probe validates
/// the current class layout and constructor before any raw Immix reservation;
/// the commit publishes the instance before changing frames. Unsupported layouts
/// and exhausted TLABs take the existing authoritative call-to-return path.
fn emit_native_class_instance_alloc(
    b: &mut FunctionBuilder<'_>,
    slow_helper: cranelift::codegen::ir::FuncRef,
    probe_helper: cranelift::codegen::ir::FuncRef,
    commit_helper: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    regs: cranelift::prelude::Value,
    function: &PreparedFunction,
    function_index: u32,
    index: usize,
    fast: Option<GcFastAllocationInfo>,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    let Some(fast) = fast.filter(|info| info.alignment.is_power_of_two() && info.bytes > 0) else {
        return call_runtime_helper(b, slow_helper, vm, regs, index as u32, abort);
    };
    if function.gc_map_by_op[index] == NO_INDEX {
        return call_runtime_helper(b, slow_helper, vm, regs, index as u32, abort);
    }
    emit_native_root_map_update(b, vm, regs, function.gc_map_by_op[index], index as u32);
    let flags = MemFlagsData::trusted();
    let func = b.ins().iconst(types::I32, function_index as i64);
    let op_idx = b.ins().iconst(types::I32, index as i64);
    let checked = b.ins().call(probe_helper, &[vm, func, op_idx, regs]);
    let allowed = b.inst_results(checked)[0];
    let allowed = b.ins().icmp_imm(IntCC::NotEqual, allowed, 0);
    let fast_block = b.create_block();
    let slow_block = b.create_block();
    let commit_block = b.create_block();
    let done_block = b.create_block();
    b.append_block_param(commit_block, types::I64);
    b.append_block_param(done_block, types::I64);
    b.ins().brif(allowed, fast_block, &[], slow_block, &[]);

    b.switch_to_block(fast_block);
    let cursor_ptr = b.ins().iconst(types::I64, fast.cursor as usize as i64);
    let limit_ptr = b.ins().iconst(types::I64, fast.limit as usize as i64);
    let cursor = b.ins().load(types::I64, flags, cursor_ptr, 0);
    let limit = b.ins().load(types::I64, flags, limit_ptr, 0);
    let biased = b.ins().iadd_imm(cursor, (fast.alignment - 1) as i64);
    let no_align_overflow = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, biased, cursor);
    let mask = b.ins().iconst(types::I64, -(fast.alignment as i64));
    let aligned = b.ins().band(biased, mask);
    let end = b.ins().iadd_imm(aligned, fast.bytes as i64);
    let no_end_overflow = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, end, aligned);
    let fits = b.ins().icmp(IntCC::UnsignedLessThanOrEqual, end, limit);
    let valid = b.ins().band(fits, no_align_overflow);
    let valid = b.ins().band(valid, no_end_overflow);
    b.ins().brif(valid, commit_block, &[BlockArg::Value(aligned)], slow_block, &[]);

    b.switch_to_block(commit_block);
    let address = b.block_params(commit_block)[0];
    let next = b.ins().iadd_imm(address, fast.bytes as i64);
    b.ins().store(flags, next, cursor_ptr, 0);
    let call = b.ins().call(commit_helper, &[vm, func, op_idx, address, regs]);
    let refreshed = b.inst_results(call)[0];
    b.ins().jump(done_block, &[BlockArg::Value(refreshed)]);

    b.switch_to_block(slow_block);
    let next_regs = call_runtime_helper(b, slow_helper, vm, regs, index as u32, abort);
    b.ins().jump(done_block, &[BlockArg::Value(next_regs)]);

    b.seal_block(fast_block);
    b.seal_block(slow_block);
    b.seal_block(commit_block);
    b.seal_block(done_block);
    b.switch_to_block(done_block);
    let result = b.block_params(done_block)[0];
    let failed = b.ins().icmp_imm(IntCC::Equal, result, 0);
    let ok = b.create_block();
    b.ins().brif(failed, abort, &[], ok, &[]);
    b.seal_block(ok);
    b.switch_to_block(ok);
    result
}

fn emit_native_typed_array_alloc(
    b: &mut FunctionBuilder<'_>,
    slow_helper: cranelift::codegen::ir::FuncRef,
    commit_helper: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    regs: cranelift::prelude::Value,
    function: &PreparedFunction,
    bytecode: &BytecodeFunction,
    function_index: u32,
    index: usize,
    fast: Option<GcFastAllocationInfo>,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    let Some(fast) = fast else {
        return call_native_value_helper(b, slow_helper, vm, function_index, index as u32, regs, abort);
    };
    let op = function.ops[index];
    let with_capacity = match bytecode.chunk.specializations.get(op.x as usize) {
        Some(crate::napalm_bytecode::SpecializedDescriptor::ArrayType { with_capacity, .. }) => *with_capacity,
        _ => return call_native_value_helper(b, slow_helper, vm, function_index, index as u32, regs, abort),
    };
    if !fast.alignment.is_power_of_two() || fast.bytes == 0 {
        return call_native_value_helper(b, slow_helper, vm, function_index, index as u32, regs, abort);
    }

    emit_native_root_map_update(b, vm, regs, function.gc_map_by_op[index], index as u32);
    let flags = MemFlagsData::trusted();
    let try_fast = b.create_block();
    let slow = b.create_block();
    let commit = b.create_block();
    let done = b.create_block();
    b.append_block_param(commit, types::I64);
    b.append_block_param(done, types::I64);
    if with_capacity {
        let capacity = load_reg(b, regs, op.b);
        let empty = b.ins().icmp_imm(IntCC::Equal, capacity, 0);
        b.ins().brif(empty, try_fast, &[], slow, &[]);
    } else {
        b.ins().jump(try_fast, &[]);
    }

    b.switch_to_block(try_fast);
    let cursor_ptr = b.ins().iconst(types::I64, fast.cursor as usize as i64);
    let limit_ptr = b.ins().iconst(types::I64, fast.limit as usize as i64);
    let cursor = b.ins().load(types::I64, flags, cursor_ptr, 0);
    let limit = b.ins().load(types::I64, flags, limit_ptr, 0);
    let biased = b.ins().iadd_imm(cursor, (fast.alignment - 1) as i64);
    let mask = b.ins().iconst(types::I64, -(fast.alignment as i64));
    let aligned = b.ins().band(biased, mask);
    let end = b.ins().iadd_imm(aligned, fast.bytes as i64);
    let fits = b.ins().icmp(IntCC::UnsignedLessThanOrEqual, end, limit);
    let no_overflow = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, end, aligned);
    let alignment_ok = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, biased, cursor);
    let valid = b.ins().band(fits, no_overflow);
    let valid = b.ins().band(valid, alignment_ok);
    b.ins().brif(valid, commit, &[BlockArg::Value(aligned)], slow, &[]);

    b.switch_to_block(commit);
    let address = b.block_params(commit)[0];
    let end = b.ins().iadd_imm(address, fast.bytes as i64);
    b.ins().store(flags, end, cursor_ptr, 0);
    let func = b.ins().iconst(types::I32, function_index as i64);
    let op_index = b.ins().iconst(types::I32, index as i64);
    let call = b.ins().call(commit_helper, &[vm, func, op_index, address]);
    let result = b.inst_results(call)[0];
    b.ins().jump(done, &[BlockArg::Value(result)]);

    b.switch_to_block(slow);
    let func = b.ins().iconst(types::I32, function_index as i64);
    let op_index = b.ins().iconst(types::I32, index as i64);
    let call = b.ins().call(slow_helper, &[vm, func, op_index, regs]);
    let result = b.inst_results(call)[0];
    b.ins().jump(done, &[BlockArg::Value(result)]);

    b.seal_block(try_fast);
    b.seal_block(commit);
    b.seal_block(slow);
    b.seal_block(done);
    b.switch_to_block(done);
    let value = b.block_params(done)[0];
    let failed = b.ins().icmp_imm(IntCC::Equal, value, WFJIT_BAILOUT as i64);
    let success = b.create_block();
    b.append_block_param(success, types::I64);
    b.ins().brif(failed, abort, &[], success, &[BlockArg::Value(value)]);
    b.seal_block(success);
    b.switch_to_block(success);
    b.block_params(success)[0]
}

fn emit_profiled_small_map_alloc(
    b: &mut FunctionBuilder<'_>,
    slow_helper: cranelift::codegen::ir::FuncRef,
    commit_helper: cranelift::codegen::ir::FuncRef,
    bind_contract_helper: cranelift::codegen::ir::FuncRef,
    vm: cranelift::prelude::Value,
    regs: cranelift::prelude::Value,
    function: &PreparedFunction,
    function_index: u32,
    index: usize,
    site: &NativeProfiledSmallMapSite,
    fast_alloc: Option<GcFastAllocationInfo>,
    abort: cranelift::prelude::Block,
) -> cranelift::prelude::Value {
    let field_count = site.field_count as usize;

    // Program-string handles cannot be recycled. Guard their exact tagged words
    // so a polymorphic/dynamic-key fused site safely bails to the baseline path.
    for i in 0..field_count {
        if !site.guard_keys { break; }
        let actual = load_reg(b, regs, site.key_regs[i]);
        let expected = b.ins().iconst(types::I64, site.key_words[i] as i64);
        let matches = b.ins().icmp(IntCC::Equal, actual, expected);
        let ok = b.create_block();
        b.ins().brif(matches, ok, &[], abort, &[]);
        b.switch_to_block(ok);
        b.seal_block(ok);
    }

    emit_native_root_map_update(
        b,
        vm,
        regs,
        function.gc_map_by_op[index],
        index as u32,
    );

    let storage = b.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        std::mem::size_of::<InlineMapValues>() as u32,
        3,
    ));
    let values = b.ins().stack_addr(types::I64, storage, 0);
    let flags = MemFlagsData::trusted();
    // Materialize the final slot order once in generated code. The site has
    // already validated every slot; neither helper needs to unpack/check it
    // again for every node. Initialize padding and unused fields as well.
    let zero = b.ins().iconst(types::I64, 0);
    b.ins().store(flags, zero, values, offset_of!(InlineMapValues, len) as i32);
    let len = b.ins().iconst(types::I8, site.value_count as i64);
    b.ins().store(flags, len, values, offset_of!(InlineMapValues, len) as i32);
    let null = b.ins().iconst(types::I64, TAG_NULL as i64);
    for slot in 0..SMALL_MAP_INLINE_FIELDS {
        // Duplicate literal keys retain their last value without duplicate
        // runtime stores. All initializer expressions have already executed.
        let value = if site.null_fields {
            null
        } else {
            (0..field_count).rfind(|&i| site.slots[i] as usize == slot)
                .map_or(null, |i| load_reg(b, regs, site.value_regs[i]))
        };
        b.ins().store(flags, value, values,
            (offset_of!(InlineMapValues, values) + slot * 8) as i32);
    }

    let shape = b.ins().iconst(types::I32, site.shape as i64);

    let value = if let Some(fast) = fast_alloc {
        debug_assert!(fast.alignment.is_power_of_two());
        debug_assert!(fast.bytes > 0);

        let fast_block = b.create_block();
        let slow_block = b.create_block();
        let commit_block = b.create_block();
        let done_block = b.create_block();
        b.append_block_param(commit_block, types::I64); // reserved object address
        b.append_block_param(done_block, types::I64); // tagged Napalm result

        // Load the current Immix TLAB state directly. MMTk's precise-stress mode
        // deliberately makes the limit fail this check, so it naturally reaches
        // the ordinary allocation helper. A concurrent collection request may
        // wait until this finite TLAB is exhausted; no GC can inspect the mutator
        // until it subsequently enters a safepoint/slow path.
        b.ins().jump(fast_block, &[]);
        b.switch_to_block(fast_block);
        let cursor_ptr = b.ins().iconst(types::I64, fast.cursor as usize as i64);
        let limit_ptr = b.ins().iconst(types::I64, fast.limit as usize as i64);
        let cursor = b.ins().load(types::I64, flags, cursor_ptr, 0);
        let limit = b.ins().load(types::I64, flags, limit_ptr, 0);
        let biased = b.ins().iadd_imm(cursor, (fast.alignment - 1) as i64);
        let align_mask = b.ins().iconst(types::I64, -(fast.alignment as i64));
        let aligned = b.ins().band(biased, align_mask);
        let new_cursor = b.ins().iadd_imm(aligned, fast.bytes as i64);
        let fits = b
            .ins()
            .icmp(IntCC::UnsignedLessThanOrEqual, new_cursor, limit);
        let without_end_overflow = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, new_cursor, aligned);
        let without_alignment_overflow = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, biased, cursor);
        let fits = b.ins().band(fits, without_end_overflow);
        let fits = b.ins().band(fits, without_alignment_overflow);
        b.ins().brif(
            fits,
            commit_block,
            &[BlockArg::Value(aligned)],
            slow_block,
            &[],
        );

        b.switch_to_block(commit_block);
        let reserved = b.block_params(commit_block)[0];
        let committed_cursor = b.ins().iadd_imm(reserved, fast.bytes as i64);
        b.ins().store(flags, committed_cursor, cursor_ptr, 0);
        let call = b.ins().call(
            commit_helper,
            &[vm, reserved, shape, values],
        );
        let committed = b.inst_results(call)[0];
        b.ins().jump(done_block, &[BlockArg::Value(committed)]);

        b.switch_to_block(slow_block);
        let call = b
            .ins()
            .call(slow_helper, &[vm, shape, values]);
        let allocated = b.inst_results(call)[0];
        b.ins().jump(done_block, &[BlockArg::Value(allocated)]);

        b.seal_block(fast_block);
        b.seal_block(commit_block);
        b.seal_block(slow_block);
        b.seal_block(done_block);
        b.switch_to_block(done_block);
        b.block_params(done_block)[0]
    } else {
        let call = b
            .ins()
            .call(slow_helper, &[vm, shape, values]);
        b.inst_results(call)[0]
    };

    let value = if site.bind_contract {
        // The complete object is published by the map helper before contract
        // validation. On failure the helper returns BAILOUT; baseline replays
        // the allocation and reports the original structural-type error.
        let function = b.ins().iconst(types::I32, function_index as i64);
        let op = b.ins().iconst(types::I32, index as i64);
        let call = b.ins().call(bind_contract_helper, &[vm, function, op, value]);
        b.inst_results(call)[0]
    } else {
        value
    };

    let failed = b.ins().icmp_imm(IntCC::Equal, value, WFJIT_BAILOUT as i64);
    let ok = b.create_block();
    b.append_block_param(ok, types::I64);
    b.ins().brif(failed, abort, &[], ok, &[BlockArg::Value(value)]);
    b.seal_block(ok);
    b.switch_to_block(ok);
    b.block_params(ok)[0]
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
        let native_direct_method_class_slots: crate::napalm_collections::HashMap<ClassId, u32> =
            vm.prepared.class_by_id.iter().map(|(id, slot)| (*id, *slot)).collect();
        let mut native_direct_method_targets = crate::napalm_collections::HashMap::default();
        for (&owner, &class_slot) in &native_direct_method_class_slots {
            let class = ReceiverClass(CLASS_KIND_INSTANCE | class_slot as u64);
            for (symbol, method) in vm.classes[class_slot as usize].instance_methods.iter().enumerate() {
                if let Some(target) = method {
                    native_direct_method_targets.insert((class, MethodSymbol(symbol as u32)), *target);
                }
            }
        }
        let mut native_plain_store_classes = crate::napalm_collections::HashMap::default();
        let prepared_for_target = &vm.prepared.functions[target as usize];
        let bytecode_for_target = &vm.program.functions[target as usize];
        for (index, op) in prepared_for_target.ops.iter().enumerate() {
            let bytecode_opcode = bytecode_for_target.chunk.code[prepared_for_target.raw_pcs[index] as usize]
                .opcode();
            if bytecode_opcode != Some(Opcode::SetFieldClass) { continue; }
            let Some(crate::napalm_bytecode::SpecializedDescriptor::FieldClass { class, unchecked_store, .. }) =
                bytecode_for_target.chunk.specializations.get(
                    prepared_for_target.original_ops[index].x as usize)
            else { continue; };
            let Some(&class_slot) = vm.prepared.class_by_id.get(class) else { continue; };
            let Some(info) = vm.classes.get(class_slot as usize) else { continue; };
            // Untyped field, or a store the compiler proved satisfies the field.
            if !*unchecked_store
                && (info.instance_field_types.get(op.x as usize) != Some(&Type::Top)
                    || info.generic_fields.get(op.x as usize) != Some(&false))
            { continue; }
            native_plain_store_classes.insert(index, CLASS_KIND_INSTANCE | class_slot as u64);
        }
        // A fixed structural layout and PIC shape identify the physical slot.
        // Shape descriptors are immutable after publication. Stores are only
        // inlined when the shape also proves a writable untyped field.
        let mut native_shape_read_pics = crate::napalm_collections::HashMap::default();
        let mut native_shape_write_pics = crate::napalm_collections::HashMap::default();
        for (index, op) in prepared_for_target.ops.iter().enumerate() {
            let opcode = bytecode_for_target.chunk.code
                [prepared_for_target.raw_pcs[index] as usize].opcode();
            if !matches!(opcode, Some(Opcode::GetFieldShape | Opcode::SetFieldShape)) {
                continue;
            }
            let Some(cache) = prepared_for_target.map_inline_caches.get(op.cache as usize) else {
                continue;
            };
            if cache.len == 0 || cache.megamorphic { continue; }
            let mut entries = smallvec::SmallVec::<[MapIcEntry; 4]>::new();
            for &entry in &cache.entries[..cache.len as usize] {
                if opcode == Some(Opcode::SetFieldShape) {
                    let Some(shape) = vm.heap.map_shapes.get_shape(entry.shape) else {
                        continue;
                    };
                    let Some(contract) = shape.slots.get(entry.slot as usize) else { continue; };
                    if contract.readonly || contract.write_runtime_checked
                        || contract.write_type.is_some()
                    { continue; }
                }
                entries.push(entry);
            }
            if entries.is_empty() { continue; }
            let site = NativeMapPropertyPic { entries };
            match opcode {
                Some(Opcode::GetFieldShape) => { native_shape_read_pics.insert(index, site); }
                Some(Opcode::SetFieldShape) => { native_shape_write_pics.insert(index, site); }
                _ => {}
            }
        }
        let small_map_fast_alloc = vm.heap.inline_map_fast_allocation_info();
        let identity_sites = native_identity_sites(vm, target);
        let prepared = vm.prepared.functions.as_ptr().add(target as usize);
        let bytecode = &vm.program.functions[target as usize] as *const BytecodeFunction;
        let constants = &vm.resolved_constants[target as usize] as *const Vec<RegisterWord>;
        if let Err(error) = unsafe {
            (*jit_ptr).compile(
                target,
                &*prepared,
                &*bytecode,
                &*constants,
                vm.method_epoch,
                &identity_sites,
                small_map_fast_alloc,
                &native_direct_method_class_slots,
                &native_direct_method_targets,
                &native_plain_store_classes,
                &native_shape_read_pics,
                &native_shape_write_pics,
            )
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

unsafe extern "C" fn napalm_wfjit_profiled_small_map_commit(
    vm: *mut c_void,
    reserved_address: u64,
    shape: u32,
    source_values: *const InlineMapValues,
) -> u64 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running
        || reserved_address == 0
        || source_values.is_null()
    {
        return WFJIT_BAILOUT;
    }

    // The generated caller initializes the complete repr(C), fixed-capacity
    // value array using a compile-time validated allocation-site layout.
    let values = unsafe { source_values.read() };
    debug_assert!(values.len() <= SMALL_MAP_INLINE_FIELDS);

    unsafe {
        vm.heap
            .alloc_inline_map_reserved(reserved_address as usize, shape, values)
            .raw_bits()
    }
}

unsafe extern "C" fn napalm_wfjit_profiled_small_map(
    vm: *mut c_void,
    shape: u32,
    source_values: *const InlineMapValues,
) -> u64 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running
        || source_values.is_null()
    {
        return WFJIT_BAILOUT;
    }

    let values = unsafe { source_values.read() };
    debug_assert!(values.len() <= SMALL_MAP_INLINE_FIELDS);

    vm.heap.alloc_inline_map(shape, values).raw_bits()
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

/// Install the structural contract after a fast native TLAB map allocation.
/// Contract checks may reject a runtime value; the native caller bails out and
/// executes the authoritative baseline error path without exposing the object.
unsafe extern "C" fn napalm_wfjit_bind_factory_contract(
    vm: *mut c_void,
    function_index: u32,
    op_index: u32,
    raw: u64,
) -> u64 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running || raw == WFJIT_BAILOUT {
        return WFJIT_BAILOUT;
    }
    let Some(function) = vm.prepared.functions.get(function_index as usize) else {
        return WFJIT_BAILOUT;
    };
    let Some(op) = function.ops.get(op_index as usize) else {
        return WFJIT_BAILOUT;
    };
    let descriptor = vm.program.functions.get(function_index as usize)
        .and_then(|function| function.chunk.specializations.get(op.x as usize));
    let Some(crate::napalm_bytecode::SpecializedDescriptor::ObjectShape {
        contract: Some(contract), ..
    }) = descriptor else {
        return WFJIT_BAILOUT;
    };
    if vm.heap.bind_contract(unsafe { Value::from_raw_bits_unchecked(raw) }, contract).is_err() {
        return WFJIT_BAILOUT;
    }
    raw
}

/// Publish a zero-capacity typed array in the Immix range reserved by
/// generated WFJIT code. This cannot trigger a managed GC collection.
unsafe extern "C" fn napalm_wfjit_commit_reserved_typed_array(
    vm: *mut c_void,
    function_index: u32,
    op_index: u32,
    reserved: u64,
) -> u64 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if reserved == 0 || vm.run_state != RunState::Running {
        return WFJIT_BAILOUT;
    }
    let Some(function) = vm.prepared.functions.get(function_index as usize) else {
        return WFJIT_BAILOUT;
    };
    let Some(&op) = function.ops.get(op_index as usize) else {
        return WFJIT_BAILOUT;
    };
    let Some(crate::napalm_bytecode::SpecializedDescriptor::ArrayType { element, .. }) = vm
        .program.functions.get(function_index as usize)
        .and_then(|f| f.chunk.specializations.get(op.x as usize))
    else { return WFJIT_BAILOUT; };
    let contract = Type::List(Some(Box::new(element.clone())));
    let value = unsafe {
        vm.heap.alloc_reserved_managed(reserved as usize, HeapObject::List(Vec::new()))
    };
    if vm.heap.bind_contract(value, &contract).is_err() {
        return WFJIT_BAILOUT;
    }
    value.raw_bits()
}

/// Directly execute a prepared specialized factory without redispatching its
/// bytecode handler. The native frame is registered with the precise GC map
/// before the heap allocation or any persistent contract installation.
/// The native class path is legal only for a finalized, zero-field, non-foreign
/// layout. Other instances require externally allocated field storage and must
/// use the ordinary GC-safe runtime allocation path.
unsafe extern "C" fn napalm_wfjit_class_factory_probe(
    vm: *mut c_void, function_index: u32, op_index: u32,
    registers: *mut RegisterWord,
) -> u32 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running || registers != vm.hot_registers {
        return 0;
    }
    let Some(prepared) = vm.prepared.functions.get(function_index as usize) else {
        return 0;
    };
    let Some(&op) = prepared.ops.get(op_index as usize) else { return 0; };
    let Some(&raw) = prepared.raw_pcs.get(op_index as usize) else { return 0; };
    if vm.program.functions.get(function_index as usize)
        .and_then(|function| function.chunk.code.get(raw as usize))
        .and_then(|word| word.opcode()) != Some(Opcode::NewClassInstance)
    { return 0; }
    if !unsafe { wfjit_set_native_root_map(vm, function_index, op_index, registers) } {
        return 0;
    }
    // Prepared Y = class slot, CACHE = constructor index (or NO_INDEX). Any
    // finalized layout qualifies: the field array is unmanaged Box storage and
    // its allocation in the commit never reaches a GC safepoint.
    let Some(layout) = vm.classes.get(op.y as usize) else { return 0; };
    u32::from(!layout.is_foreign && layout.layout_finalized)
}

/// Publish a TLAB-reserved class object and complete its constructor without
/// keeping the caller's possibly relocated register pointer across the call.
unsafe extern "C" fn napalm_wfjit_class_factory_commit(
    vm: *mut c_void, function_index: u32, op_index: u32,
    reserved: u64, registers: *mut RegisterWord,
) -> *mut RegisterWord {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    // The generated function called the probe immediately before reserving the
    // TLAB range. Rechecking it after the bump would introduce a failure path
    // with an uninitialized object in already-reserved GC space.
    debug_assert_ne!(reserved, 0);
    debug_assert_eq!(registers, vm.hot_registers);
    let caller_fiber = vm.current_fiber;
    let caller_depth = vm.fiber(caller_fiber).frames.len();
    let caller_function = vm.hot_function_index;
    let Some(&op) = vm.prepared.functions[function_index as usize].ops.get(op_index as usize) else {
        return std::ptr::null_mut();
    };
    // Prepared Y = class slot, CACHE = constructor index (or NO_INDEX).
    let args = unsafe { gather_specialized_values(vm, &op) };
    vm.mark_gc_point(&op);
    let count = vm.classes[op.y as usize].total_instance_field_count as usize;
    let instance = unsafe { vm.heap.alloc_reserved_managed(reserved as usize, HeapObject::Instance(InstanceObject::nulls(op.y, count))) };
    unsafe { vm.set_reg(op.a, instance.word()) };
    if op.cache != NO_INDEX {
        let callee = op.cache;
        let caller_pc = vm.raw_pc_for_op(&op);
        unsafe { vm.advance(&op) };
        vm.flush_hot_pc();
        let mut explicit = [RegisterWord(0); ARGUMENT_REGISTER_COUNT];
        for (index, value) in args[..op.c as usize].iter().enumerate() {
            explicit[index] = value.word();
        }
        if let Err(error) = vm.push_frame(callee, Some(instance.word()),
            &explicit[..op.c as usize], None, None)
        {
            vm.fail_at_raw_pc(caller_pc, error.message);
            return std::ptr::null_mut();
        }
        vm.current_frame_mut().caller_pc = caller_pc;
        while vm.run_state == RunState::Running
            && vm.current_fiber == caller_fiber
            && vm.fiber(caller_fiber).frames.len() > caller_depth
        {
            let pc = vm.hot_pc as usize;
            let handler = unsafe { *vm.hot_handlers.add(pc) };
            let prepared = unsafe { vm.hot_ops.add(pc) };
            unsafe { handler(vm, prepared) };
        }
    } else {
        unsafe { vm.advance(&op) };
    }
    if vm.run_state == RunState::Running && vm.current_fiber == caller_fiber
        && vm.fiber(caller_fiber).frames.len() == caller_depth
        && vm.hot_function_index == caller_function && vm.hot_pc == op.next
    {
        vm.hot_registers
    } else {
        std::ptr::null_mut()
    }
}

unsafe extern "C" fn napalm_wfjit_specialized_factory(
    vm: *mut c_void,
    function_index: u32,
    op_index: u32,
    registers: *mut RegisterWord,
) -> u64 {
    let vm = unsafe { &mut *(vm as *mut Interpreter<'static>) };
    if vm.run_state != RunState::Running || registers.is_null()
        || !unsafe { wfjit_set_native_root_map(vm, function_index, op_index, registers) }
    {
        return WFJIT_BAILOUT;
    }
    let Some(function) = vm.prepared.functions.get(function_index as usize) else {
        return WFJIT_BAILOUT;
    };
    let Some(&op) = function.ops.get(op_index as usize) else {
        return WFJIT_BAILOUT;
    };
    let Some(opcode) = vm.program.functions.get(function_index as usize)
        .and_then(|f| f.chunk.code.get(function.raw_pcs[op_index as usize] as usize))
        .and_then(|word| word.opcode()) else {
        return WFJIT_BAILOUT;
    };
    // Native-recursive shadow frames use a separate GPR bank. The object
    // factory reads the normal VM registers, so require the top-level frame.
    if registers != vm.hot_registers { return WFJIT_BAILOUT; }
    let result = match opcode {
        Opcode::NewObjectEmpty | Opcode::NewObjectShape
        | Opcode::NewObjectWithProperties | Opcode::NewInterfaceOrTypeObject =>
            allocate_specialized_object(vm, &op, opcode),
        Opcode::DefineArrayWithGenerics => {
            let descriptor = vm.program.functions[function_index as usize]
                .chunk.specializations.get(op.x as usize);
            let Some(crate::napalm_bytecode::SpecializedDescriptor::ArrayType {
                element, with_capacity,
            }) = descriptor else { return WFJIT_BAILOUT; };
            let capacity = if *with_capacity {
                let count = word_i32(unsafe { *registers.add(op.b as usize) });
                if count < 0 { return WFJIT_BAILOUT; }
                count as usize
            } else { 0 };
            let contract = Type::List(Some(Box::new(element.clone())));
            let result = vm.heap.alloc(HeapObject::List(Vec::with_capacity(capacity)));
            vm.heap.bind_contract(result, &contract).map(|()| result)
        },
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

/// Bytecode InvokeMethodDirect: the prepared op carries the target in Y.
unsafe extern "C" fn napalm_wfjit_direct_method(vm: *mut c_void, op_index: u32) -> *mut RegisterWord {
    let target = {
        let vm = unsafe { &*(vm as *const Interpreter<'static>) };
        if vm.run_state != RunState::Running || vm.hot_frame.is_null() { return std::ptr::null_mut(); }
        unsafe { vm.current_prepared().ops.get_unchecked(op_index as usize).y }
    };
    unsafe { wfjit_direct_method_call(vm, op_index, target) }
}

/// Runtime-profiled InvokeMethodDirect (a promoted Invoke): the target was the
/// profile's at compile time; native code already guarded class and epoch.
unsafe extern "C" fn napalm_wfjit_profiled_method(vm: *mut c_void, op_index: u32, target: u32) -> *mut RegisterWord {
    unsafe { wfjit_direct_method_call(vm, op_index, target) }
}

unsafe fn wfjit_direct_method_call(vm: *mut c_void, op_index: u32, target: u32) -> *mut RegisterWord {
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
    let receiver = unsafe { vm.reg(op.b) };
    unsafe { vm.set_pc(op_index); }

    // The native caller already verified the method epoch and receiver class.
    // Always use the ordinary call entry. This is the memoization firewall: if a
    // future native caller reaches an auto-memoized target, probe/miss bookkeeping
    // and return caching still happen exactly once in the existing runtime path.
    vm.enter_bytecode_call(&op, target, Some(receiver), &words[..count], None);
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
    let raw_pc = unsafe { *vm.current_prepared().raw_pcs.get_unchecked(op_index as usize) };
    let opcode = vm.program.functions[caller_function as usize].chunk.code[raw_pc as usize]
        .opcode().unwrap();

    // Constructors may grow the VM's register stack while the native caller
    // is suspended. Run the constructor to completion and return the *current*
    // GPR base so the caller never retains a dangling register-bank pointer.
    // The tracing tier has a separate inline Immix reservation for zero-field
    // instances; WFJIT currently uses the safe class-constructor slow path.
    if opcode == Opcode::NewClassInstance {
        let expected_next = vm.current_prepared().ops[op_index as usize].next;
        let result = vm.trace_execute_call_to_return();
        return match result {
            Ok(step) if step.running && !step.frame_changed
                && vm.current_fiber == caller_fiber
                && vm.fiber(caller_fiber).frames.len() == caller_depth
                && vm.hot_function_index == caller_function
                && vm.hot_pc == expected_next => vm.hot_registers,
            _ => std::ptr::null_mut(),
        };
    }

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
            | Opcode::GetFieldShape | Opcode::SetFieldShape
            | Opcode::GetFieldClass | Opcode::SetFieldClass | Opcode::InvokeMethodDirect
            | Opcode::AllocInstance | Opcode::AllocForeign) {
            op.x = prepared_op.x;
            op.y = prepared_op.y;
        }
        if matches!(opcode, Opcode::MapGet | Opcode::MapSet | Opcode::LoadProperty
            | Opcode::StoreProperty | Opcode::StorePropertyIfPresent
            | Opcode::GetFieldShape | Opcode::SetFieldShape | Opcode::InvokeMethodDirect) {
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
