//! Napalm HIR -> register-bytecode compiler.
//!
//! This module lowers the SSA Napalm IR from `napalm_ir.rs` into the executable
//! register bytecode defined by `napalm_bytecode.rs`.
//!
//! The lowering is intentionally optimized for interpreter throughput:
//! - dead pure SSA nodes are removed before allocation;
//! - block parameters are lowered as edge parallel copies, never as phi ops;
//! - a global hole-aware priority allocator reuses all 256 GPRs aggressively;
//! - weighted conservative coalescing removes move/guard/block-parameter copies;
//! - interference-aware DSATUR ordering keeps hot/high-pressure values stable;
//! - no GPR is reserved for compiler scratch state; parallel-copy cycles use
//!   `SpecialReg::Scratch0`;
//! - branches use fallthrough whenever the chosen code layout permits it;
//! - critical-edge copies are emitted in shared out-of-line trampolines;
//! - constants, guard descriptors and IC descriptors are deduplicated;
//! - precise GC maps are emitted from physical liveness at every GC-capable PC;
//! - frame-entry parameter/capture moves are encoded in `FrameLayout`, so no
//!   bytecode prologue is needed.
//!
//! Register spilling is deliberately not implemented here because the current
//! bytecode architecture has no spill-slot opcodes. If more than 256 physical
//! values interfere, compilation fails with `CompileError::RegisterPressure`.
//! In normal Napalm code this should be exceptionally rare; adding spill slots
//! later can be done without changing the HIR.

#![allow(clippy::large_enum_variant)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::fmt;

use bumpalo::{collections::Vec as BumpVec, Bump};
use smallvec::{smallvec, SmallVec};

use crate::napalm_bytecode::{
    encode_bailout_reason, function_kind_uses_this, opcode_for_f64_bin,
    opcode_for_f64_cmp, opcode_for_f64_unary, opcode_for_int_bin,
    opcode_for_int_cmp, opcode_for_int_unary, ArgReg, Assembler, BytecodeDeoptPoint,
    BytecodeFunction, BytecodeInlinedFrameState, BytecodePc, BytecodeProgram,
    BytecodeSnapshotSlot, BytecodeVerifier, CaptureBinding, ClassGuardId, ConstIndex,
    DeoptLocation, ExceptionHandlerEntry, FrameLayout, IcSlot, IncomingSource, InlineCacheDescriptor, Label,
    LiveRegisterEntry, LiveRegisterMap, Opcode, ParameterBinding, ProfileCounterKind, ReferenceMap, Reg, RegisterKind,
    ResumeEntry, RuntimeMetadata, SafepointKind, SourceMapEntry, SpecialReg,
    TypeGuardId, Word, SpecializedDescriptor, SpecializedMethod, SpecializedMethodTarget,
    GENERAL_REGISTER_COUNT, MAX_NAPALM_ARITY,
};
use crate::napalm_collections::{HashMap, HashSet};
use crate::napalm_bytecode_optimizer::optimization_passes;
use crate::napalm_ir::{
    verify_function, BlockId, ClassId, ClassSet, Constant, DeoptId, EffectBehavior, FuncId,
    Function, FunctionKind, Inst, InstId, MethodSymbol, Program, ResumeId, SiteId, SnapshotSlot,
    StringId, Type, ValueDef, ValueId,
};

// =============================================================================
// Public API
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResumeTarget {
    FunctionEntry,
    BlockStart(BlockId),
    BeforeInst(InstId),
    AfterInst(InstId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResumeBinding {
    pub function: FuncId,
    pub target: ResumeTarget,
}

#[derive(Clone, Debug)]
pub struct CompileOptions {
    /// Verify every HIR function before lowering.
    pub verify_ir: bool,
    /// Verify the complete bytecode program after all cross-function fixups.
    pub verify_bytecode: bool,
    /// Run the semantics-preserving post-lowering bytecode optimizer.
    pub optimize_bytecode: bool,
    /// Emit the specialized (super-instruction) opcodes 135..=153.
    pub emit_specialized_bytecode: bool,
    /// Verify the bytecode after every optimization pass in all build modes.
    pub verify_each_bytecode_pass: bool,
    /// Remove pure SSA nodes whose results cannot affect observable execution.
    pub eliminate_dead_code: bool,
    /// Bias physical allocation toward eliminating HIR moves and edge copies.
    pub coalesce_registers: bool,
    /// Order CFG successors using existing block profile counts.
    pub profile_guided_layout: bool,
    /// Omit unconditional jumps when the target block is physically next.
    pub omit_fallthrough_jumps: bool,
    /// Emit source ranges into the bytecode side table.
    pub emit_source_map: bool,
    /// Emit block counter bytecodes. Disabled by default for throughput.
    pub instrument_blocks: bool,
    /// Explicit mapping required for speculative HIR deoptimization snapshots.
    ///
    /// HIR resume IDs intentionally do not encode a bytecode location. Mapping
    /// them here avoids inventing an incorrect resume PC during lowering.
    pub resume_bindings: HashMap<ResumeId, ResumeBinding>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            verify_ir: true,
            verify_bytecode: true,
            optimize_bytecode: true,
            emit_specialized_bytecode: true,
            verify_each_bytecode_pass: cfg!(debug_assertions),
            eliminate_dead_code: true,
            coalesce_registers: true,
            profile_guided_layout: true,
            omit_fallthrough_jumps: true,
            emit_source_map: true,
            instrument_blocks: false,
            resume_bindings: HashMap::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FunctionCompileStats {
    pub function: FuncId,
    pub hir_nodes: usize,
    pub emitted_hir_nodes: usize,
    pub eliminated_hir_nodes: usize,
    pub allocated_values: usize,
    pub general_registers: u16,
    pub bytecode_words: usize,
    pub constants: usize,
    pub inline_caches: usize,
    pub safepoints: usize,
    pub edge_moves: usize,
    pub edge_trampolines: usize,
    /// Specialized opcodes emitted by the compiler (not regular optimizer pass).
    pub specialized_opcodes: usize,
    /// Method bindings subsumed by an atomic class definition.
    pub fused_class_method_bindings: usize,
}

#[derive(Clone, Debug, Default)]
pub struct CompileStats {
    pub functions: Vec<FunctionCompileStats>,
    pub total_bytecode_words: usize,
    pub max_general_registers: u16,
    /// Total words before the post-lowering optimizer runs.
    pub bytecode_words_before_optimization: usize,
    pub bytecode_words_removed: usize,
    pub bytecode_passes_changed: usize,
}

pub struct CompilationOutput {
    pub program: BytecodeProgram,
    pub stats: CompileStats,
}

#[derive(Debug)]
pub enum CompileError {
    IrVerification {
        function: FuncId,
        errors: Vec<String>,
    },
    UnknownFunction(FuncId),
    UnknownBlock {
        function: FuncId,
        block: BlockId,
    },
    UnknownValue {
        function: FuncId,
        value: ValueId,
    },
    InvalidFunctionSignature {
        function: FuncId,
        message: String,
    },
    InvalidEdge {
        function: FuncId,
        from: BlockId,
        to: BlockId,
        message: String,
    },
    RepresentationMismatch {
        function: FuncId,
        from: ValueId,
        to: ValueId,
        from_kind: RegisterKind,
        to_kind: RegisterKind,
    },
    RegisterPressure {
        function: FuncId,
        value: ValueId,
        simultaneously_required: usize,
    },
    MissingRegister {
        function: FuncId,
        value: ValueId,
    },
    MissingResumeBinding {
        function: FuncId,
        resume: ResumeId,
    },
    InvalidResumeBinding {
        function: FuncId,
        resume: ResumeId,
        message: String,
    },
    DeoptValueUnavailable {
        function: FuncId,
        value: ValueId,
    },
    Assemble {
        function: FuncId,
        message: String,
    },
    Metadata(String),
    BytecodeVerification(String),
    Internal {
        function: Option<FuncId>,
        message: String,
    },
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IrVerification { function, errors } => {
                write!(f, "HIR verification failed for {function:?}")?;
                for error in errors {
                    write!(f, "\n  - {error}")?;
                }
                Ok(())
            }
            Self::UnknownFunction(function) => write!(f, "unknown function {function:?}"),
            Self::UnknownBlock { function, block } => {
                write!(f, "function {function:?} references unknown block {block:?}")
            }
            Self::UnknownValue { function, value } => {
                write!(f, "function {function:?} references unknown value {value:?}")
            }
            Self::InvalidFunctionSignature { function, message } => {
                write!(f, "invalid signature for {function:?}: {message}")
            }
            Self::InvalidEdge {
                function,
                from,
                to,
                message,
            } => write!(
                f,
                "invalid CFG edge {function:?}:{from:?} -> {to:?}: {message}"
            ),
            Self::RepresentationMismatch {
                function,
                from,
                to,
                from_kind,
                to_kind,
            } => write!(
                f,
                "edge representation mismatch in {function:?}: {from:?} ({from_kind:?}) -> {to:?} ({to_kind:?})"
            ),
            Self::RegisterPressure {
                function,
                value,
                simultaneously_required,
            } => write!(
                f,
                "register pressure in {function:?} while allocating {value:?}: allocator observed {simultaneously_required}-way register pressure but the VM exposes 256 GPRs"
            ),
            Self::MissingRegister { function, value } => {
                write!(f, "no physical register assigned to {function:?}:{value:?}")
            }
            Self::MissingResumeBinding { function, resume } => write!(
                f,
                "function {function:?} uses deopt resume point {resume:?} without a CompileOptions::resume_bindings entry"
            ),
            Self::InvalidResumeBinding {
                function,
                resume,
                message,
            } => write!(
                f,
                "invalid resume binding for {function:?}/{resume:?}: {message}"
            ),
            Self::DeoptValueUnavailable { function, value } => write!(
                f,
                "deopt snapshot in {function:?} references unmaterialized value {value:?}"
            ),
            Self::Assemble { function, message } => {
                write!(f, "bytecode assembly failed for {function:?}: {message}")
            }
            Self::Metadata(message) => write!(f, "metadata lowering failed: {message}"),
            Self::BytecodeVerification(message) => {
                write!(f, "bytecode verification failed: {message}")
            }
            Self::Internal { function, message } => match function {
                Some(function) => write!(f, "internal compiler error in {function:?}: {message}"),
                None => write!(f, "internal compiler error: {message}"),
            },
        }
    }
}

impl std::error::Error for CompileError {}

pub struct BytecodeCompiler {
    options: CompileOptions,
}

impl Default for BytecodeCompiler {
    fn default() -> Self {
        Self::new(CompileOptions::default())
    }
}

impl BytecodeCompiler {
    pub fn new(options: CompileOptions) -> Self {
        Self { options }
    }

    pub fn options(&self) -> &CompileOptions {
        &self.options
    }

    pub fn compile(&self, hir: &Program) -> Result<BytecodeProgram, CompileError> {
        Ok(self.compile_with_stats(hir)?.program)
    }

    pub fn compile_with_stats(&self, hir: &Program) -> Result<CompilationOutput, CompileError> {
        compile_program_impl(hir, &self.options)
    }
}

pub fn compile_program(hir: &Program) -> Result<BytecodeProgram, CompileError> {
    BytecodeCompiler::default().compile(hir)
}

pub fn compile_program_with_options(
    hir: &Program,
    options: CompileOptions,
) -> Result<BytecodeProgram, CompileError> {
    BytecodeCompiler::new(options).compile(hir)
}

// =============================================================================
// Compact bit set used by dataflow analysis
// =============================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
struct BitSet<'arena> {
    words: BumpVec<'arena, u64>,
    bits: usize,
}

impl<'arena> BitSet<'arena> {
    fn new(bits: usize, arena: &'arena Bump) -> Self {
        Self {
            words: bumpalo::vec![in arena; 0; bits.div_ceil(64)],
            bits,
        }
    }

    #[inline]
    fn contains(&self, bit: usize) -> bool {
        debug_assert!(bit < self.bits);
        (self.words[bit / 64] & (1u64 << (bit % 64))) != 0
    }

    #[inline]
    fn insert(&mut self, bit: usize) -> bool {
        debug_assert!(bit < self.bits);
        let word = &mut self.words[bit / 64];
        let mask = 1u64 << (bit % 64);
        let changed = (*word & mask) == 0;
        *word |= mask;
        changed
    }


    fn union_with(&mut self, other: &Self) -> bool {
        debug_assert_eq!(self.words.len(), other.words.len());
        let mut changed = false;
        for (dst, src) in self.words.iter_mut().zip(&other.words) {
            let merged = *dst | *src;
            changed |= merged != *dst;
            *dst = merged;
        }
        changed
    }

    fn subtract(&mut self, other: &Self) {
        debug_assert_eq!(self.words.len(), other.words.len());
        for (dst, src) in self.words.iter_mut().zip(&other.words) {
            *dst &= !*src;
        }
    }


    fn iter_ones(&self) -> BitSetOnes<'_, 'arena> {
        BitSetOnes {
            set: self,
            word_index: 0,
            current: self.words.first().copied().unwrap_or(0),
        }
    }
}

struct BitSetOnes<'a, 'arena> {
    set: &'a BitSet<'arena>,
    word_index: usize,
    current: u64,
}

impl Iterator for BitSetOnes<'_, '_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current != 0 {
                let bit = self.current.trailing_zeros() as usize;
                self.current &= self.current - 1;
                let value = self.word_index * 64 + bit;
                if value < self.set.bits {
                    return Some(value);
                }
                return None;
            }

            self.word_index += 1;
            self.current = *self.set.words.get(self.word_index)?;
        }
    }
}

// =============================================================================
// CFG, DCE and positions
// =============================================================================

#[derive(Clone, Debug)]
struct IncomingArg {
    predecessor: BlockId,
    value: ValueId,
}

#[derive(Clone, Debug)]
struct CfgInfo {
    successors: Vec<Vec<BlockId>>,
    incoming_args: Vec<Vec<Vec<IncomingArg>>>,
}

fn build_cfg(function: &Function) -> Result<CfgInfo, CompileError> {
    let block_count = function.blocks.len();
    let mut successors = vec![Vec::new(); block_count];
    let mut incoming_args = function
        .blocks
        .iter()
        .map(|block| vec![Vec::new(); block.params.len()])
        .collect::<Vec<_>>();

    for block in &function.blocks {
        let Some(&terminator_id) = block.insts.last() else {
            continue;
        };
        let node = function
            .node(terminator_id)
            .ok_or_else(|| CompileError::Internal {
                function: Some(function.id),
                message: format!("missing terminator node {terminator_id:?}"),
            })?;

        for edge in node.kind.edges() {
            let target = function.block(edge.target).ok_or(CompileError::UnknownBlock {
                function: function.id,
                block: edge.target,
            })?;
            if edge.args.len() != target.params.len() {
                return Err(CompileError::InvalidEdge {
                    function: function.id,
                    from: block.id,
                    to: edge.target,
                    message: format!(
                        "edge supplies {} values but target expects {}",
                        edge.args.len(),
                        target.params.len()
                    ),
                });
            }

            successors[block.id.0 as usize].push(edge.target);
            for (index, &value) in edge.args.iter().enumerate() {
                incoming_args[edge.target.0 as usize][index].push(IncomingArg {
                    predecessor: block.id,
                    value,
                });
            }
        }

        successors[block.id.0 as usize].sort_unstable();
        successors[block.id.0 as usize].dedup();
    }

    // Exception regions introduce control-flow edges without ordinary SSA arguments.
    // These edges keep handler state live and make block layout aware of cold handlers.
    for region in &function.exception_regions {
        for &protected in &region.protected_blocks {
            let list = successors
                .get_mut(protected.0 as usize)
                .ok_or(CompileError::UnknownBlock {
                    function: function.id,
                    block: protected,
                })?;
            list.push(region.handler);
        }
    }
    for list in &mut successors {
        list.sort_unstable();
        list.dedup();
    }

    Ok(CfgInfo {
        successors,
        incoming_args,
    })
}

#[derive(Clone, Debug)]
struct DeadCodeInfo<'arena> {
    live_nodes: Vec<bool>,
    materialized_values: BitSet<'arena>,
}

fn mark_non_edge_operands(kind: &Inst) -> SmallVec<[ValueId; 4]> {
    match kind {
        Inst::Jump { .. } => SmallVec::new(),
        Inst::Branch { cond, .. } => smallvec![*cond],
        Inst::Switch { tag, .. } => smallvec![*tag],
        _ => kind.operands(),
    }
}

fn analyze_dead_code<'arena>(
    arena: &'arena Bump,
    function: &Function,
    cfg: &CfgInfo,
    eliminate: bool,
) -> Result<DeadCodeInfo<'arena>, CompileError> {
    let value_count = function.values.len();
    let node_count = function.nodes.len();

    if !eliminate {
        let mut needed_values = BitSet::new(value_count, arena);
        for value in 0..value_count {
            needed_values.insert(value);
        }
        return Ok(DeadCodeInfo {
            live_nodes: vec![true; node_count],
            materialized_values: needed_values,
        });
    }

    let mut live_nodes = vec![false; node_count];
    let mut needed_values = BitSet::new(value_count, arena);
    let mut node_queue = VecDeque::new();
    let mut value_queue = VecDeque::new();

    let mark_node = |id: InstId,
                     live_nodes: &mut [bool],
                     queue: &mut VecDeque<InstId>| {
        let index = id.0 as usize;
        if !live_nodes[index] {
            live_nodes[index] = true;
            queue.push_back(id);
        }
    };

    let mark_value = |id: ValueId,
                      needed_values: &mut BitSet,
                      queue: &mut VecDeque<ValueId>| {
        let index = id.0 as usize;
        if needed_values.insert(index) {
            queue.push_back(id);
        }
    };

    for node in &function.nodes {
        if node.kind.is_terminator() || node.kind.effect_behavior() != EffectBehavior::None {
            mark_node(node.id, &mut live_nodes, &mut node_queue);
        }
    }

    for point in &function.deopts {
        for slot in &point.snapshot.slots {
            mark_value(slot.value, &mut needed_values, &mut value_queue);
        }
        for frame in &point.snapshot.inlined_frames {
            for slot in &frame.slots {
                mark_value(slot.value, &mut needed_values, &mut value_queue);
            }
        }
    }

    // Handler entry values are defined by the VM rather than a normal CFG edge.
    // Keep them allocated even when the source catch binding is unused.
    for region in &function.exception_regions {
        mark_value(region.error, &mut needed_values, &mut value_queue);
    }

    while !node_queue.is_empty() || !value_queue.is_empty() {
        while let Some(id) = node_queue.pop_front() {
            let node = function.node(id).ok_or_else(|| CompileError::Internal {
                function: Some(function.id),
                message: format!("DCE worklist references missing node {id:?}"),
            })?;
            for value in mark_non_edge_operands(&node.kind) {
                mark_value(value, &mut needed_values, &mut value_queue);
            }
        }

        while let Some(id) = value_queue.pop_front() {
            let value = function.value(id).ok_or(CompileError::UnknownValue {
                function: function.id,
                value: id,
            })?;
            match value.def {
                ValueDef::InstResult(inst, _) => {
                    mark_node(inst, &mut live_nodes, &mut node_queue);
                }
                ValueDef::BlockParam(block, index) => {
                    let incoming = cfg
                        .incoming_args
                        .get(block.0 as usize)
                        .and_then(|params| params.get(index as usize))
                        .ok_or_else(|| CompileError::Internal {
                            function: Some(function.id),
                            message: format!(
                                "missing incoming-argument table for {block:?} parameter {index}"
                            ),
                        })?;
                    for arg in incoming {
                        mark_value(arg.value, &mut needed_values, &mut value_queue);
                    }
                }
                ValueDef::CaptureParam(_) => {}
            }
        }
    }

    let mut materialized_values = needed_values.clone();
    for node in &function.nodes {
        if live_nodes[node.id.0 as usize] {
            for result in &node.results {
                materialized_values.insert(result.0 as usize);
            }
        }
    }

    Ok(DeadCodeInfo {
        live_nodes,
        materialized_values,
    })
}

fn block_hotness(function: &Function, block: BlockId) -> u64 {
    function
        .profiles
        .block_counters
        .get(&block)
        .copied()
        .unwrap_or(0)
}

fn component_rpo(
    seed: BlockId,
    _cfg: &CfgInfo,
    function: &Function,
    profile_guided: bool,
    visited: &mut [bool],
) -> Vec<BlockId> {
    let mut postorder = Vec::new();
    let mut stack = vec![(seed, false)];

    while let Some((block, expanded)) = stack.pop() {
        let index = block.0 as usize;
        if expanded {
            postorder.push(block);
            continue;
        }
        if visited[index] {
            continue;
        }
        visited[index] = true;
        stack.push((block, true));

        // Layout follows only ordinary CFG edges. Exceptional successors are
        // required by liveness/register allocation, but treating cold handlers
        // as layout successors would pull catch/defer cleanup into the hot code
        // stream and hurt I-cache locality despite zero-cost exception tables.
        let mut successors = function
            .block(block)
            .and_then(|block| block.insts.last())
            .and_then(|id| function.node(*id))
            .map(|node| node.kind.edges().into_iter().map(|edge| edge.target).collect::<Vec<_>>())
            .unwrap_or_default();
        successors.sort_unstable();
        successors.dedup();
        if profile_guided {
            successors.sort_by_key(|successor| Reverse(block_hotness(function, *successor)));
        } else {
            successors.sort_by_key(|successor| Reverse(successor.0));
        }

        // The stack is LIFO. Pushing hot successors first means cold successors
        // are visited first, which places hot successors earlier after reversing
        // postorder.
        for successor in successors {
            if !visited[successor.0 as usize] {
                stack.push((successor, false));
            }
        }
    }

    postorder.reverse();
    postorder
}

fn layout_blocks(function: &Function, cfg: &CfgInfo, profile_guided: bool) -> Vec<BlockId> {
    let mut visited = vec![false; function.blocks.len()];
    let mut layout = component_rpo(
        function.entry,
        cfg,
        function,
        profile_guided,
        &mut visited,
    );

    for block in &function.blocks {
        if !visited[block.id.0 as usize] {
            layout.extend(component_rpo(
                block.id,
                cfg,
                function,
                profile_guided,
                &mut visited,
            ));
        }
    }

    layout
}

#[derive(Clone, Debug)]
struct PositionTable {
    block_start: Vec<u32>,
    block_end: Vec<u32>,
    node_use: Vec<u32>,
    node_def: Vec<u32>,
}

fn assign_positions(function: &Function, layout: &[BlockId]) -> PositionTable {
    let mut block_start = vec![0; function.blocks.len()];
    let mut block_end = vec![0; function.blocks.len()];
    let mut node_use = vec![0; function.nodes.len()];
    let mut node_def = vec![0; function.nodes.len()];
    let mut cursor = 0u32;

    for &block_id in layout {
        let block = &function.blocks[block_id.0 as usize];
        block_start[block_id.0 as usize] = cursor;
        cursor = cursor.saturating_add(2);

        for &inst in &block.insts {
            node_use[inst.0 as usize] = cursor;
            node_def[inst.0 as usize] = cursor.saturating_add(1);
            cursor = cursor.saturating_add(2);
        }

        block_end[block_id.0 as usize] = cursor;
        cursor = cursor.saturating_add(2);
    }

    PositionTable {
        block_start,
        block_end,
        node_use,
        node_def,
    }
}

// =============================================================================
// SSA liveness and physical register allocation
// =============================================================================

fn filtered_edge_operands(
    function: &Function,
    edge: &crate::napalm_ir::Edge,
    materialized: &BitSet,
) -> Result<SmallVec<[ValueId; 4]>, CompileError> {
    let target = function.block(edge.target).ok_or(CompileError::UnknownBlock {
        function: function.id,
        block: edge.target,
    })?;
    let mut out = SmallVec::new();
    for (index, &arg) in edge.args.iter().enumerate() {
        let param = target.params[index];
        if materialized.contains(param.0 as usize) {
            out.push(arg);
        }
    }
    Ok(out)
}


fn node_deopt_id(kind: &Inst) -> Option<DeoptId> {
    match kind {
        Inst::GuardType { deopt, .. }
        | Inst::GuardClass { deopt, .. }
        | Inst::GuardBounds { deopt, .. }
        | Inst::GuardNotNull { deopt, .. }
        | Inst::GuardInt32 { deopt, .. }
        | Inst::GuardF64 { deopt, .. }
        | Inst::Bailout { deopt, .. } => Some(*deopt),
        _ => None,
    }
}

fn append_deopt_uses(function: &Function, node: &crate::napalm_ir::Node, out: &mut SmallVec<[ValueId; 4]>) {
    let Some(deopt) = node_deopt_id(&node.kind) else {
        return;
    };
    let Some(point) = function.deopts.iter().find(|point| point.id == deopt) else {
        return;
    };
    out.extend(point.snapshot.slots.iter().map(|slot| slot.value));
    for frame in &point.snapshot.inlined_frames {
        out.extend(frame.slots.iter().map(|slot| slot.value));
    }
}

fn lowering_operands(
    function: &Function,
    node: &crate::napalm_ir::Node,
    materialized: &BitSet,
) -> Result<SmallVec<[ValueId; 4]>, CompileError> {
    let mut out = SmallVec::new();
    match &node.kind {
        Inst::Jump { edge } => {
            out.extend(filtered_edge_operands(function, edge, materialized)?);
        }
        Inst::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            out.push(*cond);
            out.extend(filtered_edge_operands(function, then_edge, materialized)?);
            out.extend(filtered_edge_operands(function, else_edge, materialized)?);
        }
        Inst::Switch {
            tag,
            default_edge,
            cases,
        } => {
            out.push(*tag);
            out.extend(filtered_edge_operands(function, default_edge, materialized)?);
            for case in cases {
                out.extend(filtered_edge_operands(function, &case.edge, materialized)?);
            }
        }
        _ => out.extend(node.kind.operands()),
    }
    append_deopt_uses(function, node, &mut out);
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

#[derive(Clone, Debug)]
struct LivenessInfo<'arena> {
    live_out: Vec<BitSet<'arena>>,
}

fn compute_liveness<'arena>(
    arena: &'arena Bump,
    function: &Function,
    cfg: &CfgInfo,
    dce: &DeadCodeInfo,
    layout: &[BlockId],
) -> Result<LivenessInfo<'arena>, CompileError> {
    let value_count = function.values.len();
    let block_count = function.blocks.len();
    let mut uses = (0..block_count)
        .map(|_| BitSet::new(value_count, arena))
        .collect::<Vec<_>>();
    let mut defs = (0..block_count)
        .map(|_| BitSet::new(value_count, arena))
        .collect::<Vec<_>>();

    for block in &function.blocks {
        let block_index = block.id.0 as usize;
        for &param in &block.params {
            if dce.materialized_values.contains(param.0 as usize) {
                defs[block_index].insert(param.0 as usize);
            }
        }

        if block.id == function.entry {
            for capture in &function.captures {
                if dce.materialized_values.contains(capture.value.0 as usize) {
                    defs[block_index].insert(capture.value.0 as usize);
                }
            }
        }

        for &inst_id in &block.insts {
            if !dce.live_nodes[inst_id.0 as usize] {
                continue;
            }
            let node = function.node(inst_id).ok_or_else(|| CompileError::Internal {
                function: Some(function.id),
                message: format!("liveness references missing node {inst_id:?}"),
            })?;
            for operand in lowering_operands(function, node, &dce.materialized_values)? {
                let index = operand.0 as usize;
                if !defs[block_index].contains(index) {
                    uses[block_index].insert(index);
                }
            }
            for &result in &node.results {
                if dce.materialized_values.contains(result.0 as usize) {
                    defs[block_index].insert(result.0 as usize);
                }
            }
        }
    }

    let mut live_in = (0..block_count)
        .map(|_| BitSet::new(value_count, arena))
        .collect::<Vec<_>>();
    let mut live_out = live_in.clone();
    let mut new_out = BitSet::new(value_count, arena);
    let mut new_in = BitSet::new(value_count, arena);

    loop {
        let mut changed = false;
        for &block in layout.iter().rev() {
            let index = block.0 as usize;
            new_out.words.fill(0);
            for successor in &cfg.successors[index] {
                new_out.union_with(&live_in[successor.0 as usize]);
            }

            new_in.words.copy_from_slice(&new_out.words);
            new_in.subtract(&defs[index]);
            new_in.union_with(&uses[index]);

            if new_out != live_out[index] {
                std::mem::swap(&mut live_out[index], &mut new_out);
                changed = true;
            }
            if new_in != live_in[index] {
                std::mem::swap(&mut live_in[index], &mut new_in);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    Ok(LivenessInfo { live_out })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiveRange {
    start: u32,
    end: u32,
}

impl LiveRange {
    #[inline]
    fn contains(self, position: u32) -> bool {
        self.start <= position && position <= self.end
    }
}

#[derive(Clone, Debug)]
struct LiveInterval {
    value: ValueId,
    ranges: Vec<LiveRange>,
    start: u32,
    use_weight: u64,
    use_count: u64,
}

impl LiveInterval {
    fn new(value: ValueId) -> Self {
        Self {
            value,
            ranges: Vec::new(),
            start: u32::MAX,
            use_weight: 0,
            use_count: 0,
        }
    }

    fn add_range(&mut self, range: LiveRange) {
        self.start = self.start.min(range.start);
        self.ranges.push(range);
    }

    fn normalize(&mut self) {
        if self.ranges.is_empty() {
            self.start = u32::MAX;
            return;
        }
        self.ranges.sort_by_key(|range| range.start);
        let mut merged: Vec<LiveRange> = Vec::with_capacity(self.ranges.len());
        for range in self.ranges.drain(..) {
            if let Some(last) = merged.last_mut() {
                if range.start <= last.end.saturating_add(1) {
                    last.end = last.end.max(range.end);
                    continue;
                }
            }
            merged.push(range);
        }
        self.start = merged.first().unwrap().start;
        self.ranges = merged;
    }

    fn contains(&self, position: u32) -> bool {
        self.ranges.iter().any(|range| range.contains(position))
    }
}

fn build_intervals(
    function: &Function,
    dce: &DeadCodeInfo,
    liveness: &LivenessInfo,
    positions: &PositionTable,
    layout: &[BlockId],
) -> Result<Vec<LiveInterval>, CompileError> {
    let mut intervals = function
        .values
        .iter()
        .map(|value| LiveInterval::new(value.id))
        .collect::<Vec<_>>();

    for &block_id in layout {
        let block = function.block(block_id).ok_or(CompileError::UnknownBlock {
            function: function.id,
            block: block_id,
        })?;
        let block_index = block_id.0 as usize;
        let block_start = positions.block_start[block_index];
        let block_end = positions.block_end[block_index];
        let mut local: HashMap<usize, LiveRange> = HashMap::default();

        for value in liveness.live_out[block_index].iter_ones() {
            if dce.materialized_values.contains(value) {
                local.insert(
                    value,
                    LiveRange {
                        start: block_start,
                        end: block_end,
                    },
                );
            }
        }

        for &inst_id in block.insts.iter().rev() {
            if !dce.live_nodes[inst_id.0 as usize] {
                continue;
            }
            let node = function.node(inst_id).ok_or_else(|| CompileError::Internal {
                function: Some(function.id),
                message: format!("interval construction references missing node {inst_id:?}"),
            })?;
            let use_position = positions.node_use[inst_id.0 as usize];
            let def_position = positions.node_def[inst_id.0 as usize];

            for &result in &node.results {
                let index = result.0 as usize;
                if !dce.materialized_values.contains(index) {
                    continue;
                }
                local
                    .entry(index)
                    .and_modify(|range| range.start = def_position)
                    .or_insert(LiveRange {
                        start: def_position,
                        end: def_position,
                    });
            }

            for operand in lowering_operands(function, node, &dce.materialized_values)? {
                let index = operand.0 as usize;
                if !dce.materialized_values.contains(index) {
                    continue;
                }
                local
                    .entry(index)
                    .and_modify(|range| range.end = range.end.max(use_position))
                    .or_insert(LiveRange {
                        start: block_start,
                        end: use_position,
                    });
                intervals[index].use_weight = intervals[index]
                    .use_weight
                    .saturating_add(block_hotness(function, block_id).max(1));
                intervals[index].use_count = intervals[index].use_count.saturating_add(1);
            }
        }

        for &param in &block.params {
            let index = param.0 as usize;
            if dce.materialized_values.contains(index) {
                local
                    .entry(index)
                    .and_modify(|range| range.start = block_start)
                    .or_insert(LiveRange {
                        start: block_start,
                        end: block_start,
                    });
            }
        }

        if block_id == function.entry {
            for capture in &function.captures {
                let index = capture.value.0 as usize;
                if dce.materialized_values.contains(index) {
                    local
                        .entry(index)
                        .and_modify(|range| range.start = block_start)
                        .or_insert(LiveRange {
                            start: block_start,
                            end: block_start,
                        });
                }
            }
        }

        for (value, range) in local {
            intervals[value].add_range(range);
        }
    }

    for interval in &mut intervals {
        interval.normalize();
    }
    Ok(intervals)
}

#[derive(Clone, Debug, Default)]
struct AffinityGraph {
    edges: Vec<Vec<(ValueId, u64)>>,
}

impl AffinityGraph {
    fn new(value_count: usize) -> Self {
        Self {
            edges: vec![Vec::new(); value_count],
        }
    }

    fn add(&mut self, left: ValueId, right: ValueId, weight: u64) {
        if left == right {
            return;
        }
        self.edges[left.0 as usize].push((right, weight));
        self.edges[right.0 as usize].push((left, weight));
    }
}

fn result_affinity_source(node: &crate::napalm_ir::Node) -> Option<ValueId> {
    match &node.kind {
        Inst::Move(value)
        | Inst::BindInterface { value, .. }
        | Inst::GuardType { value, .. }
        | Inst::GuardClass { value, .. }
        | Inst::GuardNotNull { value, .. }
        | Inst::GuardInt32 { value, .. }
        | Inst::GuardF64 { value, .. }
        | Inst::NumToInt32 { value }
        | Inst::NumToF64 { value }
        | Inst::Int32ToNum { value }
        | Inst::F64ToNum { value }
        | Inst::ReadBarrier { object: value } => Some(*value),
        _ => None,
    }
}

fn build_affinities(
    function: &Function,
    cfg: &CfgInfo,
    dce: &DeadCodeInfo,
    enabled: bool,
) -> AffinityGraph {
    let mut graph = AffinityGraph::new(function.values.len());
    if !enabled {
        return graph;
    }

    const COPY_WEIGHT: u64 = 1_000_000_000;
    const EDGE_WEIGHT_BASE: u64 = 1_000;

    for node in &function.nodes {
        if !dce.live_nodes[node.id.0 as usize] || node.results.len() != 1 {
            continue;
        }
        if let Some(source) = result_affinity_source(node) {
            let result = node.results[0];
            if dce.materialized_values.contains(result.0 as usize)
                && dce.materialized_values.contains(source.0 as usize)
            {
                graph.add(result, source, COPY_WEIGHT);
            }
        }
    }

    for block in &function.blocks {
        for (param_index, &param) in block.params.iter().enumerate() {
            if !dce.materialized_values.contains(param.0 as usize) {
                continue;
            }
            for incoming in &cfg.incoming_args[block.id.0 as usize][param_index] {
                if dce.materialized_values.contains(incoming.value.0 as usize) {
                    let weight = EDGE_WEIGHT_BASE.saturating_add(
                        block_hotness(function, incoming.predecessor).min(u64::MAX - EDGE_WEIGHT_BASE),
                    );
                    graph.add(param, incoming.value, weight);
                }
            }
        }
    }

    graph
}

#[derive(Clone, Debug)]
struct RegisterAllocation {
    value_regs: Vec<Option<Reg>>,
    intervals: Vec<LiveInterval>,
    frame_register_count: u16,
}

fn register_kind(ty: &Type) -> RegisterKind {
    match ty {
        Type::Int32 => RegisterKind::Int32,
        Type::F64 => RegisterKind::F64,
        Type::RawPtr => RegisterKind::RawPtr,
        _ => RegisterKind::Value,
    }
}

/// Whether a WIR type is a useful, Napalm-visible fact for the JIT tiers.
/// Internal representations (Int32/F64/RawPtr/Cell) describe register encoding,
/// not semantic values, and Top/Bottom carry no information.
fn is_trace_fact_type(ty: &Type) -> bool {
    match ty {
        Type::Top | Type::Bottom | Type::Int32 | Type::F64 | Type::RawPtr | Type::Cell(_) => false,
        Type::Union(types) => !types.is_empty() && types.iter().all(is_trace_fact_type),
        Type::List(Some(element)) | Type::Set(Some(element)) => is_trace_fact_type(element),
        Type::Builtin(_, Some(payload)) => is_trace_fact_type(payload),
        _ => true,
    }
}

fn type_may_reference(ty: &Type) -> bool {
    match ty {
        Type::Top => true,
        Type::String
        | Type::List(_)
        | Type::Map
        | Type::GenericMap(_, _)
        | Type::Set(_)
        | Type::Interface(_)
        | Type::Record(_)
        | Type::Range
        | Type::Class(_)
        | Type::Instance(_)
        | Type::Closure(_)
        | Type::Fiber
        | Type::Foreign(_)
        | Type::Module
        | Type::Builtin(_, _)
        | Type::GenericInstance { .. }
        | Type::Cell(_) => true,
        Type::Union(types) => types.iter().any(type_may_reference),
        Type::Bottom
        | Type::Null
        | Type::Bool
        | Type::Num
        | Type::Int32
        | Type::F64
        | Type::RawPtr => false,
    }
}

#[derive(Clone, Debug)]
struct InterferenceGraph {
    edges: Vec<Vec<usize>>,
}

impl InterferenceGraph {
    fn new(value_count: usize) -> Self {
        Self {
            edges: vec![Vec::new(); value_count],
        }
    }

    #[inline]
    fn interferes(&self, left: usize, right: usize) -> bool {
        self.edges[left].binary_search(&right).is_ok()
    }
}

/// Build exact interference from the disjoint live ranges rather than from each
/// interval's [start, end] envelope. This preserves lifetime holes, which is
/// especially important for SSA values that are live in only a subset of CFG
/// successors.
fn build_interference_graph(intervals: &[LiveInterval]) -> InterferenceGraph {
    // Event kind 0 is a start and 1 is an end. Starts sort before ends at the
    // same position because LiveRange endpoints are inclusive.
    let mut events = Vec::new();
    for (interval_index, interval) in intervals.iter().enumerate() {
        for range in &interval.ranges {
            events.push((range.start, 0u8, interval_index));
            events.push((range.end, 1u8, interval_index));
        }
    }
    events.sort_unstable();

    let mut graph = InterferenceGraph::new(intervals.len());
    let mut active = Vec::<usize>::new();

    for (_, kind, interval_index) in events {
        if kind == 0 {
            for &other in &active {
                if other == interval_index {
                    continue;
                }
                graph.edges[interval_index].push(other);
                graph.edges[other].push(interval_index);
            }
            active.push(interval_index);
        } else if let Some(position) = active.iter().position(|&value| value == interval_index) {
            active.swap_remove(position);
        }
    }

    for edges in &mut graph.edges {
        edges.sort_unstable();
        edges.dedup();
    }
    graph
}

#[derive(Clone, Copy, Debug)]
struct WeightedAffinity {
    left: usize,
    right: usize,
    weight: u64,
}

fn canonical_affinities(affinities: &AffinityGraph) -> Vec<WeightedAffinity> {
    let mut combined: HashMap<(usize, usize), u64> = HashMap::default();
    for (left, edges) in affinities.edges.iter().enumerate() {
        for &(right, weight) in edges {
            let right = right.0 as usize;
            if left >= right {
                continue;
            }
            let slot = combined.entry((left, right)).or_default();
            *slot = (*slot).saturating_add(weight);
        }
    }

    let mut out = combined
        .into_iter()
        .map(|((left, right), weight)| WeightedAffinity {
            left,
            right,
            weight,
        })
        .collect::<Vec<_>>();
    out.sort_unstable_by_key(|edge| (Reverse(edge.weight), edge.left, edge.right));
    out
}

#[derive(Clone, Debug)]
struct CoalescedGroups {
    groups: Vec<Vec<usize>>,
    value_group: Vec<usize>,
}

fn singleton_groups(intervals: &[LiveInterval]) -> CoalescedGroups {
    let mut groups = Vec::new();
    let mut value_group = vec![usize::MAX; intervals.len()];
    for (value, interval) in intervals.iter().enumerate() {
        if interval.ranges.is_empty() {
            continue;
        }
        value_group[value] = groups.len();
        groups.push(vec![value]);
    }
    CoalescedGroups {
        groups,
        value_group,
    }
}

/// Conservatively merge affinity-related values when every member of the two
/// groups is non-interfering. This is stronger than a one-sided register hint:
/// successful merges guarantee the copy can disappear if the grouped coloring
/// succeeds. Allocation automatically retries without these merges if they make
/// the 256-register problem harder to color.
fn coalesce_affinity_groups(
    intervals: &[LiveInterval],
    interference: &InterferenceGraph,
    weighted_affinities: &[WeightedAffinity],
) -> CoalescedGroups {
    let mut groups = intervals
        .iter()
        .enumerate()
        .map(|(value, interval)| {
            if interval.ranges.is_empty() {
                None
            } else {
                Some(vec![value])
            }
        })
        .collect::<Vec<_>>();
    let mut value_group = intervals
        .iter()
        .enumerate()
        .map(|(value, interval)| {
            if interval.ranges.is_empty() {
                usize::MAX
            } else {
                value
            }
        })
        .collect::<Vec<_>>();

    for affinity in weighted_affinities {
        let left_group = value_group[affinity.left];
        let right_group = value_group[affinity.right];
        if left_group == usize::MAX || right_group == usize::MAX || left_group == right_group {
            continue;
        }

        let compatible = {
            let left_members = groups[left_group].as_ref().unwrap();
            let right_members = groups[right_group].as_ref().unwrap();
            left_members.iter().all(|&left| {
                right_members
                    .iter()
                    .all(|&right| !interference.interferes(left, right))
            })
        };
        if !compatible {
            continue;
        }

        // Keep the larger group as the destination so updating value_group stays
        // close to union-by-size complexity without needing a separate union-find.
        let (destination, source) = if groups[left_group].as_ref().unwrap().len()
            >= groups[right_group].as_ref().unwrap().len()
        {
            (left_group, right_group)
        } else {
            (right_group, left_group)
        };

        let source_members = groups[source].take().unwrap();
        for &value in &source_members {
            value_group[value] = destination;
        }
        groups[destination]
            .as_mut()
            .unwrap()
            .extend(source_members);
    }

    // Compact sparse group IDs so all subsequent allocator structures are dense.
    let mut compact_groups = Vec::new();
    let mut old_to_new = vec![usize::MAX; groups.len()];
    for (old, group) in groups.into_iter().enumerate() {
        let Some(group) = group else {
            continue;
        };
        old_to_new[old] = compact_groups.len();
        compact_groups.push(group);
    }
    for group in &mut value_group {
        if *group != usize::MAX {
            *group = old_to_new[*group];
        }
    }

    CoalescedGroups {
        groups: compact_groups,
        value_group,
    }
}

#[derive(Clone, Debug)]
struct AllocationGroup {
    members: Vec<usize>,
    neighbors: Vec<usize>,
    weighted_uses: u64,
    use_count: u64,
    live_points: u64,
    start: u32,
}

fn build_allocation_groups(
    intervals: &[LiveInterval],
    interference: &InterferenceGraph,
    coalesced: CoalescedGroups,
) -> (Vec<AllocationGroup>, Vec<usize>) {
    let CoalescedGroups {
        groups,
        value_group,
    } = coalesced;
    let mut out = groups
        .into_iter()
        .map(|members| {
            let mut weighted_uses = 0u64;
            let mut use_count = 0u64;
            let mut live_points = 0u64;
            let mut start = u32::MAX;
            for &value in &members {
                let interval = &intervals[value];
                weighted_uses = weighted_uses.saturating_add(interval.use_weight);
                use_count = use_count.saturating_add(interval.use_count);
                live_points = live_points.saturating_add(interval.ranges.iter().fold(
                    0u64,
                    |total, range| {
                        total.saturating_add(
                            u64::from(range.end.saturating_sub(range.start)).saturating_add(1),
                        )
                    },
                ));
                start = start.min(interval.start);
            }
            AllocationGroup {
                members,
                neighbors: Vec::new(),
                weighted_uses,
                use_count,
                live_points: live_points.max(1),
                start,
            }
        })
        .collect::<Vec<_>>();

    for (left, edges) in interference.edges.iter().enumerate() {
        let left_group = value_group[left];
        if left_group == usize::MAX {
            continue;
        }
        for &right in edges {
            if left >= right {
                continue;
            }
            let right_group = value_group[right];
            if right_group == usize::MAX || left_group == right_group {
                continue;
            }
            out[left_group].neighbors.push(right_group);
            out[right_group].neighbors.push(left_group);
        }
    }
    for group in &mut out {
        group.neighbors.sort_unstable();
        group.neighbors.dedup();
    }

    (out, value_group)
}

const REGISTER_MASK_WORDS: usize = (GENERAL_REGISTER_COUNT + 63) / 64;

#[derive(Clone, Debug)]
struct RegisterMask {
    words: [u64; REGISTER_MASK_WORDS],
}

impl Default for RegisterMask {
    fn default() -> Self {
        Self {
            words: [0; REGISTER_MASK_WORDS],
        }
    }
}

impl RegisterMask {
    #[inline]
    fn contains(&self, reg: usize) -> bool {
        (self.words[reg / 64] & (1u64 << (reg % 64))) != 0
    }

    #[inline]
    fn insert(&mut self, reg: usize) -> bool {
        let word = &mut self.words[reg / 64];
        let mask = 1u64 << (reg % 64);
        let changed = (*word & mask) == 0;
        *word |= mask;
        changed
    }

    #[inline]
    fn count(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AllocationQueueEntry {
    saturation: usize,
    degree: usize,
    priority: u64,
    weighted_uses: u64,
    use_count: u64,
    live_points: u64,
    start: u32,
    group: usize,
    version: u32,
}

impl Ord for AllocationQueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        (
            self.saturation,
            self.degree,
            self.priority,
            self.weighted_uses,
            self.use_count,
            Reverse(self.live_points),
            Reverse(self.start),
            Reverse(self.group),
            self.version,
        )
            .cmp(&(
                other.saturation,
                other.degree,
                other.priority,
                other.weighted_uses,
                other.use_count,
                Reverse(other.live_points),
                Reverse(other.start),
                Reverse(other.group),
                other.version,
            ))
    }
}

impl PartialOrd for AllocationQueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug)]
enum AllocationFailure {
    Pressure {
        value: usize,
        forbidden_registers: usize,
    },
    Conflict {
        left: usize,
        right: usize,
        reg: Reg,
    },
}

fn group_priority(group: &AllocationGroup) -> u64 {
    // Prefer frequently used values with short live ranges. u128 keeps profile
    // counts from overflowing when hot functions have accumulated large counters.
    let numerator = (group.weighted_uses as u128)
        .saturating_mul(1 << 20)
        .saturating_add((group.use_count as u128) << 12);
    (numerator / u128::from(group.live_points)).min(u128::from(u64::MAX)) as u64
}

fn queue_entry(
    group: usize,
    groups: &[AllocationGroup],
    saturation: &[usize],
    versions: &[u32],
) -> AllocationQueueEntry {
    let data = &groups[group];
    AllocationQueueEntry {
        saturation: saturation[group],
        degree: data.neighbors.len(),
        priority: group_priority(data),
        weighted_uses: data.weighted_uses,
        use_count: data.use_count,
        live_points: data.live_points,
        start: data.start,
        group,
        version: versions[group],
    }
}

/// Color the interference graph with a DSATUR-style priority queue. Saturation
/// handles local pressure, degree handles difficult regions, and the weighted
/// use-density tie breakers keep hot/short-lived SSA values from being forced
/// into poor colors merely because they appear later in layout order.
fn color_allocation_groups(
    intervals: &[LiveInterval],
    interference: &InterferenceGraph,
    affinities: &AffinityGraph,
    coalesced: CoalescedGroups,
) -> Result<(Vec<Option<Reg>>, u16), AllocationFailure> {
    debug_assert!(GENERAL_REGISTER_COUNT <= 256);
    let (groups, value_group) = build_allocation_groups(intervals, interference, coalesced);
    let mut group_regs = vec![None::<usize>; groups.len()];
    let mut neighbor_colors = vec![RegisterMask::default(); groups.len()];
    let mut saturation = vec![0usize; groups.len()];
    let mut versions = vec![0u32; groups.len()];
    let mut queue = BinaryHeap::new();

    for group in 0..groups.len() {
        queue.push(queue_entry(group, &groups, &saturation, &versions));
    }

    let mut affinity_scores = vec![0u64; GENERAL_REGISTER_COUNT];
    let mut max_reg = None::<usize>;
    let mut colored = 0usize;

    while colored < groups.len() {
        let current = loop {
            let Some(entry) = queue.pop() else {
                unreachable!("register-allocation queue exhausted before coloring all groups");
            };
            if group_regs[entry.group].is_none() && versions[entry.group] == entry.version {
                break entry.group;
            }
        };

        let forbidden = &neighbor_colors[current];
        affinity_scores.fill(0);
        for &value in &groups[current].members {
            for &(other, weight) in &affinities.edges[value] {
                let other_group = value_group[other.0 as usize];
                if other_group == usize::MAX || other_group == current {
                    continue;
                }
                if let Some(reg) = group_regs[other_group] {
                    affinity_scores[reg] = affinity_scores[reg].saturating_add(weight);
                }
            }
        }

        let mut selected = None::<usize>;
        let mut selected_score = 0u64;
        for reg in 0..GENERAL_REGISTER_COUNT {
            if forbidden.contains(reg) {
                continue;
            }
            let score = affinity_scores[reg];
            if selected.is_none() || score > selected_score {
                selected = Some(reg);
                selected_score = score;
            }
        }

        let Some(reg) = selected else {
            return Err(AllocationFailure::Pressure {
                value: groups[current].members[0],
                forbidden_registers: forbidden.count(),
            });
        };

        group_regs[current] = Some(reg);
        max_reg = Some(max_reg.map_or(reg, |old| old.max(reg)));
        colored += 1;

        for &neighbor in &groups[current].neighbors {
            if group_regs[neighbor].is_some() {
                continue;
            }
            if neighbor_colors[neighbor].insert(reg) {
                saturation[neighbor] += 1;
                versions[neighbor] = versions[neighbor].wrapping_add(1);
                queue.push(queue_entry(neighbor, &groups, &saturation, &versions));
            }
        }
    }

    let mut value_regs = vec![None; intervals.len()];
    for (group, data) in groups.iter().enumerate() {
        let reg = Reg(group_regs[group].unwrap() as u8);
        for &value in &data.members {
            value_regs[value] = Some(reg);
        }
    }

    // Keep this invariant checked in release builds too. A bad register
    // assignment silently corrupts bytecode state and is much worse than a
    // compiler error.
    for (left, edges) in interference.edges.iter().enumerate() {
        let Some(left_reg) = value_regs[left] else {
            continue;
        };
        for &right in edges {
            if left >= right {
                continue;
            }
            if value_regs[right] == Some(left_reg) {
                return Err(AllocationFailure::Conflict {
                    left,
                    right,
                    reg: left_reg,
                });
            }
        }
    }

    Ok((
        value_regs,
        max_reg.map_or(0, |reg| (reg + 1) as u16),
    ))
}

fn allocate_registers(
    function: &Function,
    intervals: Vec<LiveInterval>,
    affinities: &AffinityGraph,
) -> Result<RegisterAllocation, CompileError> {
    let interference = build_interference_graph(&intervals);
    let weighted_affinities = canonical_affinities(affinities);
    let coalesced = coalesce_affinity_groups(&intervals, &interference, &weighted_affinities);
    let has_real_coalescing = coalesced.groups.iter().any(|group| group.len() > 1);

    let allocation = color_allocation_groups(
        &intervals,
        &interference,
        affinities,
        coalesced,
    )
    .or_else(|failure| match failure {
        // Coalescing can theoretically make a colorable graph harder even when
        // every merged pair is non-interfering. Never turn a copy optimization
        // into a compilation failure: retry the same priority allocator with
        // singleton groups before reporting register pressure. Internal conflict
        // failures are never hidden by the retry.
        AllocationFailure::Pressure { .. } if has_real_coalescing => color_allocation_groups(
            &intervals,
            &interference,
            affinities,
            singleton_groups(&intervals),
        ),
        other => Err(other),
    });

    let (value_regs, frame_register_count) = allocation.map_err(|failure| match failure {
        AllocationFailure::Pressure {
            value,
            forbidden_registers,
        } => CompileError::RegisterPressure {
            function: function.id,
            value: intervals[value].value,
            simultaneously_required: forbidden_registers.saturating_add(1),
        },
        AllocationFailure::Conflict { left, right, reg } => CompileError::Internal {
            function: Some(function.id),
            message: format!(
                "register allocator assigned interfering values {:?} and {:?} to {:?}",
                intervals[left].value, intervals[right].value, reg
            ),
        },
    })?;

    Ok(RegisterAllocation {
        value_regs,
        intervals,
        frame_register_count,
    })
}

fn value_reg(
    function: &Function,
    allocation: &RegisterAllocation,
    value: ValueId,
) -> Result<Reg, CompileError> {
    allocation
        .value_regs
        .get(value.0 as usize)
        .and_then(|value| *value)
        .ok_or(CompileError::MissingRegister {
            function: function.id,
            value,
        })
}

fn build_frame_layout(
    function: &Function,
    allocation: &RegisterAllocation,
    dce: &DeadCodeInfo,
) -> Result<(FrameLayout, u8), CompileError> {
    let entry = function.block(function.entry).ok_or(CompileError::UnknownBlock {
        function: function.id,
        block: function.entry,
    })?;
    if entry.params.len() != function.signature.params.len() {
        return Err(CompileError::InvalidFunctionSignature {
            function: function.id,
            message: format!(
                "entry block has {} parameters but signature has {}",
                entry.params.len(),
                function.signature.params.len()
            ),
        });
    }

    let uses_this = function_kind_uses_this(function.kind);
    if uses_this && function.signature.params.is_empty() {
        return Err(CompileError::InvalidFunctionSignature {
            function: function.id,
            message: "method-like function has no receiver parameter".to_owned(),
        });
    }
    let explicit_arity = function
        .signature
        .params
        .len()
        .saturating_sub(if uses_this { 1 } else { 0 });
    if explicit_arity > MAX_NAPALM_ARITY {
        return Err(CompileError::InvalidFunctionSignature {
            function: function.id,
            message: format!(
                "{} explicit parameters exceed the VM maximum of {}",
                explicit_arity, MAX_NAPALM_ARITY
            ),
        });
    }

    let mut parameters = Vec::new();
    for (index, (&value, ty)) in entry
        .params
        .iter()
        .zip(&function.signature.params)
        .enumerate()
    {
        if !dce.materialized_values.contains(value.0 as usize) {
            continue;
        }
        let destination = value_reg(function, allocation, value)?;
        let source = if uses_this && index == 0 {
            IncomingSource::This
        } else {
            let argument_index = index - if uses_this { 1 } else { 0 };
            let argument = ArgReg::new(argument_index as u8).ok_or_else(|| {
                CompileError::InvalidFunctionSignature {
                    function: function.id,
                    message: format!("invalid argument index {argument_index}"),
                }
            })?;
            IncomingSource::Argument(argument)
        };
        parameters.push(ParameterBinding {
            source,
            destination,
            kind: register_kind(ty),
        });
    }

    let mut captures = Vec::new();
    for (index, capture) in function.captures.iter().enumerate() {
        if !dce.materialized_values.contains(capture.value.0 as usize) {
            continue;
        }
        captures.push(CaptureBinding {
            capture_index: index as u16,
            destination: value_reg(function, allocation, capture.value)?,
            kind: register_kind(&capture.ty),
        });
    }

    let frame = FrameLayout {
        general_register_count: allocation.frame_register_count,
        parameters,
        captures,
        return_kind: register_kind(&function.signature.result),
    };
    frame
        .validate()
        .map_err(|message| CompileError::InvalidFunctionSignature {
            function: function.id,
            message,
        })?;
    Ok((frame, explicit_arity as u8))
}

// =============================================================================
// Bytecode emission helpers
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ConstantKey {
    Null,
    Bool(bool),
    Num(u64),
    String(u32),
    Int32(i32),
}

impl ConstantKey {
    fn from_constant(constant: &Constant) -> Self {
        match constant {
            Constant::Null => Self::Null,
            Constant::Bool(value) => Self::Bool(*value),
            Constant::Num(value) => Self::Num(value.to_bits()),
            Constant::String(value) => Self::String(value.0),
            Constant::Int32(value) => Self::Int32(*value),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct EdgeTransferKey {
    target: BlockId,
    moves: Vec<(u8, u8)>,
}

#[derive(Clone, Debug)]
struct DeferredTrampoline {
    label: Label,
    target: BlockId,
    moves: Vec<(Reg, Reg)>,
}

#[derive(Clone, Debug)]
struct PcMap {
    block_start: Vec<Option<BytecodePc>>,
    block_end: Vec<Option<BytecodePc>>,
    before_inst: Vec<Option<BytecodePc>>,
    after_inst: Vec<Option<BytecodePc>>,
}

#[derive(Clone, Debug)]
struct FunctionCompileInfo {
    allocation: RegisterAllocation,
    pc_map: PcMap,
}

#[derive(Default)]
struct ClassFusionPlan {
    by_creation: HashMap<InstId, Vec<SpecializedMethod>>,
    suppressed_bindings: HashSet<InstId>,
}

impl ClassFusionPlan {
    fn build(function: &Function, options: &CompileOptions) -> Self {
        let mut plan = Self::default();
        if !options.emit_specialized_bytecode {
            return plan;
        }
        let mut protected_nodes = HashSet::default();
        for binding in options.resume_bindings.values() {
            if binding.function != function.id {
                continue;
            }
            if let ResumeTarget::BeforeInst(id) | ResumeTarget::AfterInst(id) = binding.target {
                protected_nodes.insert(id);
            }
        }
        let mut binding_counts: HashMap<ValueId, usize> = HashMap::default();
        for node in &function.nodes {
            if let Inst::BindMethod { class, .. } | Inst::BindForeignMethod { class, .. } = &node.kind {
                *binding_counts.entry(*class).or_default() += 1;
            }
        }
        for block in &function.blocks {
            for (position, &creation_id) in block.insts.iter().enumerate() {
                let Some(creation) = function.node(creation_id) else { continue; };
                let Inst::CreateClass { .. } = &creation.kind else { continue; };
                if protected_nodes.contains(&creation_id) { continue; }
                let Some(&class_value) = creation.results.first() else { continue; };
                let mut collected = Vec::new();
                let mut suppressed = Vec::new();
                for &id in block.insts.iter().skip(position + 1) {
                    let Some(node) = function.node(id) else { break; };
                    if protected_nodes.contains(&id) { break; }
                    let method = match &node.kind {
                        Inst::BindMethod { class, method, function, is_static }
                            if *class == class_value => SpecializedMethod {
                                symbol: *method,
                                target: SpecializedMethodTarget::Function(*function),
                                is_static: *is_static,
                            },
                        Inst::BindForeignMethod { class, method, foreign, is_static }
                            if *class == class_value => SpecializedMethod {
                                symbol: *method,
                                target: SpecializedMethodTarget::Foreign(*foreign),
                                is_static: *is_static,
                            },
                        _ => break,
                    };
                    collected.push(method);
                    suppressed.push(id);
                }
                // Never absorb a subset of bindings: a later mutation could
                // otherwise observe a different method registration order.
                let expected = binding_counts.get(&class_value).copied().unwrap_or(0);
                if collected.len() != expected {
                    continue;
                }
                plan.by_creation.insert(creation_id, collected);
                plan.suppressed_bindings.extend(suppressed);
            }
        }
        plan
    }
}

#[derive(Clone)]
struct InstanceFusion {
    constructor: FuncId,
    args: Vec<ValueId>,
}

#[derive(Default)]
struct InstanceFusionPlan {
    by_allocation: HashMap<InstId, InstanceFusion>,
    suppressed_calls: HashSet<InstId>,
}

impl InstanceFusionPlan {
    fn build(program: &Program, function: &Function, options: &CompileOptions) -> Self {
        let mut plan = Self::default();
        if !options.emit_specialized_bytecode {
            return plan;
        }
        let mut protected = HashSet::default();
        for binding in options.resume_bindings.values() {
            if binding.function == function.id {
                if let ResumeTarget::BeforeInst(id) | ResumeTarget::AfterInst(id) = binding.target {
                    protected.insert(id);
                }
            }
        }
        for block in &function.blocks {
            for pair in block.insts.windows(2) {
                let (alloc_id, call_id) = (pair[0], pair[1]);
                if protected.contains(&alloc_id) || protected.contains(&call_id) {
                    continue;
                }
                let (Some(alloc), Some(call)) = (function.node(alloc_id), function.node(call_id)) else {
                    continue;
                };
                let Inst::AllocInstance { class } = &alloc.kind else { continue; };
                let Some(&instance) = alloc.results.first() else { continue; };
                let Inst::CallDirect { conv, callee, args } = &call.kind else { continue; };
                if *conv != crate::napalm_ir::CallConv::Napalm
                    || args.first() != Some(&instance)
                    || args[1..].contains(&instance) {
                    continue;
                }
                let Some(target) = program.functions.iter().find(|f| f.id == *callee) else {
                    continue;
                };
                if target.kind != FunctionKind::Constructor
                    || target.field_owner != Some(*class)
                    || target.is_async || target.is_concurrent
                    // The fused handler pushes the constructor frame directly
                    // (no memo probe), exactly like a non-memoized CallDirect.
                    || target.memoization != crate::napalm_ir::MemoizationKind::None
                    || target.signature.params.len() != args.len()
                {
                    // Constructors receive the instance as THIS; the explicit
                    // args are the remaining elements in the complete call frame.
                    continue;
                }
                // The original CallDirect result is ignored by the factory.
                // If it is observable anywhere, fusing would lose that value.
                let result_is_used = call.results.iter().any(|result| {
                    function.nodes.iter().any(|node| node.id != call_id &&
                        node.kind.operands().contains(result))
                    || function.deopts.iter().any(|point| {
                        point.snapshot.slots.iter().any(|slot| slot.value == *result)
                        || point.snapshot.inlined_frames.iter().any(|frame| {
                            frame.slots.iter().any(|slot| slot.value == *result)
                        })
                    })
                });
                if result_is_used { continue; }
                // A class factory is a direct allocation followed by its
                // statically known constructor. The instance does not escape
                // to an unrelated effect between these adjacent HIR nodes.
                plan.by_allocation.insert(alloc_id, InstanceFusion {
                    constructor: *callee,
                    args: args[1..].to_vec(),
                });
                plan.suppressed_calls.insert(call_id);
            }
        }
        plan
    }
}

/// Whole literal construction is fused only when every MapSet is adjacent
/// to its allocation in the same block, all keys are compile-time strings,
/// and no covered node is a resume point. Values are evaluated before
/// allocation, so this transformation preserves evaluation and exception order.
#[derive(Clone)]
struct ObjectFusion {
    names: Vec<StringId>,
    values: Vec<ValueId>,
    contract: Option<(crate::napalm_ir::StructuralLayoutId, Type)>,
    null_shape: Option<crate::napalm_ir::StructuralLayoutId>,
}

#[derive(Default)]
struct ObjectFusionPlan {
    by_allocation: HashMap<InstId, ObjectFusion>,
    suppressed_sets: HashSet<InstId>,
    suppressed_bindings: HashMap<InstId, ValueId>,
}

impl ObjectFusionPlan {
    fn constant_of<'a>(function: &'a Function, value: ValueId) -> Option<&'a Constant> {
        let ValueDef::InstResult(inst, _) = function.value(value)?.def else { return None; };
        match &function.node(inst)?.kind {
            Inst::Const(value) => Some(value),
            _ => None,
        }
    }

    fn build(program: &Program, function: &Function, options: &CompileOptions) -> Self {
        let mut result = Self::default();
        if !options.emit_specialized_bytecode { return result; }
        let mut protected = HashSet::default();
        for binding in options.resume_bindings.values() {
            if binding.function == function.id {
                if let ResumeTarget::BeforeInst(id) | ResumeTarget::AfterInst(id) = binding.target {
                    protected.insert(id);
                }
            }
        }
        for block in &function.blocks {
            let ids = &block.insts;
            for (start, &allocation_id) in ids.iter().enumerate() {
                if protected.contains(&allocation_id) || result.by_allocation.contains_key(&allocation_id) {
                    continue;
                }
                let Some(allocation) = function.node(allocation_id) else { continue; };
                if !matches!(&allocation.kind, Inst::AllocMap) { continue; }
                let Some(&map) = allocation.results.first() else { continue; };
                let mut names = Vec::new();
                let mut values = Vec::new();
                let mut sets = Vec::new();
                let mut following = start + 1;
                while following < ids.len() && values.len() < MAX_NAPALM_ARITY {
                    let id = ids[following];
                    if protected.contains(&id) || result.suppressed_sets.contains(&id) { break; }
                    let Some(node) = function.node(id) else { break; };
                    // A structural StoreProperty on the unpublished literal has
                    // exactly MapSet's semantics; its layout slot is only a hint.
                    let (receiver, key, value) = match &node.kind {
                        Inst::MapSet { map: receiver, key, value }
                        | Inst::StoreProperty { object: receiver, key, value, .. } => (receiver, key, value),
                        _ => break,
                    };
                    if *receiver != map { break; }
                    let Some(Constant::String(name)) = Self::constant_of(function, *key) else { break; };
                    names.push(*name);
                    values.push(*value);
                    sets.push(id);
                    following += 1;
                }
                let mut bound = None;
                if following < ids.len() && !protected.contains(&ids[following]) {
                    if let Some(node) = function.node(ids[following]) {
                        if let Inst::BindInterface { value, expected, layout: Some(layout) } = &node.kind {
                            if *value == map && matches!(expected, Type::Interface(_) | Type::Record(_)) {
                                if let Some(shape) = program.structural_layout(*layout) {
                                    // Preserve key insertion order and optional-field presence.
                                    if shape.properties.len() == names.len()
                                        && shape.properties.iter().zip(&names).all(|(field, name)| field.name == *name)
                                        && shape.contract == *expected
                                    {
                                        bound = Some((ids[following], *layout, expected.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
                if sets.is_empty() && bound.is_none() { continue; }
                let null_shape = if bound.is_none()
                    && !names.is_empty()
                    && values.iter().all(|&value| matches!(Self::constant_of(function, value), Some(Constant::Null)))
                {
                    program.structural_layouts.iter().find(|shape| {
                        shape.properties.len() == names.len()
                            && shape.properties.iter().zip(&names).all(|(field, name)| field.name == *name)
                    }).map(|shape| shape.id)
                } else { None };
                result.suppressed_sets.extend(sets);
                if let Some((binding, _, _)) = &bound {
                    result.suppressed_bindings.insert(*binding, map);
                }
                result.by_allocation.insert(allocation_id, ObjectFusion {
                    names, values,
                    contract: bound.map(|(_, layout, ty)| (layout, ty)),
                    null_shape,
                });
            }
        }
        result
    }
}

/// Field contracts whose runtime check also *installs* persistent state on the
/// stored value (structural objects, typed collections, generic instances):
/// a static proof alone cannot skip them.
fn field_requires_runtime_contract(ty: &Type) -> bool {
    use crate::napalm_ir::BuiltinId;
    match ty {
        Type::Interface(_) | Type::Record(_) | Type::List(Some(_)) | Type::GenericMap(_, _)
        | Type::Set(Some(_)) | Type::GenericInstance { .. } => true,
        Type::Builtin(BuiltinId::Event | BuiltinId::Emitter | BuiltinId::Channel | BuiltinId::Receiver, Some(_)) => true,
        Type::Union(types) => types.iter().any(field_requires_runtime_contract),
        _ => false,
    }
}

fn mentions_nominal_type(ty: &Type) -> bool {
    match ty {
        Type::Instance(_) | Type::Class(_) => true,
        Type::Union(types) => types.iter().any(mentions_nominal_type),
        _ => false,
    }
}

/// `DefineArrayWithGenerics` installs a *persistent* element contract at
/// allocation. That is only equivalent to the generic bytecode when the list
/// was going to receive exactly that contract anyway: an `AllocList` whose
/// unpublished result is bound by `BindInterface(List<T>)` in the same block,
/// with only appends to it and pure constants in between (no other code can
/// observe the earlier contract installation). The inferred static type of a
/// literal is NOT a contract (`const xs: any = [1]` stays untyped).
#[derive(Default)]
struct ArrayFusionPlan {
    by_allocation: HashMap<InstId, Type>,
    suppressed_bindings: HashMap<InstId, ValueId>,
}

impl ArrayFusionPlan {
    fn build(function: &Function, options: &CompileOptions) -> Self {
        let mut result = Self::default();
        if !options.emit_specialized_bytecode { return result; }
        let mut protected = HashSet::default();
        for binding in options.resume_bindings.values() {
            if binding.function == function.id {
                if let ResumeTarget::BeforeInst(id) | ResumeTarget::AfterInst(id) = binding.target {
                    protected.insert(id);
                }
            }
        }
        for block in &function.blocks {
            let ids = &block.insts;
            for (start, &allocation_id) in ids.iter().enumerate() {
                if protected.contains(&allocation_id) { continue; }
                let Some(allocation) = function.node(allocation_id) else { continue; };
                if !matches!(allocation.kind, Inst::AllocList { .. }) { continue; }
                let Some(&list) = allocation.results.first() else { continue; };
                for &id in &ids[start + 1..] {
                    if protected.contains(&id) { break; }
                    let Some(node) = function.node(id) else { break; };
                    match &node.kind {
                        Inst::Const(_) => continue,
                        Inst::ListAppend { list: target, value } if *target == list && *value != list => continue,
                        Inst::BindInterface { value, expected: expected @ Type::List(Some(_)), .. }
                            if *value == list =>
                        {
                            if let Type::List(Some(element)) = expected {
                                result.by_allocation.insert(allocation_id, (**element).clone());
                                result.suppressed_bindings.insert(id, list);
                            }
                            break;
                        }
                        _ => break,
                    }
                }
            }
        }
        result
    }
}

#[derive(Clone, Copy, Debug)]
enum CopySource {
    General(Reg),
    Scratch,
}

struct FunctionEmitter<'a> {
    program: &'a Program,
    function: &'a Function,
    options: &'a CompileOptions,
    dce: &'a DeadCodeInfo<'a>,
    allocation: &'a RegisterAllocation,
    positions: &'a PositionTable,
    layout: &'a [BlockId],
    assembler: Assembler,
    block_labels: Vec<Label>,
    pc_map: PcMap,
    constant_cache: HashMap<ConstantKey, ConstIndex>,
    type_guard_cache: HashMap<Type, TypeGuardId>,
    class_guard_cache: HashMap<ClassSet, ClassGuardId>,
    ic_cache: HashMap<(SiteId, MethodSymbol, Option<ClassId>), IcSlot>,
    trampoline_cache: HashMap<EdgeTransferKey, Label>,
    trampolines: Vec<DeferredTrampoline>,
    stats: FunctionCompileStats,
    class_fusions: ClassFusionPlan,
    instance_fusions: InstanceFusionPlan,
    object_fusions: ObjectFusionPlan,
    array_fusions: ArrayFusionPlan,
}

impl<'a> FunctionEmitter<'a> {
    fn new(
        program: &'a Program,
        function: &'a Function,
        options: &'a CompileOptions,
        dce: &'a DeadCodeInfo<'a>,
        allocation: &'a RegisterAllocation,
        positions: &'a PositionTable,
        layout: &'a [BlockId],
    ) -> Self {
        let mut assembler = Assembler::new();
        let block_labels = function
            .blocks
            .iter()
            .map(|_| assembler.label())
            .collect::<Vec<_>>();
        let class_fusions = ClassFusionPlan::build(function, options);
        let instance_fusions = InstanceFusionPlan::build(program, function, options);
        let object_fusions = ObjectFusionPlan::build(program, function, options);
        let array_fusions = ArrayFusionPlan::build(function, options);
        Self {
            program,
            function,
            options,
            dce,
            allocation,
            positions,
            layout,
            assembler,
            block_labels,
            pc_map: PcMap {
                block_start: vec![None; function.blocks.len()],
                block_end: vec![None; function.blocks.len()],
                before_inst: vec![None; function.nodes.len()],
                after_inst: vec![None; function.nodes.len()],
            },
            constant_cache: HashMap::default(),
            type_guard_cache: HashMap::default(),
            class_guard_cache: HashMap::default(),
            ic_cache: HashMap::default(),
            trampoline_cache: HashMap::default(),
            trampolines: Vec::new(),
            class_fusions,
            instance_fusions,
            object_fusions,
            array_fusions,
            stats: FunctionCompileStats {
                function: function.id,
                hir_nodes: function.nodes.len(),
                ..FunctionCompileStats::default()
            },
        }
    }

    fn reg(&self, value: ValueId) -> Result<Reg, CompileError> {
        value_reg(self.function, self.allocation, value)
    }

    fn result_reg(&self, node: &crate::napalm_ir::Node) -> Result<Reg, CompileError> {
        let value = *node.results.first().ok_or_else(|| CompileError::Internal {
            function: Some(self.function.id),
            message: format!("instruction {:?} has no result", node.id),
        })?;
        self.reg(value)
    }

    fn assemble_error(&self, error: impl fmt::Display) -> CompileError {
        CompileError::Assemble {
            function: self.function.id,
            message: error.to_string(),
        }
    }

    fn constant(&mut self, value: Constant) -> ConstIndex {
        let key = ConstantKey::from_constant(&value);
        if let Some(index) = self.constant_cache.get(&key) {
            return *index;
        }
        let index = self.assembler.chunk.push_constant(value);
        self.constant_cache.insert(key, index);
        index
    }

    fn type_guard(&mut self, ty: &Type) -> TypeGuardId {
        if let Some(id) = self.type_guard_cache.get(ty) {
            return *id;
        }
        let layout = self.program.structural_layout_id(ty);
        let id = self.assembler.chunk.push_type_guard(ty.clone(), layout);
        self.type_guard_cache.insert(ty.clone(), id);
        id
    }

    fn class_guard(&mut self, classes: &ClassSet) -> ClassGuardId {
        if let Some(id) = self.class_guard_cache.get(classes) {
            return *id;
        }
        let id = self.assembler.chunk.push_class_guard(classes.clone());
        self.class_guard_cache.insert(classes.clone(), id);
        id
    }

    fn inline_cache(
        &mut self,
        site: SiteId,
        method: MethodSymbol,
        home_class: Option<ClassId>,
    ) -> IcSlot {
        let key = (site, method, home_class);
        if let Some(slot) = self.ic_cache.get(&key) {
            return *slot;
        }
        let slot = self.assembler.chunk.push_inline_cache(InlineCacheDescriptor {
            site,
            method,
            home_class,
        });
        self.ic_cache.insert(key, slot);
        slot
    }

    fn instruction_position(&self, node: InstId) -> u32 {
        self.positions.node_use[node.0 as usize]
    }

    fn live_register_entry(&self, pc: BytecodePc, position: u32) -> LiveRegisterEntry {
        let mut entry = LiveRegisterEntry {
            pc,
            live: LiveRegisterMap::default(),
            constants: Vec::new(),
            types: Vec::new(),
        };
        // A register claimed by two intervals here would make its type ambiguous;
        // `position` is a use position so this should not happen, but never
        // publish a type fact that is not uniquely attributable.
        let mut ambiguous: Vec<Reg> = Vec::new();
        for interval in &self.allocation.intervals {
            if interval.ranges.is_empty() || !interval.contains(position) {
                continue;
            }
            if let Some(reg) = self.allocation.value_regs[interval.value.0 as usize] {
                if entry.live.general_is_live(reg) {
                    ambiguous.push(reg);
                }
                entry.live.set_general(reg, true);
                let value = &self.function.values[interval.value.0 as usize];
                if is_trace_fact_type(&value.ty) {
                    entry.types.push((reg, value.ty.clone()));
                }
                if let ValueDef::InstResult(id, 0) = value.def {
                    if let Inst::Const(constant) = &self.function.nodes[id.0 as usize].kind {
                        if !matches!(constant, Constant::String(_)) {
                            entry.constants.push((reg, constant.clone()));
                        }
                    }
                }
            }
        }
        if !ambiguous.is_empty() {
            entry.types.retain(|(reg, _)| !ambiguous.contains(reg));
        }
        entry
    }

    /// Join of the static types of every value this function can return to a
    /// synchronous caller. Each `Return` operand's WIR type is a guarantee
    /// (declared result contracts are bound before the return), so the join is
    /// a sound fact about the call result. Tail calls leave the result unknown.
    fn result_type(&self) -> Type {
        if self.function.is_async || self.function.is_concurrent {
            return Type::Top;
        }
        let mut returned = Vec::new();
        for (index, node) in self.function.nodes.iter().enumerate() {
            if !self.dce.live_nodes[index] {
                continue;
            }
            match &node.kind {
                Inst::Return(value) => match self.function.value(*value) {
                    Some(value) => returned.push(value.ty.clone()),
                    None => return Type::Top,
                },
                Inst::TailInvoke { .. }
                | Inst::TailInvokeSuper { .. }
                | Inst::TailCallDirect { .. }
                | Inst::TailCallClosure { .. } => return Type::Top,
                _ => {}
            }
        }
        let joined = Type::flow_join_many(returned);
        if is_trace_fact_type(&joined) { joined } else { Type::Top }
    }

    fn live_reference_map(&self, position: u32) -> ReferenceMap {
        let mut map = ReferenceMap::default();
        for interval in &self.allocation.intervals {
            if interval.ranges.is_empty() || !interval.contains(position) {
                continue;
            }
            let value = &self.function.values[interval.value.0 as usize];
            if !type_may_reference(&value.ty) {
                continue;
            }
            if let Some(reg) = self.allocation.value_regs[interval.value.0 as usize] {
                map.set_general(reg, true);
            }
        }

        // These special values are architectural roots of an active Napalm frame.
        // Tagged non-reference values are harmless when scanned as tagged roots.
        if function_kind_uses_this(self.function.kind) {
            map.set_special(SpecialReg::This, true);
        }
        if self.function.kind == FunctionKind::Closure || !self.function.captures.is_empty() {
            map.set_special(SpecialReg::Closure, true);
        }
        map.set_special(SpecialReg::Module, true);
        map.set_special(SpecialReg::Fiber, true);
        map.set_special(SpecialReg::CallerFiber, true);
        map.set_special(SpecialReg::Error, true);
        map.set_special(SpecialReg::ResumeValue, true);
        map
    }

    fn mark_transfer_value(
        &self,
        map: &mut ReferenceMap,
        value: ValueId,
        target: TransferTarget,
    ) -> Result<(), CompileError> {
        let ty = &self
            .function
            .value(value)
            .ok_or(CompileError::UnknownValue {
                function: self.function.id,
                value,
            })?
            .ty;
        if !type_may_reference(ty) {
            return Ok(());
        }
        match target {
            TransferTarget::This => map.set_special(SpecialReg::This, true),
            TransferTarget::Closure => map.set_special(SpecialReg::Closure, true),
            TransferTarget::Argument(index) => {
                if let Some(arg) = ArgReg::new(index as u8) {
                    map.set_argument(arg, true);
                }
            }
        }
        Ok(())
    }

    fn call_reference_map(
        &self,
        node: InstId,
        transfer: CallTransfer<'_>,
    ) -> Result<ReferenceMap, CompileError> {
        let mut map = self.live_reference_map(self.instruction_position(node));
        match transfer {
            CallTransfer::Invoke { receiver, args } => {
                self.mark_transfer_value(&mut map, receiver, TransferTarget::This)?;
                for (index, &arg) in args.iter().enumerate() {
                    self.mark_transfer_value(&mut map, arg, TransferTarget::Argument(index))?;
                }
            }
            CallTransfer::Closure { closure, args } => {
                self.mark_transfer_value(&mut map, closure, TransferTarget::Closure)?;
                for (index, &arg) in args.iter().enumerate() {
                    self.mark_transfer_value(&mut map, arg, TransferTarget::Argument(index))?;
                }
            }
            CallTransfer::Direct { callee, args } => {
                let callee = self
                    .program
                    .function(callee)
                    .ok_or(CompileError::UnknownFunction(callee))?;
                if function_kind_uses_this(callee.kind) {
                    if let Some((&receiver, explicit)) = args.split_first() {
                        self.mark_transfer_value(&mut map, receiver, TransferTarget::This)?;
                        for (index, &arg) in explicit.iter().enumerate() {
                            self.mark_transfer_value(
                                &mut map,
                                arg,
                                TransferTarget::Argument(index),
                            )?;
                        }
                    }
                } else {
                    for (index, &arg) in args.iter().enumerate() {
                        self.mark_transfer_value(&mut map, arg, TransferTarget::Argument(index))?;
                    }
                }
            }
            CallTransfer::RuntimeMethod { args } => {
                if let Some((&receiver, explicit)) = args.split_first() {
                    self.mark_transfer_value(&mut map, receiver, TransferTarget::This)?;
                    for (index, &arg) in explicit.iter().enumerate() {
                        self.mark_transfer_value(
                            &mut map,
                            arg,
                            TransferTarget::Argument(index),
                        )?;
                    }
                }
            }
            CallTransfer::None => {}
        }
        Ok(map)
    }

    fn safepoint_kind(opcode: Opcode) -> SafepointKind {
        use Opcode::*;
        match opcode {
            Safepoint => SafepointKind::ExplicitPoll,
            Invoke | InvokeSuper | InvokeSpread | CallDirect | CallClosure | Launch | CallPrimitive
            | NewClassInstance | InvokeMethodDirect => SafepointKind::Call,
            CallForeign => SafepointKind::Foreign,
            ImportModule | ImportVariable => SafepointKind::Module,
            Await | MakeFiber | FiberCall | FiberTry | FiberTransfer
            | FiberTransferError | FiberYield | FiberSuspend => SafepointKind::FiberSwitch,
            BindMethod | BindForeignMethod | ImplementInterfaceOrType
            | SetFieldShape | ListAppend | ListInsert | MapSet | MapLookupOrInsert | StoreProperty
            | StorePropertyIfPresent => {
                SafepointKind::RuntimeMutation
            }
            _ => SafepointKind::Allocation,
        }
    }

    fn record_safepoint(
        &mut self,
        node: InstId,
        pc: BytecodePc,
        transfer: CallTransfer<'_>,
    ) -> Result<(), CompileError> {
        let opcode = self.assembler.chunk.code[pc.word_index()]
            .opcode()
            .ok_or_else(|| CompileError::Internal {
                function: Some(self.function.id),
                message: format!("emitted invalid opcode at PC {}", pc.0),
            })?;
        if !opcode.may_gc() {
            return Ok(());
        }
        let references = self.call_reference_map(node, transfer)?;
        self.assembler
            .add_safepoint(pc, Self::safepoint_kind(opcode), references);
        Ok(())
    }

    fn edge_moves(
        &self,
        from: BlockId,
        edge: &crate::napalm_ir::Edge,
    ) -> Result<Vec<(Reg, Reg)>, CompileError> {
        let target = self.function.block(edge.target).ok_or(CompileError::UnknownBlock {
            function: self.function.id,
            block: edge.target,
        })?;
        if target.params.len() != edge.args.len() {
            return Err(CompileError::InvalidEdge {
                function: self.function.id,
                from,
                to: edge.target,
                message: "block parameter/argument count mismatch".to_owned(),
            });
        }

        let mut moves = Vec::new();
        for (&source_value, &target_value) in edge.args.iter().zip(&target.params) {
            if !self
                .dce
                .materialized_values
                .contains(target_value.0 as usize)
            {
                continue;
            }
            let source = self.reg(source_value)?;
            let target_reg = self.reg(target_value)?;
            let source_kind = register_kind(
                &self
                    .function
                    .value(source_value)
                    .ok_or(CompileError::UnknownValue {
                        function: self.function.id,
                        value: source_value,
                    })?
                    .ty,
            );
            let target_kind = register_kind(
                &self
                    .function
                    .value(target_value)
                    .ok_or(CompileError::UnknownValue {
                        function: self.function.id,
                        value: target_value,
                    })?
                    .ty,
            );
            if source_kind != target_kind {
                return Err(CompileError::RepresentationMismatch {
                    function: self.function.id,
                    from: source_value,
                    to: target_value,
                    from_kind: source_kind,
                    to_kind: target_kind,
                });
            }
            if source != target_reg {
                moves.push((target_reg, source));
            }
        }
        Ok(moves)
    }

    fn emit_copy_source(&mut self, dst: Reg, source: CopySource) {
        match source {
            CopySource::General(src) => {
                self.assembler.emit_move(dst, src);
            }
            CopySource::Scratch => {
                self.assembler
                    .emit_ab(Opcode::MoveFromSpecial, dst.0, SpecialReg::Scratch0 as u8);
            }
        }
    }

    fn emit_parallel_moves(&mut self, moves: &[(Reg, Reg)]) {
        #[derive(Clone, Copy)]
        struct PendingMove {
            dst: Reg,
            src: CopySource,
        }

        let mut pending = moves
            .iter()
            .copied()
            .filter(|(dst, src)| dst != src)
            .map(|(dst, src)| PendingMove {
                dst,
                src: CopySource::General(src),
            })
            .collect::<Vec<_>>();

        while !pending.is_empty() {
            if pending.len() == 2 {
                if let (CopySource::General(a_src), CopySource::General(b_src)) =
                    (pending[0].src, pending[1].src)
                {
                    if pending[0].dst == b_src && pending[1].dst == a_src {
                        self.assembler
                            .emit_ab(Opcode::Swap, pending[0].dst.0, pending[1].dst.0);
                        self.stats.edge_moves += 1;
                        pending.clear();
                        break;
                    }
                }
            }

            let safe = pending.iter().position(|candidate| {
                !pending.iter().any(|other| match other.src {
                    CopySource::General(src) => src == candidate.dst,
                    CopySource::Scratch => false,
                })
            });

            if let Some(index) = safe {
                let item = pending.swap_remove(index);
                self.emit_copy_source(item.dst, item.src);
                self.stats.edge_moves += 1;
                continue;
            }

            // A cycle remains. Save one old destination value outside the GPR
            // file, then rewrite all reads of it to the scratch special register.
            let cycle_reg = pending[0].dst;
            self.assembler.emit_ab(
                Opcode::MoveToSpecial,
                SpecialReg::Scratch0 as u8,
                cycle_reg.0,
            );
            for item in &mut pending {
                if matches!(item.src, CopySource::General(src) if src == cycle_reg) {
                    item.src = CopySource::Scratch;
                }
            }
        }
    }

    fn deferred_target(
        &mut self,
        from: BlockId,
        edge: &crate::napalm_ir::Edge,
    ) -> Result<Label, CompileError> {
        let moves = self.edge_moves(from, edge)?;
        if moves.is_empty() {
            return Ok(self.block_labels[edge.target.0 as usize]);
        }

        let key = EdgeTransferKey {
            target: edge.target,
            moves: moves.iter().map(|(dst, src)| (dst.0, src.0)).collect(),
        };
        if let Some(label) = self.trampoline_cache.get(&key) {
            return Ok(*label);
        }

        let label = self.assembler.label();
        self.trampoline_cache.insert(key, label);
        self.trampolines.push(DeferredTrampoline {
            label,
            target: edge.target,
            moves,
        });
        self.stats.edge_trampolines += 1;
        Ok(label)
    }

    fn emit_inline_edge(
        &mut self,
        from: BlockId,
        edge: &crate::napalm_ir::Edge,
        next_block: Option<BlockId>,
    ) -> Result<(), CompileError> {
        let moves = self.edge_moves(from, edge)?;
        self.emit_parallel_moves(&moves);
        if !(self.options.omit_fallthrough_jumps && next_block == Some(edge.target)) {
            self.assembler
                .emit_jump(self.block_labels[edge.target.0 as usize]);
        }
        Ok(())
    }

    fn emit_branch(
        &mut self,
        block: BlockId,
        cond: ValueId,
        then_edge: &crate::napalm_ir::Edge,
        else_edge: &crate::napalm_ir::Edge,
        next_block: Option<BlockId>,
    ) -> Result<(), CompileError> {
        let cond = self.reg(cond)?;

        let fallthrough_then = next_block == Some(then_edge.target);
        let fallthrough_else = next_block == Some(else_edge.target);
        let choose_then = if fallthrough_then != fallthrough_else {
            fallthrough_then
        } else {
            block_hotness(self.function, then_edge.target)
                >= block_hotness(self.function, else_edge.target)
        };

        if choose_then {
            let else_target = self.deferred_target(block, else_edge)?;
            self.assembler.emit_jump_if_false(cond, else_target);
            self.emit_inline_edge(block, then_edge, next_block)?;
        } else {
            let then_target = self.deferred_target(block, then_edge)?;
            self.assembler.emit_jump_if_true(cond, then_target);
            self.emit_inline_edge(block, else_edge, next_block)?;
        }
        Ok(())
    }

    fn emit_switch(
        &mut self,
        block: BlockId,
        tag: ValueId,
        default_edge: &crate::napalm_ir::Edge,
        cases: &[crate::napalm_ir::SwitchCase],
    ) -> Result<(), CompileError> {
        let default_target = self.deferred_target(block, default_edge)?;
        let mut targets = Vec::with_capacity(cases.len());
        for case in cases {
            targets.push((case.value, self.deferred_target(block, &case.edge)?));
        }
        self.assembler
            .emit_switch(self.reg(tag)?, default_target, targets);
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum TransferTarget {
    This,
    Closure,
    Argument(usize),
}

#[derive(Clone, Copy)]
enum CallTransfer<'a> {
    None,
    Invoke {
        receiver: ValueId,
        args: &'a [ValueId],
    },
    Closure {
        closure: ValueId,
        args: &'a [ValueId],
    },
    Direct {
        callee: FuncId,
        args: &'a [ValueId],
    },
    RuntimeMethod {
        args: &'a [ValueId],
    },
}

impl FunctionEmitter<'_> {
    /// Known-class field access: the function's own class, an own field at a
    /// known class-local offset, and what the static types prove about it
    /// (`stored` is the written value for SetFieldClass). Slots outside the
    /// class's own range (accessors inlined from another class) keep the
    /// generic LoadField/StoreField encoding.
    fn field_class_descriptor(&self, slot: u16, stored: Option<ValueId>) -> Option<SpecializedDescriptor> {
        if !self.options.emit_specialized_bytecode { return None; }
        let class = self.function.field_owner?;
        let info = self.program.class(class)?;
        let local = slot as usize;
        if local >= info.own_instance_field_count as usize { return None; }
        let field_type = info.own_instance_field_types.get(local).cloned().unwrap_or(Type::Top);
        let (nullable, unchecked_store) = match stored {
            None => (field_type.accepts(&Type::Null), false),
            Some(value) => {
                let generic = info.own_generic_fields.get(local).copied().unwrap_or(false);
                let value_type = &self.function.value(value)?.ty;
                // Heap contracts are installed (not merely checked) by the store.
                // Nominal (class) contracts are exact at runtime: require the
                // identical static type rather than static subtyping.
                let proven = matches!(field_type, Type::Top)
                    || (!field_requires_runtime_contract(&field_type)
                        && !matches!(value_type, Type::Top | Type::Bottom)
                        && !value_type.is_internal_representation()
                        && field_type.accepts(value_type)
                        && (!mentions_nominal_type(&field_type) || field_type == *value_type));
                (false, !generic && proven)
            }
        };
        Some(SpecializedDescriptor::FieldClass {
            class, local_slot: slot as u32, nullable, unchecked_store,
        })
    }

    fn specialize_method_call(
        &self,
        receiver: ValueId,
        site: SiteId,
        method: MethodSymbol,
        cache: IcSlot,
    ) -> Option<(Opcode, SpecializedDescriptor)> {
        let ty = &self.function.value(receiver)?.ty;
        let signature = self.program.method(method)?;
        // A monomorphic profile is an optimization hint, not a proof. The
        // interpreter must guard the concrete method identity before executing
        // this target; the exact original IC remains the failure path.
        if let Type::Instance(classes) = ty {
            let [owner] = classes.as_slice() else { return None; };
            let profile = self.function.profiles.call_sites.get(&site)?;
            let [(target, samples)] = profile.targets.as_slice() else { return None; };
            if *samples < 32 { return None; }
            let function = self.program.functions.iter().find(|f| f.id == *target)?;
            if function.kind == FunctionKind::Method && function.signature.params.len() == signature.arity as usize + 1 {
                return Some((Opcode::InvokeMethodDirect, SpecializedDescriptor::DirectMethod {
                    owner: *owner, symbol: method, target: *target, fallback_cache: cache,
                }));
            }
        }
        None
    }

    fn emit_non_control_node(
        &mut self,
        node: &crate::napalm_ir::Node,
    ) -> Result<(), CompileError> {
        use Inst::*;

        if self.instance_fusions.suppressed_calls.contains(&node.id)
            || self.object_fusions.suppressed_sets.contains(&node.id)
        {
            return Ok(());
        }
        if let Some(&original_map) = self.object_fusions.suppressed_bindings.get(&node.id)
            .or_else(|| self.array_fusions.suppressed_bindings.get(&node.id))
        {
            let destination = self.result_reg(node)?;
            let source = self.reg(original_map)?;
            if destination != source {
                self.assembler.emit_move(destination, source);
            }
            return Ok(());
        }
        if self.class_fusions.suppressed_bindings.contains(&node.id) {
            self.stats.fused_class_method_bindings += 1;
            return Ok(());
        }
        // Keep the fused constructor arguments alive until its allocation-and-call
        // opcode has been assigned a precise GC map.
        let constructor_fusion = self.instance_fusions.by_allocation.get(&node.id).cloned();
        let mut safepoint_transfer = CallTransfer::None;
        let pc = match &node.kind {
            Jump { .. } | Branch { .. } | Switch { .. } => {
                return Err(CompileError::Internal {
                    function: Some(self.function.id),
                    message: format!("control instruction {:?} reached non-control emitter", node.id),
                });
            }
            Return(value) => self.assembler.emit_a(Opcode::Return, self.reg(*value)?.0),
            Unreachable => self.assembler.emit0(Opcode::Trap),

            Safepoint => self.assembler.emit0(Opcode::Safepoint),
            OSRPoint => self.assembler.emit0(Opcode::OsrPoint),

            Const(constant) => {
                let dst = self.result_reg(node)?;
                match constant {
                    Constant::Null => self.assembler.emit_a(Opcode::LoadNull, dst.0),
                    Constant::Bool(false) => self.assembler.emit_a(Opcode::LoadFalse, dst.0),
                    Constant::Bool(true) => self.assembler.emit_a(Opcode::LoadTrue, dst.0),
                    Constant::Int32(value) if i8::try_from(*value).is_ok() => self
                        .assembler
                        .emit_ab(Opcode::LoadI8, dst.0, (*value as i8) as u8),
                    Constant::Int32(value) => self.assembler.emit_x32(
                        Opcode::LoadI32,
                        dst.0,
                        0,
                        0,
                        *value as u32,
                    ),
                    other => {
                        let constant = self.constant(other.clone());
                        self.assembler.emit_load_const(dst, constant)
                    }
                }
            }
            Move(value) => {
                let dst = self.result_reg(node)?;
                let src = self.reg(*value)?;
                if dst == src {
                    return Ok(());
                }
                self.assembler.emit_move(dst, src)
            }

            BindInterface { value, expected, .. } => {
                let dst = self.result_reg(node)?;
                let src = self.reg(*value)?;
                if self.options.emit_specialized_bytecode
                    && matches!(expected, Type::Interface(_) | Type::Record(_))
                {
                    let opcode = Opcode::ImplementInterfaceOrType;
                    let descriptor = SpecializedDescriptor::Contract {
                        name: None,
                        contract: expected.clone(),
                    };
                    self.stats.specialized_opcodes += 1;
                    self.assembler.emit_specialized(opcode, dst, src, &[], descriptor)
                        .map_err(|error| self.assemble_error(error))?
                } else {
                    let guard = self.type_guard(expected);
                    self.assembler.emit_x32(Opcode::BindInterface, dst.0, src.0, 0, guard.0)
                }
            }
            GuardType {
                value,
                expected,
                deopt,
            } => {
                let guard = self.type_guard(expected);
                self.assembler.emit_x32_y32(
                    Opcode::GuardType,
                    self.result_reg(node)?.0,
                    self.reg(*value)?.0,
                    0,
                    guard.0,
                    deopt.0,
                )
            }
            GuardClass {
                value,
                expected,
                deopt,
            } => {
                let guard = self.class_guard(expected);
                self.assembler.emit_x32_y32(
                    Opcode::GuardClass,
                    self.result_reg(node)?.0,
                    self.reg(*value)?.0,
                    0,
                    guard.0,
                    deopt.0,
                )
            }
            GuardBounds {
                index,
                length,
                deopt,
            } => self.assembler.emit_x32(
                Opcode::GuardBounds,
                self.reg(*index)?.0,
                self.reg(*length)?.0,
                0,
                deopt.0,
            ),
            GuardNotNull { value, deopt } => self.assembler.emit_x32(
                Opcode::GuardNotNull,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
                0,
                deopt.0,
            ),
            GuardInt32 { value, deopt } => self.assembler.emit_x32(
                Opcode::GuardInt32,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
                0,
                deopt.0,
            ),
            GuardF64 { value, deopt } => self.assembler.emit_x32(
                Opcode::GuardF64,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
                0,
                deopt.0,
            ),
            Bailout { reason, deopt } => self.assembler.emit_x32(
                Opcode::Bailout,
                encode_bailout_reason(*reason),
                0,
                0,
                deopt.0,
            ),

            Invoke {
                site,
                receiver,
                method,
                args,
            } => {
                let cache = self.inline_cache(*site, *method, None);
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let specialized = if self.options.emit_specialized_bytecode {
                    self.specialize_method_call(*receiver, *site, *method, cache)
                } else {
                    None
                };
                let pc = if let Some((opcode, descriptor)) = specialized {
                    self.stats.specialized_opcodes += 1;
                    self.assembler.emit_specialized(
                        opcode, self.result_reg(node)?, self.reg(*receiver)?,
                        &arg_regs, descriptor,
                    ).map_err(|error| self.assemble_error(error))?
                } else {
                    self.assembler
                        .emit_invoke(self.result_reg(node)?, self.reg(*receiver)?, &arg_regs, cache)
                        .map_err(|error| self.assemble_error(error))?
                };
                safepoint_transfer = CallTransfer::Invoke {
                    receiver: *receiver,
                    args,
                };
                pc
            }
            InvokeSuper {
                site,
                home_class,
                receiver,
                method,
                args,
            } => {
                let cache = self.inline_cache(*site, *method, Some(*home_class));
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let pc = self
                    .assembler
                    .emit_invoke_super(
                        self.result_reg(node)?,
                        self.reg(*receiver)?,
                        &arg_regs,
                        cache,
                    )
                    .map_err(|error| self.assemble_error(error))?;
                safepoint_transfer = CallTransfer::Invoke {
                    receiver: *receiver,
                    args,
                };
                pc
            }
            CallDirect { conv, callee, args } => {
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let pc = self
                    .assembler
                    .emit_call_direct(self.result_reg(node)?, *conv, &arg_regs, *callee)
                    .map_err(|error| self.assemble_error(error))?;
                safepoint_transfer = CallTransfer::Direct {
                    callee: *callee,
                    args,
                };
                pc
            }
            CallClosure {
                conv,
                closure,
                args,
            } => {
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let pc = self
                    .assembler
                    .emit_call_closure(
                        self.result_reg(node)?,
                        self.reg(*closure)?,
                        *conv,
                        &arg_regs,
                    )
                    .map_err(|error| self.assemble_error(error))?;
                safepoint_transfer = CallTransfer::Closure {
                    closure: *closure,
                    args,
                };
                pc
            }
            Launch { closure, args } => {
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let pc = self.assembler.emit_launch(
                    self.result_reg(node)?, self.reg(*closure)?, &arg_regs,
                ).map_err(|error| self.assemble_error(error))?;
                safepoint_transfer = CallTransfer::Closure {
                    closure: *closure,
                    args,
                };
                pc
            }
            // Tail calls are WIR terminators and therefore have no SSA result
            // register. Until the bytecode VM grows frame-reusing TailCall opcodes,
            // lower them exactly as the corresponding call followed by Return. R0
            // is safe as the dead destination: a tail terminator has no normal-path
            // live values after the call, and call operands are consumed before the
            // destination is written on return.
            TailInvoke {
                site,
                receiver,
                method,
                args,
            } => {
                let dst = Reg(0);
                let cache = self.inline_cache(*site, *method, None);
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let pc = self
                    .assembler
                    .emit_invoke(dst, self.reg(*receiver)?, &arg_regs, cache)
                    .map_err(|error| self.assemble_error(error))?;
                self.record_safepoint(
                    node.id,
                    pc,
                    CallTransfer::Invoke {
                        receiver: *receiver,
                        args,
                    },
                )?;
                self.assembler.emit_a(Opcode::Return, dst.0);
                return Ok(());
            }
            TailInvokeSuper {
                site,
                home_class,
                receiver,
                method,
                args,
            } => {
                let dst = Reg(0);
                let cache = self.inline_cache(*site, *method, Some(*home_class));
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let pc = self
                    .assembler
                    .emit_invoke_super(dst, self.reg(*receiver)?, &arg_regs, cache)
                    .map_err(|error| self.assemble_error(error))?;
                self.record_safepoint(
                    node.id,
                    pc,
                    CallTransfer::Invoke {
                        receiver: *receiver,
                        args,
                    },
                )?;
                self.assembler.emit_a(Opcode::Return, dst.0);
                return Ok(());
            }
            TailCallDirect { conv, callee, args } => {
                let dst = Reg(0);
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let pc = self
                    .assembler
                    .emit_call_direct(dst, *conv, &arg_regs, *callee)
                    .map_err(|error| self.assemble_error(error))?;
                self.record_safepoint(
                    node.id,
                    pc,
                    CallTransfer::Direct {
                        callee: *callee,
                        args,
                    },
                )?;
                self.assembler.emit_a(Opcode::Return, dst.0);
                return Ok(());
            }
            TailCallClosure {
                conv,
                closure,
                args,
            } => {
                let dst = Reg(0);
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let pc = self
                    .assembler
                    .emit_call_closure(dst, self.reg(*closure)?, *conv, &arg_regs)
                    .map_err(|error| self.assemble_error(error))?;
                self.record_safepoint(
                    node.id,
                    pc,
                    CallTransfer::Closure {
                        closure: *closure,
                        args,
                    },
                )?;
                self.assembler.emit_a(Opcode::Return, dst.0);
                return Ok(());
            }
            CallPrimitive { primitive, args } => {
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let pc = self
                    .assembler
                    .emit_call_primitive(self.result_reg(node)?, &arg_regs, *primitive)
                    .map_err(|error| self.assemble_error(error))?;
                safepoint_transfer = CallTransfer::RuntimeMethod { args };
                pc
            }
            CallForeign { foreign, args } => {
                let arg_regs = args
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                let pc = self
                    .assembler
                    .emit_call_foreign(self.result_reg(node)?, &arg_regs, *foreign)
                    .map_err(|error| self.assemble_error(error))?;
                safepoint_transfer = CallTransfer::RuntimeMethod { args };
                pc
            }

            CreateClass { class, superclass } => {
                let (super_reg, has_superclass) = match superclass {
                    Some(value) => (self.reg(*value)?.0, 1),
                    None => (0, 0),
                };
                if let (true, Some(methods), Some(info)) = (
                    self.options.emit_specialized_bytecode,
                    self.class_fusions.by_creation.get(&node.id),
                    self.program.classes.iter().find(|info| info.id == *class),
                ) {
                    let has_implements = !info.implements.is_empty();
                    let opcode = match (has_superclass != 0, has_implements) {
                        (false, false) => Opcode::DefineClass,
                        (true, false) => Opcode::DefineDerivedClass,
                        (false, true) => Opcode::DefineClassWithImplements,
                        (true, true) => Opcode::DefineDerivedClassWithImplements,
                    };
                    let descriptor = SpecializedDescriptor::Class {
                        class: *class,
                        methods: methods.clone(),
                        implements: info.implements.clone(),
                    };
                    self.stats.specialized_opcodes += 1;
                    self.assembler.emit_specialized(
                        opcode, self.result_reg(node)?, Reg(super_reg), &[], descriptor,
                    ).map_err(|error| self.assemble_error(error))?
                } else {
                    self.assembler.emit_x32(
                        Opcode::CreateClass,
                        self.result_reg(node)?.0,
                        super_reg,
                        has_superclass,
                        class.0,
                    )
                }
            },
            BindMethod {
                class,
                method,
                function,
                is_static,
            } => self.assembler.emit_x32_y32(
                Opcode::BindMethod,
                self.reg(*class)?.0,
                *is_static as u8,
                0,
                method.0,
                function.0,
            ),
            BindForeignMethod {
                class,
                method,
                foreign,
                is_static,
            } => self.assembler.emit_x32_y32(
                Opcode::BindForeignMethod,
                self.reg(*class)?.0,
                *is_static as u8,
                0,
                method.0,
                foreign.0,
            ),
            AllocInstance { class } => {
                if let Some(fusion) = constructor_fusion.as_ref() {
                    // The native handler transfers the newly allocated instance
                    // through THIS and the explicit parameters through ARG0+.
                    // Both banks must be rooted for a nested constructor call.
                    let instance = *node.results.first().ok_or_else(|| CompileError::Internal {
                        function: Some(self.function.id),
                        message: "fused instance allocation has no result".to_owned(),
                    })?;
                    safepoint_transfer = CallTransfer::Invoke {
                        receiver: instance,
                        args: &fusion.args,
                    };
                    let args = fusion.args.iter().map(|arg| self.reg(*arg))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.stats.specialized_opcodes += 1;
                    self.assembler.emit_specialized(
                        Opcode::NewClassInstance, self.result_reg(node)?, Reg(0),
                        &args,
                        SpecializedDescriptor::Instance {
                            class: *class,
                            constructor: Some(fusion.constructor),
                        },
                    ).map_err(|error| self.assemble_error(error))?
                } else {
                    self.assembler.emit_x32(
                        Opcode::AllocInstance, self.result_reg(node)?.0,
                        0, 0, class.0,
                    )
                }
            },
            AllocForeign { class, size } => self.assembler.emit_x32(
                Opcode::AllocForeign,
                self.result_reg(node)?.0,
                self.reg(*size)?.0,
                0,
                class.0,
            ),
            LoadField { object, slot } => {
                if let Some(descriptor) = self.field_class_descriptor(slot.0, None) {
                    self.stats.specialized_opcodes += 1;
                    self.assembler.emit_specialized(
                        Opcode::GetFieldClass, self.result_reg(node)?, self.reg(*object)?, &[],
                        descriptor,
                    ).map_err(|error| self.assemble_error(error))?
                } else {
                    self.assembler.emit_x32(
                        Opcode::LoadField, self.result_reg(node)?.0,
                        self.reg(*object)?.0, 0, slot.0 as u32,
                    )
                }
            },
            StoreField { object, slot, value } => {
                if let Some(descriptor) = self.field_class_descriptor(slot.0, Some(*value)) {
                    self.stats.specialized_opcodes += 1;
                    self.assembler.emit_specialized(
                        Opcode::SetFieldClass, self.reg(*object)?, self.reg(*value)?, &[],
                        descriptor,
                    ).map_err(|error| self.assemble_error(error))?
                } else {
                    self.assembler.emit_x32(
                        Opcode::StoreField, self.reg(*object)?.0,
                        self.reg(*value)?.0, 0, slot.0 as u32,
                    )
                }
            },
            LoadStaticField { class, slot } => self.assembler.emit_x32(
                Opcode::LoadStaticField,
                self.result_reg(node)?.0,
                self.reg(*class)?.0,
                0,
                slot.0 as u32,
            ),
            StoreStaticField { class, slot, value } => self.assembler.emit_x32(
                Opcode::StoreStaticField,
                self.reg(*class)?.0,
                self.reg(*value)?.0,
                0,
                slot.0 as u32,
            ),
            IsInstance { value, class } => self.assembler.emit_abc(
                Opcode::IsInstance,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
                self.reg(*class)?.0,
            ),

            ImportModule { module_name } => self.assembler.emit_x32(
                Opcode::ImportModule,
                self.result_reg(node)?.0,
                0,
                0,
                module_name.0,
            ),
            ImportVariable {
                module,
                variable_name,
            } => self.assembler.emit_x32(
                Opcode::ImportVariable,
                self.result_reg(node)?.0,
                self.reg(*module)?.0,
                0,
                variable_name.0,
            ),
            LoadModuleVar { module, slot } => self.assembler.emit_x32_y32(
                Opcode::LoadModuleVar,
                self.result_reg(node)?.0,
                0,
                0,
                module.0,
                slot.0,
            ),
            StoreModuleVar {
                module,
                slot,
                value,
            } => self.assembler.emit_x32_y32(
                Opcode::StoreModuleVar,
                self.reg(*value)?.0,
                0,
                0,
                module.0,
                slot.0,
            ),

            AllocCell { initial } => self.assembler.emit_ab(
                Opcode::AllocCell,
                self.result_reg(node)?.0,
                self.reg(*initial)?.0,
            ),
            LoadCell { cell } => self.assembler.emit_ab(
                Opcode::LoadCell,
                self.result_reg(node)?.0,
                self.reg(*cell)?.0,
            ),
            StoreCell { cell, value } => self.assembler.emit_ab(
                Opcode::StoreCell,
                self.reg(*cell)?.0,
                self.reg(*value)?.0,
            ),
            MakeClosure { function, captures } => {
                let capture_regs = captures
                    .iter()
                    .map(|value| self.reg(*value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.assembler
                    .emit_make_closure(self.result_reg(node)?, *function, &capture_regs)
                    .map_err(|error| self.assemble_error(error))?
            }

            AllocList { capacity } => {
                // Only lists that the generic code binds to exactly List<T>
                // (see ArrayFusionPlan); never the inferred literal type.
                let element = self.array_fusions.by_allocation.get(&node.id).cloned();
                if let Some(element) = element {
                    let capacity_reg = match capacity {
                        Some(value) => self.reg(*value)?,
                        None => Reg(0),
                    };
                    self.stats.specialized_opcodes += 1;
                    self.assembler.emit_specialized(
                        Opcode::DefineArrayWithGenerics, self.result_reg(node)?,
                        capacity_reg, &[],
                        SpecializedDescriptor::ArrayType { element, with_capacity: capacity.is_some() },
                    ).map_err(|error| self.assemble_error(error))?
                } else if let Some(capacity) = capacity {
                    self.assembler.emit_ab(Opcode::AllocListCap, self.result_reg(node)?.0, self.reg(*capacity)?.0)
                } else {
                    self.assembler.emit_a(Opcode::AllocList, self.result_reg(node)?.0)
                }
            },
            ListLen { list } => self.assembler.emit_ab(
                Opcode::ListLen,
                self.result_reg(node)?.0,
                self.reg(*list)?.0,
            ),
            ListGet { list, index } => self.assembler.emit_abc(
                Opcode::ListGet,
                self.result_reg(node)?.0,
                self.reg(*list)?.0,
                self.reg(*index)?.0,
            ),
            ListSet { list, index, value } => self.assembler.emit_abc(
                Opcode::ListSet,
                self.reg(*list)?.0,
                self.reg(*index)?.0,
                self.reg(*value)?.0,
            ),
            ListSpread { list, source } => self.assembler.emit_abc(
                Opcode::ListSpread, self.result_reg(node)?.0, self.reg(*list)?.0, self.reg(*source)?.0,
            ),
            MapSpread { map, source } => self.assembler.emit_ab(
                Opcode::MapSpread, self.reg(*map)?.0, self.reg(*source)?.0,
            ),
            DestructureCheck { source, is_array } => self.assembler.emit_abc(
                Opcode::DestructureCheck, self.result_reg(node)?.0, self.reg(*source)?.0, u8::from(*is_array),
            ),
            DestructureGet { source, key } => self.assembler.emit_abc(
                Opcode::DestructureGet, self.result_reg(node)?.0, self.reg(*source)?.0, self.reg(*key)?.0,
            ),
            DestructureRest { source, key } => self.assembler.emit_abc(
                Opcode::DestructureRest, self.result_reg(node)?.0, self.reg(*source)?.0, self.reg(*key)?.0,
            ),
            InvokeSpread { receiver, method, arguments, home_class } => self.assembler.emit_x32_y32(
                Opcode::InvokeSpread, self.result_reg(node)?.0, self.reg(*receiver)?.0,
                self.reg(*arguments)?.0, method.0, home_class.map_or(u32::MAX, |id| id.0),
            ),
            ListAppend { list, value } => self.assembler.emit_ab(
                Opcode::ListAppend,
                self.reg(*list)?.0,
                self.reg(*value)?.0,
            ),
            ListInsert { list, index, value } => self.assembler.emit_abc(
                Opcode::ListInsert,
                self.reg(*list)?.0,
                self.reg(*index)?.0,
                self.reg(*value)?.0,
            ),
            ListRemoveAt { list, index } => self.assembler.emit_abc(
                Opcode::ListRemoveAt,
                self.result_reg(node)?.0,
                self.reg(*list)?.0,
                self.reg(*index)?.0,
            ),

            AllocMap => {
                if let Some(fusion) = self.object_fusions.by_allocation.get(&node.id).cloned() {
                    self.stats.specialized_opcodes += 1;
                    if let Some((layout, contract)) = fusion.contract {
                        let args = fusion.values.iter().map(|value| self.reg(*value))
                            .collect::<Result<Vec<_>, _>>()?;
                        self.assembler.emit_specialized(
                            Opcode::NewInterfaceOrTypeObject, self.result_reg(node)?, Reg(0),
                            &args, SpecializedDescriptor::ObjectShape {
                                layout, contract: Some(contract),
                            },
                        ).map_err(|error| self.assemble_error(error))?
                    } else if let Some(layout) = fusion.null_shape {
                        self.assembler.emit_specialized(
                            Opcode::NewObjectShape, self.result_reg(node)?, Reg(0),
                            &[], SpecializedDescriptor::ObjectShape { layout, contract: None },
                        ).map_err(|error| self.assemble_error(error))?
                    } else {
                        let args = fusion.values.iter().map(|value| self.reg(*value))
                            .collect::<Result<Vec<_>, _>>()?;
                        self.assembler.emit_specialized(
                            Opcode::NewObjectWithProperties, self.result_reg(node)?, Reg(0),
                            &args, SpecializedDescriptor::ObjectProperties { names: fusion.names },
                        ).map_err(|error| self.assemble_error(error))?
                    }
                } else if self.options.emit_specialized_bytecode {
                    self.stats.specialized_opcodes += 1;
                    self.assembler.emit_a(Opcode::NewObjectEmpty, self.result_reg(node)?.0)
                } else {
                    self.assembler.emit_a(Opcode::AllocMap, self.result_reg(node)?.0)
                }
            },
            MapLen { map } => self.assembler.emit_ab(
                Opcode::MapLen,
                self.result_reg(node)?.0,
                self.reg(*map)?.0,
            ),
            MapGet { map, key } => self.assembler.emit_abc(
                Opcode::MapGet,
                self.result_reg(node)?.0,
                self.reg(*map)?.0,
                self.reg(*key)?.0,
            ),
            LoadProperty { object, key, layout, slot } => {
                if self.options.emit_specialized_bytecode {
                    self.stats.specialized_opcodes += 1;
                    self.assembler.emit_specialized_shape(
                        Opcode::GetFieldShape, self.result_reg(node)?,
                        self.reg(*object)?, self.reg(*key)?, *layout, *slot,
                    ).map_err(|error| self.assemble_error(error))?
                } else {
                    self.assembler.emit_x32_y32(
                        Opcode::LoadProperty, self.result_reg(node)?.0,
                        self.reg(*object)?.0, self.reg(*key)?.0,
                        layout.0, slot.0 as u32,
                    )
                }
            },
            MapLookupOrInsert { map, key, value } => self.assembler.emit_x32(
                Opcode::MapLookupOrInsert, self.result_reg(node)?.0,
                self.reg(*map)?.0, self.reg(*key)?.0, self.reg(*value)?.0 as u32,
            ),
            MapSet { map, key, value } => self.assembler.emit_abc(
                Opcode::MapSet,
                self.reg(*map)?.0,
                self.reg(*key)?.0,
                self.reg(*value)?.0,
            ),
            StoreProperty { object, key, value, layout, slot } => {
                if self.options.emit_specialized_bytecode {
                    self.stats.specialized_opcodes += 1;
                    self.assembler.emit_specialized_shape(
                        Opcode::SetFieldShape, self.reg(*object)?,
                        self.reg(*key)?, self.reg(*value)?, *layout, *slot,
                    ).map_err(|error| self.assemble_error(error))?
                } else {
                    self.assembler.emit_x32_y32(
                        Opcode::StoreProperty, self.reg(*object)?.0,
                        self.reg(*key)?.0, self.reg(*value)?.0,
                        layout.0, slot.0 as u32,
                    )
                }
            },
            StorePropertyIfPresent { object, key, value, present, layout, slot } => {
                let packed = slot.0 as u32 | ((self.reg(*present)?.0 as u32) << 16);
                self.assembler.emit_x32_y32(
                    Opcode::StorePropertyIfPresent,
                    self.reg(*object)?.0,
                    self.reg(*key)?.0,
                    self.reg(*value)?.0,
                    layout.0,
                    packed,
                )
            }
            MapRemove { map, key } => self.assembler.emit_abc(
                Opcode::MapRemove,
                self.result_reg(node)?.0,
                self.reg(*map)?.0,
                self.reg(*key)?.0,
            ),

            StringLen { string } => self.assembler.emit_ab(
                Opcode::StringLen,
                self.result_reg(node)?.0,
                self.reg(*string)?.0,
            ),
            StringConcat { left, right } => self.assembler.emit_abc(
                Opcode::StringConcat,
                self.result_reg(node)?.0,
                self.reg(*left)?.0,
                self.reg(*right)?.0,
            ),
            StringSlice {
                string,
                start,
                count,
            } => self.assembler.emit_x32(
                Opcode::StringSlice,
                self.result_reg(node)?.0,
                self.reg(*string)?.0,
                self.reg(*start)?.0,
                self.reg(*count)?.0 as u32,
            ),
            StringCodePointAt { string, index } => self.assembler.emit_abc(
                Opcode::StringCodePointAt,
                self.result_reg(node)?.0,
                self.reg(*string)?.0,
                self.reg(*index)?.0,
            ),
            MakeRange {
                from,
                to,
                inclusive,
            } => self.assembler.emit_x32(
                Opcode::MakeRange,
                self.result_reg(node)?.0,
                self.reg(*from)?.0,
                self.reg(*to)?.0,
                if *inclusive { 1 } else { 0 },
            ),

            IntUnary { op, value } => self.assembler.emit_ab(
                opcode_for_int_unary(*op),
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
            ),
            IntBin { op, left, right } => self.assembler.emit_abc(
                opcode_for_int_bin(*op),
                self.result_reg(node)?.0,
                self.reg(*left)?.0,
                self.reg(*right)?.0,
            ),
            F64Unary { op, value } => self.assembler.emit_ab(
                opcode_for_f64_unary(*op),
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
            ),
            F64Bin { op, left, right } => self.assembler.emit_abc(
                opcode_for_f64_bin(*op),
                self.result_reg(node)?.0,
                self.reg(*left)?.0,
                self.reg(*right)?.0,
            ),
            CmpInt { kind, left, right } => self.assembler.emit_abc(
                opcode_for_int_cmp(*kind),
                self.result_reg(node)?.0,
                self.reg(*left)?.0,
                self.reg(*right)?.0,
            ),
            CmpF64 { kind, left, right } => self.assembler.emit_abc(
                opcode_for_f64_cmp(*kind),
                self.result_reg(node)?.0,
                self.reg(*left)?.0,
                self.reg(*right)?.0,
            ),
            RefEq { left, right } => self.assembler.emit_abc(
                Opcode::RefEq,
                self.result_reg(node)?.0,
                self.reg(*left)?.0,
                self.reg(*right)?.0,
            ),
            NumToInt32 { value } => self.assembler.emit_ab(
                Opcode::NumToInt32,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
            ),
            NumToF64 { value } => self.assembler.emit_ab(
                Opcode::NumToF64,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
            ),
            Int32ToNum { value } => self.assembler.emit_ab(
                Opcode::Int32ToNum,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
            ),
            F64ToNum { value } => self.assembler.emit_ab(
                Opcode::F64ToNum,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
            ),

            Await { promise } => self.assembler.emit_ab(
                Opcode::Await,
                self.result_reg(node)?.0,
                self.reg(*promise)?.0,
            ),

            MakeFiber { closure } => self.assembler.emit_ab(
                Opcode::MakeFiber,
                self.result_reg(node)?.0,
                self.reg(*closure)?.0,
            ),
            FiberCall { fiber, argument } => self.assembler.emit_abc(
                Opcode::FiberCall,
                self.result_reg(node)?.0,
                self.reg(*fiber)?.0,
                self.reg(*argument)?.0,
            ),
            FiberTry { fiber, argument } => self.assembler.emit_abc(
                Opcode::FiberTry,
                self.result_reg(node)?.0,
                self.reg(*fiber)?.0,
                self.reg(*argument)?.0,
            ),
            FiberTransfer { fiber, argument } => self.assembler.emit_abc(
                Opcode::FiberTransfer,
                self.result_reg(node)?.0,
                self.reg(*fiber)?.0,
                self.reg(*argument)?.0,
            ),
            FiberTransferError { fiber, error } => self.assembler.emit_abc(
                Opcode::FiberTransferError,
                self.result_reg(node)?.0,
                self.reg(*fiber)?.0,
                self.reg(*error)?.0,
            ),
            FiberYield { value } => self.assembler.emit_ab(
                Opcode::FiberYield,
                self.result_reg(node)?.0,
                self.reg(*value)?.0,
            ),
            FiberSuspend => self.assembler.emit_a(Opcode::FiberSuspend, self.result_reg(node)?.0),
            FiberAbort { error } => self.assembler.emit_a(Opcode::FiberAbort, self.reg(*error)?.0),

            WriteBarrier { object, value } => self.assembler.emit_ab(
                Opcode::WriteBarrier,
                self.reg(*object)?.0,
                self.reg(*value)?.0,
            ),
            ReadBarrier { object } => self.assembler.emit_ab(
                Opcode::ReadBarrier,
                self.result_reg(node)?.0,
                self.reg(*object)?.0,
            ),
        };

        self.record_safepoint(node.id, pc, safepoint_transfer)?;
        Ok(())
    }

    fn emit_control_node(
        &mut self,
        block: BlockId,
        node: &crate::napalm_ir::Node,
        next_block: Option<BlockId>,
    ) -> Result<(), CompileError> {
        match &node.kind {
            Inst::Jump { edge } => self.emit_inline_edge(block, edge, next_block),
            Inst::Branch {
                cond,
                then_edge,
                else_edge,
            } => self.emit_branch(block, *cond, then_edge, else_edge, next_block),
            Inst::Switch {
                tag,
                default_edge,
                cases,
            } => self.emit_switch(block, *tag, default_edge, cases),
            _ => self.emit_non_control_node(node),
        }
    }

    fn exception_handler_error(&self, block: BlockId) -> Option<ValueId> {
        self.function
            .exception_regions
            .iter()
            .find(|region| region.handler == block)
            .map(|region| region.error)
    }

    fn emit_catch_entry(&mut self, block: BlockId) -> Result<(), CompileError> {
        let Some(error) = self.exception_handler_error(block) else {
            return Ok(());
        };
        let reg = self.reg(error)?;
        let pc = self.assembler.emit_abc(Opcode::Catch, reg.0, 0, 0);
        let references = self.live_reference_map(self.positions.block_start[block.0 as usize]);
        self.assembler
            .add_safepoint(pc, SafepointKind::Allocation, references);
        Ok(())
    }

    fn emit_block_profile_counter(&mut self, block: BlockId) {
        if !self.options.instrument_blocks {
            return;
        }
        let counter = self
            .assembler
            .chunk
            .push_profile_counter(ProfileCounterKind::Block, None);
        self.assembler
            .emit_x32(Opcode::ProfileBlock, 0, 0, 0, counter.0);
        let _ = block;
    }

    fn emit(mut self) -> Result<(BytecodeFunction, PcMap, FunctionCompileStats), CompileError> {
        let layout = self.layout;
        let function = self.function;
        for (layout_index, &block_id) in layout.iter().enumerate() {
            let insts = &function
                .block(block_id)
                .ok_or(CompileError::UnknownBlock {
                    function: self.function.id,
                    block: block_id,
                })?
                .insts;
            let label = self.block_labels[block_id.0 as usize];
            self.assembler.bind(label);
            self.pc_map.block_start[block_id.0 as usize] = Some(self.assembler.pc());
            self.emit_catch_entry(block_id)?;
            self.emit_block_profile_counter(block_id);
            let next_block = layout.get(layout_index + 1).copied();

            for &inst_id in insts {
                let before = self.assembler.pc();
                self.pc_map.before_inst[inst_id.0 as usize] = Some(before);
                let node = function
                    .node(inst_id)
                    .ok_or_else(|| CompileError::Internal {
                        function: Some(self.function.id),
                        message: format!("emission references missing node {inst_id:?}"),
                    })?;

                let live = if self.dce.live_nodes[inst_id.0 as usize] {
                    self.stats.emitted_hir_nodes += 1;
                    let live = self.live_register_entry(before, self.instruction_position(inst_id));
                    self.emit_control_node(block_id, node, next_block)?;
                    Some(live)
                } else {
                    self.stats.eliminated_hir_nodes += 1;
                    None
                };

                let after = self.assembler.pc();
                if before != after {
                    // Backedges and native exits may rendezvous here without an
                    // allocating opcode. No extra instruction enters the hot loop.
                    if self.assembler.chunk.safepoints.last().is_none_or(|entry| entry.pc != before) {
                        let references = self.live_reference_map(self.instruction_position(inst_id));
                        self.assembler.add_safepoint(
                            before,
                            SafepointKind::ExplicitPoll,
                            references,
                        );
                    }
                    if let Some(live) = live {
                        // Side exits resume at original bytecode instruction boundaries.
                        // Keep all physical GPR liveness separate from GC reference maps.
                        // If a single HIR node emits several bytecodes, the start PC is
                        // still the semantically valid re-entry point for that node.
                        self.assembler.chunk.live_registers.push(live);
                    }
                }
                self.pc_map.after_inst[inst_id.0 as usize] = Some(after);
                if self.options.emit_source_map && before != after {
                    if let Some(source) = node.source {
                        self.assembler.chunk.source_map.push(SourceMapEntry {
                            start_pc: before,
                            end_pc: after,
                            source,
                        });
                    }
                }
            }
            self.pc_map.block_end[block_id.0 as usize] = Some(self.assembler.pc());
        }

        // Materialize the zero-cost exception table after every block PC range is known.
        for region in &self.function.exception_regions {
            let handler_pc = self.pc_map.block_start[region.handler.0 as usize].ok_or_else(|| {
                CompileError::Internal {
                    function: Some(self.function.id),
                    message: format!("missing bytecode PC for exception handler {:?}", region.handler),
                }
            })?;
            let error_reg = self.reg(region.error)?;
            for &protected in &region.protected_blocks {
                let start_pc = self.pc_map.block_start[protected.0 as usize].ok_or_else(|| {
                    CompileError::Internal {
                        function: Some(self.function.id),
                        message: format!("missing bytecode start PC for protected block {:?}", protected),
                    }
                })?;
                let end_pc = self.pc_map.block_end[protected.0 as usize].ok_or_else(|| {
                    CompileError::Internal {
                        function: Some(self.function.id),
                        message: format!("missing bytecode end PC for protected block {:?}", protected),
                    }
                })?;
                if start_pc != end_pc {
                    self.assembler.chunk.exception_handlers.push(ExceptionHandlerEntry {
                        start_pc,
                        end_pc,
                        handler_pc,
                        error_reg,
                        depth: region.depth,
                    });
                }
            }
        }
        self.assembler.chunk.exception_handlers.sort_unstable_by_key(|entry| {
            (entry.start_pc.0, entry.end_pc.0, std::cmp::Reverse(entry.depth))
        });

        for trampoline in std::mem::take(&mut self.trampolines) {
            self.assembler.bind(trampoline.label);
            self.emit_parallel_moves(&trampoline.moves);
            self.assembler
                .emit_jump(self.block_labels[trampoline.target.0 as usize]);
        }

        let (frame, arity) = build_frame_layout(self.function, self.allocation, self.dce)?;
        let capture_count = u16::try_from(self.function.captures.len()).map_err(|_| {
            CompileError::InvalidFunctionSignature {
                function: self.function.id,
                message: format!(
                    "{} captures exceed the bytecode u16 capture count",
                    self.function.captures.len()
                ),
            }
        })?;

        let function_id = self.function.id;
        let result_type = self.result_type();
        let chunk = self.assembler.finish().map_err(|error| CompileError::Assemble {
            function: function_id,
            message: error.to_string(),
        })?;
        self.stats.allocated_values = self
            .allocation
            .value_regs
            .iter()
            .filter(|reg| reg.is_some())
            .count();
        self.stats.general_registers = self.allocation.frame_register_count;
        self.stats.bytecode_words = chunk.code.len();
        self.stats.constants = chunk.constants.len();
        self.stats.inline_caches = chunk.inline_caches.len();
        self.stats.safepoints = chunk.safepoints.len();

        let bytecode = BytecodeFunction {
            id: self.function.id,
            module: self.function.module,
            kind: self.function.kind,
            is_async: self.function.is_async,
            is_concurrent: self.function.is_concurrent,
            memoization: self.function.memoization,
            field_owner: self.function.field_owner,
            name: self.function.signature.name,
            arity,
            result_type,
            capture_count,
            frame,
            chunk,
        };
        Ok((bytecode, self.pc_map, self.stats))
    }
}

// =============================================================================
// Function/program compilation and deoptimization fixups
// =============================================================================

fn compile_function(
    program: &Program,
    function: &Function,
    options: &CompileOptions,
) -> Result<(BytecodeFunction, FunctionCompileInfo, FunctionCompileStats), CompileError> {
    if options.verify_ir {
        if let Err(errors) = verify_function(function) {
            return Err(CompileError::IrVerification {
                function: function.id,
                errors: errors.into_iter().map(|error| error.to_string()).collect(),
            });
        }
    }

    // All analysis bitsets share one function-scoped arena. No arena-backed
    // data escapes into the emitted bytecode or its metadata.
    let arena = Bump::new();
    let cfg = build_cfg(function)?;
    let dce = analyze_dead_code(&arena, function, &cfg, options.eliminate_dead_code)?;
    let layout = layout_blocks(function, &cfg, options.profile_guided_layout);
    let positions = assign_positions(function, &layout);
    let liveness = compute_liveness(&arena, function, &cfg, &dce, &layout)?;
    let intervals = build_intervals(function, &dce, &liveness, &positions, &layout)?;
    let affinities = build_affinities(function, &cfg, &dce, options.coalesce_registers);
    let allocation = allocate_registers(function, intervals, &affinities)?;

    // Validate edge representation classes before emission. Doing this after
    // allocation also catches a malformed HIR before parallel-copy generation.
    for block in &function.blocks {
        let Some(&terminator) = block.insts.last() else {
            continue;
        };
        let node = function.node(terminator).ok_or_else(|| CompileError::Internal {
            function: Some(function.id),
            message: format!("missing node {terminator:?}"),
        })?;
        for edge in node.kind.edges() {
            let target = function.block(edge.target).ok_or(CompileError::UnknownBlock {
                function: function.id,
                block: edge.target,
            })?;
            for (&from, &to) in edge.args.iter().zip(&target.params) {
                if !dce.materialized_values.contains(to.0 as usize) {
                    continue;
                }
                let from_ty = &function
                    .value(from)
                    .ok_or(CompileError::UnknownValue {
                        function: function.id,
                        value: from,
                    })?
                    .ty;
                let to_ty = &function
                    .value(to)
                    .ok_or(CompileError::UnknownValue {
                        function: function.id,
                        value: to,
                    })?
                    .ty;
                let from_kind = register_kind(from_ty);
                let to_kind = register_kind(to_ty);
                if from_kind != to_kind {
                    return Err(CompileError::RepresentationMismatch {
                        function: function.id,
                        from,
                        to,
                        from_kind,
                        to_kind,
                    });
                }
            }
        }
    }

    let (bytecode, pc_map, stats) = FunctionEmitter::new(
        program,
        function,
        options,
        &dce,
        &allocation,
        &positions,
        &layout,
    )
    .emit()?;
    Ok((bytecode, FunctionCompileInfo { allocation, pc_map }, stats))
}

fn lower_runtime_metadata(hir: &Program) -> Result<RuntimeMetadata, CompileError> {
    let mut metadata = RuntimeMetadata::default();

    let mut string_index = 0u32;
    while let Some(value) = hir.string(StringId(string_index)) {
        metadata.strings.push(value.to_owned());
        string_index = string_index
            .checked_add(1)
            .ok_or_else(|| CompileError::Metadata("string ID overflow".to_owned()))?;
    }

    let mut method_index = 0u32;
    while let Some(method) = hir.method(MethodSymbol(method_index)) {
        metadata.methods.push(method);
        method_index = method_index
            .checked_add(1)
            .ok_or_else(|| CompileError::Metadata("method ID overflow".to_owned()))?;
    }

    metadata.modules = hir.modules.clone();
    metadata.classes = hir.classes.clone();
    metadata.foreign_functions = hir.foreign_functions.clone();
    metadata.structural_layouts = hir.structural_layouts.clone();
    metadata.contract_names = hir.contract_names.clone();
    metadata
        .rebuild_indexes()
        .map_err(CompileError::Metadata)?;
    Ok(metadata)
}

fn resolve_resume_pc(
    owner: FuncId,
    resume: ResumeId,
    expected_function: FuncId,
    options: &CompileOptions,
    infos: &HashMap<FuncId, FunctionCompileInfo>,
    hir: &Program,
) -> Result<BytecodePc, CompileError> {
    let binding = options
        .resume_bindings
        .get(&resume)
        .copied()
        .ok_or(CompileError::MissingResumeBinding {
            function: owner,
            resume,
        })?;
    if binding.function != expected_function {
        return Err(CompileError::InvalidResumeBinding {
            function: owner,
            resume,
            message: format!(
                "binding targets {:?}, but snapshot expects {:?}",
                binding.function, expected_function
            ),
        });
    }

    let info = infos
        .get(&expected_function)
        .ok_or(CompileError::UnknownFunction(expected_function))?;
    let function = hir
        .function(expected_function)
        .ok_or(CompileError::UnknownFunction(expected_function))?;

    let pc = match binding.target {
        ResumeTarget::FunctionEntry => info.pc_map.block_start[function.entry.0 as usize],
        ResumeTarget::BlockStart(block) => {
            if function.block(block).is_none() {
                return Err(CompileError::InvalidResumeBinding {
                    function: owner,
                    resume,
                    message: format!("unknown target block {block:?}"),
                });
            }
            info.pc_map.block_start[block.0 as usize]
        }
        ResumeTarget::BeforeInst(inst) => {
            if function.node(inst).is_none() {
                return Err(CompileError::InvalidResumeBinding {
                    function: owner,
                    resume,
                    message: format!("unknown target instruction {inst:?}"),
                });
            }
            info.pc_map.before_inst[inst.0 as usize]
        }
        ResumeTarget::AfterInst(inst) => {
            if function.node(inst).is_none() {
                return Err(CompileError::InvalidResumeBinding {
                    function: owner,
                    resume,
                    message: format!("unknown target instruction {inst:?}"),
                });
            }
            info.pc_map.after_inst[inst.0 as usize]
        }
    };

    pc.ok_or_else(|| CompileError::InvalidResumeBinding {
        function: owner,
        resume,
        message: "target has no emitted bytecode PC".to_owned(),
    })
}

fn deopt_location_for_value(
    function: &Function,
    info: &FunctionCompileInfo,
    value: ValueId,
) -> Result<(DeoptLocation, RegisterKind), CompileError> {
    let value_info = function.value(value).ok_or(CompileError::UnknownValue {
        function: function.id,
        value,
    })?;
    let reg = info
        .allocation
        .value_regs
        .get(value.0 as usize)
        .and_then(|reg| *reg)
        .ok_or(CompileError::DeoptValueUnavailable {
            function: function.id,
            value,
        })?;
    Ok((
        DeoptLocation::General(reg),
        register_kind(&value_info.ty),
    ))
}

fn lower_snapshot_slots(
    function: &Function,
    info: &FunctionCompileInfo,
    slots: &[SnapshotSlot],
) -> Result<Vec<BytecodeSnapshotSlot>, CompileError> {
    slots
        .iter()
        .map(|slot| {
            let (location, kind) = deopt_location_for_value(function, info, slot.value)?;
            Ok(BytecodeSnapshotSlot {
                slot: slot.slot,
                location,
                kind,
            })
        })
        .collect()
}

fn finalize_deopts(
    hir: &Program,
    bytecode: &mut BytecodeProgram,
    infos: &HashMap<FuncId, FunctionCompileInfo>,
    options: &CompileOptions,
) -> Result<(), CompileError> {
    let mut function_indices = HashMap::default();
    for (index, function) in bytecode.functions.iter().enumerate() {
        function_indices.insert(function.id, index);
    }

    let mut resume_entries: HashMap<FuncId, HashMap<ResumeId, BytecodePc>> = HashMap::default();

    for function in &hir.functions {
        let info = infos
            .get(&function.id)
            .ok_or(CompileError::UnknownFunction(function.id))?;
        let output_index = *function_indices
            .get(&function.id)
            .ok_or(CompileError::UnknownFunction(function.id))?;
        let mut lowered_points = Vec::with_capacity(function.deopts.len());

        for point in &function.deopts {
            let resume_id = point.snapshot.resume.id;
            let resume_pc = resolve_resume_pc(
                function.id,
                resume_id,
                function.id,
                options,
                infos,
                hir,
            )?;
            resume_entries
                .entry(function.id)
                .or_default()
                .insert(resume_id, resume_pc);

            let slots = lower_snapshot_slots(function, info, &point.snapshot.slots)?;
            let mut inlined_frames = Vec::with_capacity(point.snapshot.inlined_frames.len());
            for frame in &point.snapshot.inlined_frames {
                let resume_pc = resolve_resume_pc(
                    function.id,
                    frame.resume.id,
                    frame.function,
                    options,
                    infos,
                    hir,
                )?;
                resume_entries
                    .entry(frame.function)
                    .or_default()
                    .insert(frame.resume.id, resume_pc);
                inlined_frames.push(BytecodeInlinedFrameState {
                    function: frame.function,
                    resume_pc,
                    slots: lower_snapshot_slots(function, info, &frame.slots)?,
                });
            }

            lowered_points.push(BytecodeDeoptPoint {
                id: point.id,
                resume_pc,
                slots,
                inlined_frames,
            });
        }

        bytecode.functions[output_index].chunk.deopts = lowered_points;
    }

    for (function, entries) in resume_entries {
        let index = *function_indices
            .get(&function)
            .ok_or(CompileError::UnknownFunction(function))?;
        let chunk = &mut bytecode.functions[index].chunk;
        chunk.resume_points = entries
            .into_iter()
            .map(|(id, pc)| ResumeEntry { id, pc })
            .collect();
        chunk.resume_points.sort_by_key(|entry| entry.pc);
    }

    Ok(())
}

/// Insert module-owned contract declarations before the first original
/// instruction. The runtime already has canonical contract_names; these
/// explicit bytecodes verify that metadata without creating new identities.
///
/// All word PCs in metadata are shifted, including cross-function inlined
/// deoptimization frames. Relative branch displacements are unchanged because
/// both source and destination move by the same constant within the function.
fn prepend_contract_declarations(
    hir: &Program,
    program: &mut BytecodeProgram,
) -> Result<(), CompileError> {
    use crate::napalm_ir::{ContractDeclarationKind, FunctionKind};

    let mut shifts: HashMap<FuncId, u32> = HashMap::default();
    for function in &mut program.functions {
        if function.kind != FunctionKind::ModuleBody { continue; }
        let declarations = hir.contract_declarations.iter()
            .filter(|decl| decl.module == function.module)
            .collect::<Vec<_>>();
        if declarations.is_empty() { continue; }

        let mut prefix = Vec::with_capacity(declarations.len() * 2);
        for decl in declarations {
            let opcode = match decl.kind {
                ContractDeclarationKind::Interface => Opcode::DefineInterface,
                ContractDeclarationKind::Type => Opcode::DefineType,
            };
            let descriptor = SpecializedDescriptor::Contract {
                name: Some(decl.name),
                contract: decl.contract.clone(),
            };
            let id = function.chunk.push_specialization(descriptor);
            prefix.push(Word::abc(opcode, 0, 0, 0));
            prefix.push(Word(id));
        }
        let shift = u32::try_from(prefix.len()).map_err(|_| CompileError::Internal {
            function: Some(function.id),
            message: "contract prelude word count overflows u32".into(),
        })?;
        if function.chunk.code.len() > (u32::MAX - shift) as usize {
            return Err(CompileError::Internal {
                function: Some(function.id),
                message: "contract prelude exceeds bytecode PC range".into(),
            });
        }
        prefix.extend(std::mem::take(&mut function.chunk.code));
        function.chunk.code = prefix;
        function.frame.general_register_count = function.frame.general_register_count.max(1);
        shifts.insert(function.id, shift);
    }

    for function in &mut program.functions {
        let shift = shifts.get(&function.id).copied().unwrap_or(0);
        let bump = |pc: &mut BytecodePc, shift: u32| {
            pc.0 += shift;
        };
        if shift != 0 {
            let chunk = &mut function.chunk;
            for table in &mut chunk.switch_tables {
                bump(&mut table.default_target, shift);
                for case in &mut table.cases { bump(&mut case.target, shift); }
            }
            for entry in &mut chunk.safepoints { bump(&mut entry.pc, shift); }
            for entry in &mut chunk.live_registers { bump(&mut entry.pc, shift); }
            for entry in &mut chunk.resume_points { bump(&mut entry.pc, shift); }
            for entry in &mut chunk.source_map {
                bump(&mut entry.start_pc, shift);
                bump(&mut entry.end_pc, shift);
            }
            for entry in &mut chunk.exception_handlers {
                bump(&mut entry.start_pc, shift);
                bump(&mut entry.end_pc, shift);
                bump(&mut entry.handler_pc, shift);
            }
        }
        for point in &mut function.chunk.deopts {
            bump(&mut point.resume_pc, shift);
            for frame in &mut point.inlined_frames {
                bump(&mut frame.resume_pc, shifts.get(&frame.function).copied().unwrap_or(0));
            }
        }
    }
    Ok(())
}

fn compile_program_impl(
    hir: &Program,
    options: &CompileOptions,
) -> Result<CompilationOutput, CompileError> {
    let mut bytecode = BytecodeProgram::new();
    bytecode.metadata = lower_runtime_metadata(hir)?;

    let mut infos = HashMap::default();
    let mut stats = CompileStats::default();

    for function in &hir.functions {
        let (compiled, info, function_stats) = compile_function(hir, function, options)?;
        infos.insert(function.id, info);
        stats.total_bytecode_words = stats
            .total_bytecode_words
            .saturating_add(function_stats.bytecode_words);
        stats.max_general_registers = stats
            .max_general_registers
            .max(function_stats.general_registers);
        stats.functions.push(function_stats);
        bytecode
            .push_function(compiled)
            .map_err(CompileError::Metadata)?;
    }

    finalize_deopts(hir, &mut bytecode, &infos, options)?;
    if options.emit_specialized_bytecode {
        prepend_contract_declarations(hir, &mut bytecode)?;
    }
    bytecode.rebuild_indexes().map_err(CompileError::Metadata)?;

    stats.total_bytecode_words = bytecode.functions.iter().map(|function| function.chunk.code.len()).sum();
    for entry in &mut stats.functions {
        if let Some(function) = bytecode.function(entry.function) {
            entry.bytecode_words = function.chunk.code.len();
            entry.general_registers = function.frame.general_register_count;
        }
    }
    stats.max_general_registers = bytecode.functions.iter()
        .map(|function| function.frame.general_register_count)
        .max().unwrap_or(0);
    stats.bytecode_words_before_optimization = stats.total_bytecode_words;
    if options.optimize_bytecode {
        let report = optimization_passes()
            .verify_after_each_pass(options.verify_each_bytecode_pass)
            .optimize_program(&mut bytecode)
            .map_err(|e| CompileError::BytecodeVerification(e.to_string()))?;
        stats.bytecode_words_removed = report.words_removed;
        stats.bytecode_passes_changed = report.passes_changed;
        stats.total_bytecode_words = report.output_words;
        for entry in &mut stats.functions {
            if let Some(f) = bytecode.function(entry.function) {
                entry.bytecode_words = f.chunk.code.len();
                entry.safepoints = f.chunk.safepoints.len();
            }
        }
    }

    if options.verify_bytecode {
        BytecodeVerifier::verify_program(&bytecode)
            .map_err(|error| CompileError::BytecodeVerification(error.to_string()))?;
    }

    Ok(CompilationOutput {
        program: bytecode,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::napalm_bytecode::{disassemble, BytecodeVerifier, Decoder};
    use crate::napalm_ir::{Builder, IntBinOp};

    fn synthetic_interval(value: u32, ranges: &[(u32, u32)], use_weight: u64) -> LiveInterval {
        let mut interval = LiveInterval::new(ValueId(value));
        for &(start, end) in ranges {
            interval.add_range(LiveRange { start, end });
        }
        interval.use_weight = use_weight;
        interval.use_count = ranges.len() as u64;
        interval.normalize();
        interval
    }

    #[test]
    fn interference_graph_preserves_lifetime_holes() {
        let intervals = vec![
            synthetic_interval(0, &[(0, 2), (8, 10)], 10),
            synthetic_interval(1, &[(4, 6)], 10),
            synthetic_interval(2, &[(2, 4)], 10),
        ];
        let graph = build_interference_graph(&intervals);

        assert!(!graph.interferes(0, 1), "a lifetime hole must stay reusable");
        assert!(graph.interferes(0, 2), "inclusive endpoints must interfere");
        assert!(graph.interferes(1, 2), "overlapping ranges must interfere");
    }

    #[test]
    fn weighted_affinity_coalescing_shares_a_register_without_conflicts() {
        let intervals = vec![
            synthetic_interval(0, &[(0, 2)], 100),
            synthetic_interval(1, &[(3, 5)], 100),
            synthetic_interval(2, &[(1, 4)], 1),
        ];
        let interference = build_interference_graph(&intervals);
        let mut affinities = AffinityGraph::new(intervals.len());
        affinities.add(ValueId(0), ValueId(1), 1_000_000);
        affinities.add(ValueId(0), ValueId(2), 1);

        let weighted = canonical_affinities(&affinities);
        let coalesced = coalesce_affinity_groups(&intervals, &interference, &weighted);
        assert_eq!(coalesced.value_group[0], coalesced.value_group[1]);
        assert_ne!(coalesced.value_group[0], coalesced.value_group[2]);

        let (regs, frame_register_count) = color_allocation_groups(
            &intervals,
            &interference,
            &affinities,
            coalesced,
        )
        .expect("synthetic graph must color");
        assert_eq!(regs[0], regs[1], "coalesced affinity must eliminate the copy");
        assert_ne!(regs[0], regs[2], "interfering values must use different registers");
        assert_eq!(frame_register_count, 2);
    }

    #[test]
    fn compiles_arithmetic_and_reuses_registers() {
        let mut hir = Program::new();
        let module = hir.add_module("main", 0);
        {
            let mut builder = Builder::new(
                &mut hir,
                module,
                "main",
                FunctionKind::ModuleBody,
                Vec::new(),
                Type::Num,
            );
            let one = builder.const_i32(1);
            let two = builder.const_i32(2);
            let sum = builder.emit1(
                Inst::IntBin {
                    op: IntBinOp::Add,
                    left: one,
                    right: two,
                },
                Type::Int32,
            );
            let result = builder.emit1(Inst::Int32ToNum { value: sum }, Type::Num);
            builder.ret(result);
            builder.finish();
        }

        // Inspect raw lowering separately: the post-bytecode optimizer is
        // allowed to replace IAdd with a precomputed LoadI8.
        let output = BytecodeCompiler::new(CompileOptions {
            optimize_bytecode: false,
            ..CompileOptions::default()
        })
            .compile_with_stats(&hir)
            .expect("HIR must compile");
        BytecodeVerifier::verify_program(&output.program).expect("bytecode must verify");
        assert_eq!(output.program.functions.len(), 1);
        assert!(output.stats.max_general_registers <= 3);
        let text = disassemble(&output.program.functions[0]).expect("must disassemble");
        assert!(text.contains("IAdd"));
        assert!(text.contains("Int32ToNum"));
        let chunk = &output.program.functions[0].chunk;
        let instructions = Decoder::new(&chunk.code).collect::<Result<Vec<_>, _>>().unwrap();
        let add = instructions.iter().find(|op| op.opcode == Opcode::IAdd).unwrap();
        assert!(chunk.constants_at(add.pc).contains(&(Reg(add.first.b()), Constant::Int32(1))));
        assert!(chunk.constants_at(add.pc).contains(&(Reg(add.first.c()), Constant::Int32(2))));
        let convert = instructions.iter().find(|op| op.opcode == Opcode::Int32ToNum).unwrap();
        assert!(chunk.constants_at(convert.pc).is_empty(),
            "a register reused for an arithmetic result must not retain its previous literal");
    }

    #[test]
    fn post_bytecode_optimizer_folds_arithmetic_and_remaps_metadata() {
        let mut hir = Program::new();
        let module = hir.add_module("main", 0);
        {
            let mut builder = Builder::new(
                &mut hir,
                module,
                "main",
                FunctionKind::ModuleBody,
                Vec::new(),
                Type::Num,
            );
            let one = builder.const_i32(1);
            let two = builder.const_i32(2);
            let sum = builder.emit1(
                Inst::IntBin { op: IntBinOp::Add, left: one, right: two },
                Type::Int32,
            );
            let result = builder.emit1(Inst::Int32ToNum { value: sum }, Type::Num);
            builder.ret(result);
            builder.finish();
        }
        let raw = BytecodeCompiler::new(CompileOptions {
            optimize_bytecode: false,
            ..CompileOptions::default()
        }).compile(&hir).expect("raw bytecode compilation");
        let optimized = BytecodeCompiler::default()
            .compile_with_stats(&hir).expect("optimized bytecode compilation");
        BytecodeVerifier::verify_program(&optimized.program).expect("optimized bytecode verifies");
        assert!(optimized.program.functions[0].chunk.code.len() <= raw.functions[0].chunk.code.len());
        assert_eq!(optimized.stats.total_bytecode_words,optimized.program.functions[0].chunk.code.len());
        let instructions = Decoder::new(&optimized.program.functions[0].chunk.code)
            .collect::<Result<Vec<_>,_>>().expect("valid decoder stream");
        assert!(instructions.iter().any(|inst|
            inst.opcode == Opcode::LoadI8 && inst.first.b() as i8 == 3));
        assert!(!instructions.iter().any(|inst|inst.opcode == Opcode::IAdd));
    }

    #[test]
    fn lowers_block_parameters_as_parallel_edge_copies() {
        let mut hir = Program::new();
        let module = hir.add_module("main", 0);
        {
            let mut builder = Builder::new(
                &mut hir,
                module,
                "select",
                FunctionKind::ModuleBody,
                Vec::new(),
                Type::Num,
            );
            let then_block = builder.new_block(&[]);
            let else_block = builder.new_block(&[]);
            let join = builder.new_block(&[]);
            let joined = builder.add_block_param(join, Type::Int32);
            let condition = builder.const_bool(true);
            builder.branch(condition, then_block, vec![], else_block, vec![]);

            builder.position_at(then_block);
            let left = builder.const_i32(10);
            builder.jump(join, vec![left]);

            builder.position_at(else_block);
            let right = builder.const_i32(20);
            builder.jump(join, vec![right]);

            builder.position_at(join);
            let result = builder.emit1(Inst::Int32ToNum { value: joined }, Type::Num);
            builder.ret(result);
            builder.finish();
        }

        let output = compile_program(&hir).expect("HIR must compile");
        BytecodeVerifier::verify_program(&output).expect("bytecode must verify");
        let chunk = &output.functions[0].chunk;
        let instructions = Decoder::new(&chunk.code).collect::<Result<Vec<_>, _>>().unwrap();
        let convert = instructions.iter().find(|op| op.opcode == Opcode::Int32ToNum).unwrap();
        assert!(chunk.constants_at(convert.pc).is_empty(),
            "a block parameter merging different literals is not itself a literal");
    }

    #[test]
    fn dce_removes_dead_pure_nodes_but_keeps_effects() {
        let mut hir = Program::new();
        let module = hir.add_module("main", 0);
        {
            let mut builder = Builder::new(
                &mut hir,
                module,
                "main",
                FunctionKind::ModuleBody,
                Vec::new(),
                Type::Null,
            );
            let _dead_a = builder.const_i32(123);
            let _dead_b = builder.const_i32(456);
            let result = builder.const_null();
            builder.ret(result);
            builder.finish();
        }

        let output = BytecodeCompiler::default()
            .compile_with_stats(&hir)
            .expect("HIR must compile");
        assert!(output.stats.functions[0].eliminated_hir_nodes >= 2);
        BytecodeVerifier::verify_program(&output.program).expect("bytecode must verify");
    }

    #[test]
    fn contract_prologue_keeps_alias_declaration_kind_and_relocates_pcs() {
        use crate::napalm_bytecode::{
            BytecodeChunk, BytecodeFunction, BytecodeDeoptPoint, FrameLayout,
            LiveRegisterMap, RegisterKind, ResumeEntry, SwitchTable, SwitchTarget,
        };
        use crate::napalm_ir::{
            ContractDeclaration, ContractDeclarationKind, MemoizationKind, ModuleId,
        };

        let mut hir = Program::new();
        let module = hir.add_module("test", 0);
        let interface_name = hir.intern_string("Contract");
        let alias_name = hir.intern_string("Alias");
        // Both resolve to Interface, but the alias must emit DefineType.
        let contract = Type::Interface(Vec::new());
        hir.contract_declarations.extend([
            ContractDeclaration {
                module, name: interface_name, contract: contract.clone(),
                kind: ContractDeclarationKind::Interface,
            },
            ContractDeclaration {
                module, name: alias_name, contract,
                kind: ContractDeclarationKind::Type,
            },
        ]);

        let mut chunk = BytecodeChunk::default();
        chunk.code = vec![
            Word::abc(Opcode::LoadNull, 0, 0, 0),
            Word::abc(Opcode::Return, 0, 0, 0),
        ];
        chunk.resume_points.push(ResumeEntry {
            id: ResumeId(0), pc: BytecodePc(1),
        });
        chunk.deopts.push(BytecodeDeoptPoint {
            id: DeoptId(0), resume_pc: BytecodePc(1),
            slots: vec![], inlined_frames: vec![],
        });
        chunk.live_registers.push(LiveRegisterEntry {
            pc: BytecodePc(0), live: LiveRegisterMap::default(),
            constants: vec![], types: vec![],
        });
        chunk.switch_tables.push(SwitchTable {
            default_target: BytecodePc(0),
            cases: vec![SwitchTarget { value: 1, target: BytecodePc(1) }],
        });

        let mut bytecode = BytecodeProgram::new();
        bytecode.push_function(BytecodeFunction {
            id: FuncId(0), module, kind: FunctionKind::ModuleBody,
            is_async: false, is_concurrent: false,
            memoization: MemoizationKind::None, field_owner: None,
            name: interface_name, arity: 0, result_type: Type::Top,
            capture_count: 0,
            frame: FrameLayout {
                general_register_count: 0, parameters: vec![], captures: vec![],
                return_kind: RegisterKind::Value,
            },
            chunk,
        }).unwrap();
        prepend_contract_declarations(&hir, &mut bytecode).unwrap();
        let function = &bytecode.functions[0];
        assert_eq!(function.chunk.code[0].opcode(), Some(Opcode::DefineInterface));
        assert_eq!(function.chunk.code[2].opcode(), Some(Opcode::DefineType));
        assert_eq!(function.chunk.code[4].opcode(), Some(Opcode::LoadNull));
        assert_eq!(function.chunk.code[5].opcode(), Some(Opcode::Return));
        assert_eq!(function.frame.general_register_count, 1);
        assert_eq!(function.chunk.resume_points[0].pc, BytecodePc(5));
        assert_eq!(function.chunk.deopts[0].resume_pc, BytecodePc(5));
        assert_eq!(function.chunk.live_registers[0].pc, BytecodePc(4));
        assert_eq!(function.chunk.switch_tables[0].default_target, BytecodePc(4));
        assert_eq!(function.chunk.switch_tables[0].cases[0].target, BytecodePc(5));
    }

}
