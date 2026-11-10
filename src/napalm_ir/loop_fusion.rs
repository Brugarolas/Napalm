//! Semantics-preserving fusion of adjacent SSA counted-reduction loops.
//!
//! Accepts two canonical single-body loops whose headers compare Int32
//! induction parameters against an identical, loop-invariant bound, with the
//! same initial index, comparison and constant nonzero stride. The loops must
//! be adjacent and have no observable effects,
//! calls, guards, deopts, safepoints, or external dependencies on the first loop.
//!
//! The second loop's side-effect-free body is moved into the first loop after
//! the original first body. Its loop-carried SSA state becomes additional block
//! parameters of the first header; the second header is reduced to a jump that
//! forwards the fused loop's final state to the original continuation.
//!
//! No new opcode or bytecode lowering support is required. This deliberately
//! rejects memory-writing loops: without alias, bounds and exception proofs,
//! interleaving even two ListSets can change observable behavior. Pure ListGet
//! reads are supported when both loops scan one statically typed List over
//! `[0, ListLen(list))` with no intervening mutations.

use crate::napalm_ir::{
    BlockId, CmpKind, Constant, Edge, Function, Inst, InstId, IntBinOp,
    Type, Value, ValueDef, ValueId,
};
use crate::napalm_ir_optimizer::{repair_effect_chain, FunctionOptimization};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoopFusionStats {
    pub fused_pairs: usize,
    pub merged_body_instructions: usize,
    pub merged_carried_values: usize,
}

#[derive(Default)]
pub struct LoopFusion {
    pub stats: LoopFusionStats,
}

struct Plan {
    preheader: usize,
    first_header: usize,
    first_body: usize,
    gap: Option<usize>,
    gap_preparation: Vec<InstId>,
    second_header: usize,
    second_body: usize,
    first_index: usize,
    second_index: usize,
    second_initial: Vec<ValueId>,
    second_next: Vec<ValueId>,
    second_exit: Edge,
    moved: Vec<InstId>,
}

impl FunctionOptimization for LoopFusion {
    fn optimize_function(&mut self, function: &mut Function) -> bool {
        let mut changed = false;
        loop {
            let mut selected = None;
            let preds = predecessors(function);
            for header in 0..function.blocks.len() {
                if let Some(plan) = find_plan(function, header, &preds) {
                    let counts = (plan.moved.len(), plan.second_initial.len().saturating_sub(1));
                    let mut candidate = function.clone();
                    apply_plan(&mut candidate, plan);
                    if crate::napalm_bytecodecompiler::check_register_allocation(&candidate).is_ok() {
                        selected = Some((candidate, counts));
                        break;
                    }
                }
            }
            let Some((candidate, (instructions, carried))) = selected else { break };
            self.stats.fused_pairs += 1;
            self.stats.merged_body_instructions += instructions;
            self.stats.merged_carried_values += carried;
            *function = candidate;
            changed = true;
        }
        changed
    }
}

fn predecessors(function: &Function) -> Vec<Vec<usize>> {
    let mut preds = vec![Vec::<usize>::new(); function.blocks.len()];
    for (block_index, block) in function.blocks.iter().enumerate() {
        let Some(term) = block.insts.last().and_then(|id| function.node(*id)) else { continue };
        for edge in term.kind.edges() {
            if let Some(entry) = preds.get_mut(edge.target.0 as usize) {
                if !entry.contains(&block_index) {
                    entry.push(block_index);
                }
            }
        }
    }
    for region in &function.exception_regions {
        for block in &region.protected_blocks {
            let entry = &mut preds[region.handler.0 as usize];
            let source = block.0 as usize;
            if !entry.contains(&source) { entry.push(source); }
        }
    }
    preds
}

fn is_handler(function: &Function, block: usize) -> bool {
    let id = function.blocks[block].id;
    function.exception_regions.iter().any(|region| {
        region.handler == id
    })
}

fn terminator(function: &Function, block: usize) -> Option<&Inst> {
    function.blocks[block].insts.last().and_then(|id| function.node(*id)).map(|node| &node.kind)
}

fn cmp_header(function: &Function, block: usize) -> Option<(CmpKind, ValueId, ValueId, Edge, Edge)> {
    let ids = &function.blocks[block].insts;
    let &[cmp, branch] = ids.as_slice() else { return None };
    let Inst::CmpInt { kind, left, right } = &function.node(cmp)?.kind else { return None };
    let [result] = function.node(cmp)?.results.as_slice() else { return None };
    let Inst::Branch { cond, then_edge, else_edge } = &function.node(branch)?.kind else { return None };
    if *result != *cond {
        return None;
    }
    Some((*kind, *left, *right, then_edge.clone(), else_edge.clone()))
}

pub(super) fn i32_literal(function: &Function, mut value: ValueId) -> Option<i32> {
    for _ in 0..function.values.len() {
        let ValueDef::InstResult(id, 0) = function.value(value)?.def else { return None };
        match &function.node(id)?.kind {
            Inst::Const(Constant::Int32(x)) => return Some(*x),
            Inst::Move(next) => value = *next,
            _ => return None,
        }
    }
    None
}

fn equal_scalar(function: &Function, a: ValueId, b: ValueId) -> bool {
    a == b || i32_literal(function, a).zip(i32_literal(function, b)).is_some_and(|(a, b)| a == b)
}

fn available_at_preheader(function: &Function, value: ValueId, preheader: usize) -> bool {
    let Some(value) = function.value(value) else { return false };
    let (owner, instruction) = match value.def {
        ValueDef::CaptureParam(_) => return true,
        ValueDef::BlockParam(block, _) => (block.0 as usize, false),
        ValueDef::InstResult(id, _) => match function.blocks.iter().position(|block| block.insts.contains(&id)) {
            Some(owner) => (owner, true),
            None => return false,
        },
    };
    // Walk every incoming path, including exceptional control flow. An
    // instruction in a protected block is not necessarily executed before
    // that block throws into a handler; block parameters, however, are set.
    let preds = predecessors(function);
    let mut pending = vec![preheader];
    let mut visited = vec![false; function.blocks.len()];
    let mut reaches_owner = false;
    while let Some(block) = pending.pop() {
        if block == owner { reaches_owner = true; continue; }
        if visited[block] { continue; }
        visited[block] = true;
        if block == function.entry.0 as usize || preds[block].is_empty() { return false; }
        if instruction && function.exception_regions.iter().any(|region| {
            region.handler.0 as usize == block
                && region.protected_blocks.contains(&BlockId(owner as u32))
        }) { return false; }
        pending.extend_from_slice(&preds[block]);
    }
    reaches_owner
}

fn constant_stride(function: &Function, index: ValueId, next: ValueId) -> Option<i32> {
    let Some(Value { def: ValueDef::InstResult(id, 0), .. }) = function.value(next) else {
        return None;
    };
    match function.node(*id).map(|node| &node.kind) {
        Some(Inst::IntBin { op: IntBinOp::Add, left, right }) => {
            if *left == index { i32_literal(function, *right) }
            else if *right == index { i32_literal(function, *left) }
            else { None }
        }
        Some(Inst::IntBin { op: IntBinOp::Sub, left, right }) if *left == index => {
            i32_literal(function, *right).map(i32::wrapping_neg)
        }
        _ => None,
    }
}

fn safe_body_instruction(
    function: &Function,
    kind: &Inst,
    induction: ValueId,
    bounded_list: Option<ValueId>,
) -> bool {
    match kind {
        Inst::ListGet { list, index } => bounded_list == Some(*list) && *index == induction,
        _ => super::deforestation::movable_computation(function, kind),
    }
}

fn body_is_pure(
    function: &Function,
    block: usize,
    induction: ValueId,
    bounded_list: Option<ValueId>,
) -> bool {
    let ids = &function.blocks[block].insts;
    ids.len() > 1 && ids[..ids.len() - 1].iter().all(|&id| {
        let node = &function.nodes[id.0 as usize];
        safe_body_instruction(
            function, &node.kind, induction, bounded_list,
        )
    })
}

/// A moved operand is either a loop-carried parameter of the second header,
/// a local instruction result already produced in that body, or a value that
/// exists before entering the first loop. Nothing from the first loop's current
/// or completed iterations may be read by the moved body.
/// Recognize a proven in-bounds read-only List iteration, so two scalar
/// reductions of the same List can be fused without reordering exceptions.
fn bounded_list(function: &Function, bound: ValueId, start: ValueId, preheader: usize) -> Option<ValueId> {
    if i32_literal(function, start) != Some(0) { return None; }
    let ValueDef::InstResult(id, 0) = function.value(bound)?.def else { return None };
    let Inst::ListLen { list } = &function.node(id)?.kind else { return None };
    if !matches!(function.value(*list).map(|value| &value.ty), Some(Type::List(_)))
        || !available_at_preheader(function, *list, preheader)
    {
        return None;
    }
    Some(*list)
}

fn body_inputs_available(
    function: &Function,
    body: usize,
    second_header: usize,
    preheader: usize,
    available: &[ValueId],
) -> bool {
    let mut seen = Vec::<ValueId>::new();
    // Include the backedge arguments: they become operands of the first loop
    // latch and must also dominate it after the body is relocated.
    for &id in &function.blocks[body].insts {
        let node = &function.nodes[id.0 as usize];
        for operand in node.kind.operands() {
            if seen.contains(&operand) || available.contains(&operand) {
                continue;
            }
            match function.value(operand).map(|value| value.def) {
                Some(ValueDef::BlockParam(owner, _)) if owner.0 as usize == second_header => {}
                _ if available_at_preheader(function, operand, preheader) => {}
                _ => return false,
            }
        }
        seen.extend_from_slice(&node.results);
    }
    true
}

fn body_results_do_not_escape(function: &Function, body: usize) -> bool {
    let ids = &function.blocks[body].insts;
    let mut results = Vec::new();
    for &id in &ids[..ids.len().saturating_sub(1)] {
        results.extend_from_slice(&function.nodes[id.0 as usize].results);
    }
    if results.is_empty() { return true; }
    if function.blocks.iter().enumerate().any(|(other, block)| {
        other != body && block.insts.iter().any(|id| {
            function.nodes[id.0 as usize].kind.operands().iter().any(|operand| results.contains(operand))
        })
    }) {
        return false;
    }
    !function.deopts.iter().any(|deopt| {
        deopt.snapshot.slots.iter().any(|slot| results.contains(&slot.value))
            || deopt.snapshot.inlined_frames.iter().any(|frame| frame.slots.iter().any(|slot| results.contains(&slot.value)))
    })
}

fn find_plan(function: &Function, first_header: usize, preds: &[Vec<usize>]) -> Option<Plan> {
    let (cmp1, i1, bound1, first_body_edge, first_exit_edge) = cmp_header(function, first_header)?;
    let first_body = first_body_edge.target.0 as usize;
    let fallthrough = first_exit_edge.target.0 as usize;
    if first_body == fallthrough || first_body == first_header || fallthrough == first_header
        || !first_body_edge.args.is_empty()
        || !function.blocks.get(first_body)?.params.is_empty()
    {
        return None;
    }
    // `ControlFlowCleanup` threads empty Jump blocks before this pass runs.
    // Accept an empty gap, a gap containing independent total initialization,
    // or a direct H1 -> H2 edge. Initialization is hoisted before both loops.
    let (gap, second_start) = if function.blocks.get(fallthrough)?.params.is_empty()
        && first_exit_edge.args.is_empty()
    {
        match terminator(function, fallthrough)? {
            Inst::Jump { edge } => (Some(fallthrough), edge.clone()),
            _ => (None, first_exit_edge.clone()),
        }
    } else {
        (None, first_exit_edge.clone())
    };
    let second_header = second_start.target.0 as usize;
    if second_header == first_header || second_header == first_body {
        return None;
    }
    let (cmp2, i2, bound2, second_body_edge, second_exit) = cmp_header(function, second_header)?;
    let second_body = second_body_edge.target.0 as usize;
    if second_body == second_header || Some(second_body) == gap
        || second_body == first_body || second_body == first_header
        || !second_body_edge.args.is_empty()
        || !function.blocks.get(second_body)?.params.is_empty()
    {
        return None;
    }
    let first_params = &function.blocks[first_header].params;
    let second_params = &function.blocks[second_header].params;
    let first_index = first_params.iter().position(|value| *value == i1)?;
    let second_index = second_params.iter().position(|value| *value == i2)?;
    if !matches!(function.value(i1).map(|v| &v.ty), Some(Type::Int32))
        || !matches!(function.value(i2).map(|v| &v.ty), Some(Type::Int32))
        || !equal_scalar(function, bound1, bound2)
        || cmp1 != cmp2
    {
        return None;
    }
    let Inst::Jump { edge: first_back } = terminator(function, first_body)? else { return None };
    let Inst::Jump { edge: second_back } = terminator(function, second_body)? else { return None };
    if first_back.target.0 as usize != first_header
        || second_back.target.0 as usize != second_header
        || first_back.args.len() != first_params.len()
        || second_back.args.len() != second_params.len()
        || second_start.args.len() != second_params.len()
    {
        return None;
    }
    let stride = constant_stride(function, i1, first_back.args[first_index])?;
    if stride == 0 || constant_stride(function, i2, second_back.args[second_index]) != Some(stride) {
        return None;
    }

    let candidates = preds.get(first_header)?;
    if candidates.len() != 2 || !candidates.contains(&first_body) { return None; }
    let preheader = *candidates.iter().find(|&&id| id != first_body)?;
    if preds.get(first_body)?.as_slice() != [first_header]
        || gap.is_some_and(|gap| preds.get(gap).is_none_or(|parents| parents.as_slice() != [first_header]))
        || preds.get(second_header)?.len() != 2
        || !preds[second_header].contains(&gap.unwrap_or(first_header))
        || !preds[second_header].contains(&second_body)
        || preds.get(second_body)?.as_slice() != [second_header]
    {
        return None;
    }
    let Inst::Jump { edge: first_start } = terminator(function, preheader)? else { return None };
    let gap_preparation = if let Some(gap) = gap {
        let ids = &function.blocks[gap].insts;
        if !ids[..ids.len() - 1].iter().all(|&id| {
            super::deforestation::movable_computation(function, &function.nodes[id.0 as usize].kind)
        }) || !body_inputs_available(function, gap, gap, preheader, &[]) {
            return None;
        }
        ids[..ids.len() - 1].to_vec()
    } else { Vec::new() };
    let gap_values: Vec<_> = gap_preparation.iter()
        .flat_map(|id| function.nodes[id.0 as usize].results.iter().copied()).collect();
    if first_start.target.0 as usize != first_header
        || first_start.args.len() != first_params.len()
        || !equal_scalar(function, first_start.args[first_index], second_start.args[second_index])
        || !available_at_preheader(function, bound1, preheader)
        || (!gap_values.contains(&bound2) && !available_at_preheader(function, bound2, preheader))
    {
        return None;
    }
    if [Some(first_header), Some(first_body), gap, Some(second_header), Some(second_body)]
        .into_iter().flatten().any(|block| is_handler(function, block))
        || function.deopts.iter().any(|deopt| {
            // Moving a block can invalidate liveness/location at an OSR/deopt
            // point. Reject any snapshots until virtual-state remapping exists.
            deopt.snapshot.slots.iter().any(|slot| {
                second_params.contains(&slot.value)
                    || first_params.contains(&slot.value)
            }) || deopt.snapshot.inlined_frames.iter().any(|frame| frame.slots.iter().any(|slot| {
                second_params.contains(&slot.value) || first_params.contains(&slot.value)
            }))
        })
    {
        return None;
    }
    let read_list = if cmp1 == CmpKind::Lt && stride == 1 {
        bounded_list(function, bound1, first_start.args[first_index], preheader)
    } else { None };
    if !body_is_pure(function, first_body, i1, read_list)
        || !body_is_pure(function, second_body, i2, read_list)
        || !body_inputs_available(function, second_body, second_header, preheader, &gap_values)
        || !body_results_do_not_escape(function, second_body)
    {
        return None;
    }
    // Initial reduction values must be available before either loop, not a
    // result from the preceding loop. All remaining state is passed through SSA.
    if second_start.args.iter().any(|arg| !gap_values.contains(arg) && !available_at_preheader(function, *arg, preheader)) {
        return None;
    }
    let moved = function.blocks[second_body].insts[..function.blocks[second_body].insts.len() - 1].to_vec();
    Some(Plan {
        preheader, first_header, first_body, gap, gap_preparation, second_header, second_body,
        first_index, second_index,
        second_initial: second_start.args.clone(),
        second_next: second_back.args.clone(),
        second_exit, moved,
    })
}

fn replace_param(operand: &mut ValueId, from: &[ValueId], to: &[ValueId]) {
    if let Some(index) = from.iter().position(|value| value == operand) {
        *operand = to[index];
    }
}

fn apply_plan(function: &mut Function, plan: Plan) {
    let first_header_id = BlockId(plan.first_header as u32);
    let second_header_id = BlockId(plan.second_header as u32);
    let second_params = function.blocks[plan.second_header].params.clone();
    let first_params = function.blocks[plan.first_header].params.clone();
    let mut mapped = Vec::with_capacity(second_params.len());
    for (index, &param) in second_params.iter().enumerate() {
        if index == plan.second_index {
            mapped.push(first_params[plan.first_index]);
        } else {
            let result = ValueId(function.values.len() as u32);
            let param_index = function.blocks[plan.first_header].params.len();
            let ty = function.values[param.0 as usize].ty.clone();
            function.values.push(Value {
                id: result,
                ty,
                def: ValueDef::BlockParam(first_header_id, param_index as u16),
            });
            function.blocks[plan.first_header].params.push(result);
            mapped.push(result);
        }
    }

    // All second-body instructions are pure, and all their external inputs have
    // been proven available before the first loop. Rebind only loop-carried SSA
    // operands; local SSA result identities remain stable as nodes are moved.
    for &id in &plan.moved {
        function.nodes[id.0 as usize].kind.for_each_operand_mut(|operand| {
            replace_param(operand, &second_params, &mapped);
        });
    }
    let mut back_values = plan.second_next;
    for arg in &mut back_values { replace_param(arg, &second_params, &mapped); }

    // New entry and latch edges carry each independent second-loop accumulator.
    let initial_values = plan.second_initial;
    let old_first_back = *function.blocks[plan.first_body].insts.last().expect("body jump");
    let old_first_start = *function.blocks[plan.preheader].insts.last().expect("preheader jump");
    if let Inst::Jump { edge } = &mut function.nodes[old_first_start.0 as usize].kind {
        edge.args.extend(initial_values.iter().enumerate().filter_map(|(ix, value)| {
            (ix != plan.second_index).then_some(*value)
        }));
    }
    if let Inst::Jump { edge } = &mut function.nodes[old_first_back.0 as usize].kind {
        edge.args.extend(back_values.iter().enumerate().filter_map(|(ix, value)| {
            (ix != plan.second_index).then_some(*value)
        }));
    }

    let first_branch = *function.blocks[plan.first_header].insts.last().expect("first branch");
    if let Inst::Branch { else_edge, .. } = &mut function.nodes[first_branch.0 as usize].kind {
        else_edge.target = second_header_id;
        else_edge.args = mapped;
    }

    let second_branch = *function.blocks[plan.second_header].insts.last().expect("second branch");
    function.nodes[second_branch.0 as usize].kind = Inst::Jump { edge: plan.second_exit };

    // Transfer the second body without duplicating InstIds. The old backedge
    // remains only as an unreachable terminator of the now-unreachable block.
    let old_second_back = *function.blocks[plan.second_body].insts.last().expect("second backedge");
    function.blocks[plan.second_body].insts = vec![old_second_back];
    function.nodes[old_second_back.0 as usize].kind = Inst::Unreachable;
    let insert_at = function.blocks[plan.first_body].insts.len() - 1;
    function.blocks[plan.first_body].insts.splice(insert_at..insert_at, plan.moved);

    // The old gap is unreachable. Prevent it from contributing a stale initial
    // state through the second header after the fallthrough redirect.
    if let Some(gap) = plan.gap {
        let old_gap_jump = *function.blocks[gap].insts.last().expect("gap jump");
        function.blocks[gap].insts = vec![old_gap_jump];
        function.nodes[old_gap_jump.0 as usize].kind = Inst::Unreachable;
        let insert_at = function.blocks[plan.preheader].insts.len() - 1;
        function.blocks[plan.preheader].insts.splice(insert_at..insert_at, plan.gap_preparation);
    }

    for index in [Some(plan.preheader), Some(plan.first_header), Some(plan.first_body),
        plan.gap, Some(plan.second_header), Some(plan.second_body)].into_iter().flatten()
    {
        repair_effect_chain(function, index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::napalm_ir::{Builder, F64BinOp, FunctionKind, Program};

    fn two_reductions(second_has_observable_effect: bool) -> (
        Program, crate::napalm_ir::FuncId, BlockId, BlockId, BlockId,
    ) {
        let mut program = Program::new();
        let module = program.add_module("fusion", 1);
        let mut builder = Builder::new(
            &mut program,
            module,
            "two_reductions",
            FunctionKind::ModuleBody,
            vec![],
            Type::Num,
        );
        let bound = builder.const_i32(5);
        let zero_index = builder.const_i32(0);
        let one = builder.const_i32(1);
        let zero_number = builder.const_num(0.0);
        let one_number = builder.const_num(1.0);
        let two_number = builder.const_num(2.0);
        let zero_acc = builder.emit1(Inst::NumToF64 { value: zero_number }, Type::F64);
        let one_float = builder.emit1(Inst::NumToF64 { value: one_number }, Type::F64);
        let two_float = builder.emit1(Inst::NumToF64 { value: two_number }, Type::F64);
        let first_header = builder.new_block(&[Type::Int32, Type::F64]);
        let first_body = builder.new_block(&[]);
        let gap = builder.new_block(&[]);
        let second_header = builder.new_block(&[Type::Int32, Type::F64]);
        let second_body = builder.new_block(&[]);
        let exit = builder.new_block(&[]);
        builder.jump(first_header, vec![zero_index, zero_acc]);

        builder.position_at(first_header);
        let i = builder.block_param(first_header, 0);
        let first_acc = builder.block_param(first_header, 1);
        let first_cond = builder.emit1(Inst::CmpInt { kind: CmpKind::Lt, left: i, right: bound }, Type::Bool);
        builder.branch(first_cond, first_body, vec![], gap, vec![]);
        builder.position_at(first_body);
        let first_sum = builder.emit1(Inst::F64Bin { op: F64BinOp::Add, left: first_acc, right: one_float }, Type::F64);
        let next_i = builder.emit1(Inst::IntBin { op: IntBinOp::Add, left: i, right: one }, Type::Int32);
        builder.jump(first_header, vec![next_i, first_sum]);

        builder.position_at(gap);
        builder.jump(second_header, vec![zero_index, zero_acc]);
        builder.position_at(second_header);
        let j = builder.block_param(second_header, 0);
        let second_acc = builder.block_param(second_header, 1);
        let second_cond = builder.emit1(Inst::CmpInt { kind: CmpKind::Lt, left: j, right: bound }, Type::Bool);
        builder.branch(second_cond, second_body, vec![], exit, vec![]);

        builder.position_at(second_body);
        let next_acc = builder.emit1(Inst::F64Bin { op: F64BinOp::Add, left: second_acc, right: two_float }, Type::F64);
        if second_has_observable_effect {
            builder.emit0(Inst::StoreModuleVar {
                module,
                slot: crate::napalm_ir::ModuleSlot(0),
                value: next_acc,
            });
        }
        let next_j = builder.emit1(Inst::IntBin { op: IntBinOp::Add, left: j, right: one }, Type::Int32);
        builder.jump(second_header, vec![next_j, next_acc]);
        builder.position_at(exit);
        let total = builder.emit1(Inst::F64Bin { op: F64BinOp::Add, left: first_acc, right: second_acc }, Type::F64);
        let result = builder.emit1(Inst::F64ToNum { value: total }, Type::Num);
        builder.ret(result);
        let id = builder.finish();
        (program, id, first_header, first_body, second_header)
    }

    #[test]
    fn fuses_independent_reductions_and_preserves_ssa_edges() {
        let (mut program, id, first_header, first_body, second_header) = two_reductions(false);
        let function = &mut program.functions[id.0 as usize];
        crate::napalm_ir::verify_function(function).unwrap();
        let body_before = function.blocks[first_body.0 as usize].insts.len();
        let mut pass = LoopFusion::default();
        assert!(pass.optimize_function(function));
        assert_eq!(pass.stats.fused_pairs, 1);
        assert_eq!(pass.stats.merged_carried_values, 1);
        assert!(function.blocks[first_body.0 as usize].insts.len() > body_before);
        assert_eq!(function.blocks[first_header.0 as usize].params.len(), 3);
        assert!(matches!(terminator(function, second_header.0 as usize), Some(Inst::Jump { .. })));
        crate::napalm_ir::verify_function(function).unwrap();
        assert!(!pass.optimize_function(function));
    }

    fn two_list_reductions(second_reads_other_list: bool) -> (Program, crate::napalm_ir::FuncId) {
        let mut program = Program::new();
        let module = program.add_module("list_fusion", 0);
        let mut builder = Builder::new(
            &mut program, module, "two_list_reductions", FunctionKind::Closure,
            vec![Type::List(Some(Box::new(Type::F64))), Type::List(Some(Box::new(Type::F64)))],
            Type::F64,
        );
        let list = builder.param(0);
        let other = builder.param(1);
        let bound = builder.emit1(Inst::ListLen { list }, Type::Int32);
        let zero = builder.const_i32(0);
        let one = builder.const_i32(1);
        let number = builder.const_num(0.0);
        let empty = builder.emit1(Inst::NumToF64 { value: number }, Type::F64);
        let h1 = builder.new_block(&[Type::Int32, Type::F64]);
        let b1 = builder.new_block(&[]);
        let gap = builder.new_block(&[]);
        let h2 = builder.new_block(&[Type::Int32, Type::F64]);
        let b2 = builder.new_block(&[]);
        let exit = builder.new_block(&[]);
        builder.jump(h1, vec![zero, empty]);
        builder.position_at(h1);
        let i = builder.block_param(h1, 0);
        let a = builder.block_param(h1, 1);
        let check = builder.emit1(Inst::CmpInt { kind: CmpKind::Lt, left: i, right: bound }, Type::Bool);
        builder.branch(check, b1, vec![], gap, vec![]);
        builder.position_at(b1);
        let read = builder.emit1(Inst::ListGet { list, index: i }, Type::F64);
        let next_a = builder.emit1(Inst::F64Bin { op: F64BinOp::Add, left: a, right: read }, Type::F64);
        let step = builder.emit1(Inst::IntBin { op: IntBinOp::Add, left: i, right: one }, Type::Int32);
        builder.jump(h1, vec![step, next_a]);
        builder.position_at(gap);
        builder.jump(h2, vec![zero, empty]);
        builder.position_at(h2);
        let j = builder.block_param(h2, 0);
        let b = builder.block_param(h2, 1);
        let check = builder.emit1(Inst::CmpInt { kind: CmpKind::Lt, left: j, right: bound }, Type::Bool);
        builder.branch(check, b2, vec![], exit, vec![]);
        builder.position_at(b2);
        let read = builder.emit1(Inst::ListGet { list: if second_reads_other_list { other } else { list }, index: j }, Type::F64);
        let next_b = builder.emit1(Inst::F64Bin { op: F64BinOp::Add, left: b, right: read }, Type::F64);
        let step = builder.emit1(Inst::IntBin { op: IntBinOp::Add, left: j, right: one }, Type::Int32);
        builder.jump(h2, vec![step, next_b]);
        builder.position_at(exit);
        let result = builder.emit1(Inst::F64Bin { op: F64BinOp::Add, left: a, right: b }, Type::F64);
        builder.ret(result);
        let id = builder.finish();
        (program, id)
    }

    #[test]
    fn fuses_after_cfg_cleanup_threads_the_gap() {
        use crate::napalm_ir_optimizing::ControlFlowCleanup;
        let (mut program, id, first_header, _, second_header) = two_reductions(false);
        let function = &mut program.functions[id.0 as usize];
        assert!(ControlFlowCleanup::default().optimize_function(function));
        let head_term = terminator(function, first_header.0 as usize).unwrap();
        assert!(matches!(head_term, Inst::Branch { else_edge, .. } if else_edge.target == second_header));
        assert!(LoopFusion::default().optimize_function(function));
        crate::napalm_ir::verify_function(function).unwrap();
    }

    #[test]
    fn fuses_proven_in_bounds_read_only_list_reductions() {
        let (mut program, id) = two_list_reductions(false);
        let function = &mut program.functions[id.0 as usize];
        assert!(LoopFusion::default().optimize_function(function));
        crate::napalm_ir::verify_function(function).unwrap();
    }

    #[test]
    fn does_not_interleave_list_reads_with_unproven_bounds() {
        let (mut program, id) = two_list_reductions(true);
        let function = &mut program.functions[id.0 as usize];
        assert!(!LoopFusion::default().optimize_function(function));
        crate::napalm_ir::verify_function(function).unwrap();
    }

    #[test]
    fn refuses_to_interleave_observable_writes() {
        let (mut program, id, _, _, _) = two_reductions(true);
        let function = &mut program.functions[id.0 as usize];
        assert!(!LoopFusion::default().optimize_function(function));
        crate::napalm_ir::verify_function(function).unwrap();
    }

    fn run_module(program: &Program, entry: crate::napalm_ir::FuncId) -> f64 {
        use crate::napalm_bytecodecompiler::BytecodeCompiler;
        use crate::napalm_interpreter::{ExecutionResult, Interpreter};
        let code = BytecodeCompiler::default().compile(program).unwrap();
        let mut vm = Interpreter::new(&code).unwrap();
        vm.set_jit_options(None);
        let ExecutionResult::Returned(value) = vm.run_module(entry).unwrap() else {
            panic!("module did not return");
        };
        value.as_num().expect("expected numeric result")
    }

    #[test]
    fn fused_and_unfused_loops_execute_identically_in_the_vm() {
        let (unoptimized, original_id, _, _, _) = two_reductions(false);
        let before = run_module(&unoptimized, original_id);
        let (mut optimized, optimized_id, _, _, _) = two_reductions(false);
        assert!(LoopFusion::default().optimize_function(
            &mut optimized.functions[optimized_id.0 as usize]
        ));
        crate::napalm_ir::verify_function(&optimized.functions[optimized_id.0 as usize]).unwrap();
        let after = run_module(&optimized, optimized_id);
        assert_eq!(before, 15.0);
        assert_eq!(after, before);
    }

    #[test]
    fn refuses_different_iteration_bounds() {
        let (mut program, id, first_header, _, _) = two_reductions(false);
        let function = &mut program.functions[id.0 as usize];
        // Replace the first header's loop bound with an independent constant,
        // so the iteration spaces are no longer provably identical.
        let cmp_id = function.blocks[first_header.0 as usize].insts[0];
        let index = match &function.nodes[cmp_id.0 as usize].kind {
            Inst::CmpInt { left, .. } => *left,
            _ => panic!("expected comparison"),
        };
        if let Inst::CmpInt { right, .. } = &mut function.nodes[cmp_id.0 as usize].kind {
            *right = index;
        }
        assert!(!LoopFusion::default().optimize_function(function));
        crate::napalm_ir::verify_function(function).unwrap();
    }

    fn integer_reductions(
        carried: usize, work: usize, comparison: CmpKind,
        start: i32, limit: i32, stride: i32, protected: bool, first_temporaries: usize,
    ) -> (Program, crate::napalm_ir::FuncId) {
        let mut program = Program::new();
        let module = program.add_module("integer_fusion", 0);
        let mut b = Builder::new(
            &mut program, module, "integer_fusion", FunctionKind::ModuleBody, vec![], Type::Num,
        );
        let bound = b.const_i32(limit);
        let initial = b.const_i32(start);
        let zero = b.const_i32(0);
        let one = b.const_i32(1);
        let step = b.const_i32(stride.abs());
        let first_carried = if first_temporaries == 0 { carried } else { 1 };
        // Definitions are in an ancestor of the immediate preheader.
        let preheader = b.new_block(&[]);
        let h1 = b.new_block(&vec![Type::Int32; first_carried + 1]);
        let b1 = b.new_block(&[]);
        let gap = b.new_block(&[]);
        let h2 = b.new_block(&vec![Type::Int32; carried + 1]);
        let b2 = b.new_block(&[]);
        let exit = b.new_block(&[]);
        let handler = b.new_block(&[Type::Top]);
        if protected {
            let error = b.block_param(handler, 0);
            b.add_exception_region(vec![BlockId(0), preheader, h1, b1, gap, h2, b2, exit], handler, error, 0);
        }
        b.jump(preheader, vec![]);
        b.position_at(preheader);
        let mut args = vec![zero; first_carried + 1];
        args[0] = initial;
        b.jump(h1, args);
        let mut final_values = Vec::new();
        for (header, body, next, count) in [(h1, b1, gap, first_carried), (h2, b2, exit, carried)] {
            if header == h2 {
                b.position_at(gap);
                // Independent initialization between loops must be hoisted.
                let second_zero = b.const_i32(0);
                let second_initial = b.const_i32(start);
                let mut args = vec![second_zero; count + 1];
                args[0] = second_initial;
                b.jump(h2, args);
            }
            b.position_at(header);
            let index = b.block_param(header, 0);
            let cond = b.emit1(Inst::CmpInt { kind: comparison, left: index, right: bound }, Type::Bool);
            b.branch(cond, body, vec![], next, vec![]);
            b.position_at(body);
            let mut temporary_values = Vec::new();
            if header == h1 {
                for offset in 0..first_temporaries {
                    let constant = b.const_i32(offset as i32);
                    temporary_values.push(b.emit1(Inst::IntBin {
                        op: IntBinOp::Add, left: index, right: constant,
                    }, Type::Int32));
                }
            }
            let mut values = Vec::new();
            for accumulator in 1..=count {
                let param = b.block_param(header, accumulator);
                final_values.push(param);
                let mut value = b.emit1(Inst::IntBin { op: IntBinOp::Add, left: param, right: index }, Type::Int32);
                for _ in 0..work {
                    value = b.emit1(Inst::IntBin { op: IntBinOp::Add, left: value, right: one }, Type::Int32);
                }
                for &temporary in &temporary_values {
                    value = b.emit1(Inst::IntBin { op: IntBinOp::Add, left: value, right: temporary }, Type::Int32);
                }
                values.push(value);
            }
            let next_index = b.emit1(Inst::IntBin {
                op: if stride > 0 { IntBinOp::Add } else { IntBinOp::Sub }, left: index, right: step,
            }, Type::Int32);
            values.insert(0, next_index);
            b.jump(header, values);
        }
        b.position_at(exit);
        let mut sum = zero;
        for value in final_values {
            sum = b.emit1(Inst::IntBin { op: IntBinOp::Add, left: sum, right: value }, Type::Int32);
        }
        let result = b.emit1(Inst::Int32ToNum { value: sum }, Type::Num);
        b.ret(result);
        b.position_at(handler);
        let failed = b.const_num(-999.0);
        b.ret(failed);
        let entry = b.finish();
        (program, entry)
    }

    #[test]
    fn fuses_large_integer_loops_in_try_with_dominating_and_gap_initialization() {
        let (mut program, entry) = integer_reductions(8, 8, CmpKind::Lt, 0, 5, 1, true, 0);
        let before = run_module(&program, entry);
        assert_eq!(before, 800.0);
        let mut pass = LoopFusion::default();
        assert!(pass.optimize_function(&mut program.functions[entry.0 as usize]));
        assert_eq!(pass.stats.fused_pairs, 1);
        assert_eq!(pass.stats.merged_carried_values, 8);
        crate::napalm_ir::verify_function(&program.functions[entry.0 as usize]).unwrap();
        assert_eq!(run_module(&program, entry), before);
    }

    #[test]
    fn fuses_descending_inclusive_strided_and_empty_iteration_spaces() {
        for (comparison, start, bound, stride, expected) in [
            (CmpKind::Gt, 5, 0, -1, 40.0),
            (CmpKind::Ge, 4, 0, -2, 18.0),
            (CmpKind::Le, 0, 4, 2, 18.0),
            (CmpKind::Ne, 0, 5, 1, 30.0),
            (CmpKind::Lt, 5, 0, 1, 0.0),
        ] {
            let (mut program, entry) = integer_reductions(1, 1, comparison, start, bound, stride, false, 0);
            assert_eq!(run_module(&program, entry), expected);
            assert!(LoopFusion::default().optimize_function(&mut program.functions[entry.0 as usize]));
            crate::napalm_ir::verify_function(&program.functions[entry.0 as usize]).unwrap();
            assert_eq!(run_module(&program, entry), expected);
        }
    }

    #[test]
    fn retains_separate_loops_when_fusion_would_exhaust_bytecode_registers() {
        let (mut program, entry) = integer_reductions(100, 1, CmpKind::Lt, 0, 5, 1, false, 180);
        let before = run_module(&program, entry);
        let function = &mut program.functions[entry.0 as usize];
        // The transformation is semantically legal, but its simultaneous live
        // values exceed the real backend limit. Keep the compilable original.
        let plan = find_plan(function, 2, &predecessors(function)).expect("legal fusion candidate");
        let mut candidate = function.clone();
        apply_plan(&mut candidate, plan);
        assert!(matches!(crate::napalm_bytecodecompiler::check_register_allocation(&candidate),
            Err(crate::napalm_bytecodecompiler::CompileError::RegisterPressure { .. })));
        assert!(!LoopFusion::default().optimize_function(function));
        assert_eq!(run_module(&program, entry), before);
    }

    #[test]
    fn exceptional_predecessor_does_not_make_late_instruction_available() {
        let mut program = Program::new();
        let module = program.add_module("availability", 0);
        let mut b = Builder::new(&mut program, module, "availability", FunctionKind::Closure, vec![], Type::Int32);
        let handler = b.new_block(&[Type::Top]);
        let error = b.block_param(handler, 0);
        let late = b.const_i32(42);
        b.ret(late);
        b.add_exception_region(vec![BlockId(0)], handler, error, 0);
        b.position_at(handler);
        let fallback = b.const_i32(0);
        b.ret(fallback);
        let entry = b.finish();
        assert!(!available_at_preheader(&program.functions[entry.0 as usize], late, handler.0 as usize));
    }
}
