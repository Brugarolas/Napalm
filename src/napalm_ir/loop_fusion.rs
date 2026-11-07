//! Semantics-preserving fusion of adjacent SSA counted-reduction loops.
//!
//! Accepts two canonical single-body loops whose headers compare Int32
//! induction parameters against an identical, loop-invariant bound, with the
//! same initial index and +1 latch. The two loops must be adjacent through an
//! empty fallthrough block and have no exceptional regions, observable effects,
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

// Bound the combined live SSA state and native register pressure.
const MAX_CARRIED_VALUES: usize = 12;
const MAX_COMBINED_BODY_NODES: usize = 96;

struct Plan {
    preheader: usize,
    first_header: usize,
    first_body: usize,
    gap: Option<usize>,
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
                    selected = Some(plan);
                    break;
                }
            }
            let Some(plan) = selected else { break };
            self.stats.fused_pairs += 1;
            self.stats.merged_body_instructions += plan.moved.len();
            self.stats.merged_carried_values += plan.second_initial.len().saturating_sub(1);
            apply_plan(function, plan);
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
    preds
}

fn protected(function: &Function, block: usize) -> bool {
    let id = function.blocks[block].id;
    function.exception_regions.iter().any(|region| {
        region.handler == id || region.protected_blocks.contains(&id)
    })
}

fn terminator(function: &Function, block: usize) -> Option<&Inst> {
    function.blocks[block].insts.last().and_then(|id| function.node(*id)).map(|node| &node.kind)
}

fn cmp_header(function: &Function, block: usize) -> Option<(ValueId, ValueId, Edge, Edge)> {
    let ids = &function.blocks[block].insts;
    let &[cmp, branch] = ids.as_slice() else { return None };
    let Inst::CmpInt { kind: CmpKind::Lt, left, right } = &function.node(cmp)?.kind else { return None };
    let [result] = function.node(cmp)?.results.as_slice() else { return None };
    let Inst::Branch { cond, then_edge, else_edge } = &function.node(branch)?.kind else { return None };
    if *result != *cond {
        return None;
    }
    Some((*left, *right, then_edge.clone(), else_edge.clone()))
}

fn i32_literal(function: &Function, mut value: ValueId) -> Option<i32> {
    for _ in 0..8 {
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
    match value.def {
        ValueDef::CaptureParam(_) => true,
        ValueDef::BlockParam(block, _) => block == function.entry || block.0 as usize == preheader,
        ValueDef::InstResult(id, _) => {
            function.blocks[preheader].insts.iter().any(|candidate| *candidate == id)
        }
    }
}

fn incremented_by_one(function: &Function, index: ValueId, next: ValueId) -> bool {
    let Some(Value { def: ValueDef::InstResult(id, 0), .. }) = function.value(next) else {
        return false;
    };
    match function.node(*id).map(|node| &node.kind) {
        Some(Inst::IntBin { op: IntBinOp::Add, left, right }) => {
            *left == index && i32_literal(function, *right) == Some(1)
        }
        _ => false,
    }
}

fn safe_body_instruction(
    kind: &Inst,
    result: Option<ValueId>,
    index_update: ValueId,
    induction: ValueId,
    bounded_list: Option<ValueId>,
) -> bool {
    match kind {
        Inst::Const(_) | Inst::Move(_) | Inst::CmpInt { .. } | Inst::CmpF64 { .. }
        | Inst::RefEq { .. } | Inst::F64Bin { .. } | Inst::F64Unary { .. }
        | Inst::Int32ToNum { .. } | Inst::F64ToNum { .. } => true,
        Inst::ListGet { list, index } => bounded_list == Some(*list) && *index == induction,
        Inst::IntBin { op, .. } => {
            matches!(op, IntBinOp::BitAnd | IntBinOp::BitOr | IntBinOp::BitXor)
                || (matches!(op, IntBinOp::Add) && result == Some(index_update))
        }
        _ => false,
    }
}

fn body_is_pure(
    function: &Function,
    block: usize,
    index_update: ValueId,
    induction: ValueId,
    bounded_list: Option<ValueId>,
) -> bool {
    let ids = &function.blocks[block].insts;
    ids.len() > 1 && ids[..ids.len() - 1].iter().all(|&id| {
        let node = &function.nodes[id.0 as usize];
        safe_body_instruction(
            &node.kind, node.results.first().copied(), index_update, induction, bounded_list,
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
) -> bool {
    let mut seen = Vec::<ValueId>::new();
    // Include the backedge arguments: they become operands of the first loop
    // latch and must also dominate it after the body is relocated.
    for &id in &function.blocks[body].insts {
        let node = &function.nodes[id.0 as usize];
        for operand in node.kind.operands() {
            if seen.contains(&operand) {
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
    let (i1, bound1, first_body_edge, first_exit_edge) = cmp_header(function, first_header)?;
    let first_body = first_body_edge.target.0 as usize;
    let fallthrough = first_exit_edge.target.0 as usize;
    if first_body == fallthrough || first_body == first_header || fallthrough == first_header
        || !first_body_edge.args.is_empty()
        || !function.blocks.get(first_body)?.params.is_empty()
    {
        return None;
    }
    // `ControlFlowCleanup` threads empty Jump blocks before this pass runs.
    // Accept both the original pre-cleanup gap and a direct H1 -> H2 edge.
    let (gap, second_start) = if function.blocks.get(fallthrough)?.params.is_empty()
        && function.blocks[fallthrough].insts.len() == 1
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
    let (i2, bound2, second_body_edge, second_exit) = cmp_header(function, second_header)?;
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
        || !incremented_by_one(function, i1, first_back.args[first_index])
        || !incremented_by_one(function, i2, second_back.args[second_index])
    {
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
    if first_start.target.0 as usize != first_header
        || first_start.args.len() != first_params.len()
        || !equal_scalar(function, first_start.args[first_index], second_start.args[second_index])
        || !available_at_preheader(function, bound1, preheader)
        || !available_at_preheader(function, bound2, preheader)
    {
        return None;
    }
    if [Some(preheader), Some(first_header), Some(first_body), gap, Some(second_header), Some(second_body)]
        .into_iter().flatten().any(|block| protected(function, block))
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
    if first_params.len() + second_params.len() > MAX_CARRIED_VALUES
        || function.blocks[first_body].insts.len() + function.blocks[second_body].insts.len() > MAX_COMBINED_BODY_NODES
    {
        return None;
    }
    let read_list = bounded_list(function, bound1, first_start.args[first_index], preheader);
    if !body_is_pure(function, first_body, first_back.args[first_index], i1, read_list)
        || !body_is_pure(function, second_body, second_back.args[second_index], i2, read_list)
        || !body_inputs_available(function, second_body, second_header, preheader)
        || !body_results_do_not_escape(function, second_body)
    {
        return None;
    }
    // Initial reduction values must be available before either loop, not a
    // result from the preceding loop. All remaining state is passed through SSA.
    if second_start.args.iter().any(|arg| !available_at_preheader(function, *arg, preheader)) {
        return None;
    }
    let moved = function.blocks[second_body].insts[..function.blocks[second_body].insts.len() - 1].to_vec();
    Some(Plan {
        preheader, first_header, first_body, gap, second_header, second_body,
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
        function.nodes[old_gap_jump.0 as usize].kind = Inst::Unreachable;
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
}
