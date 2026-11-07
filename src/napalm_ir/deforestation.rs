//! Conservative, effect-preserving deforestation of short-lived List builders.
//!
//! The frontend emits concrete `AllocList`, `ListAppend` and `ListSpread` operations
//! even when the intermediate list is only used to append into another newly
//! allocated list. This pass removes that intermediate allocation and its copy.
//!
//! Supported producers, in one basic block:
//!   * `AllocList(None); ListAppend(tmp, value)*`
//!   * `AllocList(None); ListSpread(tmp, known_list); ListAppend(tmp, value)*`
//! followed by exactly one `ListSpread(dst, tmp)` consumer.
//!
//! The destination must be a different fresh, uncontracted list. Intermediate
//! instructions must be total, effect-free computations; no invokes, guards,
//! safepoints, handlers or observable aliases are permitted. All writes are
//! delayed to the original consumer site, preserving evaluation-before-write
//! ordering. The source's spread is emitted exactly once, before appended values.
//!
//! The producer ListSpread result is replaced with `true` only when its source
//! has a guaranteed List semantic type; an arbitrary iterable is never folded.
//! All other escape/alias patterns deliberately fall back to the original IR.

use crate::napalm_ir::{
    Constant, Function, Inst, InstId, Type, ValueDef, ValueId,
};
use crate::napalm_ir_optimizer::{repair_effect_chain, FunctionOptimization};

const MAX_APPENDS: usize = 64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeforestationStats {
    pub removed_lists: usize,
    pub eliminated_copies: usize,
    pub transferred_appends: usize,
}

#[derive(Default)]
pub struct Deforestation {
    pub stats: DeforestationStats,
}

#[derive(Clone)]
struct Plan {
    block: usize,
    allocation: InstId,
    producer_spread: Option<(InstId, ValueId)>,
    appends: Vec<InstId>,
    consumer: InstId,
    destination: ValueId,
}

impl FunctionOptimization for Deforestation {
    fn optimize_function(&mut self, function: &mut Function) -> bool {
        let mut changed = false;
        // Reanalyse after every rewrite. This avoids stale instruction positions
        // and naturally handles nested/adjacent intermediate list builders.
        loop {
            let uses = operand_use_counts(function);
            let plan = (0..function.blocks.len())
                .find_map(|block| find_plan(function, block, &uses));
            let Some(plan) = plan else { break };
            apply_plan(function, &plan);
            self.stats.removed_lists += 1;
            self.stats.eliminated_copies += 1;
            self.stats.transferred_appends += plan.appends.len();
            changed = true;
        }
        changed
    }
}

/// Only operations whose execution cannot throw, deoptimize, suspend, allocate,
/// mutate state or invoke language code may be crossed by moved list writes.
fn movable_computation(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Const(_)
            | Inst::Move(_)
            | Inst::IntBin { op: crate::napalm_ir::IntBinOp::BitAnd
                | crate::napalm_ir::IntBinOp::BitOr
                | crate::napalm_ir::IntBinOp::BitXor, .. }
            | Inst::CmpInt { .. }
            | Inst::CmpF64 { .. }
            | Inst::RefEq { .. }
            | Inst::F64Unary { .. }
            | Inst::F64Bin { .. }
            | Inst::Int32ToNum { .. }
            | Inst::F64ToNum { .. }
    )
}

fn is_fresh_list(function: &Function, value: ValueId, block: usize, before: usize) -> Option<usize> {
    let value_info = function.value(value)?;
    if !matches!(&value_info.ty, Type::List(None)) {
        return None;
    }
    let ValueDef::InstResult(id, 0) = value_info.def else {
        return None;
    };
    if !matches!(&function.node(id)?.kind, Inst::AllocList { .. }) {
        return None;
    }
    function.blocks[block].insts[..before]
        .iter().position(|candidate| *candidate == id)
}

fn is_unaliased_external_list(function: &Function, source: ValueId, block: usize, destination_at: usize) -> bool {
    if !matches!(function.value(source).map(|value| &value.ty), Some(Type::List(_))) {
        return false;
    }
    // A source allocated earlier in this same block could be the destination
    // through aliases. Restrict this fast path to an SSA value that cannot be
    // the destination newly allocated in this block.
    match function.value(source).map(|value| value.def) {
        Some(ValueDef::CaptureParam(_)) | Some(ValueDef::BlockParam(_, _)) => true,
        Some(ValueDef::InstResult(id, _)) => !function.blocks[block].insts[..=destination_at].contains(&id)
            && !function.blocks[block].insts[destination_at + 1..].contains(&id),
        None => false,
    }
}

fn has_snapshot_use(function: &Function, value: ValueId) -> bool {
    function.deopts.iter().any(|deopt| {
        deopt.snapshot.slots.iter().any(|slot| slot.value == value)
            || deopt.snapshot.inlined_frames.iter().any(|frame| {
                frame.slots.iter().any(|slot| slot.value == value)
            })
    })
}

fn operand_use_counts(function: &Function) -> Vec<usize> {
    let mut counts = vec![0usize; function.values.len()];
    for block in &function.blocks {
        for &id in &block.insts {
            for operand in function.nodes[id.0 as usize].kind.operands() {
                if let Some(count) = counts.get_mut(operand.0 as usize) {
                    *count = count.saturating_add(1);
                }
            }
        }
    }
    counts
}

fn find_plan(function: &Function, block_ix: usize, uses: &[usize]) -> Option<Plan> {
    if function.exception_regions.iter().any(|region| region.protected_blocks.contains(&function.blocks[block_ix].id)) {
        return None;
    }
    let ids = &function.blocks[block_ix].insts;
    for (start, &allocation) in ids.iter().enumerate() {
        let alloc_node = function.node(allocation)?;
        if !matches!(&alloc_node.kind, Inst::AllocList { capacity: None }) {
            continue;
        }
        let &[tmp] = alloc_node.results.as_slice() else { continue };
        if !matches!(function.value(tmp).map(|value| &value.ty), Some(Type::List(None)))
            || has_snapshot_use(function, tmp)
        {
            continue;
        }

        let mut appends = Vec::new();
        let mut producer_spread = None;
        let mut consumer = None;
        for (index, &id) in ids.iter().enumerate().skip(start + 1) {
            let kind = &function.node(id)?.kind;
            if let Inst::ListSpread { list, source } = kind {
                if *source == tmp {
                    consumer = Some((index, id, *list));
                    break;
                }
                if *list == tmp && producer_spread.is_none() && appends.is_empty() {
                    producer_spread = Some((id, *source));
                    continue;
                }
            }
            if let Inst::ListAppend { list, .. } = kind {
                if *list == tmp && appends.len() < MAX_APPENDS {
                    appends.push(id);
                    continue;
                }
            }
            if !movable_computation(kind) {
                break;
            }
            // Disallow computing aliases of the temporary list even if the
            // resulting Move would be unused: the proof stays explicit.
            if kind.operands().contains(&tmp) {
                break;
            }
        }
        let Some((end, consumer_id, destination)) = consumer else { continue };
        if destination == tmp || has_snapshot_use(function, destination) {
            continue;
        }
        let Some(dst_start) = is_fresh_list(function, destination, block_ix, start) else {
            continue;
        };
        // No earlier alias, contract installation, observable read or escape of
        // destination: an append into it cannot be observed before the consumer.
        if ids[dst_start + 1..start].iter().any(|&id| {
            let kind = &function.nodes[id.0 as usize].kind;
            kind.operands().contains(&destination)
                && !matches!(kind, Inst::ListAppend { list, value } if *list == destination && *value != destination)
        }) {
            continue;
        }
        // No other SSA use of the temporary or of the producer spread's result
        // requires its identity. A producer ListSpread result is a guaranteed true
        // Bool and may be used by pure computations before the final copy.
        let source = producer_spread.map(|(_, value)| value);
        if let Some(source) = source {
            if source == destination || source == tmp
                || !is_unaliased_external_list(function, source, block_ix, dst_start)
            {
                continue;
            }
        }
        // Every use must be the List operand of a recognized initializing
        // append/spread or the source operand of the final copy. Counting raw
        // occurrences also catches self-appends (tmp is used twice).
        let expected_uses = appends.len() + usize::from(producer_spread.is_some()) + 1;
        if uses.get(tmp.0 as usize).copied() != Some(expected_uses) {
            continue;
        }
        if let Some((id, _)) = producer_spread {
            if !matches!(function.node(id)?.results.as_slice(), [_]) {
                continue;
            }
        }
        if !matches!(function.node(consumer_id)?.results.as_slice(), [_]) {
            continue;
        }
        // Ensure even the destination's uses *between* its allocation and the
        // copy are either its own initial appends or this consumer.
        if ids[start..end].iter().any(|&id| {
            let kind = &function.nodes[id.0 as usize].kind;
            kind.operands().contains(&destination)
                && !matches!(kind, Inst::ListAppend { list, value } if *list == destination && *value != destination)
        }) {
            continue;
        }
        return Some(Plan {
            block: block_ix,
            allocation,
            producer_spread,
            appends,
            consumer: consumer_id,
            destination,
        });
    }
    None
}

fn apply_plan(function: &mut Function, plan: &Plan) {
    let ids = &mut function.blocks[plan.block].insts;
    ids.retain(|id| *id != plan.allocation && !plan.appends.contains(id));

    let position = ids.iter().position(|id| *id == plan.consumer)
        .expect("deforestation consumer must remain live");

    // For a copied input List, the original consumer becomes a direct spread
    // of the input into the destination. Otherwise it becomes the successful
    // Bool result of appending a known List. Both preserve the result ValueId.
    if let Some((producer, source)) = plan.producer_spread {
        function.nodes[plan.consumer.0 as usize].kind = Inst::ListSpread {
            list: plan.destination,
            source,
        };
        function.nodes[producer.0 as usize].kind = Inst::Const(Constant::Bool(true));
    } else {
        function.nodes[plan.consumer.0 as usize].kind = Inst::Const(Constant::Bool(true));
    }

    for &id in &plan.appends {
        if let Inst::ListAppend { list, .. } = &mut function.nodes[id.0 as usize].kind {
            *list = plan.destination;
        }
    }

    // Place the delayed appends *after* any directly-copied input List.
    let insert_at = if plan.producer_spread.is_some() { position + 1 } else { position };
    function.blocks[plan.block].insts.splice(insert_at..insert_at, plan.appends.iter().copied());
    repair_effect_chain(function, plan.block);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::napalm_ir::{Builder, FunctionKind, Program};

    fn function_with_builder(
        source_ty: Type,
        add_observable_use: bool,
    ) -> (Program, crate::napalm_ir::FuncId, ValueId, ValueId) {
        let mut program = Program::new();
        let module = program.add_module("deforestation", 0);
        let mut builder = Builder::new(
            &mut program,
            module,
            "test",
            FunctionKind::Closure,
            vec![source_ty],
            Type::Bool,
        );
        let source = builder.param(0);
        let destination = builder.emit1(Inst::AllocList { capacity: None }, Type::List(None));
        let temporary = builder.emit1(Inst::AllocList { capacity: None }, Type::List(None));
        let spread = builder.emit1(Inst::ListSpread { list: temporary, source }, Type::Bool);
        if add_observable_use {
            builder.emit1(Inst::ListLen { list: temporary }, Type::Int32);
        }
        let number = builder.const_num(42.0);
        builder.emit0(Inst::ListAppend { list: temporary, value: number });
        let copied = builder.emit1(Inst::ListSpread { list: destination, source: temporary }, Type::Bool);
        builder.ret(copied);
        let id = builder.finish();
        let _ = spread;
        (program, id, source, destination)
    }

    #[test]
    fn fuses_typed_spread_and_appends_without_materializing_intermediate_list() {
        let (mut program, id, source, destination) = function_with_builder(Type::List(None), false);
        let function = &mut program.functions[id.0 as usize];
        let initial = function.blocks[0].insts.len();
        let mut pass = Deforestation::default();
        assert!(pass.optimize_function(function));
        assert_eq!(pass.stats.removed_lists, 1);
        assert_eq!(pass.stats.transferred_appends, 1);
        assert_eq!(function.blocks[0].insts.len(), initial - 1);
        assert_eq!(function.blocks[0].insts.iter().filter(|&&id| {
            matches!(&function.nodes[id.0 as usize].kind, Inst::AllocList { .. })
        }).count(), 1);
        assert!(function.blocks[0].insts.iter().any(|&id| matches!(
            &function.nodes[id.0 as usize].kind,
            Inst::ListSpread { list, source: input } if *list == destination && *input == source
        )));
        assert!(function.blocks[0].insts.iter().any(|&id| matches!(
            &function.nodes[id.0 as usize].kind,
            Inst::ListAppend { list, .. } if *list == destination
        )));
        crate::napalm_ir::verify_function(function).unwrap();
        assert!(!pass.optimize_function(function));
    }

    #[test]
    fn does_not_assume_an_unknown_iterable_is_a_list() {
        let (mut program, id, _, _) = function_with_builder(Type::Top, false);
        let function = &mut program.functions[id.0 as usize];
        assert!(!Deforestation::default().optimize_function(function));
        crate::napalm_ir::verify_function(function).unwrap();
    }

    #[test]
    fn preserves_intermediate_observable_reads() {
        let (mut program, id, _, _) = function_with_builder(Type::List(None), true);
        let function = &mut program.functions[id.0 as usize];
        assert!(!Deforestation::default().optimize_function(function));
        crate::napalm_ir::verify_function(function).unwrap();
    }

    fn list_copy_program() -> (Program, crate::napalm_ir::FuncId) {
        let mut program = Program::new();
        let module = program.add_module("list_copy", 0);
        let mut builder = Builder::new(
            &mut program, module, "list_copy", FunctionKind::ModuleBody,
            vec![], Type::Num,
        );
        let destination = builder.emit1(Inst::AllocList { capacity: None }, Type::List(None));
        let staging = builder.emit1(Inst::AllocList { capacity: None }, Type::List(None));
        let one = builder.const_num(1.0);
        let two = builder.const_num(2.0);
        builder.emit0(Inst::ListAppend { list: staging, value: one });
        builder.emit0(Inst::ListAppend { list: staging, value: two });
        builder.emit1(Inst::ListSpread { list: destination, source: staging }, Type::Bool);
        let len = builder.emit1(Inst::ListLen { list: destination }, Type::Int32);
        let result = builder.emit1(Inst::Int32ToNum { value: len }, Type::Num);
        builder.ret(result);
        let id = builder.finish();
        (program, id)
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
    fn deforested_list_copy_executes_identically_in_the_vm() {
        let (before_program, before_entry) = list_copy_program();
        let baseline = run_module(&before_program, before_entry);
        let (mut optimized_program, optimized_entry) = list_copy_program();
        assert!(Deforestation::default().optimize_function(
            &mut optimized_program.functions[optimized_entry.0 as usize],
        ));
        let optimized_function = &optimized_program.functions[optimized_entry.0 as usize];
        crate::napalm_ir::verify_function(optimized_function).unwrap();
        let after = run_module(&optimized_program, optimized_entry);
        assert_eq!(baseline, 2.0);
        assert_eq!(after, baseline);
    }

    #[test]
    fn eliminates_an_append_only_staging_list() {
        let mut program = Program::new();
        let module = program.add_module("deforestation", 0);
        let mut builder = Builder::new(
            &mut program, module, "append_only", FunctionKind::Closure,
            vec![Type::Num, Type::Num], Type::Bool,
        );
        let first = builder.param(0);
        let second = builder.param(1);
        let destination = builder.emit1(Inst::AllocList { capacity: None }, Type::List(None));
        let temporary = builder.emit1(Inst::AllocList { capacity: None }, Type::List(None));
        builder.emit0(Inst::ListAppend { list: temporary, value: first });
        builder.emit0(Inst::ListAppend { list: temporary, value: second });
        let flag = builder.emit1(Inst::ListSpread { list: destination, source: temporary }, Type::Bool);
        builder.ret(flag);
        let id = builder.finish();
        let function = &mut program.functions[id.0 as usize];
        assert!(Deforestation::default().optimize_function(function));
        assert_eq!(function.blocks[0].insts.iter().filter(|&&id| matches!(
            function.nodes[id.0 as usize].kind, Inst::ListAppend { list, .. } if list == destination
        )).count(), 2);
        let ValueDef::InstResult(result_inst, 0) = function.values[flag.0 as usize].def else { panic!("missing result producer") };
        assert!(matches!(&function.nodes[result_inst.0 as usize].kind, Inst::Const(Constant::Bool(true))));
        crate::napalm_ir::verify_function(function).unwrap();
    }
}
