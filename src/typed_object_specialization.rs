//! Semantic type/closed-object propagation for WIR.
//!
//! The frontend deliberately allows many CFG merge parameters to start broad.
//! This pass reconstructs the strongest *guaranteed* source type from SSA flow
//! without inserting runtime checks. The resulting `Value::ty` facts feed the
//! existing guard eliminator, numeric specialization, class devirtualization and
//! tracing JIT recorder.

use crate::napalm_ir::{ClassSet, Function, Inst, Type, ValueDef, ValueId};
use crate::napalm_ir_optimizer::FunctionOptimization;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Fact {
    /// No reachable predecessor has provided information yet. This is distinct
    /// from dynamic `any` so loop headers can be seeded by their preheader edge.
    Unreached,
    /// A source-level/runtime guarantee valid on every path reaching the value.
    Known(Type),
    /// Reachable, but no useful semantic type is guaranteed (`any`).
    Dynamic,
}

impl Fact {
    fn from_declared(ty: &Type) -> Self {
        if matches!(ty, Type::Top) {
            Self::Dynamic
        } else {
            Self::Known(ty.clone())
        }
    }

    fn with_declared(self, declared: &Type) -> Self {
        if matches!(declared, Type::Top) {
            return self;
        }
        match self {
            Fact::Known(derived) if declared.accepts(&derived) => Fact::Known(derived),
            Fact::Unreached | Fact::Dynamic | Fact::Known(_) => Fact::Known(declared.clone()),
        }
    }
}

fn join_fact(lhs: Fact, rhs: Fact) -> Fact {
    match (lhs, rhs) {
        (Fact::Unreached, rhs) => rhs,
        (lhs, Fact::Unreached) => lhs,
        (Fact::Dynamic, _) | (_, Fact::Dynamic) => Fact::Dynamic,
        (Fact::Known(lhs), Fact::Known(rhs)) => {
            let joined = lhs.flow_join(&rhs);
            if matches!(joined, Type::Top) {
                Fact::Dynamic
            } else {
                Fact::Known(joined)
            }
        }
    }
}

fn join_incoming<'a>(values: impl IntoIterator<Item = &'a Fact>) -> Fact {
    values
        .into_iter()
        .cloned()
        .fold(Fact::Unreached, join_fact)
}

fn without_null(ty: &Type) -> Option<Type> {
    match ty {
        Type::Null => Some(Type::Bottom),
        Type::Union(types) => {
            let kept = types
                .iter()
                .filter(|ty| !matches!(ty, Type::Null))
                .cloned()
                .collect::<Vec<_>>();
            (!kept.is_empty()).then(|| Type::flow_join_many(kept))
        }
        Type::Top => None,
        ty => Some(ty.clone()),
    }
}

fn intersect_class_sets(lhs: &ClassSet, rhs: &ClassSet) -> ClassSet {
    if lhs.is_unknown() {
        return rhs.clone();
    }
    if rhs.is_unknown() {
        return lhs.clone();
    }
    ClassSet::from_iter(
        lhs.as_slice()
            .iter()
            .copied()
            .filter(|class| rhs.contains(*class)),
    )
}

fn refine_class_type(ty: &Type, expected: &ClassSet) -> Option<Type> {
    let refine = |actual: &ClassSet, ctor: fn(ClassSet) -> Type| {
        let intersection = intersect_class_sets(actual, expected);
        if !actual.is_unknown() && !expected.is_unknown() && intersection.is_unknown() {
            Some(Type::Bottom)
        } else {
            Some(ctor(intersection))
        }
    };

    match ty {
        Type::Instance(actual) => refine(actual, Type::Instance),
        Type::Class(actual) => refine(actual, Type::Class),
        Type::Foreign(Some(class)) if expected.is_unknown() || expected.contains(*class) => {
            Some(Type::Foreign(Some(*class)))
        }
        Type::Union(types) => {
            let refined = types
                .iter()
                .filter_map(|ty| refine_class_type(ty, expected))
                .collect::<Vec<_>>();
            (!refined.is_empty()).then(|| Type::flow_join_many(refined))
        }
        Type::Bottom => Some(Type::Bottom),
        _ => None,
    }
}

fn guard_type_fact(source: &Fact, expected: &Type) -> Fact {
    match source {
        Fact::Known(actual) if expected.accepts(actual) => Fact::Known(actual.clone()),
        Fact::Known(actual) if actual.accepts(expected) => Fact::Known(expected.clone()),
        Fact::Known(_) | Fact::Dynamic | Fact::Unreached => Fact::Known(expected.clone()),
    }
}

fn result_fact(
    function: &Function,
    result: ValueId,
    facts: &[Fact],
    incoming: &[Vec<ValueId>],
) -> Fact {
    let Some(value) = function.value(result) else {
        return Fact::Dynamic;
    };
    let declared = &value.ty;

    let derived = match value.def {
        ValueDef::CaptureParam(_) => return Fact::from_declared(declared),
        ValueDef::BlockParam(block, _) => {
            if block == function.entry {
                return Fact::from_declared(declared);
            }
            let incoming = incoming
                .get(result.0 as usize)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if incoming.is_empty() {
                // Exception-handler parameters and deliberately synthetic blocks
                // have no normal CFG edge carrying their runtime value.
                Fact::Dynamic
            } else {
                join_incoming(
                    incoming
                        .iter()
                        .filter_map(|value| facts.get(value.0 as usize)),
                )
            }
        }
        ValueDef::InstResult(inst, result_index) => {
            let Some(node) = function.node(inst) else {
                return Fact::from_declared(declared);
            };
            if result_index as usize >= node.results.len() {
                return Fact::from_declared(declared);
            }
            match &node.kind {
                Inst::Move(source) => facts
                    .get(source.0 as usize)
                    .cloned()
                    .unwrap_or(Fact::Dynamic),
                Inst::BindInterface { expected, .. } => Fact::Known(expected.clone()),
                Inst::GuardType {
                    value, expected, ..
                } => guard_type_fact(
                    facts.get(value.0 as usize).unwrap_or(&Fact::Dynamic),
                    expected,
                ),
                Inst::GuardClass {
                    value, expected, ..
                } => match facts.get(value.0 as usize) {
                    Some(Fact::Known(ty)) => refine_class_type(ty, expected)
                        .map(Fact::Known)
                        .unwrap_or(Fact::Dynamic),
                    Some(Fact::Unreached) => Fact::Unreached,
                    _ => Fact::Dynamic,
                },
                Inst::GuardNotNull { value, .. } => match facts.get(value.0 as usize) {
                    Some(Fact::Known(ty)) => without_null(ty)
                        .map(Fact::Known)
                        .unwrap_or(Fact::Dynamic),
                    Some(Fact::Unreached) => Fact::Unreached,
                    _ => Fact::Dynamic,
                },
                Inst::GuardInt32 { .. } | Inst::NumToInt32 { .. } => {
                    Fact::Known(Type::Int32)
                }
                Inst::GuardF64 { .. } | Inst::NumToF64 { .. } => Fact::Known(Type::F64),
                Inst::Int32ToNum { .. } | Inst::F64ToNum { .. } => Fact::Known(Type::Num),
                Inst::LoadCell { cell } => match facts.get(cell.0 as usize) {
                    Some(Fact::Known(Type::Cell(inner))) => Fact::Known((**inner).clone()),
                    Some(Fact::Unreached) => Fact::Unreached,
                    _ => Fact::Dynamic,
                },
                Inst::ListGet { list, .. } => match facts.get(list.0 as usize) {
                    Some(Fact::Known(Type::List(Some(element)))) => {
                        Fact::Known((**element).clone())
                    }
                    Some(Fact::Unreached) => Fact::Unreached,
                    _ => Fact::Dynamic,
                },
                Inst::DestructureGet { source, .. } => match facts.get(source.0 as usize) {
                    Some(Fact::Known(Type::List(Some(element)))) => {
                        Fact::Known(element.flow_join(&Type::Null))
                    }
                    Some(Fact::Unreached) => Fact::Unreached,
                    _ => Fact::Dynamic,
                },
                _ => Fact::from_declared(declared),
            }
        }
    };

    derived.with_declared(declared)
}

/// Propagate exact closed/open structural contracts, class sets and collection
/// element types through WIR's SSA graph. This pass never emits a guard: it only
/// records facts already guaranteed by source typing or a successful guard/bind.
#[derive(Default)]
pub struct TypedObjectSpecialization;

impl FunctionOptimization for TypedObjectSpecialization {
    fn optimize_function(&mut self, function: &mut Function) -> bool {
        if function.values.is_empty() {
            return false;
        }

        // Precompute the SSA operands feeding every block parameter. Doing this
        // once keeps the fixed point O(values + CFG edges) per round instead of
        // rescanning every terminator for every loop/header parameter.
        let mut incoming = vec![Vec::<ValueId>::new(); function.values.len()];
        for block in &function.blocks {
            let Some(terminator) = block.insts.last().and_then(|id| function.node(*id)) else {
                continue;
            };
            for edge in terminator.kind.edges() {
                let Some(target) = function.block(edge.target) else {
                    continue;
                };
                for (&param, &argument) in target.params.iter().zip(edge.args.iter()) {
                    if let Some(values) = incoming.get_mut(param.0 as usize) {
                        values.push(argument);
                    }
                }
            }
        }

        let mut facts = vec![Fact::Unreached; function.values.len()];
        // Most SSA producers precede their users, so each round consumes facts
        // already computed earlier in the same value-order sweep. Only true CFG
        // cycles need another round. If the bounded solver does not converge we
        // conservatively abandon the refinement instead of installing a fact that
        // has not yet seen every backedge.
        let max_rounds = function
            .blocks
            .len()
            .saturating_mul(2)
            .saturating_add(8)
            .max(16)
            .min(128);
        let mut converged = false;
        for _ in 0..max_rounds {
            let mut next = facts.clone();
            for index in 0..function.values.len() {
                let fact = result_fact(function, ValueId(index as u32), &next, &incoming);
                next[index] = fact;
            }
            if next == facts {
                facts = next;
                converged = true;
                break;
            }
            facts = next;
        }
        if !converged {
            return false;
        }

        let mut changed = false;
        for (value, fact) in function.values.iter_mut().zip(facts) {
            let Fact::Known(ty) = fact else {
                continue;
            };
            if ty == value.ty {
                continue;
            }
            // Never weaken an existing verifier guarantee. A refinement must be a
            // subtype accepted by the type already attached to the SSA value.
            if value.ty.accepts(&ty) {
                value.ty = ty;
                changed = true;
            }
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::napalm_ir::{Builder, FunctionKind, Program};

    #[test]
    fn preserves_record_contract_through_loop_parameter() {
        let mut program = Program::new();
        let module = program.add_module("typed", 0);
        let record = Type::Record(vec![]);
        let mut builder = Builder::new(
            &mut program,
            module,
            "f",
            FunctionKind::Closure,
            vec![record.clone()],
            Type::Null,
        );
        let entry_value = builder.param(0);
        let header = builder.new_block(&[Type::Top]);
        let exit = builder.new_block(&[]);
        builder.jump(header, vec![entry_value]);

        builder.position_at(header);
        let param = builder.block_param(header, 0);
        let result = builder.emit1(Inst::Move(param), Type::Top);
        let keep_looping = builder.const_bool(false);
        builder.branch(keep_looping, header, vec![result], exit, vec![]);
        builder.seal_block(header);

        builder.position_at(exit);
        builder.seal_block(exit);
        let null = builder.const_null();
        builder.ret(null);
        let function = builder.finish();

        let function = &mut program.functions[function.0 as usize];
        assert!(TypedObjectSpecialization.optimize_function(function));
        assert_eq!(function.value(param).unwrap().ty, record);
        assert_eq!(function.value(result).unwrap().ty, Type::Record(vec![]));
    }

    #[test]
    fn successful_structural_guard_refines_any() {
        let mut program = Program::new();
        let module = program.add_module("typed", 0);
        let expected = Type::Interface(vec![]);
        let mut builder = Builder::new(
            &mut program,
            module,
            "f",
            FunctionKind::Closure,
            vec![Type::Top],
            Type::Top,
        );
        let input = builder.param(0);
        let guarded = builder.emit1(
            Inst::GuardType {
                value: input,
                expected: expected.clone(),
                deopt: crate::napalm_ir::DeoptId(0),
            },
            Type::Top,
        );
        builder.ret(guarded);
        let function = builder.finish();
        let function = &mut program.functions[function.0 as usize];
        // The synthetic test does not verify deopt metadata; this pass itself only
        // reasons about the successful SSA path.
        assert!(TypedObjectSpecialization.optimize_function(function));
        assert_eq!(function.value(guarded).unwrap().ty, expected);
    }

    #[test]
    fn mutable_list_flow_join_drops_incompatible_element_contracts() {
        let numbers = Type::List(Some(Box::new(Type::Num)));
        let strings = Type::List(Some(Box::new(Type::String)));
        assert_eq!(numbers.flow_join(&strings), Type::List(None));
    }

    #[test]
    fn joins_exact_classes_without_falling_back_to_any() {
        let a = crate::napalm_ir::ClassId(1);
        let b = crate::napalm_ir::ClassId(2);
        let joined = Type::Instance(ClassSet::singleton(a))
            .flow_join(&Type::Instance(ClassSet::singleton(b)));
        assert_eq!(
            joined,
            Type::Instance(ClassSet::from_iter([a, b]))
        );
    }
}
