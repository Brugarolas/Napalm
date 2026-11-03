//! Optimization passes for Napalm MX IR.
//!
//! The pipeline is deliberately split into logical simplification, predicate and
//! projection movement, storage access selection, physical-property exploitation,
//! and final delivery/transaction specialization. It targets SurrealMX's strongest
//! primitives: point reads, ordered range scans, reverse scans, key-only scans,
//! zero-copy visitor scans and range counts.

#![allow(dead_code)]

use crate::napalm_collections::{HashMap, HashSet};
use crate::napalm_mx_ir::*;
use crate::napalm_mx_optimizer::{
    replace_expr_uses, Optimize, QueryOptimization,
};

pub fn optimization_passes() -> Optimize {
    Optimize::new().with_query_pass(QueryOptimizationBundle::default())
}

pub fn aggressive_optimization_passes() -> Optimize {
    optimization_passes()
}

#[derive(Default)]
pub struct QueryOptimizationBundle;

impl QueryOptimization for QueryOptimizationBundle {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let mut changed_any = false;

        let mut expressions = ExpressionSimplify::default();
        let mut cse = ExpressionCse::default();
        let mut in_lists = InListCanonicalization::default();
        let mut or_to_in = OrToIn::default();
        let mut constraints = ConstraintPropagation::default();
        let mut ranges = RangeCanonicalization::default();
        let mut logical = LogicalSimplify::default();
        let mut contradictions = ContradictionElimination::default();
        let mut outer_join = OuterJoinReduction::default();
        let mut join_predicates = JoinPredicateDecomposition::default();
        let mut join_constraints = JoinConstraintPropagation::default();
        let mut join_reorder = JoinReordering::default();
        let mut pushdown = PredicatePushdown::default();
        let mut prune = ColumnPruning::default();
        let mut scan_fusion = ScanPipelineFusion::default();
        let mut decode_pruning = PhysicalDecodePruning::default();
        let mut union_limits = UnionLimitPushdown::default();
        let mut materialize_shared = SharedSubplanMaterialization::default();
        let mut materialize_cleanup = MaterializationCleanup::default();
        let mut memory_cost = MemoryCostAnalysis::default();
        let mut reorder = PredicateReordering::default();
        let mut index_union = IndexUnionSelection::default();
        let mut index_intersection = IndexIntersectionSelection::default();
        let mut multi_point = MultiPointAccessSelection::default();
        let mut access = AccessPathSelection::default();
        let mut unique_index = UniqueIndexSpecialization::default();
        let mut covering = CoveringIndexPromotion::default();
        let mut min_max = MinMaxOptimization::default();
        let mut ordered_top_n = OrderedTopNPushdown::default();
        let mut sort = SortElimination::default();
        let mut distinct = DistinctElimination::default();
        let mut limits = LimitPushdown::default();
        let mut exists = ExistencePushdown::default();
        let mut aggregate_distinct = AggregateDistinctElimination::default();
        let mut count_args = CountArgumentElimination::default();
        let mut unique_groups = UniqueGroupAggregateElimination::default();
        let mut aggregate_sorts = SortUnderAggregateElimination::default();
        let mut empty_aggregates = EmptyAggregateFolding::default();
        let mut count = CountOptimization::default();
        let mut late_fetch = LateRowFetch::default();
        let mut joins = JoinSelection::default();
        let mut write_noops = WriteNoOpElimination::default();
        let mut writes = WriteLowering::default();
        let mut write_coalesce = WriteBatchCoalescing::default();
        let mut delivery = DeliverySelection::default();
        let mut transactions = TransactionSpecialization::default();
        let mut properties = PropertyAnalysis::default();

        // Phase 1: expression and local relational fixed point.
        for _ in 0..3 {
            let mut round = false;
            round |= expressions.optimize_query(query, catalog);
            round |= cse.optimize_query(query, catalog);
            round |= in_lists.optimize_query(query, catalog);
            round |= or_to_in.optimize_query(query, catalog);
            round |= constraints.optimize_query(query, catalog);
            round |= ranges.optimize_query(query, catalog);
            round |= logical.optimize_query(query, catalog);
            round |= contradictions.optimize_query(query, catalog);
            changed_any |= round;
            if !round {
                break;
            }
        }

        // Phase 2: normalize joins, propagate relational constraints, move predicates
        // down and shrink rows before access selection. Join-derived constraints are
        // intentionally exposed while children are still logical TableScan/Filter
        // trees so the ordinary access-path selector can consume them later.
        for _ in 0..3 {
            let mut round = false;
            round |= outer_join.optimize_query(query, catalog);
            round |= pushdown.optimize_query(query, catalog);
            round |= join_predicates.optimize_query(query, catalog);
            round |= join_constraints.optimize_query(query, catalog);
            round |= constraints.optimize_query(query, catalog);
            round |= ranges.optimize_query(query, catalog);
            round |= prune.optimize_query(query, catalog);
            round |= reorder.optimize_query(query, catalog);
            round |= logical.optimize_query(query, catalog);
            round |= properties.optimize_query(query, catalog);
            changed_any |= round;
            if !round {
                break;
            }
        }

        // Convert row/cardinality properties into a RAM-oriented cost immediately
        // before bounded DP join enumeration.
        changed_any |= memory_cost.optimize_query(query, catalog);

        // Reorder connected inner-join components only after local predicates and
        // cardinality estimates have reached a fixed point. The DP is bounded to
        // small join graphs, so compile-time remains predictable.
        changed_any |= join_reorder.optimize_query(query, catalog);
        changed_any |= properties.optimize_query(query, catalog);
        changed_any |= memory_cost.optimize_query(query, catalog);

        // Phase 3: choose ordered KV access paths.
        // Multi-index rewrites must see the original Filter(TableScan) shape. OR can
        // become an index union; selective independent AND terms can become a PK
        // intersection. Small IN predicates are expanded before the generic selector.
        changed_any |= index_union.optimize_query(query, catalog);
        changed_any |= index_intersection.optimize_query(query, catalog);
        changed_any |= multi_point.optimize_query(query, catalog);
        changed_any |= access.optimize_query(query, catalog);
        changed_any |= unique_index.optimize_query(query, catalog);
        // Access-path extraction removes consumed predicate fields from the visible row shape.
        changed_any |= prune.optimize_query(query, catalog);
        changed_any |= covering.optimize_query(query, catalog);
        changed_any |= properties.optimize_query(query, catalog);
        changed_any |= memory_cost.optimize_query(query, catalog);

        // Phase 4: exploit physical order/cardinality, fuse scan pipelines, prune
        // decoded fields aggressively and lower joins/writes.
        for _ in 0..3 {
            let mut round = false;
            round |= properties.optimize_query(query, catalog);
            round |= scan_fusion.optimize_query(query, catalog);
            round |= decode_pruning.optimize_query(query, catalog);
            round |= distinct.optimize_query(query, catalog);
            // MIN/MAX and ORDER BY ... LIMIT can turn an ordered index/range scan
            // into a bounded forward/reverse scan without changing aggregate/empty semantics.
            round |= min_max.optimize_query(query, catalog);
            round |= ordered_top_n.optimize_query(query, catalog);
            round |= sort.optimize_query(query, catalog);
            round |= union_limits.optimize_query(query, catalog);
            round |= limits.optimize_query(query, catalog);
            round |= exists.optimize_query(query, catalog);
            round |= aggregate_sorts.optimize_query(query, catalog);
            round |= aggregate_distinct.optimize_query(query, catalog);
            round |= count_args.optimize_query(query, catalog);
            round |= unique_groups.optimize_query(query, catalog);
            round |= empty_aggregates.optimize_query(query, catalog);
            round |= count.optimize_query(query, catalog);
            // Delay primary-row materialization until filters/order/limits have been
            // absorbed by the index scan. Re-check covering after the rewrite.
            round |= late_fetch.optimize_query(query, catalog);
            round |= decode_pruning.optimize_query(query, catalog);
            round |= covering.optimize_query(query, catalog);
            round |= joins.optimize_query(query, catalog);
            round |= write_noops.optimize_query(query, catalog);
            round |= writes.optimize_query(query, catalog);
            round |= write_coalesce.optimize_query(query, catalog);
            round |= logical.optimize_query(query, catalog);
            round |= properties.optimize_query(query, catalog);
            changed_any |= round;
            if !round {
                break;
            }
        }

        changed_any |= memory_cost.optimize_query(query, catalog);

        // Phase 5: decide whether repeated subplans deserve one buffered materialization,
        // eliminate redundant materialization, then select zero-copy/key-only delivery
        // and the narrowest transaction shape.
        changed_any |= materialize_cleanup.optimize_query(query, catalog);
        changed_any |= properties.optimize_query(query, catalog);
        changed_any |= memory_cost.optimize_query(query, catalog);
        changed_any |= materialize_shared.optimize_query(query, catalog);
        changed_any |= properties.optimize_query(query, catalog);
        changed_any |= delivery.optimize_query(query, catalog);
        changed_any |= transactions.optimize_query(query, catalog);
        changed_any |= properties.optimize_query(query, catalog);
        changed_any |= memory_cost.optimize_query(query, catalog);

        changed_any
    }
}

// -----------------------------------------------------------------------------
// Expression simplification / CSE
// -----------------------------------------------------------------------------

#[derive(Default)]
pub struct ExpressionSimplify;

impl QueryOptimization for ExpressionSimplify {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let mut changed_any = false;
        for _ in 0..4 {
            let mut changed = false;
            let len = query.exprs.len();
            for index in 0..len {
                let id = ExprId(index as u32);
                let Some(rewrite) = simplify_expr(query, id) else { continue; };
                match rewrite {
                    ExprRewrite::Alias(replacement) => {
                        changed |= replace_expr_uses(query, id, replacement);
                    }
                    ExprRewrite::Replace(kind, ty, nullable) => {
                        let node = &mut query.exprs[index];
                        if node.kind != kind || node.ty != ty || node.nullable != nullable {
                            node.kind = kind;
                            node.ty = ty;
                            node.nullable = nullable;
                            changed = true;
                        }
                    }
                }
            }
            changed_any |= changed;
            if !changed {
                break;
            }
        }
        changed_any
    }
}

enum ExprRewrite {
    Alias(ExprId),
    Replace(ExprKind, ScalarType, bool),
}

fn simplify_expr(query: &Query, id: ExprId) -> Option<ExprRewrite> {
    let node = query.expr(id)?;
    match &node.kind {
        ExprKind::Unary { op, value } => {
            if let Some(constant) = constant(query, *value) {
                match (op, constant) {
                    (UnaryOp::Not, ScalarValue::Bool(value)) => {
                        return Some(const_rewrite(ScalarValue::Bool(!value)));
                    }
                    (UnaryOp::Neg, ScalarValue::Number(value)) => {
                        return Some(const_rewrite(ScalarValue::Number(NumberBits::from_f64(-value.to_f64()))));
                    }
                    _ => {}
                }
            }
            if *op == UnaryOp::Not {
                if let Some(ExprKind::Unary { op: UnaryOp::Not, value: inner }) = producer(query, *value) {
                    return Some(ExprRewrite::Alias(*inner));
                }
            }
        }
        ExprKind::Binary { op, left, right } => {
            if op.is_commutative() && left.0 > right.0 {
                return Some(ExprRewrite::Replace(
                    ExprKind::Binary { op: *op, left: *right, right: *left },
                    node.ty,
                    node.nullable,
                ));
            }

            if left == right {
                let value = query.expr(*left)?;
                let reflexive = !value.nullable && !matches!(value.ty, ScalarType::Number | ScalarType::Any);
                match op {
                    BinaryOp::Eq if reflexive => {
                        return Some(const_rewrite(ScalarValue::Bool(true)));
                    }
                    BinaryOp::Ne if reflexive => {
                        return Some(const_rewrite(ScalarValue::Bool(false)));
                    }
                    BinaryOp::And | BinaryOp::Or => return Some(ExprRewrite::Alias(*left)),
                    _ => {}
                }
            }

            if let (Some(lhs), Some(rhs)) = (constant(query, *left), constant(query, *right)) {
                if let Some(value) = fold_binary(*op, lhs, rhs) {
                    return Some(const_rewrite_as(value, node.ty));
                }
            }

            match op {
                BinaryOp::And => {
                    if bool_const(query, *left) == Some(true) { return Some(ExprRewrite::Alias(*right)); }
                    if bool_const(query, *right) == Some(true) { return Some(ExprRewrite::Alias(*left)); }
                    if bool_const(query, *left) == Some(false) || bool_const(query, *right) == Some(false) {
                        return Some(const_rewrite(ScalarValue::Bool(false)));
                    }
                }
                BinaryOp::Or => {
                    if bool_const(query, *left) == Some(false) { return Some(ExprRewrite::Alias(*right)); }
                    if bool_const(query, *right) == Some(false) { return Some(ExprRewrite::Alias(*left)); }
                    if bool_const(query, *left) == Some(true) || bool_const(query, *right) == Some(true) {
                        return Some(const_rewrite(ScalarValue::Bool(true)));
                    }
                }
                BinaryOp::Add => {
                    if number_const(query, *right) == Some(0.0) { return Some(ExprRewrite::Alias(*left)); }
                    if number_const(query, *left) == Some(0.0) { return Some(ExprRewrite::Alias(*right)); }
                }
                BinaryOp::Sub => {
                    if number_const(query, *right) == Some(0.0) { return Some(ExprRewrite::Alias(*left)); }
                }
                BinaryOp::Mul => {
                    if number_const(query, *right) == Some(1.0) { return Some(ExprRewrite::Alias(*left)); }
                    if number_const(query, *left) == Some(1.0) { return Some(ExprRewrite::Alias(*right)); }
                }
                BinaryOp::Div => {
                    if number_const(query, *right) == Some(1.0) { return Some(ExprRewrite::Alias(*left)); }
                }
                _ => {}
            }

            // Canonicalize sargable comparisons to `Field op bound`.
            if op.is_comparison() && is_runtime_bound(query, *left) && matches!(producer(query, *right), Some(ExprKind::Column(_))) {
                let reversed = reverse_comparison(*op)?;
                return Some(ExprRewrite::Replace(
                    ExprKind::Binary { op: reversed, left: *right, right: *left },
                    node.ty,
                    node.nullable,
                ));
            }
        }
        ExprKind::IsNull { value, negated } => {
            if let Some(value) = constant(query, *value) {
                let is_null = matches!(value, ScalarValue::Null);
                return Some(const_rewrite(ScalarValue::Bool(if *negated { !is_null } else { is_null })));
            }
            if !query.expr(*value)?.nullable {
                return Some(const_rewrite(ScalarValue::Bool(*negated)));
            }
        }
        ExprKind::Between { value, low, high, negated } => {
            if let (Some(value), Some(low), Some(high)) = (
                constant(query, *value),
                constant(query, *low),
                constant(query, *high),
            ) {
                if matches!(&value, ScalarValue::Null)
                    || matches!(&low, ScalarValue::Null)
                    || matches!(&high, ScalarValue::Null)
                {
                    return Some(const_rewrite_as(ScalarValue::Null, ScalarType::Bool));
                }
                if let (Some(ge), Some(le)) = (
                    fold_binary(BinaryOp::Ge, value.clone(), low),
                    fold_binary(BinaryOp::Le, value, high),
                ) {
                    if let (ScalarValue::Bool(ge), ScalarValue::Bool(le)) = (ge, le) {
                        let inside = ge && le;
                        return Some(const_rewrite(ScalarValue::Bool(if *negated { !inside } else { inside })));
                    }
                }
            }
        }
        ExprKind::InList { value, items, negated } => {
            if items.is_empty() {
                return Some(const_rewrite(ScalarValue::Bool(*negated)));
            }
            if items.len() == 1 {
                let op = if *negated { BinaryOp::Ne } else { BinaryOp::Eq };
                return Some(ExprRewrite::Replace(
                    ExprKind::Binary { op, left: *value, right: items[0] },
                    ScalarType::Bool,
                    query.expr(*value)?.nullable || query.expr(items[0])?.nullable,
                ));
            }
            if let Some(value) = constant(query, *value) {
                let mut all_const = true;
                let mut found = false;
                let mut saw_null = matches!(&value, ScalarValue::Null);
                for item in items {
                    let Some(item) = constant(query, *item) else {
                        all_const = false;
                        break;
                    };
                    if matches!(&item, ScalarValue::Null) {
                        saw_null = true;
                        continue;
                    }
                    if !matches!(&value, ScalarValue::Null) && scalar_eq(&value, &item) {
                        found = true;
                        break;
                    }
                }
                if found {
                    return Some(const_rewrite(ScalarValue::Bool(!*negated)));
                }
                if all_const {
                    if saw_null {
                        return Some(const_rewrite_as(ScalarValue::Null, ScalarType::Bool));
                    }
                    return Some(const_rewrite(ScalarValue::Bool(*negated)));
                }
            }
        }
        ExprKind::Call { function, args } => {
            if args.len() == 1 {
                if let Some(value) = number_const(query, args[0]) {
                    let result = match function {
                        BuiltinFn::Abs => Some(value.abs()),
                        BuiltinFn::Floor => Some(value.floor()),
                        BuiltinFn::Ceil => Some(value.ceil()),
                        BuiltinFn::Round => Some(value.round()),
                        _ => None,
                    };
                    if let Some(result) = result {
                        return Some(const_rewrite(number(result)));
                    }
                }
            }
        }
        ExprKind::Const(_) | ExprKind::Param(_) | ExprKind::Column(_) => {}
    }
    None
}

fn const_rewrite(value: ScalarValue) -> ExprRewrite {
    let ty = value.ty();
    ExprRewrite::Replace(ExprKind::Const(value), ty, ty == ScalarType::Null)
}

fn const_rewrite_as(value: ScalarValue, expected: ScalarType) -> ExprRewrite {
    if matches!(&value, ScalarValue::Null) {
        ExprRewrite::Replace(ExprKind::Const(value), expected, true)
    } else {
        const_rewrite(value)
    }
}

fn number(value: f64) -> ScalarValue {
    ScalarValue::Number(NumberBits::from_f64(value))
}

fn producer(query: &Query, id: ExprId) -> Option<&ExprKind> {
    query.expr(id).map(|node| &node.kind)
}

fn constant(query: &Query, id: ExprId) -> Option<ScalarValue> {
    match producer(query, id)? {
        ExprKind::Const(value) => Some(value.clone()),
        _ => None,
    }
}

fn bool_const(query: &Query, id: ExprId) -> Option<bool> {
    match constant(query, id)? {
        ScalarValue::Bool(value) => Some(value),
        _ => None,
    }
}

fn is_null_const(query: &Query, id: ExprId) -> bool {
    matches!(constant(query, id), Some(ScalarValue::Null))
}

fn number_const(query: &Query, id: ExprId) -> Option<f64> {
    match constant(query, id)? {
        ScalarValue::Number(value) => Some(value.to_f64()),
        _ => None,
    }
}

fn scalar_eq(lhs: &ScalarValue, rhs: &ScalarValue) -> bool {
    match (lhs, rhs) {
        (ScalarValue::Number(a), ScalarValue::Number(b)) => a.to_f64() == b.to_f64(),
        _ => lhs == rhs,
    }
}

fn fold_binary(op: BinaryOp, lhs: ScalarValue, rhs: ScalarValue) -> Option<ScalarValue> {
    use ScalarValue::Null;

    match op {
        BinaryOp::And => match (lhs, rhs) {
            (ScalarValue::Bool(false), _) | (_, ScalarValue::Bool(false)) => {
                Some(ScalarValue::Bool(false))
            }
            (ScalarValue::Bool(true), ScalarValue::Bool(value))
            | (ScalarValue::Bool(value), ScalarValue::Bool(true)) => Some(ScalarValue::Bool(value)),
            (Null, ScalarValue::Bool(true))
            | (ScalarValue::Bool(true), Null)
            | (Null, Null) => Some(Null),
            _ => None,
        },
        BinaryOp::Or => match (lhs, rhs) {
            (ScalarValue::Bool(true), _) | (_, ScalarValue::Bool(true)) => {
                Some(ScalarValue::Bool(true))
            }
            (ScalarValue::Bool(false), ScalarValue::Bool(value))
            | (ScalarValue::Bool(value), ScalarValue::Bool(false)) => Some(ScalarValue::Bool(value)),
            (Null, ScalarValue::Bool(false))
            | (ScalarValue::Bool(false), Null)
            | (Null, Null) => Some(Null),
            _ => None,
        },
        BinaryOp::Eq | BinaryOp::Ne if matches!(&lhs, Null) || matches!(&rhs, Null) => Some(Null),
        BinaryOp::Eq => Some(ScalarValue::Bool(scalar_eq(&lhs, &rhs))),
        BinaryOp::Ne => Some(ScalarValue::Bool(!scalar_eq(&lhs, &rhs))),
        _ if matches!(&lhs, Null) || matches!(&rhs, Null) => Some(Null),
        _ => {
            let (ScalarValue::Number(a), ScalarValue::Number(b)) = (lhs, rhs) else { return None; };
            let (a, b) = (a.to_f64(), b.to_f64());
            Some(match op {
                BinaryOp::Add => number(a + b),
                BinaryOp::Sub => number(a - b),
                BinaryOp::Mul => number(a * b),
                BinaryOp::Div => number(a / b),
                BinaryOp::Rem => number(a % b),
                BinaryOp::Lt => ScalarValue::Bool(a < b),
                BinaryOp::Le => ScalarValue::Bool(a <= b),
                BinaryOp::Gt => ScalarValue::Bool(a > b),
                BinaryOp::Ge => ScalarValue::Bool(a >= b),
                BinaryOp::Eq | BinaryOp::Ne | BinaryOp::And | BinaryOp::Or => unreachable!(),
            })
        }
    }
}

fn reverse_comparison(op: BinaryOp) -> Option<BinaryOp> {
    Some(match op {
        BinaryOp::Eq => BinaryOp::Eq,
        BinaryOp::Ne => BinaryOp::Ne,
        BinaryOp::Lt => BinaryOp::Gt,
        BinaryOp::Le => BinaryOp::Ge,
        BinaryOp::Gt => BinaryOp::Lt,
        BinaryOp::Ge => BinaryOp::Le,
        _ => return None,
    })
}

fn is_runtime_bound(query: &Query, id: ExprId) -> bool {
    matches!(producer(query, id), Some(ExprKind::Const(_) | ExprKind::Param(_)))
}

#[derive(Default)]
pub struct ExpressionCse;

impl QueryOptimization for ExpressionCse {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let mut values: HashMap<(ExprKind, ScalarType, bool), ExprId> = HashMap::default();
        let mut aliases = Vec::new();
        for node in &query.exprs {
            let key = (node.kind.clone(), node.ty, node.nullable);
            if let Some(previous) = values.get(&key).copied() {
                if previous != node.id {
                    aliases.push((node.id, previous));
                }
            } else {
                values.insert(key, node.id);
            }
        }
        let mut changed = false;
        for (old, new) in aliases {
            changed |= replace_expr_uses(query, old, new);
        }
        changed
    }
}


#[derive(Default)]
pub struct InListCanonicalization;

impl QueryOptimization for InListCanonicalization {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let mut rewrites = Vec::new();

        for node in &query.exprs {
            let ExprKind::InList { value, items, negated } = &node.kind else {
                continue;
            };

            let mut seen_constants: HashSet<ScalarValue> = HashSet::default();
            let mut seen_params: HashSet<ParamId> = HashSet::default();
            let mut seen_other: HashSet<ExprId> = HashSet::default();
            let mut canonical = Vec::with_capacity(items.len());

            for &item in items {
                let unique = match producer(query, item) {
                    Some(ExprKind::Const(value)) => seen_constants.insert(value.clone()),
                    Some(ExprKind::Param(param)) => seen_params.insert(*param),
                    _ => seen_other.insert(item),
                };
                if unique {
                    canonical.push(item);
                }
            }

            if canonical != *items {
                rewrites.push((
                    node.id,
                    ExprKind::InList {
                        value: *value,
                        items: canonical,
                        negated: *negated,
                    },
                ));
            }
        }

        if rewrites.is_empty() {
            return false;
        }

        for (id, kind) in rewrites {
            query.expr_mut(id).unwrap().kind = kind;
        }
        true
    }
}

#[derive(Default)]
pub struct OrToIn;

impl QueryOptimization for OrToIn {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let mut changed = false;
        let len = query.exprs.len();
        for index in 0..len {
            let id = ExprId(index as u32);
            let mut terms = Vec::new();
            flatten_boolean(query, id, BinaryOp::Or, &mut terms);
            if terms.len() < 2 {
                continue;
            }
            let mut field = None;
            let mut items = Vec::new();
            let mut valid = true;
            for term in terms {
                let Some((candidate, value)) = equality_column_bound(query, term) else {
                    valid = false;
                    break;
                };
                if field.is_some_and(|current| current != candidate) {
                    valid = false;
                    break;
                }
                field = Some(candidate);
                items.push(value);
            }
            if !valid || items.len() < 2 {
                continue;
            }
            items.sort_unstable_by_key(|expr| expr.0);
            items.dedup();
            let Some(column) = field else { continue; };
            let ty = column_type(query, column).unwrap_or(ScalarType::Any);
            let nullable = query.column(column).is_none_or(|column| column.nullable);
            let column_expr = query.add_expr(ExprKind::Column(column), ty, nullable);
            let nullable_result = query.exprs[index].nullable;
            query.exprs[index].kind = ExprKind::InList { value: column_expr, items, negated: false };
            query.exprs[index].ty = ScalarType::Bool;
            query.exprs[index].nullable = nullable_result;
            changed = true;
        }
        changed
    }
}

fn column_type(query: &Query, column: ColumnId) -> Option<ScalarType> {
    query.column(column).map(|column| column.ty)
}


// -----------------------------------------------------------------------------
// Constraint propagation / predicate range canonicalization
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ComparisonSignature {
    column: ColumnId,
    op: BinaryOp,
    bound: ExprId,
}

fn comparison_column_bound(query: &Query, id: ExprId) -> Option<(ColumnId, BinaryOp, ExprId)> {
    let ExprKind::Binary { op, left, right } = producer(query, id)? else {
        return None;
    };
    if !op.is_comparison() {
        return None;
    }
    if let Some(ExprKind::Column(column)) = producer(query, *left) {
        if is_runtime_bound(query, *right) {
            return Some((*column, *op, *right));
        }
    }
    if let Some(ExprKind::Column(column)) = producer(query, *right) {
        if is_runtime_bound(query, *left) {
            return Some((*column, reverse_comparison(*op)?, *left));
        }
    }
    None
}

fn equality_columns(query: &Query, id: ExprId) -> Option<(ColumnId, ColumnId)> {
    let ExprKind::Binary { op: BinaryOp::Eq, left, right } = producer(query, id)? else {
        return None;
    };
    let ExprKind::Column(left) = producer(query, *left)? else {
        return None;
    };
    let ExprKind::Column(right) = producer(query, *right)? else {
        return None;
    };
    Some((*left, *right))
}

fn union_find_root(parent: &[usize], mut index: usize) -> usize {
    while parent[index] != index {
        index = parent[index];
    }
    index
}

fn union_find_join(parent: &mut [usize], left: usize, right: usize) {
    let left = union_find_root(parent, left);
    let right = union_find_root(parent, right);
    if left != right {
        let (root, child) = if left < right { (left, right) } else { (right, left) };
        parent[child] = root;
    }
}

#[derive(Default)]
pub struct ConstraintPropagation;

impl QueryOptimization for ConstraintPropagation {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let filters: Vec<_> = query
            .live_plans_preorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::Filter { input, predicate }) => Some((id, input, predicate)),
                _ => None,
            })
            .collect();

        let mut changed = false;
        for (plan, input, predicate) in filters {
            let mut terms = Vec::new();
            flatten_boolean(query, predicate, BinaryOp::And, &mut terms);
            if terms.len() < 2 || query.columns.is_empty() {
                continue;
            }

            let mut parent: Vec<usize> = (0..query.columns.len()).collect();
            for &term in &terms {
                let Some((left, right)) = equality_columns(query, term) else {
                    continue;
                };
                union_find_join(&mut parent, left.0 as usize, right.0 as usize);
            }

            let mut groups: HashMap<usize, Vec<ColumnId>> = HashMap::default();
            for index in 0..query.columns.len() {
                let root = union_find_root(&parent, index);
                groups.entry(root).or_default().push(ColumnId(index as u32));
            }

            let mut existing: HashSet<ComparisonSignature> = HashSet::default();
            for &term in &terms {
                if let Some((column, op, bound)) = comparison_column_bound(query, term) {
                    existing.insert(ComparisonSignature { column, op, bound });
                }
            }

            let source_terms = terms.clone();
            let mut additions = Vec::new();
            for term in source_terms {
                let Some((column, op, bound)) = comparison_column_bound(query, term) else {
                    continue;
                };
                if !matches!(op, BinaryOp::Eq | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge) {
                    continue;
                }

                let root = union_find_root(&parent, column.0 as usize);
                let Some(members) = groups.get(&root) else {
                    continue;
                };
                for &member in members {
                    if member == column {
                        continue;
                    }
                    let signature = ComparisonSignature {
                        column: member,
                        op,
                        bound,
                    };
                    if !existing.insert(signature) {
                        continue;
                    }
                    let Some(column_def) = query.column(member).cloned() else {
                        continue;
                    };
                    let column_expr = query.add_expr(
                        ExprKind::Column(member),
                        column_def.ty,
                        column_def.nullable,
                    );
                    let nullable = column_def.nullable
                        || query.expr(bound).is_none_or(|bound| bound.nullable);
                    let derived = query.add_expr(
                        ExprKind::Binary {
                            op,
                            left: column_expr,
                            right: bound,
                        },
                        ScalarType::Bool,
                        nullable,
                    );
                    additions.push(derived);
                }
            }

            if additions.is_empty() {
                continue;
            }
            terms.extend(additions);
            let new_predicate = conjunction(query, &terms).unwrap();
            if let Some(node) = query.plan_mut(plan) {
                node.kind = PlanKind::Filter {
                    input,
                    predicate: new_predicate,
                };
                changed = true;
            }
        }
        changed
    }
}

#[derive(Clone)]
struct ConstantComparison {
    term_index: usize,
    op: BinaryOp,
    bound: ExprId,
    value: ScalarValue,
}

fn scalar_order(left: &ScalarValue, right: &ScalarValue) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    match (left, right) {
        (ScalarValue::Bool(left), ScalarValue::Bool(right)) => Some(left.cmp(right)),
        (ScalarValue::Number(left), ScalarValue::Number(right)) => {
            left.to_f64().partial_cmp(&right.to_f64())
        }
        _ if scalar_eq(left, right) => Some(Ordering::Equal),
        _ => None,
    }
}

fn lower_is_tighter(candidate: &ConstantComparison, current: &ConstantComparison) -> bool {
    match scalar_order(&candidate.value, &current.value) {
        Some(std::cmp::Ordering::Greater) => true,
        Some(std::cmp::Ordering::Equal) => {
            candidate.op == BinaryOp::Gt && current.op == BinaryOp::Ge
        }
        _ => false,
    }
}

fn upper_is_tighter(candidate: &ConstantComparison, current: &ConstantComparison) -> bool {
    match scalar_order(&candidate.value, &current.value) {
        Some(std::cmp::Ordering::Less) => true,
        Some(std::cmp::Ordering::Equal) => {
            candidate.op == BinaryOp::Lt && current.op == BinaryOp::Le
        }
        _ => false,
    }
}

fn equality_satisfies_comparison(value: &ScalarValue, constraint: &ConstantComparison) -> Option<bool> {
    let result = fold_binary(constraint.op, value.clone(), constraint.value.clone())?;
    match result {
        ScalarValue::Bool(value) => Some(value),
        ScalarValue::Null => Some(false),
        _ => None,
    }
}

fn is_null_column(query: &Query, id: ExprId) -> Option<(ColumnId, bool)> {
    let ExprKind::IsNull { value, negated } = producer(query, id)? else {
        return None;
    };
    let ExprKind::Column(column) = producer(query, *value)? else {
        return None;
    };
    Some((*column, *negated))
}

#[derive(Default)]
pub struct RangeCanonicalization;

impl QueryOptimization for RangeCanonicalization {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let filters: Vec<_> = query
            .live_plans_preorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::Filter { input, predicate }) => Some((id, input, predicate)),
                _ => None,
            })
            .collect();

        let mut changed_any = false;
        for (plan, input, predicate) in filters {
            let mut raw_terms = Vec::new();
            flatten_boolean(query, predicate, BinaryOp::And, &mut raw_terms);

            // BETWEEN is useful syntax, but two ordinary range predicates compose
            // better with equivalence propagation and the access-path extractor.
            let mut terms = Vec::with_capacity(raw_terms.len() + 2);
            let mut expanded_between = false;
            for term in raw_terms {
                let Some(ExprKind::Between {
                    value,
                    low,
                    high,
                    negated: false,
                }) = producer(query, term).cloned()
                else {
                    terms.push(term);
                    continue;
                };
                let column = match producer(query, value) {
                    Some(ExprKind::Column(column)) => *column,
                    _ => {
                        terms.push(term);
                        continue;
                    }
                };
                if !is_runtime_bound(query, low) || !is_runtime_bound(query, high) {
                    terms.push(term);
                    continue;
                }
                let Some(column_def) = query.column(column).cloned() else {
                    terms.push(term);
                    continue;
                };
                let low_column = query.add_expr(
                    ExprKind::Column(column),
                    column_def.ty,
                    column_def.nullable,
                );
                let high_column = query.add_expr(
                    ExprKind::Column(column),
                    column_def.ty,
                    column_def.nullable,
                );
                let low_nullable = column_def.nullable
                    || query.expr(low).is_none_or(|expr| expr.nullable);
                let high_nullable = column_def.nullable
                    || query.expr(high).is_none_or(|expr| expr.nullable);
                let ge = query.add_expr(
                    ExprKind::Binary {
                        op: BinaryOp::Ge,
                        left: low_column,
                        right: low,
                    },
                    ScalarType::Bool,
                    low_nullable,
                );
                let le = query.add_expr(
                    ExprKind::Binary {
                        op: BinaryOp::Le,
                        left: high_column,
                        right: high,
                    },
                    ScalarType::Bool,
                    high_nullable,
                );
                terms.push(ge);
                terms.push(le);
                expanded_between = true;
            }

            let mut remove = vec![false; terms.len()];
            let mut seen_terms: HashSet<ExprId> = HashSet::default();
            for (index, &term) in terms.iter().enumerate() {
                if !seen_terms.insert(term) {
                    remove[index] = true;
                }
            }

            let mut comparisons: HashMap<ColumnId, Vec<ConstantComparison>> = HashMap::default();
            let mut comparison_columns: HashSet<ColumnId> = HashSet::default();
            let mut is_null_terms: HashMap<ColumnId, Vec<usize>> = HashMap::default();
            let mut is_not_null_terms: HashMap<ColumnId, Vec<usize>> = HashMap::default();
            let mut contradiction = false;

            for (index, &term) in terms.iter().enumerate() {
                if let Some((column, negated)) = is_null_column(query, term) {
                    if negated {
                        is_not_null_terms.entry(column).or_default().push(index);
                    } else {
                        is_null_terms.entry(column).or_default().push(index);
                    }
                    continue;
                }

                let Some((column, op, bound)) = comparison_column_bound(query, term) else {
                    continue;
                };
                comparison_columns.insert(column);
                let Some(value) = constant(query, bound) else {
                    continue;
                };
                if matches!(value, ScalarValue::Null) {
                    contradiction = true;
                    break;
                }
                comparisons.entry(column).or_default().push(ConstantComparison {
                    term_index: index,
                    op,
                    bound,
                    value,
                });
            }

            if !contradiction {
                for column in &comparison_columns {
                    if is_null_terms.contains_key(column) {
                        // Every ordinary comparison is null-rejecting, while IS NULL
                        // requires the opposite. Their conjunction cannot be TRUE.
                        contradiction = true;
                        break;
                    }
                    if let Some(indices) = is_not_null_terms.get(column) {
                        for &index in indices {
                            remove[index] = true;
                        }
                    }
                }
            }

            if !contradiction {
                for (column, nulls) in &is_null_terms {
                    if let Some(not_nulls) = is_not_null_terms.get(column) {
                        if !nulls.is_empty() && !not_nulls.is_empty() {
                            contradiction = true;
                            break;
                        }
                    }
                    for &duplicate in nulls.iter().skip(1) {
                        remove[duplicate] = true;
                    }
                }
                for not_nulls in is_not_null_terms.values() {
                    for &duplicate in not_nulls.iter().skip(1) {
                        remove[duplicate] = true;
                    }
                }
            }

            let mut additions = Vec::new();
            if !contradiction {
                for (&column, constraints) in &comparisons {
                    let equalities: Vec<_> = constraints
                        .iter()
                        .filter(|constraint| constraint.op == BinaryOp::Eq)
                        .collect();

                    if let Some(first) = equalities.first().copied() {
                        for equality in equalities.iter().skip(1).copied() {
                            if !scalar_eq(&first.value, &equality.value) {
                                contradiction = true;
                                break;
                            }
                            remove[equality.term_index] = true;
                        }
                        if contradiction {
                            break;
                        }

                        for constraint in constraints {
                            if constraint.term_index == first.term_index
                                || constraint.op == BinaryOp::Eq
                            {
                                continue;
                            }
                            if let Some(satisfied) = equality_satisfies_comparison(&first.value, constraint) {
                                if satisfied {
                                    remove[constraint.term_index] = true;
                                } else {
                                    contradiction = true;
                                    break;
                                }
                            }
                        }
                        if contradiction {
                            break;
                        }
                        continue;
                    }

                    let mut lower: Option<&ConstantComparison> = None;
                    let mut upper: Option<&ConstantComparison> = None;
                    let mut not_equal: HashMap<ScalarValue, usize> = HashMap::default();

                    for constraint in constraints {
                        match constraint.op {
                            BinaryOp::Gt | BinaryOp::Ge => match lower {
                                None => lower = Some(constraint),
                                Some(current) if lower_is_tighter(constraint, current) => {
                                    remove[current.term_index] = true;
                                    lower = Some(constraint);
                                }
                                Some(current)
                                    if scalar_order(&constraint.value, &current.value).is_some() =>
                                {
                                    remove[constraint.term_index] = true;
                                }
                                _ => {}
                            },
                            BinaryOp::Lt | BinaryOp::Le => match upper {
                                None => upper = Some(constraint),
                                Some(current) if upper_is_tighter(constraint, current) => {
                                    remove[current.term_index] = true;
                                    upper = Some(constraint);
                                }
                                Some(current)
                                    if scalar_order(&constraint.value, &current.value).is_some() =>
                                {
                                    remove[constraint.term_index] = true;
                                }
                                _ => {}
                            },
                            BinaryOp::Ne => {
                                if not_equal.insert(constraint.value.clone(), constraint.term_index).is_some() {
                                    remove[constraint.term_index] = true;
                                }
                            }
                            _ => {}
                        }
                    }

                    if let (Some(lower), Some(upper)) = (lower, upper) {
                        match scalar_order(&lower.value, &upper.value) {
                            Some(std::cmp::Ordering::Greater) => {
                                contradiction = true;
                                break;
                            }
                            Some(std::cmp::Ordering::Equal) => {
                                let inclusive = lower.op == BinaryOp::Ge && upper.op == BinaryOp::Le;
                                if !inclusive {
                                    contradiction = true;
                                    break;
                                }
                                remove[lower.term_index] = true;
                                remove[upper.term_index] = true;
                                let Some(column_def) = query.column(column).cloned() else {
                                    continue;
                                };
                                let column_expr = query.add_expr(
                                    ExprKind::Column(column),
                                    column_def.ty,
                                    column_def.nullable,
                                );
                                let equality = query.add_expr(
                                    ExprKind::Binary {
                                        op: BinaryOp::Eq,
                                        left: column_expr,
                                        right: lower.bound,
                                    },
                                    ScalarType::Bool,
                                    column_def.nullable,
                                );
                                additions.push(equality);
                            }
                            _ => {}
                        }
                    }
                }
            }

            if contradiction {
                let columns = output_columns(query, catalog, input);
                query.plan_mut(plan).unwrap().kind = PlanKind::Empty { columns };
                changed_any = true;
                continue;
            }

            let changed = expanded_between
                || !additions.is_empty()
                || remove.iter().copied().any(|removed| removed);
            if !changed {
                continue;
            }

            let mut canonical_terms: Vec<_> = terms
                .into_iter()
                .enumerate()
                .filter_map(|(index, term)| (!remove[index]).then_some(term))
                .collect();
            canonical_terms.extend(additions);

            if canonical_terms.is_empty() {
                changed_any |= replace_plan_uses(query, plan, input);
                continue;
            }

            let predicate = conjunction(query, &canonical_terms).unwrap();
            query.plan_mut(plan).unwrap().kind = PlanKind::Filter { input, predicate };
            changed_any = true;
        }

        changed_any
    }
}

// -----------------------------------------------------------------------------
// Logical rewrites
// -----------------------------------------------------------------------------

#[derive(Default)]
pub struct LogicalSimplify;

impl QueryOptimization for LogicalSimplify {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let mut changed = false;
        for id in query.live_plans_postorder() {
            let Some(kind) = query.plan(id).map(|node| node.kind.clone()) else { continue; };
            match kind {
                PlanKind::Filter { input, predicate } => {
                    if bool_const(query, predicate) == Some(true) {
                        changed |= replace_plan_uses(query, id, input);
                        continue;
                    }
                    if bool_const(query, predicate) == Some(false)
                        || is_null_const(query, predicate)
                        || matches!(query.plan(input).map(|n| &n.kind), Some(PlanKind::Empty { .. }))
                    {
                        let columns = output_columns(query, catalog, input);
                        query.plan_mut(id).unwrap().kind = PlanKind::Empty { columns };
                        changed = true;
                        continue;
                    }
                    if let Some(PlanKind::Filter { input: grandchild, predicate: inner }) = query.plan(input).map(|n| n.kind.clone()) {
                        let combined = query.add_expr(
                            ExprKind::Binary { op: BinaryOp::And, left: inner, right: predicate },
                            ScalarType::Bool,
                            false,
                        );
                        query.plan_mut(id).unwrap().kind = PlanKind::Filter { input: grandchild, predicate: combined };
                        changed = true;
                    }
                }
                PlanKind::Project { input, expressions } => {
                    if matches!(query.plan(input).map(|n| &n.kind), Some(PlanKind::Empty { .. })) {
                        let columns = expressions.iter().map(|projection| projection.column).collect();
                        query.plan_mut(id).unwrap().kind = PlanKind::Empty { columns };
                        changed = true;
                        continue;
                    }
                    let input_fields = output_columns(query, catalog, input);
                    let projected_fields: Vec<_> = expressions.iter().map(|projection| projection.column).collect();
                    if projected_fields == input_fields
                        && expressions.iter().all(|projection| matches!(producer(query, projection.expr), Some(ExprKind::Column(field)) if *field == projection.column))
                    {
                        changed |= replace_plan_uses(query, id, input);
                    }
                }
                PlanKind::Sort { input, keys } => {
                    if matches!(query.plan(input).map(|n| &n.kind), Some(PlanKind::Empty { .. })) {
                        let columns = output_columns(query, catalog, input);
                        query.plan_mut(id).unwrap().kind = PlanKind::Empty { columns };
                        changed = true;
                    } else if let Some(PlanKind::Sort { input: inner_input, keys: inner_keys }) = query.plan(input).map(|n| n.kind.clone()) {
                        if keys == inner_keys {
                            query.plan_mut(id).unwrap().kind = PlanKind::Sort { input: inner_input, keys };
                            changed = true;
                        }
                    }
                }
                PlanKind::Limit { input, skip, limit } => {
                    if limit == Some(0) || matches!(query.plan(input).map(|n| &n.kind), Some(PlanKind::Empty { .. })) {
                        let columns = output_columns(query, catalog, input);
                        query.plan_mut(id).unwrap().kind = PlanKind::Empty { columns };
                        changed = true;
                        continue;
                    }
                    if let Some(PlanKind::Limit { input: inner_input, skip: inner_skip, limit: inner_limit }) = query.plan(input).map(|n| n.kind.clone()) {
                        let merged_skip = inner_skip.saturating_add(skip);
                        let inner_remaining = inner_limit.map(|value| value.saturating_sub(skip));
                        let merged_limit = match (limit, inner_remaining) {
                            (Some(a), Some(b)) => Some(a.min(b)),
                            (Some(a), None) => Some(a),
                            (None, Some(b)) => Some(b),
                            (None, None) => None,
                        };
                        query.plan_mut(id).unwrap().kind = PlanKind::Limit {
                            input: inner_input,
                            skip: merged_skip,
                            limit: merged_limit,
                        };
                        changed = true;
                    }
                }
                PlanKind::Distinct { input, columns } => {
                    if let Some(PlanKind::Distinct { input: inner, columns: inner_columns }) = query.plan(input).map(|n| n.kind.clone()) {
                        if columns == inner_columns {
                            query.plan_mut(id).unwrap().kind = PlanKind::Distinct { input: inner, columns };
                            changed = true;
                        }
                    }
                }
                PlanKind::UnionAll { inputs, columns } => {
                    let mut flattened = Vec::new();
                    for input in &inputs {
                        match query.plan(*input).map(|n| n.kind.clone()) {
                            Some(PlanKind::Empty { .. }) => {}
                            Some(PlanKind::UnionAll { inputs: nested, .. }) => flattened.extend(nested),
                            _ => flattened.push(*input),
                        }
                    }
                    if flattened.is_empty() {
                        query.plan_mut(id).unwrap().kind = PlanKind::Empty { columns };
                        changed = true;
                    } else if flattened.len() == 1 {
                        // Preserve the union-owned output identities unless they are
                        // already exactly the child's output identities.
                        if output_columns(query, catalog, flattened[0]) == columns {
                            changed |= replace_plan_uses(query, id, flattened[0]);
                        } else if flattened != inputs {
                            query.plan_mut(id).unwrap().kind = PlanKind::UnionAll { inputs: flattened, columns };
                            changed = true;
                        }
                    } else if flattened != inputs {
                        query.plan_mut(id).unwrap().kind = PlanKind::UnionAll { inputs: flattened, columns };
                        changed = true;
                    }
                }
                PlanKind::FetchRows { input, columns, .. } => {
                    if matches!(query.plan(input).map(|n| &n.kind), Some(PlanKind::Empty { .. })) {
                        query.plan_mut(id).unwrap().kind = PlanKind::Empty { columns };
                        changed = true;
                    }
                }
                PlanKind::Materialize { input, .. } => {
                    if matches!(query.plan(input).map(|n| &n.kind), Some(PlanKind::Empty { .. })) {
                        let columns = output_columns(query, catalog, input);
                        query.plan_mut(id).unwrap().kind = PlanKind::Empty { columns };
                        changed = true;
                    }
                }
                _ => {}
            }
        }
        changed
    }
}

fn replace_plan_uses(query: &mut Query, old: PlanId, new: PlanId) -> bool {
    if old == new {
        return false;
    }
    let mut changed = false;
    if query.root == old {
        query.root = new;
        changed = true;
    }
    let live = query.live_plans_preorder();
    for parent in live {
        if parent == old {
            continue;
        }
        let Some(node) = query.plan_mut(parent) else { continue; };
        changed |= replace_child(&mut node.kind, old, new);
    }
    changed
}

fn replace_child(kind: &mut PlanKind, old: PlanId, new: PlanId) -> bool {
    let replace = |slot: &mut PlanId| {
        if *slot == old { *slot = new; true } else { false }
    };
    match kind {
        PlanKind::Filter { input, .. }
        | PlanKind::Project { input, .. }
        | PlanKind::Sort { input, .. }
        | PlanKind::Limit { input, .. }
        | PlanKind::Distinct { input, .. }
        | PlanKind::Aggregate { input, .. }
        | PlanKind::Exists { input, .. }
        | PlanKind::Update { input, .. }
        | PlanKind::Delete { input, .. }
        | PlanKind::FetchRows { input, .. }
        | PlanKind::Materialize { input, .. } => replace(input),
        PlanKind::Join { left, right, .. } | PlanKind::PhysicalJoin { left, right, .. } => replace(left) | replace(right),
        PlanKind::UnionAll { inputs, .. } => inputs.iter_mut().fold(false, |changed, input| replace(input) | changed),
        _ => false,
    }
}

#[derive(Default)]
pub struct ContradictionElimination;

impl QueryOptimization for ContradictionElimination {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let mut changed = false;
        for id in query.live_plans_preorder() {
            let Some(PlanKind::Filter { input, predicate }) = query.plan(id).map(|n| n.kind.clone()) else { continue; };
            let mut equalities: HashMap<ColumnId, ScalarValue> = HashMap::default();
            let mut terms = Vec::new();
            flatten_boolean(query, predicate, BinaryOp::And, &mut terms);
            let mut contradiction = false;
            for term in terms {
                let Some((field, bound)) = equality_column_bound(query, term) else { continue; };
                let Some(value) = constant(query, bound) else { continue; };
                if matches!(&value, ScalarValue::Null) {
                    contradiction = true;
                    break;
                }
                if let Some(previous) = equalities.get(&field) {
                    if !scalar_eq(previous, &value) {
                        contradiction = true;
                        break;
                    }
                } else {
                    equalities.insert(field, value);
                }
            }
            if contradiction {
                let columns = output_columns(query, catalog, input);
                query.plan_mut(id).unwrap().kind = PlanKind::Empty { columns };
                changed = true;
            }
        }
        changed
    }
}

#[derive(Default)]
pub struct PredicatePushdown;

impl QueryOptimization for PredicatePushdown {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let mut changed = false;
        let live = query.live_plans_preorder();
        for id in live {
            let Some(PlanKind::Filter { input, predicate }) = query.plan(id).map(|n| n.kind.clone()) else { continue; };
            let Some(input_kind) = query.plan(input).map(|n| n.kind.clone()) else { continue; };

            match input_kind {
                PlanKind::Sort { input: child, keys } => {
                    let filtered = query.add_plan(PlanKind::Filter { input: child, predicate });
                    query.plan_mut(id).unwrap().kind = PlanKind::Sort { input: filtered, keys };
                    changed = true;
                }
                PlanKind::Join { left, right, kind: JoinKind::Inner, on } => {
                    let left_fields = output_columns(query, catalog, left);
                    let right_fields = output_columns(query, catalog, right);
                    let mut terms = Vec::new();
                    flatten_boolean(query, predicate, BinaryOp::And, &mut terms);
                    let mut left_terms = Vec::new();
                    let mut right_terms = Vec::new();
                    let mut residual = Vec::new();
                    for term in terms {
                        let fields = columns_in_expr(query, term);
                        if !fields.is_empty() && fields.iter().all(|field| left_fields.contains(field)) {
                            left_terms.push(term);
                        } else if !fields.is_empty() && fields.iter().all(|field| right_fields.contains(field)) {
                            right_terms.push(term);
                        } else {
                            residual.push(term);
                        }
                    }
                    if left_terms.is_empty() && right_terms.is_empty() {
                        continue;
                    }
                    let new_left = if left_terms.is_empty() {
                        left
                    } else {
                        let predicate = conjunction(query, &left_terms).unwrap();
                        query.add_plan(PlanKind::Filter { input: left, predicate })
                    };
                    let new_right = if right_terms.is_empty() {
                        right
                    } else {
                        let predicate = conjunction(query, &right_terms).unwrap();
                        query.add_plan(PlanKind::Filter { input: right, predicate })
                    };
                    let join = query.add_plan(PlanKind::Join {
                        left: new_left,
                        right: new_right,
                        kind: JoinKind::Inner,
                        on,
                    });
                    if let Some(predicate) = conjunction(query, &residual) {
                        query.plan_mut(id).unwrap().kind = PlanKind::Filter { input: join, predicate };
                    } else {
                        replace_plan_uses(query, id, join);
                    }
                    changed = true;
                }
                _ => {}
            }
        }
        changed
    }
}

#[derive(Default)]
pub struct PredicateReordering;

impl QueryOptimization for PredicateReordering {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let mut targets = Vec::new();
        for id in query.live_plans_preorder() {
            if let Some(node) = query.plan(id) {
                match &node.kind {
                    PlanKind::Filter { predicate, .. } => targets.push((id, *predicate, 0u8)),
                    PlanKind::PointGet { residual: Some(predicate), .. }
                    | PlanKind::PrimaryRangeScan { residual: Some(predicate), .. }
                    | PlanKind::IndexRangeScan { residual: Some(predicate), .. } => targets.push((id, *predicate, 1u8)),
                    _ => {}
                }
            }
        }

        let mut changed = false;
        for (plan, predicate, slot) in targets {
            let mut terms = Vec::new();
            flatten_boolean(query, predicate, BinaryOp::And, &mut terms);
            if terms.len() < 2 {
                continue;
            }
            let before = terms.clone();
            terms.sort_by_key(|term| predicate_cost(query, *term));
            if terms == before {
                continue;
            }
            let new_predicate = conjunction(query, &terms).unwrap();
            let node = query.plan_mut(plan).unwrap();
            match (&mut node.kind, slot) {
                (PlanKind::Filter { predicate, .. }, 0) => *predicate = new_predicate,
                (PlanKind::PointGet { residual, .. }, 1)
                | (PlanKind::PrimaryRangeScan { residual, .. }, 1)
                | (PlanKind::IndexRangeScan { residual, .. }, 1) => *residual = Some(new_predicate),
                _ => continue,
            }
            changed = true;
        }
        changed
    }
}

fn predicate_cost(query: &Query, id: ExprId) -> u16 {
    match producer(query, id) {
        Some(ExprKind::Binary { op, left, right }) if op.is_comparison() => {
            let field_bound = matches!(producer(query, *left), Some(ExprKind::Column(_))) && is_runtime_bound(query, *right);
            if field_bound { 1 } else { 4 }
        }
        Some(ExprKind::IsNull { .. }) => 1,
        Some(ExprKind::InList { items, .. }) => 2 + items.len().min(32) as u16,
        Some(ExprKind::Call { .. }) => 16,
        Some(ExprKind::Binary { op: BinaryOp::And | BinaryOp::Or, .. }) => 8,
        _ => 6,
    }
}

// -----------------------------------------------------------------------------
// Join normalization, transitive constraints and bounded join-order enumeration
// -----------------------------------------------------------------------------

/// Convert a LEFT join to an INNER join when a WHERE predicate rejects every
/// null-extended right-side row. This exposes substantially more predicate pushdown
/// and index-selection opportunities while preserving outer-join semantics.
#[derive(Default)]
pub struct OuterJoinReduction;

impl QueryOptimization for OuterJoinReduction {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let mut changed = false;
        let filters: Vec<_> = query
            .live_plans_preorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::Filter { input, predicate }) => Some((input, predicate)),
                _ => None,
            })
            .collect();

        for (input, predicate) in filters {
            let Some(PlanKind::Join {
                left,
                right,
                kind: JoinKind::Left,
                on,
            }) = query.plan(input).map(|node| node.kind.clone())
            else {
                continue;
            };

            let right_columns: HashSet<_> = output_columns(query, catalog, right).into_iter().collect();
            if right_columns.is_empty()
                || !predicate_null_rejects_columns(query, predicate, &right_columns)
            {
                continue;
            }

            query.plan_mut(input).unwrap().kind = PlanKind::Join {
                left,
                right,
                kind: JoinKind::Inner,
                on,
            };
            changed = true;
        }

        changed
    }
}

fn predicate_null_rejects_columns(
    query: &Query,
    predicate: ExprId,
    columns: &HashSet<ColumnId>,
) -> bool {
    let Some(kind) = producer(query, predicate) else {
        return false;
    };

    match kind {
        ExprKind::Binary {
            op: BinaryOp::And,
            left,
            right,
        } => {
            predicate_null_rejects_columns(query, *left, columns)
                || predicate_null_rejects_columns(query, *right, columns)
        }
        ExprKind::Binary {
            op: BinaryOp::Or,
            left,
            right,
        } => {
            predicate_null_rejects_columns(query, *left, columns)
                && predicate_null_rejects_columns(query, *right, columns)
        }
        ExprKind::Binary { op, left, right } if op.is_comparison() => {
            expr_references_any_column(query, *left, columns)
                || expr_references_any_column(query, *right, columns)
        }
        ExprKind::IsNull {
            value,
            negated: true,
        } => expr_references_any_column(query, *value, columns),
        ExprKind::Between {
            value,
            low,
            high,
            ..
        } => {
            expr_references_any_column(query, *value, columns)
                || expr_references_any_column(query, *low, columns)
                || expr_references_any_column(query, *high, columns)
        }
        ExprKind::InList { value, .. } => expr_references_any_column(query, *value, columns),
        ExprKind::Unary {
            op: UnaryOp::Not,
            value,
        } => match producer(query, *value) {
            Some(ExprKind::IsNull { value, negated }) => {
                !*negated && expr_references_any_column(query, *value, columns)
            }
            Some(ExprKind::Binary { op, left, right }) if op.is_comparison() => {
                expr_references_any_column(query, *left, columns)
                    || expr_references_any_column(query, *right, columns)
            }
            Some(ExprKind::Between { value, low, high, .. }) => {
                expr_references_any_column(query, *value, columns)
                    || expr_references_any_column(query, *low, columns)
                    || expr_references_any_column(query, *high, columns)
            }
            Some(ExprKind::InList { value, .. }) => {
                expr_references_any_column(query, *value, columns)
            }
            _ => false,
        },
        _ => false,
    }
}

fn expr_references_any_column(
    query: &Query,
    expression: ExprId,
    columns: &HashSet<ColumnId>,
) -> bool {
    columns_in_expr(query, expression)
        .iter()
        .any(|column| columns.contains(column))
}

/// Move one-sided ON terms of an INNER join into the corresponding input. This is
/// deliberately separate from WHERE pushdown: predicates originating in ON become
/// ordinary child Filters before access-path selection and can therefore become
/// point/range/index probes using the normal machinery.
#[derive(Default)]
pub struct JoinPredicateDecomposition;

impl QueryOptimization for JoinPredicateDecomposition {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let joins: Vec<_> = query
            .live_plans_postorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::Join {
                    left,
                    right,
                    kind: JoinKind::Inner,
                    on,
                }) => Some((id, left, right, on)),
                _ => None,
            })
            .collect();

        let mut changed = false;
        for (id, left, right, on) in joins {
            let left_columns: HashSet<_> = output_columns(query, catalog, left).into_iter().collect();
            let right_columns: HashSet<_> = output_columns(query, catalog, right).into_iter().collect();
            if left_columns.is_empty() || right_columns.is_empty() {
                continue;
            }

            let mut terms = Vec::new();
            flatten_boolean(query, on, BinaryOp::And, &mut terms);
            let mut left_terms = Vec::new();
            let mut right_terms = Vec::new();
            let mut join_terms = Vec::new();

            for term in terms {
                let referenced = columns_in_expr(query, term);
                if !referenced.is_empty()
                    && referenced.iter().all(|column| left_columns.contains(column))
                {
                    left_terms.push(term);
                } else if !referenced.is_empty()
                    && referenced.iter().all(|column| right_columns.contains(column))
                {
                    right_terms.push(term);
                } else {
                    join_terms.push(term);
                }
            }

            if left_terms.is_empty() && right_terms.is_empty() {
                continue;
            }

            let new_left = add_filter_terms(query, left, &left_terms);
            let new_right = add_filter_terms(query, right, &right_terms);
            let new_on = conjunction(query, &join_terms).unwrap_or_else(|| {
                query.add_expr(
                    ExprKind::Const(ScalarValue::Bool(true)),
                    ScalarType::Bool,
                    false,
                )
            });

            query.plan_mut(id).unwrap().kind = PlanKind::Join {
                left: new_left,
                right: new_right,
                kind: JoinKind::Inner,
                on: new_on,
            };
            changed = true;
        }

        changed
    }
}

fn add_filter_terms(query: &mut Query, input: PlanId, terms: &[ExprId]) -> PlanId {
    let Some(predicate) = conjunction(query, terms) else {
        return input;
    };
    query.add_plan(PlanKind::Filter { input, predicate })
}

/// Propagate parameter/constant comparisons through equi-join keys. For example,
/// `a.id = b.id AND a.id = $id` synthesizes `b.id = $id`, allowing both sides to
/// choose indexed access. Directionality is conservative for non-inner joins.
#[derive(Default)]
pub struct JoinConstraintPropagation;

impl QueryOptimization for JoinConstraintPropagation {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let joins: Vec<_> = query
            .live_plans_postorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::Join {
                    left,
                    right,
                    kind,
                    on,
                }) => Some((id, left, right, kind, on)),
                _ => None,
            })
            .collect();

        let mut changed = false;
        for (id, mut left, mut right, kind, on) in joins {
            let mut local_changed = false;
            let pairs = oriented_join_equalities(query, catalog, left, right, on);
            if pairs.is_empty() {
                continue;
            }

            let left_constraints = local_plan_comparisons(query, left);
            let right_constraints = local_plan_comparisons(query, right);

            let left_to_right = matches!(
                kind,
                JoinKind::Inner | JoinKind::Left | JoinKind::Semi | JoinKind::Anti
            );
            let right_to_left = matches!(kind, JoinKind::Inner | JoinKind::Semi);

            if left_to_right {
                let right_existing = local_plan_comparison_set(query, right);
                let additions = propagated_join_comparisons(
                    query,
                    &left_constraints,
                    &right_existing,
                    pairs.iter().map(|(left, right)| (*left, *right)),
                );
                if !additions.is_empty() {
                    right = add_filter_terms(query, right, &additions);
                    local_changed = true;
                }
            }

            if right_to_left {
                let left_existing = local_plan_comparison_set(query, left);
                let additions = propagated_join_comparisons(
                    query,
                    &right_constraints,
                    &left_existing,
                    pairs.iter().map(|(left, right)| (*right, *left)),
                );
                if !additions.is_empty() {
                    left = add_filter_terms(query, left, &additions);
                    local_changed = true;
                }
            }

            if local_changed {
                changed = true;
                if let Some(node) = query.plan_mut(id) {
                    node.kind = PlanKind::Join {
                        left,
                        right,
                        kind,
                        on,
                    };
                }
            }
        }

        changed
    }
}

fn local_plan_comparisons(query: &Query, plan: PlanId) -> Vec<ComparisonSignature> {
    let mut out = Vec::new();
    let mut current = Some(plan);
    let mut seen = HashSet::default();

    while let Some(id) = current {
        if !seen.insert(id) {
            break;
        }
        match query.plan(id).map(|node| &node.kind) {
            Some(PlanKind::Filter { input, predicate }) => {
                let mut terms = Vec::new();
                flatten_boolean(query, *predicate, BinaryOp::And, &mut terms);
                out.extend(terms.into_iter().filter_map(|term| {
                    comparison_column_bound(query, term).map(|(column, op, bound)| {
                        ComparisonSignature { column, op, bound }
                    })
                }));
                current = Some(*input);
            }
            _ => break,
        }
    }

    out
}

fn local_plan_comparison_set(query: &Query, plan: PlanId) -> HashSet<ComparisonSignature> {
    local_plan_comparisons(query, plan).into_iter().collect()
}

fn propagated_join_comparisons(
    query: &mut Query,
    source: &[ComparisonSignature],
    existing: &HashSet<ComparisonSignature>,
    mappings: impl IntoIterator<Item = (ColumnId, ColumnId)>,
) -> Vec<ExprId> {
    let mappings: Vec<_> = mappings.into_iter().collect();
    let mut emitted = HashSet::default();
    let mut out = Vec::new();

    for &(source_column, target_column) in &mappings {
        let Some(target_def) = query.column(target_column).cloned() else {
            continue;
        };
        for constraint in source.iter().filter(|constraint| constraint.column == source_column) {
            let signature = ComparisonSignature {
                column: target_column,
                op: constraint.op,
                bound: constraint.bound,
            };
            if existing.contains(&signature) || !emitted.insert(signature) {
                continue;
            }

            let column_expr = query.add_expr(
                ExprKind::Column(target_column),
                target_def.ty,
                target_def.nullable,
            );
            let nullable = target_def.nullable
                || query
                    .expr(constraint.bound)
                    .is_none_or(|bound| bound.nullable);
            out.push(query.add_expr(
                ExprKind::Binary {
                    op: constraint.op,
                    left: column_expr,
                    right: constraint.bound,
                },
                ScalarType::Bool,
                nullable,
            ));
        }
    }

    out
}

fn oriented_join_equalities(
    query: &Query,
    catalog: &Catalog,
    left: PlanId,
    right: PlanId,
    predicate: ExprId,
) -> Vec<(ColumnId, ColumnId)> {
    let left_columns: HashSet<_> = output_columns(query, catalog, left).into_iter().collect();
    let right_columns: HashSet<_> = output_columns(query, catalog, right).into_iter().collect();
    let mut terms = Vec::new();
    flatten_boolean(query, predicate, BinaryOp::And, &mut terms);

    let mut out = Vec::new();
    for term in terms {
        let Some((a, b)) = equality_columns(query, term) else {
            continue;
        };
        let pair = if left_columns.contains(&a) && right_columns.contains(&b) {
            Some((a, b))
        } else if left_columns.contains(&b) && right_columns.contains(&a) {
            Some((b, a))
        } else {
            None
        };
        if let Some(pair) = pair {
            if !out.contains(&pair) {
                out.push(pair);
            }
        }
    }
    out
}

const MAX_JOIN_REORDER_RELATIONS: usize = 8;

#[derive(Clone)]
struct JoinOrderState {
    cost: f64,
    rows: u64,
    order: Vec<usize>,
}

#[derive(Clone, Copy)]
struct JoinOrderTerm {
    expr: ExprId,
    relation_mask: u16,
}

/// Bounded dynamic-programming enumeration of connected INNER-join orders. The
/// selected tree is left-deep because that maps naturally onto hash/merge/index-NL
/// execution, but unlike a greedy sorter the DP accounts for intermediate result
/// cardinalities. Components larger than eight relations keep their source order.
#[derive(Default)]
pub struct JoinReordering;

impl QueryOptimization for JoinReordering {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let parents = query.parents();
        let candidates: Vec<_> = query
            .live_plans_postorder()
            .into_iter()
            .filter(|id| {
                matches!(
                    query.plan(*id).map(|node| &node.kind),
                    Some(PlanKind::Join {
                        kind: JoinKind::Inner,
                        ..
                    })
                ) && !parents.get(id).into_iter().flatten().any(|parent| {
                    matches!(
                        query.plan(*parent).map(|node| &node.kind),
                        Some(PlanKind::Join {
                            kind: JoinKind::Inner,
                            ..
                        })
                    )
                })
            })
            .collect();

        let mut changed = false;
        for root in candidates {
            let original_columns = output_columns(query, catalog, root);
            let mut leaves = Vec::new();
            let mut predicate_terms = Vec::new();
            flatten_inner_join_component(query, root, &mut leaves, &mut predicate_terms);

            if leaves.len() < 3 || leaves.len() > MAX_JOIN_REORDER_RELATIONS {
                continue;
            }
            let mut unique_leaves = HashSet::default();
            if !leaves.iter().all(|leaf| unique_leaves.insert(*leaf)) {
                continue;
            }

            let mut column_owner = HashMap::default();
            let mut disjoint = true;
            for (index, leaf) in leaves.iter().copied().enumerate() {
                for column in output_columns(query, catalog, leaf) {
                    if column_owner.insert(column, index).is_some() {
                        disjoint = false;
                        break;
                    }
                }
                if !disjoint {
                    break;
                }
            }
            if !disjoint {
                continue;
            }

            let terms: Vec<_> = predicate_terms
                .iter()
                .copied()
                .map(|expr| {
                    let mut mask = 0u16;
                    for column in columns_in_expr(query, expr) {
                        if let Some(owner) = column_owner.get(&column).copied() {
                            mask |= 1u16 << owner;
                        }
                    }
                    JoinOrderTerm {
                        expr,
                        relation_mask: mask,
                    }
                })
                .collect();

            let Some(best_order) = best_join_order(query, catalog, &leaves, &terms) else {
                continue;
            };
            let current_order: Vec<_> = (0..leaves.len()).collect();
            if best_order == current_order {
                continue;
            }

            let Some(replacement) = build_reordered_join(
                query,
                catalog,
                &leaves,
                &terms,
                &best_order,
                &original_columns,
            ) else {
                continue;
            };
            changed |= replace_plan_uses(query, root, replacement);
        }

        changed
    }
}

fn flatten_inner_join_component(
    query: &Query,
    plan: PlanId,
    leaves: &mut Vec<PlanId>,
    terms: &mut Vec<ExprId>,
) {
    match query.plan(plan).map(|node| &node.kind) {
        Some(PlanKind::Join {
            left,
            right,
            kind: JoinKind::Inner,
            on,
        }) => {
            flatten_inner_join_component(query, *left, leaves, terms);
            flatten_inner_join_component(query, *right, leaves, terms);
            flatten_boolean(query, *on, BinaryOp::And, terms);
        }
        _ => leaves.push(plan),
    }
}

fn best_join_order(
    query: &Query,
    catalog: &Catalog,
    leaves: &[PlanId],
    terms: &[JoinOrderTerm],
) -> Option<Vec<usize>> {
    let count = leaves.len();
    let state_count = 1usize << count;
    let mut states: Vec<Option<JoinOrderState>> = vec![None; state_count];

    for (index, leaf) in leaves.iter().copied().enumerate() {
        let rows = estimated_rows(query, catalog, leaf).max(1);
        let cost = query
            .plan(leaf)
            .map(|node| node.props.estimated_cost)
            .filter(|cost| *cost > 0.0)
            .unwrap_or(rows as f64);
        states[1usize << index] = Some(JoinOrderState {
            cost,
            rows,
            order: vec![index],
        });
    }

    for mask in 1usize..state_count {
        let Some(state) = states[mask].clone() else {
            continue;
        };
        for next in 0..count {
            let next_bit = 1usize << next;
            if mask & next_bit != 0 {
                continue;
            }
            let new_mask = mask | next_bit;
            let connecting = terms
                .iter()
                .filter(|term| {
                    let term_mask = term.relation_mask as usize;
                    term_mask != 0
                        && term_mask & mask != 0
                        && term_mask & next_bit != 0
                        && term_mask & !new_mask == 0
                })
                .count();
            if connecting == 0 {
                continue;
            }

            let next_rows = estimated_rows(query, catalog, leaves[next]).max(1);
            let divisor = 10u64.saturating_pow(connecting.min(6) as u32).max(1);
            let joined_rows = state
                .rows
                .saturating_mul(next_rows)
                .saturating_add(divisor - 1)
                / divisor;
            let joined_rows = joined_rows.max(1);
            let next_cost = query
                .plan(leaves[next])
                .map(|node| node.props.estimated_cost)
                .filter(|cost| *cost > 0.0)
                .unwrap_or(next_rows as f64);
            let cost = state.cost + next_cost + joined_rows as f64;
            let mut order = state.order.clone();
            order.push(next);

            let replace = match states[new_mask].as_ref() {
                None => true,
                Some(previous) => {
                    cost < previous.cost
                        || (cost == previous.cost && order < previous.order)
                }
            };
            if replace {
                states[new_mask] = Some(JoinOrderState {
                    cost,
                    rows: joined_rows,
                    order,
                });
            }
        }
    }

    states[state_count - 1].as_ref().map(|state| state.order.clone())
}

fn build_reordered_join(
    query: &mut Query,
    catalog: &Catalog,
    leaves: &[PlanId],
    terms: &[JoinOrderTerm],
    order: &[usize],
    original_columns: &[ColumnId],
) -> Option<PlanId> {
    let (&first, rest) = order.split_first()?;
    let mut current = leaves[first];
    let mut joined_mask = 1u16 << first;
    let mut used = vec![false; terms.len()];

    for &next in rest {
        let next_bit = 1u16 << next;
        let new_mask = joined_mask | next_bit;
        let mut join_terms = Vec::new();
        for (index, term) in terms.iter().enumerate() {
            if used[index]
                || term.relation_mask == 0
                || term.relation_mask & joined_mask == 0
                || term.relation_mask & next_bit == 0
                || term.relation_mask & !new_mask != 0
            {
                continue;
            }
            used[index] = true;
            join_terms.push(term.expr);
        }
        let on = conjunction(query, &join_terms)?;
        current = query.add_plan(PlanKind::Join {
            left: current,
            right: leaves[next],
            kind: JoinKind::Inner,
            on,
        });
        joined_mask = new_mask;
    }

    let residual: Vec<_> = terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| (!used[index]).then_some(term.expr))
        .collect();
    if let Some(predicate) = conjunction(query, &residual) {
        current = query.add_plan(PlanKind::Filter {
            input: current,
            predicate,
        });
    }

    let new_columns = output_columns(query, catalog, current);
    if new_columns != original_columns {
        let expressions = projection_aliases(query, original_columns);
        current = query.add_plan(PlanKind::Project {
            input: current,
            expressions,
        });
    }

    Some(current)
}

// -----------------------------------------------------------------------------
// Column pruning
// -----------------------------------------------------------------------------

#[derive(Default)]
pub struct ColumnPruning;

impl QueryOptimization for ColumnPruning {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let required = required_columns_by_plan(query, catalog);

        let mut changed = false;
        for id in query.live_plans_preorder() {
            let demand = required.get(&id).cloned().unwrap_or_default();
            let Some(kind) = query.plan(id).map(|node| node.kind.clone()) else { continue; };
            let replacement = match kind {
                PlanKind::TableScan { table, columns } => {
                    let desired = retain_scan_columns(query, table, &columns, &demand);
                    (desired != columns).then_some(PlanKind::TableScan { table, columns: desired })
                }
                PlanKind::PointGet { table, key, columns, residual, delivery } => {
                    let mut needed = demand.clone();
                    if let Some(predicate) = residual { needed.extend(columns_in_expr(query, predicate)); }
                    let desired = retain_scan_columns(query, table, &columns, &needed);
                    (desired != columns).then_some(PlanKind::PointGet {
                        table, key, columns: desired, residual, delivery,
                    })
                }
                PlanKind::PrimaryRangeScan { table, range, columns, residual, direction, skip, limit, delivery } => {
                    let mut needed = demand.clone();
                    if let Some(predicate) = residual { needed.extend(columns_in_expr(query, predicate)); }
                    let desired = retain_scan_columns(query, table, &columns, &needed);
                    (desired != columns).then_some(PlanKind::PrimaryRangeScan {
                        table, range, columns: desired, residual, direction, skip, limit, delivery,
                    })
                }
                PlanKind::IndexRangeScan {
                    table, index, range, columns, residual, direction, skip, limit, delivery, covering,
                } => {
                    let mut needed_columns = demand.clone();
                    if let Some(predicate) = residual { needed_columns.extend(columns_in_expr(query, predicate)); }
                    let desired = retain_scan_columns(query, table, &columns, &needed_columns);
                    let physical = storage_fields_for_columns(query, table, &desired);
                    let next_covering = match (catalog.index(index), catalog.table(table), physical) {
                        (Some(index_def), Some(table_def), Some(fields)) => index_def.covers(&fields, table_def),
                        _ => false,
                    };
                    (desired != columns || next_covering != covering).then_some(PlanKind::IndexRangeScan {
                        table,
                        index,
                        range,
                        columns: desired,
                        residual,
                        direction,
                        skip,
                        limit,
                        delivery,
                        covering: next_covering,
                    })
                }
                PlanKind::Project { input, mut expressions } => {
                    let before = expressions.len();
                    if !demand.is_empty() {
                        expressions.retain(|projection| demand.contains(&projection.column));
                    } else {
                        expressions.clear();
                    }
                    (expressions.len() != before).then_some(PlanKind::Project { input, expressions })
                }
                _ => None,
            };
            if let Some(kind) = replacement {
                query.plan_mut(id).unwrap().kind = kind;
                changed = true;
            }
        }
        changed
    }
}

fn required_columns_by_plan(
    query: &Query,
    catalog: &Catalog,
) -> HashMap<PlanId, HashSet<ColumnId>> {
    let mut required: HashMap<PlanId, HashSet<ColumnId>> = HashMap::default();
    required.insert(
        query.root,
        output_columns(query, catalog, query.root).into_iter().collect(),
    );

    let mut queue = vec![query.root];
    while let Some(id) = queue.pop() {
        let demand = required.get(&id).cloned().unwrap_or_default();
        let Some(kind) = query.plan(id).map(|node| node.kind.clone()) else {
            continue;
        };
        for (child, columns) in child_demands(query, catalog, &kind, &demand) {
            let entry = required.entry(child).or_default();
            let before = entry.len();
            entry.extend(columns);
            if entry.len() != before {
                queue.push(child);
            }
        }
    }

    required
}

fn child_demands(
    query: &Query,
    catalog: &Catalog,
    kind: &PlanKind,
    parent_demand: &HashSet<ColumnId>,
) -> Vec<(PlanId, HashSet<ColumnId>)> {
    let one = |input: PlanId, extra: HashSet<ColumnId>| {
        let mut demand = parent_demand.clone();
        demand.extend(extra);
        vec![(input, demand)]
    };

    match kind {
        PlanKind::Filter { input, predicate } => one(*input, columns_in_expr(query, *predicate)),
        PlanKind::Project { input, expressions } => {
            let mut demand = HashSet::default();
            for projection in expressions {
                if parent_demand.contains(&projection.column) {
                    demand.extend(columns_in_expr(query, projection.expr));
                }
            }
            vec![(*input, demand)]
        }
        PlanKind::Sort { input, keys } => {
            let mut extra = HashSet::default();
            for key in keys { extra.extend(columns_in_expr(query, key.expr)); }
            one(*input, extra)
        }
        PlanKind::Limit { input, .. } | PlanKind::Materialize { input, .. } => one(*input, HashSet::default()),
        PlanKind::Distinct { input, columns } => {
            let mut extra: HashSet<_> = columns.iter().copied().collect();
            extra.extend(parent_demand.iter().copied());
            vec![(*input, extra)]
        }
        PlanKind::Exists { input, .. } => vec![(*input, HashSet::default())],
        PlanKind::Aggregate { input, group_by, aggregates } => {
            let mut demand = HashSet::default();
            for projection in group_by { demand.extend(columns_in_expr(query, projection.expr)); }
            for aggregate in aggregates {
                if parent_demand.contains(&aggregate.column) {
                    if let Some(expr) = aggregate.argument { demand.extend(columns_in_expr(query, expr)); }
                }
            }
            vec![(*input, demand)]
        }
        PlanKind::Join { left, right, on, .. } | PlanKind::PhysicalJoin { left, right, on, .. } => {
            let left_columns: HashSet<_> = output_columns(query, catalog, *left).into_iter().collect();
            let right_columns: HashSet<_> = output_columns(query, catalog, *right).into_iter().collect();
            let on_columns = columns_in_expr(query, *on);
            let mut left_demand = HashSet::default();
            let mut right_demand = HashSet::default();
            for column in parent_demand.iter().chain(on_columns.iter()) {
                if left_columns.contains(column) { left_demand.insert(*column); }
                if right_columns.contains(column) { right_demand.insert(*column); }
            }
            vec![(*left, left_demand), (*right, right_demand)]
        }
        PlanKind::UnionAll { inputs, columns } => {
            inputs.iter().map(|input| {
                let child_columns = output_columns(query, catalog, *input);
                let mut demand = HashSet::default();
                for (ordinal, output) in columns.iter().enumerate() {
                    if parent_demand.contains(output) {
                        if let Some(child) = child_columns.get(ordinal) { demand.insert(*child); }
                    }
                }
                (*input, demand)
            }).collect()
        },
        PlanKind::Update { input, table, assignments } => {
            let mut extra = HashSet::default();
            for (_, expr) in assignments { extra.extend(columns_in_expr(query, *expr)); }
            extra.extend(primary_key_columns(query, catalog, *input, *table));
            one(*input, extra)
        }
        PlanKind::Delete { input, table } | PlanKind::FetchRows { input, table, .. } => {
            vec![(*input, primary_key_columns(query, catalog, *input, *table))]
        }
        _ => Vec::new(),
    }
}

fn retain_scan_columns(
    query: &Query,
    table: TableId,
    available: &[ColumnId],
    demand: &HashSet<ColumnId>,
) -> Vec<ColumnId> {
    available
        .iter()
        .copied()
        .filter(|column| {
            demand.contains(column)
                && matches!(query.storage_origin(*column), Some((origin_table, _)) if origin_table == table)
        })
        .collect()
}

fn storage_fields_for_columns(query: &Query, table: TableId, columns: &[ColumnId]) -> Option<Vec<FieldId>> {
    columns.iter().map(|column| {
        let (origin_table, field) = query.storage_origin(*column)?;
        (origin_table == table).then_some(field)
    }).collect()
}

fn primary_key_columns(query: &Query, catalog: &Catalog, input: PlanId, table: TableId) -> HashSet<ColumnId> {
    let Some(table_def) = catalog.table(table) else { return HashSet::default(); };
    let outputs = output_columns(query, catalog, input);
    outputs.into_iter().filter(|column| {
        matches!(query.storage_origin(*column), Some((origin_table, field)) if origin_table == table && table_def.primary_key.contains(&field))
    }).collect()
}


// -----------------------------------------------------------------------------
// Runtime-aware scan fusion and physical decode pruning
// -----------------------------------------------------------------------------

/// Fuse a relational Filter directly into a physical point/range access. Keeping the
/// predicate as a scan residual lets the executor evaluate it inside the SurrealMX
/// visitor callback, avoiding an intermediate row/materialization boundary.
#[derive(Default)]
pub struct ScanPipelineFusion;

impl QueryOptimization for ScanPipelineFusion {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let mut changed = false;
        for id in query.live_plans_postorder() {
            let Some(PlanKind::Filter { input, predicate }) = query.plan(id).map(|node| node.kind.clone()) else {
                continue;
            };
            let Some(kind) = query.plan(input).map(|node| node.kind.clone()) else {
                continue;
            };

            let replacement = match kind {
                PlanKind::PointGet { table, key, columns, residual, delivery } => {
                    Some(PlanKind::PointGet {
                        table,
                        key,
                        columns,
                        residual: fuse_residual(query, residual, predicate),
                        delivery,
                    })
                }
                PlanKind::PrimaryRangeScan {
                    table,
                    range,
                    columns,
                    residual,
                    direction,
                    skip,
                    limit,
                    delivery,
                } => Some(PlanKind::PrimaryRangeScan {
                    table,
                    range,
                    columns,
                    residual: fuse_residual(query, residual, predicate),
                    direction,
                    skip,
                    limit,
                    delivery,
                }),
                PlanKind::IndexRangeScan {
                    table,
                    index,
                    range,
                    columns,
                    residual,
                    direction,
                    skip,
                    limit,
                    delivery,
                    covering,
                } => Some(PlanKind::IndexRangeScan {
                    table,
                    index,
                    range,
                    columns,
                    residual: fuse_residual(query, residual, predicate),
                    direction,
                    skip,
                    limit,
                    delivery,
                    covering,
                }),
                _ => None,
            };

            let Some(replacement) = replacement else { continue; };
            query.plan_mut(input).unwrap().kind = replacement;
            changed |= replace_plan_uses(query, id, input);
        }
        changed
    }
}

fn fuse_residual(query: &mut Query, current: Option<ExprId>, extra: ExprId) -> Option<ExprId> {
    match current {
        None => Some(extra),
        Some(current) if current == extra => Some(current),
        Some(current) => {
            let nullable = query.expr(current).is_some_and(|expr| expr.nullable)
                || query.expr(extra).is_some_and(|expr| expr.nullable);
            Some(query.add_expr(
                ExprKind::Binary {
                    op: BinaryOp::And,
                    left: current,
                    right: extra,
                },
                ScalarType::Bool,
                nullable,
            ))
        }
    }
}

/// Re-run demand propagation after physical lowering. The ordinary ColumnPruning
/// pass already shrinks scans and Projects; this pass additionally shrinks FetchRows
/// so late materialization decodes only columns that survive above the fetch.
#[derive(Default)]
pub struct PhysicalDecodePruning;

impl QueryOptimization for PhysicalDecodePruning {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let mut changed = ColumnPruning.optimize_query(query, catalog);
        let required = required_columns_by_plan(query, catalog);

        for id in query.live_plans_preorder() {
            let Some(PlanKind::FetchRows { input, table, columns }) =
                query.plan(id).map(|node| node.kind.clone())
            else {
                continue;
            };
            let demand = required.get(&id).cloned().unwrap_or_default();
            let desired: Vec<_> = columns
                .iter()
                .copied()
                .filter(|column| demand.contains(column))
                .collect();
            if desired != columns {
                query.plan_mut(id).unwrap().kind = PlanKind::FetchRows {
                    input,
                    table,
                    columns: desired,
                };
                changed = true;
            }
        }

        changed
    }
}

// -----------------------------------------------------------------------------
// Multi-index access rewrites
// -----------------------------------------------------------------------------

const MAX_INDEX_UNION_BRANCHES: usize = 8;
const MIN_INDEX_INTERSECTION_ROWS: u64 = 256;

/// Expand a disjunction whose independent arms can each be satisfied by a bounded
/// primary/index access path. Branches are de-duplicated by primary key so a row
/// satisfying multiple OR arms is still produced exactly once.
#[derive(Default)]
pub struct IndexUnionSelection;

impl QueryOptimization for IndexUnionSelection {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let filters: Vec<_> = query
            .live_plans_postorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::Filter { input, predicate }) => Some((id, input, predicate)),
                _ => None,
            })
            .collect();

        let mut changed = false;
        for (plan, input, predicate) in filters {
            let Some(PlanKind::TableScan { table, columns }) =
                query.plan(input).map(|node| node.kind.clone())
            else {
                continue;
            };
            let Some(table_def) = catalog.table(table) else {
                continue;
            };

            let mut terms = Vec::new();
            flatten_boolean(query, predicate, BinaryOp::Or, &mut terms);
            if !(2..=MAX_INDEX_UNION_BRANCHES).contains(&terms.len()) {
                continue;
            }

            // Same-column equality ORs should already have become IN and are better
            // handled by MultiPointAccessSelection. Keep this pass for genuinely
            // independent access paths.
            let mut equality_columns_seen = HashSet::default();
            let all_same_equality_column = terms.iter().all(|term| {
                equality_column_bound(query, *term)
                    .map(|(column, _)| {
                        equality_columns_seen.insert(column);
                        true
                    })
                    .unwrap_or(false)
            }) && equality_columns_seen.len() == 1;
            if all_same_equality_column {
                continue;
            }

            let original_columns = columns.clone();
            let branch_columns = ensure_primary_key_columns(query, table_def, table, &columns);
            let primary_columns = primary_key_columns_from_slice(query, table_def, table, &branch_columns);
            if primary_columns.len() != table_def.primary_key.len() {
                continue;
            }

            let mut branch_kinds = Vec::with_capacity(terms.len());
            let mut useful = true;
            for term in terms {
                let kind = choose_access(query, catalog, table, branch_columns.clone(), Some(term));
                if !fully_indexed_or_branch(&kind) {
                    useful = false;
                    break;
                }
                branch_kinds.push(kind);
            }
            if !useful || branch_kinds.len() < 2 {
                continue;
            }

            let branches: Vec<_> = branch_kinds
                .into_iter()
                .map(|kind| primary_key_only_access(kind, &primary_columns))
                .map(|kind| query.add_plan(kind))
                .collect();
            let union = query.add_plan(PlanKind::UnionAll {
                inputs: branches,
                columns: primary_columns.clone(),
            });
            let distinct = query.add_plan(PlanKind::Distinct {
                input: union,
                columns: primary_columns.clone(),
            });

            if original_columns == primary_columns {
                changed |= replace_plan_uses(query, plan, distinct);
            } else if original_columns.iter().all(|column| primary_columns.contains(column)) {
                let expressions = projection_aliases(query, &original_columns);
                query.plan_mut(plan).unwrap().kind = PlanKind::Project {
                    input: distinct,
                    expressions,
                };
            } else {
                query.plan_mut(plan).unwrap().kind = PlanKind::FetchRows {
                    input: distinct,
                    table,
                    columns: original_columns,
                };
            }
            changed = true;
        }
        changed
    }
}

fn primary_key_only_access(kind: PlanKind, primary_columns: &[ColumnId]) -> PlanKind {
    match kind {
        PlanKind::PointGet { table, key, residual, delivery, .. } => PlanKind::PointGet {
            table,
            key,
            columns: primary_columns.to_vec(),
            residual,
            delivery,
        },
        PlanKind::PrimaryRangeScan {
            table,
            range,
            residual,
            direction,
            skip,
            limit,
            delivery,
            ..
        } => PlanKind::PrimaryRangeScan {
            table,
            range,
            columns: primary_columns.to_vec(),
            residual,
            direction,
            skip,
            limit,
            delivery,
        },
        PlanKind::IndexRangeScan {
            table,
            index,
            range,
            residual,
            direction,
            skip,
            limit,
            delivery,
            ..
        } => PlanKind::IndexRangeScan {
            table,
            index,
            range,
            columns: primary_columns.to_vec(),
            residual,
            direction,
            skip,
            limit,
            delivery,
            covering: true,
        },
        other => other,
    }
}

fn fully_indexed_or_branch(kind: &PlanKind) -> bool {
    match kind {
        PlanKind::PointGet { residual: None, .. } => true,
        PlanKind::PrimaryRangeScan { range, residual: None, .. }
        | PlanKind::IndexRangeScan { range, residual: None, .. } => range_is_bounded(range),
        _ => false,
    }
}

fn range_is_bounded(range: &KeyRange) -> bool {
    !range.prefix.is_empty()
        || !matches!(range.lower, BoundValue::Unbounded)
        || !matches!(range.upper, BoundValue::Unbounded)
}

/// Intersect two selective independent secondary-index predicates by joining their
/// primary-key streams. The primary row is fetched only after both indexes agree.
/// This is a deliberately bounded two-way transform; Wave 3 can generalize it into
/// full access-path enumeration together with join DP.
#[derive(Default)]
pub struct IndexIntersectionSelection;

impl QueryOptimization for IndexIntersectionSelection {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let filters: Vec<_> = query
            .live_plans_postorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::Filter { input, predicate }) => Some((id, input, predicate)),
                _ => None,
            })
            .collect();

        let mut changed = false;
        for (plan, input, predicate) in filters {
            let Some(PlanKind::TableScan { table, columns }) =
                query.plan(input).map(|node| node.kind.clone())
            else {
                continue;
            };
            let Some(table_def) = catalog.table(table) else {
                continue;
            };
            if table_def.stats.rows < MIN_INDEX_INTERSECTION_ROWS || table_def.primary_key.is_empty() {
                continue;
            }

            let mut terms = Vec::new();
            flatten_boolean(query, predicate, BinaryOp::And, &mut terms);
            if terms.len() < 2 {
                continue;
            }

            let mut candidates = Vec::new();
            for (term_index, &term) in terms.iter().enumerate() {
                let Some((index, candidate, rows)) = best_secondary_candidate_for_term(
                    query,
                    catalog,
                    table,
                    table_def,
                    &columns,
                    term,
                ) else {
                    continue;
                };
                candidates.push((term_index, index, candidate, rows));
            }
            if candidates.len() < 2 {
                continue;
            }
            candidates.sort_by_key(|candidate| candidate.3);

            let first = candidates[0].clone();
            let Some(second) = candidates
                .iter()
                .skip(1)
                .find(|candidate| candidate.1 != first.1)
                .cloned()
            else {
                continue;
            };

            let total = table_def.stats.rows.max(1);
            let intersection_rows = first
                .3
                .saturating_mul(second.3)
                .saturating_div(total)
                .max(1);
            let best_single = first.3.min(second.3).max(1);
            if intersection_rows.saturating_mul(2) >= best_single {
                continue;
            }

            let left_pk = ensure_primary_key_columns(query, table_def, table, &columns);
            let left_pk = primary_key_columns_from_slice(query, table_def, table, &left_pk);
            if left_pk.len() != table_def.primary_key.len() {
                continue;
            }
            let right_pk = clone_storage_columns(query, &left_pk);

            let left_scan = query.add_plan(index_scan_for_candidate(
                catalog,
                table,
                first.1,
                first.2,
                left_pk.clone(),
            ));
            let right_scan = query.add_plan(index_scan_for_candidate(
                catalog,
                table,
                second.1,
                second.2,
                right_pk.clone(),
            ));

            let Some(on) = primary_key_equality(query, &left_pk, &right_pk) else {
                continue;
            };
            let intersection = query.add_plan(PlanKind::PhysicalJoin {
                left: left_scan,
                right: right_scan,
                // A semi join is exactly a set intersection on primary keys and
                // keeps only the left PK stream, avoiding duplicate PK columns.
                kind: JoinKind::Semi,
                on,
                algorithm: JoinAlgorithm::Hash,
            });

            // Fetch the complete logical row only for PKs surviving both index probes.
            let fetched = query.add_plan(PlanKind::FetchRows {
                input: intersection,
                table,
                columns: columns.clone(),
            });

            let residual_terms: Vec<_> = terms
                .into_iter()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index != first.0 && index != second.0).then_some(term)
                })
                .collect();
            if let Some(residual) = conjunction(query, &residual_terms) {
                query.plan_mut(plan).unwrap().kind = PlanKind::Filter {
                    input: fetched,
                    predicate: residual,
                };
            } else {
                query.plan_mut(plan).unwrap().kind = PlanKind::FetchRows {
                    input: intersection,
                    table,
                    columns,
                };
            }
            changed = true;
        }
        changed
    }
}

fn best_secondary_candidate_for_term(
    query: &Query,
    catalog: &Catalog,
    table: TableId,
    table_def: &TableDef,
    scan_columns: &[ColumnId],
    term: ExprId,
) -> Option<(IndexId, AccessCandidate, u64)> {
    let constraints = collect_constraints(query, &[term]);
    let mut best: Option<(IndexId, AccessCandidate, u64)> = None;
    for index in catalog.indexes_for(table) {
        let physical_fields: Vec<_> = index.columns.iter().map(|column| column.field).collect();
        let Some(candidate) = candidate_for_fields(
            query,
            table,
            &physical_fields,
            scan_columns,
            &constraints,
            AccessSource::Index(index.id),
            index.unique,
            true,
            table_def,
        ) else {
            continue;
        };
        if !candidate.consumed.contains(&term) {
            continue;
        }
        let rows = estimate_access_candidate_rows(table_def, &physical_fields, &candidate);
        if best.as_ref().is_none_or(|(_, previous, previous_rows)| {
            rows < *previous_rows || (rows == *previous_rows && candidate.score > previous.score)
        }) {
            best = Some((index.id, candidate, rows));
        }
    }
    best
}

fn estimate_access_candidate_rows(
    table: &TableDef,
    physical_fields: &[FieldId],
    candidate: &AccessCandidate,
) -> u64 {
    let mut rows = table.stats.rows.max(1);
    for field in physical_fields.iter().take(candidate.range.prefix.len()) {
        let distinct = table.stats.distinct.get(field).copied().unwrap_or(10).max(1);
        rows = rows.saturating_add(distinct - 1) / distinct;
    }
    if !matches!(candidate.range.lower, BoundValue::Unbounded)
        || !matches!(candidate.range.upper, BoundValue::Unbounded)
    {
        rows = rows.saturating_add(3) / 4;
    }
    if candidate.point_key.is_some() {
        rows = 1;
    }
    rows.max(1)
}

fn index_scan_for_candidate(
    catalog: &Catalog,
    table: TableId,
    index: IndexId,
    candidate: AccessCandidate,
    columns: Vec<ColumnId>,
) -> PlanKind {
    let _ = catalog;
    let limit = candidate.point_key.is_some().then_some(1);
    PlanKind::IndexRangeScan {
        table,
        index,
        range: candidate.range,
        columns,
        residual: None,
        direction: ScanDirection::Forward,
        skip: 0,
        limit,
        delivery: ScanDelivery::Pairs,
        // Primary-key fields are implicitly available in every secondary entry.
        covering: true,
    }
}

fn ensure_primary_key_columns(
    query: &mut Query,
    table_def: &TableDef,
    table: TableId,
    columns: &[ColumnId],
) -> Vec<ColumnId> {
    let mut out = columns.to_vec();
    for &field in &table_def.primary_key {
        if out.iter().any(|column| {
            matches!(query.storage_origin(*column), Some((origin_table, origin_field)) if origin_table == table && origin_field == field)
        }) {
            continue;
        }
        let Some(field_def) = table_def.field(field) else {
            continue;
        };
        out.push(query.add_column(
            None,
            field_def.ty,
            field_def.nullable,
            ColumnOrigin::Storage { table, field },
        ));
    }
    out
}

fn primary_key_columns_from_slice(
    query: &Query,
    table_def: &TableDef,
    table: TableId,
    columns: &[ColumnId],
) -> Vec<ColumnId> {
    table_def
        .primary_key
        .iter()
        .filter_map(|field| {
            columns.iter().copied().find(|column| {
                matches!(query.storage_origin(*column), Some((origin_table, origin_field)) if origin_table == table && origin_field == *field)
            })
        })
        .collect()
}

fn clone_storage_columns(query: &mut Query, columns: &[ColumnId]) -> Vec<ColumnId> {
    let defs: Vec<_> = columns
        .iter()
        .filter_map(|column| query.column(*column).cloned())
        .collect();
    let mut out = Vec::with_capacity(defs.len());
    for column in defs {
        out.push(query.add_column(column.name, column.ty, column.nullable, column.origin));
    }
    out
}

fn primary_key_equality(query: &mut Query, left: &[ColumnId], right: &[ColumnId]) -> Option<ExprId> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let mut terms = Vec::with_capacity(left.len());
    for (&left, &right) in left.iter().zip(right) {
        let left_def = query.column(left)?.clone();
        let right_def = query.column(right)?.clone();
        let left_expr = query.add_expr(ExprKind::Column(left), left_def.ty, left_def.nullable);
        let right_expr = query.add_expr(ExprKind::Column(right), right_def.ty, right_def.nullable);
        terms.push(query.add_expr(
            ExprKind::Binary {
                op: BinaryOp::Eq,
                left: left_expr,
                right: right_expr,
            },
            ScalarType::Bool,
            left_def.nullable || right_def.nullable,
        ));
    }
    conjunction(query, &terms)
}

fn projection_aliases(query: &mut Query, columns: &[ColumnId]) -> Vec<ProjectionExpr> {
    let mut out = Vec::with_capacity(columns.len());
    for &column in columns {
        let Some(def) = query.column(column).cloned() else {
            continue;
        };
        let expr = query.add_expr(ExprKind::Column(column), def.ty, def.nullable);
        out.push(ProjectionExpr { column, expr });
    }
    out
}

// -----------------------------------------------------------------------------
// Access-path selection / range extraction
// -----------------------------------------------------------------------------


const MAX_MULTI_POINT_PROBES: usize = 32;

#[derive(Default)]
pub struct MultiPointAccessSelection;

impl QueryOptimization for MultiPointAccessSelection {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let filters: Vec<_> = query
            .live_plans_postorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::Filter { input, predicate }) => Some((id, input, predicate)),
                _ => None,
            })
            .collect();

        let mut changed = false;
        for (plan, input, predicate) in filters {
            let Some(PlanKind::TableScan { table, columns }) =
                query.plan(input).map(|node| node.kind.clone())
            else {
                continue;
            };
            let Some(table_def) = catalog.table(table) else {
                continue;
            };

            let mut terms = Vec::new();
            flatten_boolean(query, predicate, BinaryOp::And, &mut terms);

            let mut selected = None;
            for (term_index, &term) in terms.iter().enumerate() {
                let Some(ExprKind::InList {
                    value,
                    items,
                    negated: false,
                }) = producer(query, term)
                else {
                    continue;
                };
                let Some(ExprKind::Column(column)) = producer(query, *value) else {
                    continue;
                };
                let Some((origin_table, field)) = query.storage_origin(*column) else {
                    continue;
                };
                if origin_table != table {
                    continue;
                }

                let mut probes = Vec::new();
                let mut valid = true;
                for &item in items {
                    if !is_runtime_bound(query, item) {
                        valid = false;
                        break;
                    }
                    // NULL can never make an IN predicate TRUE. At Filter level it
                    // is therefore safe to omit it from the physical probe set.
                    if !is_null_const(query, item) {
                        probes.push(item);
                    }
                }
                if !valid || probes.len() < 2 || probes.len() > MAX_MULTI_POINT_PROBES {
                    continue;
                }

                let primary = table_def.primary_key.len() == 1
                    && table_def.primary_key[0] == field;
                let index = if primary {
                    None
                } else {
                    catalog
                        .indexes_for(table)
                        .filter(|index| {
                            index
                                .columns
                                .first()
                                .is_some_and(|candidate| candidate.field == field)
                        })
                        .max_by_key(|index| {
                            let requested_fields = storage_fields_for_columns(query, table, &columns)
                                .unwrap_or_default();
                            let covering = index.covers(&requested_fields, table_def);
                            (index.unique as u8, covering as u8, usize::MAX - index.columns.len())
                        })
                        .map(|index| index.id)
                };

                if !primary && index.is_none() {
                    continue;
                }
                selected = Some((term_index, probes, primary, index));
                break;
            }

            let Some((term_index, probes, primary, index)) = selected else {
                continue;
            };

            let residual_terms: Vec<_> = terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| (index != term_index).then_some(term))
                .collect();
            let residual = conjunction(query, &residual_terms);
            let mut branches = Vec::with_capacity(probes.len());

            for probe in probes {
                let branch = if primary {
                    query.add_plan(PlanKind::PointGet {
                        table,
                        key: vec![probe],
                        columns: columns.clone(),
                        residual,
                        delivery: PointDelivery::Owned,
                    })
                } else {
                    let index = index.unwrap();
                    let mut needed_columns = columns.clone();
                    if let Some(predicate) = residual {
                        needed_columns.extend(columns_in_expr(query, predicate));
                        needed_columns.sort_unstable_by_key(|column| column.0);
                        needed_columns.dedup();
                    }
                    let needed_fields = storage_fields_for_columns(query, table, &needed_columns)
                        .unwrap_or_default();
                    let index_def = catalog.index(index).unwrap();
                    let covering = index_def.covers(&needed_fields, table_def);
                    query.add_plan(PlanKind::IndexRangeScan {
                        table,
                        index,
                        range: KeyRange {
                            prefix: vec![probe],
                            ..Default::default()
                        },
                        columns: columns.clone(),
                        residual,
                        direction: ScanDirection::Forward,
                        skip: 0,
                        limit: (index_def.unique && index_def.columns.len() == 1).then_some(1),
                        delivery: ScanDelivery::Pairs,
                        covering,
                    })
                };
                branches.push(branch);
            }

            if branches.is_empty() {
                query.plan_mut(plan).unwrap().kind = PlanKind::Empty {
                    columns: columns.clone(),
                };
            } else if branches.len() == 1 {
                changed |= replace_plan_uses(query, plan, branches[0]);
                continue;
            } else {
                query.plan_mut(plan).unwrap().kind = PlanKind::UnionAll {
                    inputs: branches,
                    columns: columns.clone(),
                };
            }
            changed = true;
        }

        changed
    }
}

#[derive(Clone, Default)]
struct ColumnConstraint {
    equality: Option<(ExprId, ExprId)>, // bound, source term
    lower: Option<(BoundValue, ExprId)>,
    upper: Option<(BoundValue, ExprId)>,
}

#[derive(Clone)]
struct AccessCandidate {
    source: AccessSource,
    range: KeyRange,
    consumed: HashSet<ExprId>,
    score: i64,
    estimated_cost: f64,
    point_key: Option<Vec<ExprId>>,
}

#[derive(Clone, Copy)]
enum AccessSource {
    Primary,
    Index(IndexId),
}

#[derive(Default)]
pub struct AccessPathSelection;

impl QueryOptimization for AccessPathSelection {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let mut changed = false;

        // Filter(TableScan) is the best case: absorb the predicate into the access path.
        for id in query.live_plans_postorder() {
            let Some(PlanKind::Filter { input, predicate }) = query.plan(id).map(|n| n.kind.clone()) else { continue; };
            let Some(PlanKind::TableScan { table, columns }) = query.plan(input).map(|n| n.kind.clone()) else { continue; };
            let replacement = choose_access(query, catalog, table, columns, Some(predicate));
            query.plan_mut(id).unwrap().kind = replacement;
            changed = true;
        }

        // Remaining bare logical scans become full ordered primary scans.
        for id in query.live_plans_postorder() {
            let Some(PlanKind::TableScan { table, columns }) = query.plan(id).map(|n| n.kind.clone()) else { continue; };
            query.plan_mut(id).unwrap().kind = PlanKind::PrimaryRangeScan {
                table,
                range: KeyRange::default(),
                columns,
                residual: None,
                direction: ScanDirection::Forward,
                skip: 0,
                limit: None,
                delivery: ScanDelivery::Pairs,
            };
            changed = true;
        }

        changed
    }
}

fn choose_access(
    query: &mut Query,
    catalog: &Catalog,
    table: TableId,
    columns: Vec<ColumnId>,
    predicate: Option<ExprId>,
) -> PlanKind {
    let Some(table_def) = catalog.table(table) else {
        return PlanKind::Empty { columns };
    };
    let requested_fields = storage_fields_for_columns(query, table, &columns).unwrap_or_default();
    let Some(predicate) = predicate else {
        return PlanKind::PrimaryRangeScan {
            table,
            range: KeyRange::default(),
            columns,
            residual: None,
            direction: ScanDirection::Forward,
            skip: 0,
            limit: None,
            delivery: ScanDelivery::Pairs,
        };
    };

    let mut terms = Vec::new();
    flatten_boolean(query, predicate, BinaryOp::And, &mut terms);
    let constraints = collect_constraints(query, &terms);

    let mut candidates = Vec::new();
    if let Some(candidate) = candidate_for_fields(
        query,
        table,
        &table_def.primary_key,
        &columns,
        &constraints,
        AccessSource::Primary,
        true,
        false,
        table_def,
    ) {
        candidates.push(candidate);
    }
    for index in catalog.indexes_for(table) {
        let physical_fields: Vec<_> = index.columns.iter().map(|column| column.field).collect();
        let covering = index.covers(&requested_fields, table_def);
        if let Some(candidate) = candidate_for_fields(
            query,
            table,
            &physical_fields,
            &columns,
            &constraints,
            AccessSource::Index(index.id),
            index.unique,
            covering,
            table_def,
        ) {
            candidates.push(candidate);
        }
    }

    let Some(best) = candidates.into_iter().min_by(|left, right| {
        left.estimated_cost
            .total_cmp(&right.estimated_cost)
            .then_with(|| right.score.cmp(&left.score))
    }) else {
        return PlanKind::PrimaryRangeScan {
            table,
            range: KeyRange::default(),
            columns,
            residual: Some(predicate),
            direction: ScanDirection::Forward,
            skip: 0,
            limit: None,
            delivery: ScanDelivery::Pairs,
        };
    };

    let residual_terms: Vec<_> = terms.into_iter().filter(|term| !best.consumed.contains(term)).collect();
    let residual = conjunction(query, &residual_terms);

    let exact_unique_key = best.point_key.clone();
    if let Some(key) = exact_unique_key.as_ref() {
        if matches!(best.source, AccessSource::Primary) {
            return PlanKind::PointGet {
                table,
                key: key.clone(),
                columns,
                residual,
                delivery: PointDelivery::Owned,
            };
        }
    }

    match best.source {
        AccessSource::Primary => PlanKind::PrimaryRangeScan {
            table,
            range: best.range,
            columns,
            residual,
            direction: ScanDirection::Forward,
            skip: 0,
            limit: None,
            delivery: ScanDelivery::Pairs,
        },
        AccessSource::Index(index) => {
            let mut needed_columns = columns.clone();
            if let Some(predicate) = residual {
                needed_columns.extend(columns_in_expr(query, predicate));
                needed_columns.sort_unstable_by_key(|column| column.0);
                needed_columns.dedup();
            }
            let needed_fields = storage_fields_for_columns(query, table, &needed_columns).unwrap_or_default();
            let covering = catalog.index(index).is_some_and(|index| index.covers(&needed_fields, table_def));
            PlanKind::IndexRangeScan {
                table,
                index,
                range: best.range,
                columns,
                residual,
                direction: ScanDirection::Forward,
                skip: 0,
                limit: exact_unique_key.is_some().then_some(1),
                delivery: ScanDelivery::Pairs,
                covering,
            }
        }
    }
}

fn collect_constraints(query: &Query, terms: &[ExprId]) -> HashMap<ColumnId, ColumnConstraint> {
    let mut out: HashMap<ColumnId, ColumnConstraint> = HashMap::default();
    for &term in terms {
        match producer(query, term) {
            Some(ExprKind::Binary { op, left, right }) if op.is_comparison() => {
                let Some(ExprKind::Column(column)) = producer(query, *left) else { continue; };
                if !is_runtime_bound(query, *right) { continue; }
                let entry = out.entry(*column).or_default();
                match op {
                    BinaryOp::Eq => { entry.equality.get_or_insert((*right, term)); }
                    BinaryOp::Gt => { entry.lower.get_or_insert((BoundValue::Exclusive(*right), term)); }
                    BinaryOp::Ge => { entry.lower.get_or_insert((BoundValue::Inclusive(*right), term)); }
                    BinaryOp::Lt => { entry.upper.get_or_insert((BoundValue::Exclusive(*right), term)); }
                    BinaryOp::Le => { entry.upper.get_or_insert((BoundValue::Inclusive(*right), term)); }
                    _ => {}
                }
            }
            Some(ExprKind::Between { value, low, high, negated: false }) => {
                let Some(ExprKind::Column(column)) = producer(query, *value) else { continue; };
                if !is_runtime_bound(query, *low) || !is_runtime_bound(query, *high) { continue; }
                let entry = out.entry(*column).or_default();
                entry.lower.get_or_insert((BoundValue::Inclusive(*low), term));
                entry.upper.get_or_insert((BoundValue::Inclusive(*high), term));
            }
            _ => {}
        }
    }
    out
}

fn candidate_for_fields(
    query: &Query,
    table: TableId,
    physical_fields: &[FieldId],
    scan_columns: &[ColumnId],
    constraints: &HashMap<ColumnId, ColumnConstraint>,
    source: AccessSource,
    unique: bool,
    covering: bool,
    table_def: &TableDef,
) -> Option<AccessCandidate> {
    if physical_fields.is_empty() {
        return None;
    }
    let mut range = KeyRange::default();
    let mut consumed = HashSet::default();
    let mut point_key = Vec::new();
    let mut equality_prefix = 0usize;
    let mut has_range = false;

    for (position, physical_field) in physical_fields.iter().enumerate() {
        let Some(column) = scan_columns.iter().copied().find(|column| {
            matches!(query.storage_origin(*column), Some((origin_table, field)) if origin_table == table && field == *physical_field)
        }) else { break; };
        let Some(constraint) = constraints.get(&column) else { break; };
        if let Some((bound, term)) = constraint.equality {
            range.prefix.push(bound);
            point_key.push(bound);
            consumed.insert(term);
            equality_prefix += 1;
            continue;
        }
        if position == equality_prefix {
            if let Some((bound, term)) = &constraint.lower {
                range.lower = bound.clone();
                consumed.insert(*term);
                has_range = true;
            }
            if let Some((bound, term)) = &constraint.upper {
                range.upper = bound.clone();
                consumed.insert(*term);
                has_range = true;
            }
        }
        break;
    }

    if equality_prefix == 0 && !has_range {
        return None;
    }

    let point = unique && equality_prefix == physical_fields.len() && !has_range;
    let mut score = (equality_prefix as i64) * 120 + (has_range as i64) * 45;
    if point { score += 1_000; }
    if covering { score += 80; }

    // Prefer high-cardinality equality prefixes when two indexes consume the same length.
    let mut estimated_rows = table_def.stats.rows.max(1);
    for field in physical_fields.iter().take(equality_prefix) {
        if let Some(distinct) = table_def.stats.distinct.get(field) {
            let distinct = (*distinct).max(1);
            score += (distinct.ilog2() as i64).min(40);
            estimated_rows = estimated_rows.saturating_add(distinct - 1) / distinct;
            estimated_rows = estimated_rows.max(1);
        } else {
            estimated_rows = (estimated_rows / 10).max(1);
        }
    }
    if has_range {
        estimated_rows = (estimated_rows / 4).max(1);
    }
    if point {
        estimated_rows = 1;
    }

    // This is intentionally a memory-first model: touching primary-row bytes and
    // performing a second primary lookup matter more than any fictional disk seek.
    let touched_bytes = if covering {
        16u64.saturating_add((physical_fields.len() as u64).saturating_mul(8))
    } else {
        table_def.stats.average_row_bytes.max(8) as u64
    };
    let source_penalty = match source {
        AccessSource::Primary => 0.75,
        AccessSource::Index(_) if covering => 0.9,
        AccessSource::Index(_) => 1.8,
    };
    let estimated_cost = estimated_rows as f64
        * source_penalty
        * (1.0 + touched_bytes as f64 / 64.0);

    Some(AccessCandidate {
        source,
        range,
        consumed,
        score,
        estimated_cost,
        point_key: point.then_some(point_key),
    })
}

// -----------------------------------------------------------------------------
// Secondary-index specialization / late materialization
// -----------------------------------------------------------------------------

/// Exact equality on every component of a unique secondary index can produce at
/// most one entry. Keep that cardinality bound explicit so EXISTS/LIMIT and later
/// costing never treat it as a general range scan.
#[derive(Default)]
pub struct UniqueIndexSpecialization;

impl QueryOptimization for UniqueIndexSpecialization {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let scans: Vec<_> = query
            .live_plans_preorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::IndexRangeScan {
                    table,
                    index,
                    range,
                    columns,
                    residual,
                    direction,
                    skip,
                    limit,
                    delivery,
                    covering,
                }) => Some((
                    id, table, index, range, columns, residual, direction, skip, limit, delivery,
                    covering,
                )),
                _ => None,
            })
            .collect();

        let mut changed = false;
        for (id, table, index, range, columns, residual, direction, skip, limit, delivery, covering) in scans {
            let Some(index_def) = catalog.index(index) else {
                continue;
            };
            let exact_unique = index_def.unique
                && range.prefix.len() == index_def.columns.len()
                && matches!(range.lower, BoundValue::Unbounded)
                && matches!(range.upper, BoundValue::Unbounded);
            if !exact_unique {
                continue;
            }
            if skip > 0 || limit == Some(0) {
                query.plan_mut(id).unwrap().kind = PlanKind::Empty { columns };
                changed = true;
                continue;
            }
            let next_limit = Some(limit.map_or(1, |current| current.min(1)));
            if next_limit != limit {
                query.plan_mut(id).unwrap().kind = PlanKind::IndexRangeScan {
                    table,
                    index,
                    range,
                    columns,
                    residual,
                    direction,
                    skip: 0,
                    limit: next_limit,
                    delivery,
                    covering,
                };
                changed = true;
            }
        }
        changed
    }
}

/// Re-evaluate covering status after predicate/column pruning and eliminate an
/// already-inserted primary-row fetch when all requested data is available from
/// the secondary index entry (index key, implicit PK, or included fields).
#[derive(Default)]
pub struct CoveringIndexPromotion;

impl QueryOptimization for CoveringIndexPromotion {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let mut changed = false;

        let scans: Vec<_> = query
            .live_plans_preorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::IndexRangeScan {
                    table,
                    index,
                    columns,
                    residual,
                    covering,
                    ..
                }) => Some((id, table, index, columns, residual, covering)),
                _ => None,
            })
            .collect();

        for (id, table, index, columns, residual, covering) in scans {
            let Some(table_def) = catalog.table(table) else {
                continue;
            };
            let Some(index_def) = catalog.index(index) else {
                continue;
            };
            let mut needed = columns.clone();
            if let Some(residual) = residual {
                needed.extend(columns_in_expr(query, residual));
                sort_dedup_columns(&mut needed);
            }
            let Some(fields) = storage_fields_for_columns(query, table, &needed) else {
                continue;
            };
            let next = index_def.covers(&fields, table_def);
            if next != covering {
                if let Some(PlanKind::IndexRangeScan { covering, .. }) =
                    query.plan_mut(id).map(|node| &mut node.kind)
                {
                    *covering = next;
                    changed = true;
                }
            }
        }

        let fetches: Vec<_> = query
            .live_plans_preorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::FetchRows { input, table, columns }) => {
                    Some((id, input, table, columns))
                }
                _ => None,
            })
            .collect();

        for (fetch, input, table, final_columns) in fetches {
            let Some(PlanKind::IndexRangeScan {
                table: scan_table,
                index,
                range,
                columns: _,
                residual,
                direction,
                skip,
                limit,
                delivery,
                covering: _,
            }) = query.plan(input).map(|node| node.kind.clone())
            else {
                continue;
            };
            if scan_table != table {
                continue;
            }
            let (Some(table_def), Some(index_def)) = (catalog.table(table), catalog.index(index)) else {
                continue;
            };

            let mut needed = final_columns.clone();
            if let Some(residual) = residual {
                needed.extend(columns_in_expr(query, residual));
                sort_dedup_columns(&mut needed);
            }
            let Some(fields) = storage_fields_for_columns(query, table, &needed) else {
                continue;
            };
            if !index_def.covers(&fields, table_def) {
                continue;
            }

            query.plan_mut(input).unwrap().kind = PlanKind::IndexRangeScan {
                table,
                index,
                range,
                columns: needed.clone(),
                residual,
                direction,
                skip,
                limit,
                delivery,
                covering: true,
            };

            if needed == final_columns {
                changed |= replace_plan_uses(query, fetch, input);
            } else {
                let expressions = projection_aliases(query, &final_columns);
                query.plan_mut(fetch).unwrap().kind = PlanKind::Project {
                    input,
                    expressions,
                };
                changed = true;
            }
        }

        changed
    }
}

/// For a non-covering secondary index, evaluate every predicate that the index can
/// satisfy first, apply scan skip/limit there, and only then fetch primary rows.
/// Residual predicates requiring non-index fields are intentionally left alone: a
/// fetch before such a predicate would require moving LIMIT/OFFSET as well and is
/// better handled by a later general physical-plan enumerator.
#[derive(Default)]
pub struct LateRowFetch;

impl QueryOptimization for LateRowFetch {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let scans: Vec<_> = query
            .live_plans_preorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::IndexRangeScan {
                    table,
                    index,
                    range,
                    columns,
                    residual,
                    direction,
                    skip,
                    limit,
                    delivery,
                    covering: false,
                }) => Some((
                    id, table, index, range, columns, residual, direction, skip, limit, delivery,
                )),
                _ => None,
            })
            .collect();

        let mut changed = false;
        for (id, table, index, range, final_columns, residual, direction, skip, limit, delivery) in scans {
            let (Some(table_def), Some(index_def)) = (catalog.table(table), catalog.index(index)) else {
                continue;
            };

            let residual_columns: Vec<_> = residual
                .map(|predicate| columns_in_expr(query, predicate).into_iter().collect())
                .unwrap_or_default();
            let Some(residual_fields) = storage_fields_for_columns(query, table, &residual_columns) else {
                continue;
            };
            if !index_def.covers(&residual_fields, table_def) {
                continue;
            }

            let mut before_fetch = ensure_primary_key_columns(query, table_def, table, &[]);
            before_fetch.extend(residual_columns);
            sort_dedup_columns(&mut before_fetch);
            let Some(before_fields) = storage_fields_for_columns(query, table, &before_fetch) else {
                continue;
            };
            if !index_def.covers(&before_fields, table_def) {
                continue;
            }

            let scan = query.add_plan(PlanKind::IndexRangeScan {
                table,
                index,
                range,
                columns: before_fetch,
                residual,
                direction,
                skip,
                limit,
                delivery,
                covering: true,
            });
            query.plan_mut(id).unwrap().kind = PlanKind::FetchRows {
                input: scan,
                table,
                columns: final_columns,
            };
            changed = true;
        }
        changed
    }
}

fn sort_dedup_columns(columns: &mut Vec<ColumnId>) {
    columns.sort_unstable_by_key(|column| column.0);
    columns.dedup();
}

// -----------------------------------------------------------------------------
// Physical order, bounded scans, limits, count, joins, writes
// -----------------------------------------------------------------------------

/// Push MIN/MAX into an already ordered access path while retaining the Aggregate
/// node. Keeping the aggregate is important: on an empty input SQL-style MIN/MAX
/// still produces the aggregate's empty-input result, whereas replacing it by a
/// bare LIMIT 1 would incorrectly produce zero rows.
#[derive(Default)]
pub struct MinMaxOptimization;

impl QueryOptimization for MinMaxOptimization {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let aggregates: Vec<_> = query
            .live_plans_preorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::Aggregate { input, group_by, aggregates })
                    if group_by.is_empty() && aggregates.len() == 1 =>
                {
                    Some((id, input, aggregates[0].clone()))
                }
                _ => None,
            })
            .collect();

        let mut changed = false;
        for (_aggregate_plan, input, aggregate) in aggregates {
            if aggregate.distinct || !matches!(aggregate.function, AggregateFn::Min | AggregateFn::Max) {
                continue;
            }
            let Some(argument) = aggregate.argument else {
                continue;
            };
            let Some(ExprKind::Column(column)) = producer(query, argument) else {
                continue;
            };
            // Nullable MIN/MAX needs null-skipping semantics. Leave it to the normal
            // aggregate until the IR has an explicit IS-NOT-NULL range bound.
            if query.column(*column).is_none_or(|column| column.nullable) {
                continue;
            }
            let Some(input_node) = query.plan(input) else {
                continue;
            };
            let Some((ordered_column, direction, _)) = input_node.props.ordering.keys.first().copied() else {
                continue;
            };
            if ordered_column != *column || passthrough_scan_id(query, input).is_none() {
                continue;
            }
            let desired = match aggregate.function {
                AggregateFn::Min => SortDirection::Asc,
                AggregateFn::Max => SortDirection::Desc,
                _ => unreachable!(),
            };
            if direction != desired {
                if !flip_scan_direction_through_passthrough(query, input) {
                    continue;
                }
                changed = true;
            }
            changed |= push_limit_into_scan_through_passthrough(query, input, 0, Some(1));
        }
        changed
    }
}

/// Fuse ORDER BY + LIMIT/OFFSET into an already ordered primary/index scan. This
/// is the no-sort Top-N path: the storage engine stops after the requested window
/// instead of materializing and sorting the complete relation.
#[derive(Default)]
pub struct OrderedTopNPushdown;

impl QueryOptimization for OrderedTopNPushdown {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let limits: Vec<_> = query
            .live_plans_preorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::Limit { input, skip, limit }) => Some((id, input, skip, limit)),
                _ => None,
            })
            .collect();

        let mut changed = false;
        for (limit_plan, sort_plan, skip, limit) in limits {
            let Some(PlanKind::Sort { input, keys }) =
                query.plan(sort_plan).map(|node| node.kind.clone())
            else {
                continue;
            };
            let Some(requested) = sort_columns(query, &keys) else {
                continue;
            };
            let actual = query
                .plan(input)
                .map(|node| node.props.ordering.keys.clone())
                .unwrap_or_default();
            if passthrough_scan_id(query, input).is_none() {
                continue;
            }

            let mut reverse = false;
            if ordering_prefix_matches(&actual, &requested) {
                // Already in the requested order.
            } else if ordering_prefix_reverse_matches(&actual, &requested) {
                reverse = true;
            } else {
                continue;
            }

            if reverse && !flip_scan_direction_through_passthrough(query, input) {
                continue;
            }
            let Some(window_changed) = merge_scan_window_through_passthrough(query, input, skip, limit) else {
                if reverse {
                    let _ = flip_scan_direction_through_passthrough(query, input);
                }
                continue;
            };
            changed |= window_changed;
            changed |= replace_plan_uses(query, limit_plan, input);
        }
        changed
    }
}

fn ordering_prefix_matches(
    actual: &[(ColumnId, SortDirection, NullOrder)],
    requested: &[(ColumnId, SortDirection, NullOrder)],
) -> bool {
    requested.len() <= actual.len()
        && requested.iter().zip(actual.iter()).all(|(requested, actual)| requested == actual)
}

fn ordering_prefix_reverse_matches(
    actual: &[(ColumnId, SortDirection, NullOrder)],
    requested: &[(ColumnId, SortDirection, NullOrder)],
) -> bool {
    requested.len() <= actual.len()
        && requested.iter().zip(actual.iter()).all(
            |((requested_column, requested_direction, requested_nulls), (actual_column, actual_direction, actual_nulls))| {
                requested_column == actual_column
                    && *requested_direction == opposite_direction(*actual_direction)
                    && *requested_nulls == opposite_null_order(*actual_nulls)
            },
        )
}

fn passthrough_scan_id(query: &Query, mut id: PlanId) -> Option<PlanId> {
    loop {
        match query.plan(id).map(|node| &node.kind)? {
            PlanKind::PrimaryRangeScan { .. } | PlanKind::IndexRangeScan { .. } => return Some(id),
            PlanKind::Project { input, .. }
            | PlanKind::FetchRows { input, .. }
            | PlanKind::Materialize { input, .. } => id = *input,
            _ => return None,
        }
    }
}

fn flip_scan_direction_through_passthrough(query: &mut Query, id: PlanId) -> bool {
    let Some(scan) = passthrough_scan_id(query, id) else {
        return false;
    };
    match &mut query.plan_mut(scan).unwrap().kind {
        PlanKind::PrimaryRangeScan { direction, .. }
        | PlanKind::IndexRangeScan { direction, .. } => {
            *direction = match *direction {
                ScanDirection::Forward => ScanDirection::Reverse,
                ScanDirection::Reverse => ScanDirection::Forward,
            };
            true
        }
        _ => false,
    }
}

fn merge_scan_window_through_passthrough(
    query: &mut Query,
    id: PlanId,
    outer_skip: u64,
    outer_limit: Option<u64>,
) -> Option<bool> {
    let scan = passthrough_scan_id(query, id)?;
    match &mut query.plan_mut(scan)?.kind {
        PlanKind::PrimaryRangeScan { skip, limit, .. }
        | PlanKind::IndexRangeScan { skip, limit, .. } => {
            let (next_skip, next_limit) = merge_limits(*skip, *limit, outer_skip, outer_limit);
            let changed = next_skip != *skip || next_limit != *limit;
            *skip = next_skip;
            *limit = next_limit;
            Some(changed)
        }
        _ => None,
    }
}

fn push_limit_into_scan_through_passthrough(
    query: &mut Query,
    id: PlanId,
    outer_skip: u64,
    outer_limit: Option<u64>,
) -> bool {
    merge_scan_window_through_passthrough(query, id, outer_skip, outer_limit).unwrap_or(false)
}

#[derive(Default)]
pub struct SortElimination;

impl QueryOptimization for SortElimination {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let mut changed = false;
        for id in query.live_plans_preorder() {
            let Some(PlanKind::Sort { input, keys }) = query.plan(id).map(|n| n.kind.clone()) else { continue; };
            if keys.is_empty() || matches!(query.plan(input).map(|n| &n.kind), Some(PlanKind::PointGet { .. })) {
                changed |= replace_plan_uses(query, id, input);
                continue;
            }
            let Some(requested) = sort_columns(query, &keys) else { continue; };

            // Direct scans can be matched from catalog/index metadata even before a
            // property refresh. This remains the cheapest/common path.
            if let Some((forward, reverse)) = scan_order_matches(query, catalog, input, &requested) {
                let direction = if forward {
                    Some(ScanDirection::Forward)
                } else if reverse {
                    Some(ScanDirection::Reverse)
                } else {
                    None
                };
                if let Some(direction) = direction {
                    match &mut query.plan_mut(input).unwrap().kind {
                        PlanKind::PrimaryRangeScan { direction: scan_direction, .. }
                        | PlanKind::IndexRangeScan { direction: scan_direction, .. } => {
                            *scan_direction = direction;
                            changed |= replace_plan_uses(query, id, input);
                            continue;
                        }
                        _ => {}
                    }
                }
            }

            // Property-based matching also sees through cardinality-preserving
            // projections/fetches. If the requested order is the exact reverse,
            // flip the underlying ordered scan instead of allocating a Sort.
            let actual = query
                .plan(input)
                .map(|node| node.props.ordering.keys.clone())
                .unwrap_or_default();
            if ordering_prefix_matches(&actual, &requested) {
                changed |= replace_plan_uses(query, id, input);
            } else if ordering_prefix_reverse_matches(&actual, &requested)
                && flip_scan_direction_through_passthrough(query, input)
            {
                changed |= replace_plan_uses(query, id, input);
            }
        }
        changed
    }
}

fn sort_columns(query: &Query, keys: &[SortKey]) -> Option<Vec<(ColumnId, SortDirection, NullOrder)>> {
    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        let ExprKind::Column(column) = producer(query, key.expr)? else { return None; };
        out.push((*column, key.direction, key.nulls));
    }
    Some(out)
}

fn scan_order_matches(
    query: &Query,
    catalog: &Catalog,
    input: PlanId,
    requested: &[(ColumnId, SortDirection, NullOrder)],
) -> Option<(bool, bool)> {
    let kind = &query.plan(input)?.kind;
    let (table, physical_order, fixed_prefix) = match kind {
        PlanKind::PrimaryRangeScan { table, range, .. } => {
            let table_def = catalog.table(*table)?;
            (
                *table,
                table_def.primary_key.iter().map(|field| (*field, SortDirection::Asc, NullOrder::First)).collect::<Vec<_>>(),
                range.prefix.len(),
            )
        }
        PlanKind::IndexRangeScan { table, index, range, .. } => {
            let index = catalog.index(*index)?;
            (
                *table,
                index.columns.iter().map(|column| {
                    (column.field, if column.descending { SortDirection::Desc } else { SortDirection::Asc }, column.nulls)
                }).collect::<Vec<_>>(),
                range.prefix.len(),
            )
        }
        _ => return None,
    };

    let outputs = output_columns(query, catalog, input);
    let mut order = Vec::with_capacity(physical_order.len().saturating_sub(fixed_prefix));
    for (position, (field, direction, nulls)) in physical_order.into_iter().enumerate() {
        // Equality-constrained leading columns are constants. They may have been
        // pruned from the row entirely and do not participate in ORDER BY.
        if position < fixed_prefix {
            continue;
        }
        let Some(column) = outputs.iter().copied().find(|column| {
            matches!(query.storage_origin(*column), Some((origin_table, origin_field)) if origin_table == table && origin_field == field)
        }) else { break; };
        order.push((column, direction, nulls));
    }
    if requested.len() > order.len() {
        return Some((false, false));
    }
    let forward = requested.iter().zip(order.iter()).all(|(a, b)| a == b);
    let reverse = requested.iter().zip(order.iter()).all(|((ac, ad, an), (bc, bd, bn))| {
        ac == bc && *ad == opposite_direction(*bd) && *an == opposite_null_order(*bn)
    });
    Some((forward, reverse))
}

fn opposite_direction(direction: SortDirection) -> SortDirection {
    match direction {
        SortDirection::Asc => SortDirection::Desc,
        SortDirection::Desc => SortDirection::Asc,
    }
}

fn opposite_null_order(order: NullOrder) -> NullOrder {
    match order {
        NullOrder::First => NullOrder::Last,
        NullOrder::Last => NullOrder::First,
    }
}

#[derive(Default)]
pub struct DistinctElimination;

impl QueryOptimization for DistinctElimination {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let mut changed = false;
        for id in query.live_plans_postorder() {
            let Some(PlanKind::Distinct { input, columns }) = query.plan(id).map(|node| node.kind.clone()) else {
                continue;
            };
            let Some(input_node) = query.plan(input) else { continue; };

            // If any already-proven unique key is contained in the DISTINCT key,
            // two input rows can never agree on every DISTINCT column.
            let redundant = input_node.props.unique_on.iter().any(|unique| {
                !unique.is_empty() && unique.iter().all(|column| columns.contains(column))
            });
            if redundant {
                changed |= replace_plan_uses(query, id, input);
            }
        }
        changed
    }
}

#[derive(Default)]
pub struct LimitPushdown;

impl QueryOptimization for LimitPushdown {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let mut changed = false;
        for id in query.live_plans_preorder() {
            let Some(PlanKind::Limit { input, skip, limit }) = query.plan(id).map(|n| n.kind.clone()) else { continue; };
            let Some(input_kind) = query.plan(input).map(|n| n.kind.clone()) else { continue; };
            match input_kind {
                PlanKind::PointGet { .. } => {
                    if skip > 0 || limit == Some(0) {
                        let columns = output_columns(query, catalog, input);
                        query.plan_mut(id).unwrap().kind = PlanKind::Empty { columns };
                    } else {
                        replace_plan_uses(query, id, input);
                    }
                    changed = true;
                }
                PlanKind::PrimaryRangeScan { table, range, columns, residual, direction, skip: inner_skip, limit: inner_limit, delivery } => {
                    let (merged_skip, merged_limit) = merge_limits(inner_skip, inner_limit, skip, limit);
                    query.plan_mut(input).unwrap().kind = PlanKind::PrimaryRangeScan {
                        table, range, columns, residual, direction, skip: merged_skip, limit: merged_limit, delivery,
                    };
                    changed |= replace_plan_uses(query, id, input);
                }
                PlanKind::IndexRangeScan { table, index, range, columns, residual, direction, skip: inner_skip, limit: inner_limit, delivery, covering } => {
                    let (merged_skip, merged_limit) = merge_limits(inner_skip, inner_limit, skip, limit);
                    query.plan_mut(input).unwrap().kind = PlanKind::IndexRangeScan {
                        table, index, range, columns, residual, direction, skip: merged_skip, limit: merged_limit, delivery, covering,
                    };
                    changed |= replace_plan_uses(query, id, input);
                }
                PlanKind::Project { input: child, expressions } => {
                    let pushed = query.add_plan(PlanKind::Limit { input: child, skip, limit });
                    query.plan_mut(id).unwrap().kind = PlanKind::Project { input: pushed, expressions };
                    changed = true;
                }
                _ => {}
            }
        }
        changed
    }
}

fn merge_limits(inner_skip: u64, inner_limit: Option<u64>, outer_skip: u64, outer_limit: Option<u64>) -> (u64, Option<u64>) {
    let skip = inner_skip.saturating_add(outer_skip);
    let available = inner_limit.map(|value| value.saturating_sub(outer_skip));
    let limit = match (available, outer_limit) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };
    (skip, limit)
}


/// Bound each UNION ALL branch by the maximum number of rows the outer LIMIT can
/// possibly consume. The outer Limit remains because rows from earlier branches can
/// still contribute to OFFSET, but no branch needs to produce more than skip+limit.
#[derive(Default)]
pub struct UnionLimitPushdown;

impl QueryOptimization for UnionLimitPushdown {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let mut changed = false;
        for id in query.live_plans_preorder() {
            let Some(PlanKind::Limit { input, skip, limit: Some(limit) }) =
                query.plan(id).map(|node| node.kind.clone())
            else {
                continue;
            };
            let cap = skip.saturating_add(limit);
            if cap == u64::MAX {
                continue;
            }
            let Some(PlanKind::UnionAll { inputs, columns }) =
                query.plan(input).map(|node| node.kind.clone())
            else {
                continue;
            };

            let mut next_inputs = Vec::with_capacity(inputs.len());
            let mut local_changed = false;
            for child in inputs {
                let already_bounded = plan_produces_at_most(query, child, cap);
                if already_bounded {
                    next_inputs.push(child);
                } else {
                    next_inputs.push(query.add_plan(PlanKind::Limit {
                        input: child,
                        skip: 0,
                        limit: Some(cap),
                    }));
                    local_changed = true;
                }
            }
            if local_changed {
                query.plan_mut(input).unwrap().kind = PlanKind::UnionAll {
                    inputs: next_inputs,
                    columns,
                };
                changed = true;
            }
        }
        changed
    }
}

fn plan_produces_at_most(query: &Query, id: PlanId, cap: u64) -> bool {
    match query.plan(id).map(|node| &node.kind) {
        Some(PlanKind::Empty { .. }) => true,
        Some(PlanKind::Values { rows, .. }) => rows.len() as u64 <= cap,
        Some(PlanKind::PointGet { .. } | PlanKind::RangeCount { .. } | PlanKind::Exists { .. }) => {
            cap >= 1
        }
        Some(PlanKind::Limit {
            skip: 0,
            limit: Some(limit),
            ..
        }) => *limit <= cap,
        Some(PlanKind::PrimaryRangeScan { limit: Some(limit), .. }
        | PlanKind::IndexRangeScan { limit: Some(limit), .. }) => *limit <= cap,
        Some(PlanKind::Project { input, .. } | PlanKind::Materialize { input, .. }) => {
            plan_produces_at_most(query, *input, cap)
        }
        _ => false,
    }
}

#[derive(Default)]
pub struct ExistencePushdown;

impl QueryOptimization for ExistencePushdown {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let mut changed = false;
        for id in query.live_plans_preorder() {
            let Some(PlanKind::Exists { input, .. }) = query.plan(id).map(|node| node.kind.clone()) else {
                continue;
            };
            let Some(kind) = query.plan(input).map(|node| node.kind.clone()) else { continue; };
            match kind {
                PlanKind::PrimaryRangeScan {
                    table, range, columns, residual: None, direction, skip, limit, delivery,
                } => {
                    let next_limit = Some(limit.map_or(1, |value| value.min(1)));
                    if next_limit != limit {
                        query.plan_mut(input).unwrap().kind = PlanKind::PrimaryRangeScan {
                            table, range, columns, residual: None, direction, skip, limit: next_limit, delivery,
                        };
                        changed = true;
                    }
                }
                PlanKind::IndexRangeScan {
                    table, index, range, columns, residual: None, direction, skip, limit, delivery, covering,
                } => {
                    let next_limit = Some(limit.map_or(1, |value| value.min(1)));
                    if next_limit != limit {
                        query.plan_mut(input).unwrap().kind = PlanKind::IndexRangeScan {
                            table, index, range, columns, residual: None, direction, skip, limit: next_limit, delivery, covering,
                        };
                        changed = true;
                    }
                }
                _ => {}
            }
        }
        changed
    }
}

// -----------------------------------------------------------------------------
// Aggregate simplification from ordering, uniqueness and nullability facts
// -----------------------------------------------------------------------------

/// Sorting an input before COUNT/SUM/MIN/MAX/AVG cannot change the aggregate
/// result. Removing the sort is especially important when it would otherwise force
/// materialization of a large SurrealMX scan.
#[derive(Default)]
pub struct SortUnderAggregateElimination;

impl QueryOptimization for SortUnderAggregateElimination {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let mut changed = false;
        for id in query.live_plans_postorder() {
            let Some(PlanKind::Aggregate {
                input,
                group_by,
                aggregates,
            }) = query.plan(id).map(|node| node.kind.clone())
            else {
                continue;
            };
            let Some(PlanKind::Sort { input: child, .. }) =
                query.plan(input).map(|node| node.kind.clone())
            else {
                continue;
            };
            query.plan_mut(id).unwrap().kind = PlanKind::Aggregate {
                input: child,
                group_by,
                aggregates,
            };
            changed = true;
        }
        changed
    }
}

/// DISTINCT inside a single-argument aggregate is redundant when the grouping
/// columns plus that argument contain an already-proven unique key of the input.
/// This is stronger than only checking globally-unique argument columns.
#[derive(Default)]
pub struct AggregateDistinctElimination;

impl QueryOptimization for AggregateDistinctElimination {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let aggregate_plans: Vec<_> = query
            .live_plans_postorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::Aggregate {
                    input,
                    group_by,
                    aggregates,
                }) => Some((id, input, group_by, aggregates)),
                _ => None,
            })
            .collect();

        let mut changed = false;
        for (id, input, group_by, mut aggregates) in aggregate_plans {
            let Some(input_node) = query.plan(input) else {
                continue;
            };
            if input_node.props.unique_on.is_empty() {
                continue;
            }

            let group_columns: Vec<_> = group_by
                .iter()
                .filter_map(|projection| match producer(query, projection.expr) {
                    Some(ExprKind::Column(column)) => Some(*column),
                    _ => None,
                })
                .collect();

            let mut local_changed = false;
            for aggregate in &mut aggregates {
                if !aggregate.distinct {
                    continue;
                }
                let Some(argument) = aggregate.argument else {
                    continue;
                };
                let Some(ExprKind::Column(argument_column)) = producer(query, argument) else {
                    continue;
                };

                let mut determinant = group_columns.clone();
                if !determinant.contains(argument_column) {
                    determinant.push(*argument_column);
                }
                let unique = input_node.props.unique_on.iter().any(|key| {
                    !key.is_empty()
                        && key
                            .iter()
                            .all(|column| determinant.contains(column))
                });
                if unique {
                    aggregate.distinct = false;
                    local_changed = true;
                }
            }

            if local_changed {
                query.plan_mut(id).unwrap().kind = PlanKind::Aggregate {
                    input,
                    group_by,
                    aggregates,
                };
                changed = true;
            }
        }
        changed
    }
}

/// COUNT(expr) is COUNT(*) when the expression is statically non-nullable. Once
/// rewritten, the existing RangeCount lowering can use SurrealMX's count path.
#[derive(Default)]
pub struct CountArgumentElimination;

impl QueryOptimization for CountArgumentElimination {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let aggregate_plans: Vec<_> = query
            .live_plans_postorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::Aggregate {
                    input,
                    group_by,
                    aggregates,
                }) => Some((id, input, group_by, aggregates)),
                _ => None,
            })
            .collect();

        let mut changed = false;
        for (id, input, group_by, mut aggregates) in aggregate_plans {
            let mut local_changed = false;
            for aggregate in &mut aggregates {
                if aggregate.function != AggregateFn::Count || aggregate.distinct {
                    continue;
                }
                let Some(argument) = aggregate.argument else {
                    continue;
                };
                if query.expr(argument).is_some_and(|expr| !expr.nullable) {
                    aggregate.argument = None;
                    local_changed = true;
                }
            }
            if local_changed {
                query.plan_mut(id).unwrap().kind = PlanKind::Aggregate {
                    input,
                    group_by,
                    aggregates,
                };
                changed = true;
            }
        }
        changed
    }
}

/// If GROUP BY contains a unique key, every group contains at most one input row.
/// Order-insensitive aggregates then reduce to scalar projections: MIN/MAX/SUM/AVG
/// return their argument and COUNT returns one when its argument cannot be NULL.
#[derive(Default)]
pub struct UniqueGroupAggregateElimination;

impl QueryOptimization for UniqueGroupAggregateElimination {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let aggregate_plans: Vec<_> = query
            .live_plans_postorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::Aggregate {
                    input,
                    group_by,
                    aggregates,
                }) if !group_by.is_empty() => Some((id, input, group_by, aggregates)),
                _ => None,
            })
            .collect();

        let mut changed = false;
        for (id, input, group_by, aggregates) in aggregate_plans {
            let Some(input_node) = query.plan(input) else {
                continue;
            };
            let group_sources: Vec<_> = group_by
                .iter()
                .filter_map(|projection| match producer(query, projection.expr) {
                    Some(ExprKind::Column(column)) => Some(*column),
                    _ => None,
                })
                .collect();
            if group_sources.len() != group_by.len() {
                continue;
            }
            let unique_group = input_node.props.unique_on.iter().any(|key| {
                !key.is_empty()
                    && key.iter().all(|column| group_sources.contains(column))
            });
            if !unique_group {
                continue;
            }

            let mut projections = group_by.clone();
            let mut supported = true;
            for aggregate in &aggregates {
                let expression = match aggregate.function {
                    AggregateFn::Count => match aggregate.argument {
                        None => Some(query.add_expr(
                            ExprKind::Const(number(1.0)),
                            ScalarType::Number,
                            false,
                        )),
                        Some(argument)
                            if query.expr(argument).is_some_and(|expr| !expr.nullable) =>
                        {
                            Some(query.add_expr(
                                ExprKind::Const(number(1.0)),
                                ScalarType::Number,
                                false,
                            ))
                        }
                        Some(_) => None,
                    },
                    AggregateFn::Min
                    | AggregateFn::Max
                    | AggregateFn::Sum
                    | AggregateFn::Avg => aggregate.argument,
                };
                let Some(expression) = expression else {
                    supported = false;
                    break;
                };
                projections.push(ProjectionExpr {
                    column: aggregate.column,
                    expr: expression,
                });
            }
            if !supported {
                continue;
            }

            query.plan_mut(id).unwrap().kind = PlanKind::Project {
                input,
                expressions: projections,
            };
            changed = true;
        }

        changed
    }
}

/// Fold aggregates over a statically empty relation. Grouped aggregation remains
/// empty; a global aggregate produces exactly one row (COUNT=0, other stock
/// aggregates=NULL), matching SQL-style empty-input semantics used by this IR.
#[derive(Default)]
pub struct EmptyAggregateFolding;

impl QueryOptimization for EmptyAggregateFolding {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let aggregate_plans: Vec<_> = query
            .live_plans_postorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::Aggregate {
                    input,
                    group_by,
                    aggregates,
                }) if matches!(
                    query.plan(input).map(|node| &node.kind),
                    Some(PlanKind::Empty { .. })
                ) => Some((id, group_by, aggregates)),
                _ => None,
            })
            .collect();

        let mut changed = false;
        for (id, group_by, aggregates) in aggregate_plans {
            let mut columns: Vec<_> = group_by.iter().map(|projection| projection.column).collect();
            columns.extend(aggregates.iter().map(|aggregate| aggregate.column));

            if !group_by.is_empty() {
                query.plan_mut(id).unwrap().kind = PlanKind::Empty { columns };
                changed = true;
                continue;
            }

            let mut row = Vec::with_capacity(aggregates.len());
            let mut valid = true;
            for aggregate in &aggregates {
                let Some(output) = query.column(aggregate.column).cloned() else {
                    valid = false;
                    break;
                };
                let expr = match aggregate.function {
                    AggregateFn::Count => query.add_expr(
                        ExprKind::Const(number(0.0)),
                        ScalarType::Number,
                        false,
                    ),
                    AggregateFn::Sum
                    | AggregateFn::Min
                    | AggregateFn::Max
                    | AggregateFn::Avg => query.add_expr(
                        ExprKind::Const(ScalarValue::Null),
                        output.ty,
                        true,
                    ),
                };
                row.push(expr);
            }
            if !valid {
                continue;
            }
            query.plan_mut(id).unwrap().kind = PlanKind::Values {
                columns,
                rows: vec![row],
            };
            changed = true;
        }

        changed
    }
}

#[derive(Default)]
pub struct CountOptimization;

impl QueryOptimization for CountOptimization {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let mut changed = false;
        for id in query.live_plans_preorder() {
            let Some(PlanKind::Aggregate { input, group_by, aggregates }) = query.plan(id).map(|n| n.kind.clone()) else { continue; };
            if !group_by.is_empty() || aggregates.len() != 1 {
                continue;
            }
            let aggregate = &aggregates[0];
            if aggregate.function != AggregateFn::Count || aggregate.argument.is_some() || aggregate.distinct {
                continue;
            }
            let Some(input_kind) = query.plan(input).map(|n| n.kind.clone()) else { continue; };
            let replacement = match input_kind {
                PlanKind::PrimaryRangeScan { table, range, residual: None, skip, limit, .. } => {
                    Some(PlanKind::RangeCount { output: aggregate.column, table, index: None, range, skip, limit })
                }
                PlanKind::IndexRangeScan { table, index, range, residual: None, skip, limit, .. } => {
                    Some(PlanKind::RangeCount { output: aggregate.column, table, index: Some(index), range, skip, limit })
                }
                _ => None,
            };
            if let Some(replacement) = replacement {
                query.plan_mut(id).unwrap().kind = replacement;
                changed = true;
            }
        }
        changed
    }
}

#[derive(Default)]
pub struct JoinSelection;

impl QueryOptimization for JoinSelection {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let joins: Vec<_> = query
            .live_plans_postorder()
            .into_iter()
            .filter_map(|id| match query.plan(id).map(|node| node.kind.clone()) {
                Some(PlanKind::Join {
                    left,
                    right,
                    kind,
                    on,
                }) => Some((id, left, right, kind, on)),
                _ => None,
            })
            .collect();

        let mut changed = false;
        for (id, mut left, mut right, kind, on) in joins {
            let original_columns = output_columns(query, catalog, id);
            let mut pairs = oriented_join_equalities(query, catalog, left, right, on);
            let mut algorithm = JoinAlgorithm::Hash;
            let mut swapped = false;

            if !pairs.is_empty() {
                let right_index = best_join_index(catalog, right, &pairs, false, query);
                let left_index = if kind == JoinKind::Inner {
                    best_join_index(catalog, left, &pairs, true, query)
                } else {
                    None
                };

                match (right_index, left_index) {
                    (Some((_index, right_score)), Some((left_index, left_score)))
                        if left_score > right_score =>
                    {
                        std::mem::swap(&mut left, &mut right);
                        for pair in &mut pairs {
                            *pair = (pair.1, pair.0);
                        }
                        algorithm = JoinAlgorithm::IndexNestedLoop(left_index);
                        swapped = true;
                    }
                    (Some((index, _)), _) => {
                        algorithm = JoinAlgorithm::IndexNestedLoop(index);
                    }
                    (None, Some((index, _))) => {
                        std::mem::swap(&mut left, &mut right);
                        for pair in &mut pairs {
                            *pair = (pair.1, pair.0);
                        }
                        algorithm = JoinAlgorithm::IndexNestedLoop(index);
                        swapped = true;
                    }
                    (None, None) => {
                        if merge_join_prefix_len(query, left, right, &pairs) > 0 {
                            algorithm = JoinAlgorithm::Merge;
                        }
                    }
                }
            }

            if matches!(algorithm, JoinAlgorithm::Hash) && kind == JoinKind::Inner {
                let left_rows = estimated_rows(query, catalog, left);
                let right_rows = estimated_rows(query, catalog, right);
                // Hash join builds the right side in the executor; keep the smaller
                // input there, but preserve the source-visible column order below.
                if right_rows > left_rows {
                    std::mem::swap(&mut left, &mut right);
                    swapped = true;
                }
            }

            if swapped && kind == JoinKind::Inner {
                let physical = query.add_plan(PlanKind::PhysicalJoin {
                    left,
                    right,
                    kind,
                    on,
                    algorithm,
                });
                let expressions = projection_aliases(query, &original_columns);
                query.plan_mut(id).unwrap().kind = PlanKind::Project {
                    input: physical,
                    expressions,
                };
            } else {
                query.plan_mut(id).unwrap().kind = PlanKind::PhysicalJoin {
                    left,
                    right,
                    kind,
                    on,
                    algorithm,
                };
            }
            changed = true;
        }
        changed
    }
}

fn best_join_index(
    catalog: &Catalog,
    indexed_plan: PlanId,
    pairs: &[(ColumnId, ColumnId)],
    indexed_is_left: bool,
    query: &Query,
) -> Option<(IndexId, i64)> {
    let table = plan_table(query, indexed_plan)?;
    let table_def = catalog.table(table)?;
    let mut best = None;

    for index in catalog.indexes_for(table) {
        let mut prefix = 0usize;
        for index_column in &index.columns {
            let matched = pairs.iter().any(|(left, right)| {
                let indexed_column = if indexed_is_left { *left } else { *right };
                matches!(
                    query.storage_origin(indexed_column),
                    Some((origin_table, field))
                        if origin_table == table && field == index_column.field
                )
            });
            if !matched {
                break;
            }
            prefix += 1;
        }
        if prefix == 0 {
            continue;
        }

        let full = prefix == index.columns.len();
        let mut score = (prefix as i64) * 1_000;
        if full && index.unique {
            score += 10_000;
        }
        for column in index.columns.iter().take(prefix) {
            if let Some(distinct) = table_def.stats.distinct.get(&column.field) {
                score += ((*distinct).max(1).ilog2() as i64).min(64);
            }
        }

        if best.is_none_or(|(_, previous_score)| score > previous_score) {
            best = Some((index.id, score));
        }
    }

    best
}

fn merge_join_prefix_len(
    query: &Query,
    left: PlanId,
    right: PlanId,
    pairs: &[(ColumnId, ColumnId)],
) -> usize {
    let Some(left_node) = query.plan(left) else {
        return 0;
    };
    let Some(right_node) = query.plan(right) else {
        return 0;
    };
    let mut matched = 0usize;
    for ((left_column, left_direction, left_nulls), (right_column, right_direction, right_nulls)) in
        left_node
            .props
            .ordering
            .keys
            .iter()
            .zip(right_node.props.ordering.keys.iter())
    {
        if left_direction != right_direction || left_nulls != right_nulls {
            break;
        }
        if pairs
            .iter()
            .any(|(left, right)| left == left_column && right == right_column)
        {
            matched += 1;
        } else {
            break;
        }
    }
    matched
}



/// Remove assignments that provably store a field back into itself. If every
/// assignment disappears the Update has no observable storage effect and collapses
/// to its input relation.
#[derive(Default)]
pub struct WriteNoOpElimination;

impl QueryOptimization for WriteNoOpElimination {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let mut changed = false;
        for id in query.live_plans_postorder() {
            let Some(PlanKind::Update { input, table, assignments }) =
                query.plan(id).map(|node| node.kind.clone())
            else {
                continue;
            };
            let available: HashSet<_> = output_columns(query, catalog, input).into_iter().collect();
            let original_len = assignments.len();
            let mut next = Vec::with_capacity(original_len);
            for (field, expr) in assignments {
                let identity = match producer(query, expr) {
                    Some(ExprKind::Column(column)) if available.contains(column) => {
                        matches!(
                            query.storage_origin(*column),
                            Some((origin_table, origin_field))
                                if origin_table == table && origin_field == field
                        )
                    }
                    _ => false,
                };
                if !identity {
                    next.push((field, expr));
                }
            }

            if next.is_empty() {
                changed |= replace_plan_uses(query, id, input);
            } else if next.len() != original_len {
                query.plan_mut(id).unwrap().kind = PlanKind::Update {
                    input,
                    table,
                    assignments: next,
                };
                changed = true;
            }
        }
        changed
    }
}

#[derive(Default)]
pub struct WriteLowering;

impl QueryOptimization for WriteLowering {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let mut changed = false;
        for id in query.live_plans_preorder() {
            let Some(kind) = query.plan(id).map(|n| n.kind.clone()) else { continue; };
            if let PlanKind::Insert { table, rows } = kind {
                query.plan_mut(id).unwrap().kind = PlanKind::WriteBatch {
                    table,
                    mode: MutationMode::Insert,
                    rows,
                };
                changed = true;
            }
        }
        changed
    }
}


/// Coalesce repeated SET writes to the exact same runtime-bound primary key inside
/// one batch. Only SET is eligible: INSERT must preserve duplicate-key failures and
/// DELETE/UPDATE may have result-count semantics the IR does not currently encode.
#[derive(Default)]
pub struct WriteBatchCoalescing;

impl QueryOptimization for WriteBatchCoalescing {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let mut changed = false;
        for id in query.live_plans_preorder() {
            let Some(PlanKind::WriteBatch {
                table,
                mode: MutationMode::Set,
                rows,
            }) = query.plan(id).map(|node| node.kind.clone())
            else {
                continue;
            };
            let Some(table_def) = catalog.table(table) else { continue; };
            if table_def.primary_key.is_empty() || rows.len() < 2 {
                continue;
            }

            let mut seen: HashSet<Vec<ExprKind>> = HashSet::default();
            let mut kept = Vec::with_capacity(rows.len());
            for row in rows.iter().rev() {
                let Some(signature) = runtime_bound_primary_key_signature(
                    query,
                    row,
                    &table_def.primary_key,
                ) else {
                    kept.push(row.clone());
                    continue;
                };
                if seen.insert(signature) {
                    kept.push(row.clone());
                } else {
                    changed = true;
                }
            }
            if kept.len() != rows.len() {
                kept.reverse();
                query.plan_mut(id).unwrap().kind = PlanKind::WriteBatch {
                    table,
                    mode: MutationMode::Set,
                    rows: kept,
                };
            }
        }
        changed
    }
}

fn runtime_bound_primary_key_signature(
    query: &Query,
    row: &[(FieldId, ExprId)],
    primary_key: &[FieldId],
) -> Option<Vec<ExprKind>> {
    let mut signature = Vec::with_capacity(primary_key.len());
    for field in primary_key {
        let expr = row.iter().find_map(|(candidate, expr)| (*candidate == *field).then_some(*expr))?;
        if !is_runtime_bound(query, expr) {
            return None;
        }
        signature.push(producer(query, expr)?.clone());
    }
    Some(signature)
}


// -----------------------------------------------------------------------------
// Shared-subplan materialization and cleanup
// -----------------------------------------------------------------------------

const SHARED_MATERIALIZE_MIN_COST: f64 = 256.0;
const SHARED_MATERIALIZE_MIN_ROWS: u64 = 32;

/// Insert one buffered materialization when the same expensive, read-only plan node
/// feeds multiple parents. The query graph already supports DAGs; Materialize makes
/// the executor's reuse requirement explicit instead of re-running the subtree.
#[derive(Default)]
pub struct SharedSubplanMaterialization;

impl QueryOptimization for SharedSubplanMaterialization {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let parents = query.parents();
        let mut candidates = Vec::new();
        for id in query.live_plans_postorder() {
            if id == query.root {
                continue;
            }
            let mut users = parents.get(&id).cloned().unwrap_or_default();
            users.sort_unstable_by_key(|parent| parent.0);
            users.dedup();
            if users.len() < 2 {
                continue;
            }
            let Some(node) = query.plan(id) else { continue; };
            if matches!(node.kind, PlanKind::Materialize { .. })
                || node.props.estimated_cost < SHARED_MATERIALIZE_MIN_COST
                || node.props.estimated_rows.unwrap_or(0) < SHARED_MATERIALIZE_MIN_ROWS
                || !materializable_read_subplan(query, id, 0)
            {
                continue;
            }
            candidates.push((id, users));
        }

        let mut changed = false;
        for (id, users) in candidates {
            // A previous candidate rewrite may have detached some users.
            let live_users: Vec<_> = users
                .into_iter()
                .filter(|parent| {
                    query
                        .plan(*parent)
                        .is_some_and(|node| node.kind.children().contains(&id))
                })
                .collect();
            if live_users.len() < 2 {
                continue;
            }
            let materialized = query.add_plan(PlanKind::Materialize {
                input: id,
                mode: Materialization::Buffered,
            });
            for parent in live_users {
                if let Some(node) = query.plan_mut(parent) {
                    changed |= replace_child(&mut node.kind, id, materialized);
                }
            }
        }
        changed
    }
}

fn materializable_read_subplan(query: &Query, id: PlanId, depth: u8) -> bool {
    if depth >= 48 {
        return false;
    }
    let Some(node) = query.plan(id) else { return false; };
    match &node.kind {
        PlanKind::Insert { .. }
        | PlanKind::Update { .. }
        | PlanKind::Delete { .. }
        | PlanKind::WriteBatch { .. } => false,
        PlanKind::Materialize { .. } => false,
        kind => kind
            .children()
            .into_iter()
            .all(|child| materializable_read_subplan(query, child, depth + 1)),
    }
}

/// Streaming Materialize is an identity. Buffered materialization is also redundant
/// for a non-root single-row subtree; expensive shared buffered nodes are preserved.
#[derive(Default)]
pub struct MaterializationCleanup;

impl QueryOptimization for MaterializationCleanup {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let parents = query.parents();
        let mut changed = false;
        for id in query.live_plans_postorder() {
            let Some(PlanKind::Materialize { input, mode }) =
                query.plan(id).map(|node| node.kind.clone())
            else {
                continue;
            };
            let remove = match mode {
                Materialization::Streaming => true,
                Materialization::Buffered => {
                    id != query.root
                        && query
                            .plan(input)
                            .and_then(|node| node.props.estimated_rows)
                            .is_some_and(|rows| rows <= 1)
                        && parents.get(&id).map_or(0, |users| users.len()) <= 1
                }
            };
            if remove {
                changed |= replace_plan_uses(query, id, input);
            }
        }
        changed
    }
}

// -----------------------------------------------------------------------------
// Delivery and transaction specialization
// -----------------------------------------------------------------------------

#[derive(Default)]
pub struct DeliverySelection;

impl QueryOptimization for DeliverySelection {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let parents = query.parents();
        let mut changed = false;
        for id in query.live_plans_preorder() {
            let is_root = query.root == id;
            let parent_kinds: Vec<_> = parents
                .get(&id)
                .into_iter()
                .flatten()
                .filter_map(|parent| query.plan(*parent).map(|node| node.kind.clone()))
                .collect();
            let streaming = !is_root
                && parent_kinds.iter().all(parent_accepts_borrowed_rows);
            let key_only = query
                .plan(id)
                .map(|node| scan_key_only_possible(query, catalog, &node.kind))
                .unwrap_or(false);

            let Some(node) = query.plan_mut(id) else { continue; };
            match &mut node.kind {
                PlanKind::PointGet { columns, residual, delivery, .. } => {
                    let next = if parent_kinds.iter().any(|kind| matches!(kind, PlanKind::Exists { .. })) && residual.is_none() {
                        PointDelivery::ExistsOnly
                    } else if streaming {
                        PointDelivery::Borrowed
                    } else {
                        PointDelivery::Owned
                    };
                    if *delivery != next { *delivery = next; changed = true; }
                    if columns.is_empty() && residual.is_none() && *delivery == PointDelivery::Owned {
                        *delivery = PointDelivery::ExistsOnly;
                        changed = true;
                    }
                }
                PlanKind::PrimaryRangeScan { delivery, .. }
                | PlanKind::IndexRangeScan { delivery, .. } => {
                    let next = match (streaming, key_only) {
                        (true, true) => ScanDelivery::BorrowedKeys,
                        (true, false) => ScanDelivery::BorrowedPairs,
                        (false, true) => ScanDelivery::Keys,
                        (false, false) => ScanDelivery::Pairs,
                    };
                    if *delivery != next { *delivery = next; changed = true; }
                }
                _ => {}
            }
        }
        changed
    }
}

fn parent_accepts_borrowed_rows(kind: &PlanKind) -> bool {
    matches!(
        kind,
        PlanKind::Filter { .. }
            | PlanKind::Project { .. }
            | PlanKind::Aggregate { .. }
            | PlanKind::Exists { .. }
            | PlanKind::Limit { .. }
            | PlanKind::Distinct { .. }
            | PlanKind::PhysicalJoin { .. }
            | PlanKind::FetchRows { .. }
            | PlanKind::UnionAll { .. }
            | PlanKind::Materialize {
                mode: Materialization::Streaming,
                ..
            }
    )
}

fn scan_key_only_possible(query: &Query, catalog: &Catalog, kind: &PlanKind) -> bool {
    let (table, allowed_fields, columns, residual) = match kind {
        PlanKind::PrimaryRangeScan { table, columns, residual, .. } => {
            let Some(table_def) = catalog.table(*table) else { return false; };
            (*table, table_def.primary_key.clone(), columns, *residual)
        }
        PlanKind::IndexRangeScan { table, index, columns, residual, .. } => {
            let (Some(table_def), Some(index_def)) = (catalog.table(*table), catalog.index(*index)) else {
                return false;
            };
            let mut fields: Vec<_> = index_def.columns.iter().map(|column| column.field).collect();
            fields.extend(table_def.primary_key.iter().copied());
            fields.sort_unstable_by_key(|field| field.0);
            fields.dedup();
            (*table, fields, columns, *residual)
        }
        _ => return false,
    };

    let mut needed = columns.clone();
    if let Some(predicate) = residual {
        needed.extend(columns_in_expr(query, predicate));
        sort_dedup_columns(&mut needed);
    }
    let Some(fields) = storage_fields_for_columns(query, table, &needed) else {
        return false;
    };
    fields.into_iter().all(|field| allowed_fields.contains(&field))
}

#[derive(Default)]
pub struct TransactionSpecialization;

impl QueryOptimization for TransactionSpecialization {
    fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
        let has_write = query.live_plans_preorder().iter().any(|id| {
            query.plan(*id).is_some_and(|node| matches!(
                &node.kind,
                PlanKind::Insert { .. }
                    | PlanKind::Update { .. }
                    | PlanKind::Delete { .. }
                    | PlanKind::WriteBatch { .. }
            ))
        });
        let next = if has_write {
            TransactionMode::ReadWrite
        } else if !plan_touches_database(query, query.root, 0)
            || direct_database_plan(query, query.root)
        {
            // Constant/VALUES-only plans need no transaction at all. Direct point
            // reads and scans can use SurrealMX Database auto-commit methods.
            TransactionMode::Auto
        } else {
            TransactionMode::ReadOnlySnapshot
        };
        if query.transaction != next {
            query.transaction = next;
            true
        } else {
            false
        }
    }
}

fn plan_touches_database(query: &Query, id: PlanId, depth: u8) -> bool {
    if depth >= 64 {
        return true;
    }
    let Some(node) = query.plan(id) else { return true; };
    match &node.kind {
        PlanKind::Empty { .. } | PlanKind::Values { .. } => false,
        PlanKind::PointGet { .. }
        | PlanKind::PrimaryRangeScan { .. }
        | PlanKind::IndexRangeScan { .. }
        | PlanKind::FetchRows { .. }
        | PlanKind::RangeCount { .. }
        | PlanKind::TableScan { .. }
        | PlanKind::Insert { .. }
        | PlanKind::Update { .. }
        | PlanKind::Delete { .. }
        | PlanKind::WriteBatch { .. } => true,
        kind => kind
            .children()
            .into_iter()
            .any(|child| plan_touches_database(query, child, depth + 1)),
    }
}

fn direct_database_plan(query: &Query, id: PlanId) -> bool {
    match query.plan(id).map(|node| &node.kind) {
        Some(PlanKind::PointGet { .. }
            | PlanKind::PrimaryRangeScan { .. }
            | PlanKind::IndexRangeScan { .. }
            | PlanKind::RangeCount { .. }) => true,
        Some(PlanKind::Project { input, .. } | PlanKind::Limit { input, .. } | PlanKind::Exists { input, .. }) => direct_database_plan(query, *input),
        _ => false,
    }
}

// -----------------------------------------------------------------------------
// Property / cost analysis
// -----------------------------------------------------------------------------


/// Re-score live plans with a memory-first cost model. SurrealMX keeps the hot data
/// structure in RAM, so the useful currency is rows visited, bytes decoded/copied,
/// hash/sort work and primary-row fetches rather than random-disk I/O.
#[derive(Default)]
pub struct MemoryCostAnalysis;

impl QueryOptimization for MemoryCostAnalysis {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let mut changed = false;
        for id in query.live_plans_postorder() {
            let next = memory_cost_for_plan(query, catalog, id);
            let Some(node) = query.plan_mut(id) else { continue; };
            if (node.props.estimated_cost - next).abs() > 1e-9 {
                node.props.estimated_cost = next;
                changed = true;
            }
        }
        changed
    }
}

fn memory_cost_for_plan(query: &Query, catalog: &Catalog, id: PlanId) -> f64 {
    let Some(node) = query.plan(id) else { return 0.0; };
    let rows = node.props.estimated_rows.unwrap_or(1_000).max(1) as f64;
    let child_cost = |child: PlanId| {
        query
            .plan(child)
            .map(|node| node.props.estimated_cost.max(0.0))
            .unwrap_or(0.0)
    };

    match &node.kind {
        PlanKind::Empty { .. } => 0.0,
        PlanKind::Values { rows, .. } => rows.len() as f64 * 0.02,
        PlanKind::TableScan { table, columns } => {
            rows * scan_touch_cost(catalog, *table, columns.len(), false, false)
        }
        PlanKind::PointGet { table, columns, residual, delivery, .. } => {
            let borrowed = matches!(delivery, PointDelivery::Borrowed | PointDelivery::ExistsOnly);
            let decode = scan_touch_cost(catalog, *table, columns.len(), borrowed, false);
            decode + residual.map_or(0.0, |expr| expression_runtime_cost(query, expr) * 0.02)
        }
        PlanKind::PrimaryRangeScan {
            table,
            columns,
            residual,
            delivery,
            ..
        } => {
            let borrowed = matches!(delivery, ScanDelivery::BorrowedPairs | ScanDelivery::BorrowedKeys);
            let key_only = matches!(delivery, ScanDelivery::Keys | ScanDelivery::BorrowedKeys);
            let per_row = scan_touch_cost(catalog, *table, columns.len(), borrowed, key_only);
            let predicate = residual.map_or(0.0, |expr| expression_runtime_cost(query, expr) * 0.02);
            rows * (per_row + predicate)
        }
        PlanKind::IndexRangeScan {
            table,
            columns,
            residual,
            delivery,
            covering,
            ..
        } => {
            let borrowed = matches!(delivery, ScanDelivery::BorrowedPairs | ScanDelivery::BorrowedKeys);
            let key_only = matches!(delivery, ScanDelivery::Keys | ScanDelivery::BorrowedKeys);
            let mut per_row = scan_touch_cost(catalog, *table, columns.len(), borrowed, key_only) * 0.55;
            if !*covering {
                // A non-covering scan generally incurs a second primary-key lookup.
                per_row += 0.9;
            }
            let predicate = residual.map_or(0.0, |expr| expression_runtime_cost(query, expr) * 0.02);
            rows * (per_row + predicate)
        }
        PlanKind::FetchRows { input, table, columns } => {
            child_cost(*input)
                + rows * (0.8 + scan_touch_cost(catalog, *table, columns.len(), false, false))
        }
        PlanKind::RangeCount { .. } => 0.15 + rows * 0.01,
        PlanKind::Filter { input, predicate } => {
            child_cost(*input) + rows * expression_runtime_cost(query, *predicate) * 0.025
        }
        PlanKind::Project { input, expressions } => {
            let expr_cost: f64 = expressions
                .iter()
                .map(|projection| expression_runtime_cost(query, projection.expr))
                .sum();
            child_cost(*input) + rows * expr_cost * 0.01
        }
        PlanKind::Sort { input, .. } => {
            let nlogn = rows * rows.max(2.0).log2();
            child_cost(*input) + nlogn * 0.08 * row_width_factor(query, catalog, id)
        }
        PlanKind::Limit { input, .. }
        | PlanKind::Distinct { input, .. }
        | PlanKind::Exists { input, .. } => child_cost(*input) + rows * 0.02,
        PlanKind::Aggregate { input, .. } => child_cost(*input) + rows * 0.18,
        PlanKind::UnionAll { inputs, .. } => inputs.iter().map(|input| child_cost(*input)).sum::<f64>() + rows * 0.01,
        PlanKind::Join { left, right, .. } => {
            child_cost(*left) + child_cost(*right) + rows * 0.5
        }
        PlanKind::PhysicalJoin { left, right, algorithm, .. } => {
            let left_rows = query.plan(*left).and_then(|node| node.props.estimated_rows).unwrap_or(1_000) as f64;
            let right_rows = query.plan(*right).and_then(|node| node.props.estimated_rows).unwrap_or(1_000) as f64;
            let join_work = match algorithm {
                JoinAlgorithm::Hash => (left_rows + right_rows) * 0.38,
                JoinAlgorithm::Merge => (left_rows + right_rows) * 0.12,
                JoinAlgorithm::IndexNestedLoop(_) => left_rows.min(right_rows) * 0.95,
            };
            child_cost(*left) + child_cost(*right) + join_work
        }
        PlanKind::Insert { rows, .. } | PlanKind::WriteBatch { rows, .. } => rows.len() as f64 * 1.5,
        PlanKind::Update { input, assignments, .. } => {
            child_cost(*input) + rows * (1.0 + assignments.len() as f64 * 0.08)
        }
        PlanKind::Delete { input, .. } => child_cost(*input) + rows * 0.8,
        PlanKind::Materialize { input, mode } => match mode {
            Materialization::Streaming => child_cost(*input),
            Materialization::Buffered => {
                child_cost(*input) + rows * 0.12 * row_width_factor(query, catalog, *input)
            }
        },
    }
}

fn scan_touch_cost(
    catalog: &Catalog,
    table: TableId,
    requested_columns: usize,
    borrowed: bool,
    key_only: bool,
) -> f64 {
    if key_only {
        return if borrowed { 0.06 } else { 0.12 };
    }
    let Some(table_def) = catalog.table(table) else { return 1.0; };
    let total_fields = table_def.fields.len().max(1) as f64;
    let requested = requested_columns.min(table_def.fields.len()).max(1) as f64;
    let bytes = table_def.stats.average_row_bytes.max(8) as f64 * (requested / total_fields);
    let copy = if borrowed { 0.08 } else { 0.28 };
    0.12 + copy + bytes / 96.0
}

fn expression_runtime_cost(query: &Query, root: ExprId) -> f64 {
    fn visit(query: &Query, id: ExprId, seen: &mut HashSet<ExprId>) -> f64 {
        if !seen.insert(id) {
            return 0.0;
        }
        let Some(node) = query.expr(id) else { return 0.0; };
        let own = match &node.kind {
            ExprKind::Const(_) | ExprKind::Param(_) | ExprKind::Column(_) => 0.05,
            ExprKind::Unary { .. } | ExprKind::IsNull { .. } => 0.15,
            ExprKind::Binary { op, .. } if op.is_comparison() => 0.2,
            ExprKind::Binary { .. } => 0.3,
            ExprKind::Between { .. } => 0.35,
            ExprKind::InList { items, .. } => 0.2 + items.len().min(64) as f64 * 0.04,
            ExprKind::Call { function, .. } => match function {
                BuiltinFn::Abs | BuiltinFn::Floor | BuiltinFn::Ceil | BuiltinFn::Round => 0.4,
                BuiltinFn::StringLength | BuiltinFn::BytesLength => 0.6,
                BuiltinFn::LowerAscii | BuiltinFn::UpperAscii => 1.4,
            },
        };
        own + node
            .kind
            .operands()
            .into_iter()
            .map(|operand| visit(query, operand, seen))
            .sum::<f64>()
    }

    visit(query, root, &mut HashSet::default())
}

fn row_width_factor(query: &Query, catalog: &Catalog, id: PlanId) -> f64 {
    let columns = output_columns(query, catalog, id);
    if columns.is_empty() {
        return 0.25;
    }
    let mut bytes = 0usize;
    for column in columns {
        bytes += match query.column(column).map(|column| column.ty) {
            Some(ScalarType::Null) => 1,
            Some(ScalarType::Bool) => 1,
            Some(ScalarType::Number) => 8,
            Some(ScalarType::String)
            | Some(ScalarType::Bytes)
            | Some(ScalarType::Any)
            | None => 24,
        };
    }
    (bytes.max(8) as f64 / 64.0).max(0.125)
}

#[derive(Default)]
pub struct PropertyAnalysis;

impl QueryOptimization for PropertyAnalysis {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool {
        let mut changed = false;
        for id in query.live_plans_postorder() {
            let mut next = derive_properties(query, catalog, id);
            let node = query.plan_mut(id).unwrap();
            // Physical costing is owned by MemoryCostAnalysis. Preserve its score
            // while refreshing logical properties so the two analyses do not fight
            // each other in fixed-point pipelines.
            next.estimated_cost = node.props.estimated_cost;
            if node.props.row_shape != next.row_shape
                || node.props.ordering != next.ordering
                || node.props.unique_on != next.unique_on
                || node.props.estimated_rows != next.estimated_rows
            {
                node.props = next;
                changed = true;
            }
        }
        changed
    }
}

fn derive_properties(query: &Query, catalog: &Catalog, id: PlanId) -> PlanProperties {
    let Some(node) = query.plan(id) else { return PlanProperties::default(); };
    match &node.kind {
        PlanKind::Empty { columns } => PlanProperties { row_shape: RowShape { columns: columns.clone() }, estimated_rows: Some(0), ..Default::default() },
        PlanKind::Values { columns, rows } => PlanProperties {
            row_shape: RowShape { columns: columns.clone() },
            estimated_rows: Some(rows.len() as u64),
            estimated_cost: rows.len() as f64,
            ..Default::default()
        },
        PlanKind::TableScan { table, columns } => table_scan_props(query, catalog, *table, columns),
        PlanKind::PointGet { table, columns, residual, .. } => {
            let mut props = table_scan_props(query, catalog, *table, columns);
            props.estimated_rows = Some(1);
            props.estimated_cost = if residual.is_some() { 1.2 } else { 1.0 };
            props
        }
        PlanKind::PrimaryRangeScan { table, range, columns, direction, residual, limit, .. } => {
            let mut props = table_scan_props(query, catalog, *table, columns);
            if let Some(table_def) = catalog.table(*table) {
                props.ordering.keys = physical_ordering(
                    query,
                    *table,
                    columns,
                    table_def.primary_key.iter().map(|field| (*field, SortDirection::Asc, NullOrder::First)),
                    range.prefix.len(),
                    *direction,
                );
                let rows = estimate_range_rows(table_def.stats.rows, range.prefix.len(), residual.is_some(), *limit);
                props.estimated_rows = Some(rows);
                props.estimated_cost = rows as f64 + 2.0;
            }
            props
        }
        PlanKind::IndexRangeScan { table, index, range, columns, direction, residual, limit, covering, .. } => {
            let mut props = table_scan_props(query, catalog, *table, columns);
            if let (Some(table_def), Some(index_def)) = (catalog.table(*table), catalog.index(*index)) {
                props.ordering.keys = physical_ordering(
                    query,
                    *table,
                    columns,
                    index_def.columns.iter().map(|column| {
                        (column.field, if column.descending { SortDirection::Desc } else { SortDirection::Asc }, column.nulls)
                    }),
                    range.prefix.len(),
                    *direction,
                );
                let rows = estimate_range_rows(table_def.stats.rows, range.prefix.len(), residual.is_some(), *limit);
                props.estimated_rows = Some(rows);
                props.estimated_cost = rows as f64 * if *covering { 0.55 } else { 1.25 } + 2.0;
                if index_def.unique {
                    let unique: Vec<_> = index_def.columns.iter().filter_map(|index_column| {
                        columns.iter().copied().find(|column| {
                            matches!(query.storage_origin(*column), Some((origin_table, field)) if origin_table == *table && field == index_column.field)
                        })
                    }).collect();
                    if unique.len() == index_def.columns.len() {
                        props.unique_on.push(unique);
                    }
                }
            }
            props
        }
        PlanKind::RangeCount { output, .. } => PlanProperties {
            row_shape: RowShape { columns: vec![*output] },
            estimated_rows: Some(1),
            estimated_cost: 1.0,
            ..Default::default()
        },
        PlanKind::Filter { input, .. } => {
            let mut props = child_props(query, *input);
            props.estimated_rows = props.estimated_rows.map(|rows| rows / 3);
            props.estimated_cost += props.estimated_rows.unwrap_or(10) as f64 * 0.1;
            props
        }
        PlanKind::Project { input, expressions } => {
            let mut props = child_props(query, *input);
            let mut aliases: HashMap<ColumnId, ColumnId> = HashMap::default();
            for projection in expressions {
                if let Some(ExprKind::Column(source)) = producer(query, projection.expr) {
                    aliases.insert(*source, projection.column);
                }
            }

            let mut ordering = Vec::new();
            for (column, direction, nulls) in &props.ordering.keys {
                let Some(output) = aliases.get(column).copied() else { break; };
                ordering.push((output, *direction, *nulls));
            }
            let unique_on = props
                .unique_on
                .iter()
                .filter_map(|key| {
                    key.iter()
                        .map(|column| aliases.get(column).copied())
                        .collect::<Option<Vec<_>>>()
                })
                .collect();

            props.row_shape.columns = expressions.iter().map(|projection| projection.column).collect();
            props.ordering.keys = ordering;
            props.unique_on = unique_on;
            props
        }
        PlanKind::Sort { input, keys } => {
            let mut props = child_props(query, *input);
            if let Some(order) = sort_columns(query, keys) { props.ordering.keys = order; }
            let rows = props.estimated_rows.unwrap_or(1000).max(1) as f64;
            props.estimated_cost += rows * rows.log2();
            props
        }
        PlanKind::Limit { input, skip, limit } => {
            let mut props = child_props(query, *input);
            props.estimated_rows = props.estimated_rows.map(|rows| {
                let after_skip = rows.saturating_sub(*skip);
                limit.map_or(after_skip, |limit| after_skip.min(limit))
            });
            props
        }
        PlanKind::Distinct { input, columns } => {
            let mut props = child_props(query, *input);
            props.unique_on.push(columns.clone());
            props.estimated_rows = props.estimated_rows.map(|rows| rows / 2 + (rows > 0) as u64);
            props
        }
        PlanKind::Join { left, right, kind, on }
        | PlanKind::PhysicalJoin { left, right, kind, on, .. } => {
            let left_props = child_props(query, *left);
            let right_props = child_props(query, *right);
            let left_rows = left_props.estimated_rows.unwrap_or(1000);
            let right_rows = right_props.estimated_rows.unwrap_or(1000);
            let pairs = oriented_join_equalities(query, catalog, *left, *right, *on);
            let left_join_columns: Vec<_> = pairs.iter().map(|(left, _)| *left).collect();
            let right_join_columns: Vec<_> = pairs.iter().map(|(_, right)| *right).collect();
            let left_unique = unique_key_covered_by_columns(
                &left_props.unique_on,
                &left_join_columns,
            );
            let right_unique = unique_key_covered_by_columns(
                &right_props.unique_on,
                &right_join_columns,
            );

            let rows = match kind {
                JoinKind::Inner if left_unique && right_unique => left_rows.min(right_rows).max(1),
                JoinKind::Inner if right_unique => left_rows.max(1),
                JoinKind::Inner if left_unique => right_rows.max(1),
                JoinKind::Inner => left_rows.saturating_mul(right_rows).saturating_div(10).max(1),
                JoinKind::Left if right_unique => left_rows,
                JoinKind::Left => left_rows.max(left_rows.saturating_mul(right_rows).saturating_div(10)),
                JoinKind::Semi | JoinKind::Anti => left_rows / 2,
            };
            if matches!(kind, JoinKind::Semi | JoinKind::Anti) {
                let mut props = left_props;
                props.estimated_rows = Some(rows);
                props.estimated_cost += right_props.estimated_cost + rows as f64;
                return props;
            }

            let mut columns = left_props.row_shape.columns.clone();
            columns.extend(right_props.row_shape.columns.iter().copied());
            let mut unique_on = Vec::new();
            if right_unique {
                unique_on.extend(left_props.unique_on.iter().cloned());
            }
            if *kind == JoinKind::Inner && left_unique {
                unique_on.extend(right_props.unique_on.iter().cloned());
            }
            if *kind == JoinKind::Inner {
                for left_key in &left_props.unique_on {
                    for right_key in &right_props.unique_on {
                        let mut combined = left_key.clone();
                        combined.extend(right_key.iter().copied());
                        combined.sort_unstable_by_key(|column| column.0);
                        combined.dedup();
                        if !combined.is_empty() {
                            unique_on.push(combined);
                        }
                    }
                }
            }
            unique_on.sort_by(|a, b| {
                a.len()
                    .cmp(&b.len())
                    .then_with(|| a.iter().map(|column| column.0).cmp(b.iter().map(|column| column.0)))
            });
            unique_on.dedup();

            PlanProperties {
                row_shape: RowShape { columns },
                unique_on,
                estimated_rows: Some(rows),
                estimated_cost: left_props.estimated_cost + right_props.estimated_cost + rows as f64,
                ..Default::default()
            }
        }
        PlanKind::Aggregate { input, group_by, aggregates } => {
            let input_props = child_props(query, *input);
            let group_columns: Vec<_> = group_by.iter().map(|projection| projection.column).collect();
            let mut columns = group_columns.clone();
            columns.extend(aggregates.iter().map(|aggregate| aggregate.column));
            let estimated_rows = if group_by.is_empty() {
                1
            } else {
                input_props.estimated_rows.unwrap_or(1000).min(1000)
            };
            PlanProperties {
                row_shape: RowShape { columns },
                unique_on: (!group_columns.is_empty()).then_some(group_columns).into_iter().collect(),
                estimated_rows: Some(estimated_rows),
                estimated_cost: input_props.estimated_cost + input_props.estimated_rows.unwrap_or(1000) as f64,
                ..Default::default()
            }
        }
        PlanKind::UnionAll { inputs, columns } => {
            let rows = inputs.iter().map(|input| child_props(query, *input).estimated_rows.unwrap_or(0)).sum();
            let columns = columns.clone();
            PlanProperties {
                row_shape: RowShape { columns },
                estimated_rows: Some(rows),
                estimated_cost: rows as f64,
                ..Default::default()
            }
        }
        PlanKind::Exists { output, .. } => PlanProperties {
            row_shape: RowShape { columns: vec![*output] },
            estimated_rows: Some(1),
            estimated_cost: 1.0,
            ..Default::default()
        },
        PlanKind::Insert { rows, .. } | PlanKind::WriteBatch { rows, .. } => PlanProperties {
            estimated_rows: Some(rows.len() as u64),
            estimated_cost: rows.len() as f64,
            ..Default::default()
        },
        PlanKind::FetchRows { input, columns, .. } => {
            let mut props = child_props(query, *input);
            props.row_shape.columns = columns.clone();
            props.ordering.keys.retain(|(column, _, _)| columns.contains(column));
            props.unique_on.retain(|key| key.iter().all(|column| columns.contains(column)));
            props.estimated_cost += props.estimated_rows.unwrap_or(1000) as f64 * 0.75;
            props
        }
        PlanKind::Update { input, .. }
        | PlanKind::Delete { input, .. }
        | PlanKind::Materialize { input, .. } => child_props(query, *input),
    }
}

fn unique_key_covered_by_columns(
    unique_keys: &[Vec<ColumnId>],
    columns: &[ColumnId],
) -> bool {
    unique_keys.iter().any(|key| {
        !key.is_empty() && key.iter().all(|column| columns.contains(column))
    })
}

fn child_props(query: &Query, id: PlanId) -> PlanProperties {
    query.plan(id).map(|node| node.props.clone()).unwrap_or_default()
}

fn table_scan_props(query: &Query, catalog: &Catalog, table: TableId, columns: &[ColumnId]) -> PlanProperties {
    let Some(table_def) = catalog.table(table) else { return PlanProperties::default(); };
    let primary_columns: Vec<_> = table_def.primary_key.iter().filter_map(|field| {
        columns.iter().copied().find(|column| {
            matches!(query.storage_origin(*column), Some((origin_table, origin_field)) if origin_table == table && origin_field == *field)
        })
    }).collect();
    let unique_on = if primary_columns.len() == table_def.primary_key.len() {
        vec![primary_columns]
    } else {
        Vec::new()
    };
    PlanProperties {
        row_shape: RowShape { columns: columns.to_vec() },
        unique_on,
        estimated_rows: Some(table_def.stats.rows),
        estimated_cost: table_def.stats.rows as f64,
        ..Default::default()
    }
}

fn physical_ordering(
    query: &Query,
    table: TableId,
    columns: &[ColumnId],
    physical: impl IntoIterator<Item = (FieldId, SortDirection, NullOrder)>,
    fixed_prefix: usize,
    direction: ScanDirection,
) -> Vec<(ColumnId, SortDirection, NullOrder)> {
    let mut out = Vec::new();
    for (position, (field, base_direction, base_nulls)) in physical.into_iter().enumerate() {
        if position < fixed_prefix {
            continue;
        }
        let Some(column) = columns.iter().copied().find(|column| {
            matches!(query.storage_origin(*column), Some((origin_table, origin_field)) if origin_table == table && origin_field == field)
        }) else { break; };
        let actual = if direction == ScanDirection::Forward { base_direction } else { opposite_direction(base_direction) };
        let nulls = if direction == ScanDirection::Forward { base_nulls } else { opposite_null_order(base_nulls) };
        out.push((column, actual, nulls));
    }
    out
}

fn estimate_range_rows(total: u64, equality_prefix: usize, residual: bool, limit: Option<u64>) -> u64 {
    let divisor = 10u64.saturating_pow(equality_prefix.min(8) as u32).max(1);
    let mut rows = (total / divisor).max((total > 0) as u64);
    if residual { rows = (rows / 3).max((rows > 0) as u64); }
    if let Some(limit) = limit { rows = rows.min(limit); }
    rows
}

fn estimated_rows(query: &Query, catalog: &Catalog, id: PlanId) -> u64 {
    query.plan(id).and_then(|node| node.props.estimated_rows).unwrap_or_else(|| {
        plan_table(query, id).and_then(|table| catalog.table(table).map(|table| table.stats.rows)).unwrap_or(1000)
    })
}

// -----------------------------------------------------------------------------
// Shared analysis helpers
// -----------------------------------------------------------------------------

fn flatten_boolean(query: &Query, id: ExprId, op: BinaryOp, out: &mut Vec<ExprId>) {
    if let Some(ExprKind::Binary { op: candidate, left, right }) = producer(query, id) {
        if *candidate == op {
            flatten_boolean(query, *left, op, out);
            flatten_boolean(query, *right, op, out);
            return;
        }
    }
    out.push(id);
}

fn conjunction(query: &mut Query, terms: &[ExprId]) -> Option<ExprId> {
    let mut iter = terms.iter().copied();
    let mut current = iter.next()?;
    for term in iter {
        current = query.add_expr(
            ExprKind::Binary { op: BinaryOp::And, left: current, right: term },
            ScalarType::Bool,
            false,
        );
    }
    Some(current)
}

fn equality_column_bound(query: &Query, id: ExprId) -> Option<(ColumnId, ExprId)> {
    let ExprKind::Binary { op: BinaryOp::Eq, left, right } = producer(query, id)? else { return None; };
    if let Some(ExprKind::Column(field)) = producer(query, *left) {
        if is_runtime_bound(query, *right) { return Some((*field, *right)); }
    }
    if let Some(ExprKind::Column(field)) = producer(query, *right) {
        if is_runtime_bound(query, *left) { return Some((*field, *left)); }
    }
    None
}

fn columns_in_expr(query: &Query, root: ExprId) -> HashSet<ColumnId> {
    let mut out = HashSet::default();
    let mut seen = HashSet::default();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) { continue; }
        let Some(node) = query.expr(id) else { continue; };
        if let ExprKind::Column(field) = &node.kind { out.insert(*field); }
        stack.extend(node.kind.operands());
    }
    out
}

fn output_columns(query: &Query, catalog: &Catalog, id: PlanId) -> Vec<ColumnId> {
    let Some(node) = query.plan(id) else { return Vec::new(); };
    if !node.props.row_shape.columns.is_empty() {
        return node.props.row_shape.columns.clone();
    }
    match &node.kind {
        PlanKind::TableScan { columns, .. }
        | PlanKind::PointGet { columns, .. }
        | PlanKind::PrimaryRangeScan { columns, .. }
        | PlanKind::IndexRangeScan { columns, .. } => columns.clone(),
        PlanKind::Filter { input, .. }
        | PlanKind::Sort { input, .. }
        | PlanKind::Limit { input, .. }
        | PlanKind::Distinct { input, .. }
        | PlanKind::Materialize { input, .. } => output_columns(query, catalog, *input),
        PlanKind::FetchRows { columns, .. } => columns.clone(),
        PlanKind::Project { expressions, .. } => expressions.iter().map(|projection| projection.column).collect(),
        PlanKind::Join { left, right, kind, .. } | PlanKind::PhysicalJoin { left, right, kind, .. } => {
            let mut columns = output_columns(query, catalog, *left);
            if !matches!(kind, JoinKind::Semi | JoinKind::Anti) {
                columns.extend(output_columns(query, catalog, *right));
            }
            columns
        }
        PlanKind::Aggregate { group_by, aggregates, .. } => {
            let mut columns: Vec<_> = group_by.iter().map(|projection| projection.column).collect();
            columns.extend(aggregates.iter().map(|aggregate| aggregate.column));
            columns
        }
        PlanKind::Values { columns, .. } => columns.clone(),
        PlanKind::Exists { output, .. } | PlanKind::RangeCount { output, .. } => vec![*output],
        PlanKind::UnionAll { columns, .. } => columns.clone(),
        PlanKind::Update { input, .. } | PlanKind::Delete { input, .. } => output_columns(query, catalog, *input),
        _ => Vec::new(),
    }
}

fn plan_table(query: &Query, id: PlanId) -> Option<TableId> {
    match &query.plan(id)?.kind {
        PlanKind::TableScan { table, .. }
        | PlanKind::PointGet { table, .. }
        | PlanKind::PrimaryRangeScan { table, .. }
        | PlanKind::IndexRangeScan { table, .. }
        | PlanKind::RangeCount { table, .. }
        | PlanKind::FetchRows { table, .. }
        | PlanKind::Insert { table, .. }
        | PlanKind::Update { table, .. }
        | PlanKind::Delete { table, .. }
        | PlanKind::WriteBatch { table, .. } => Some(*table),
        PlanKind::Filter { input, .. }
        | PlanKind::Project { input, .. }
        | PlanKind::Sort { input, .. }
        | PlanKind::Limit { input, .. }
        | PlanKind::Distinct { input, .. }
        | PlanKind::Aggregate { input, .. }
        | PlanKind::Exists { input, .. }
        | PlanKind::Materialize { input, .. } => plan_table(query, *input),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::napalm_mx_optimizer::{MxOptimizer, Optimize};

    fn run_pass<P: QueryOptimization + 'static>(
        pass: P,
        query: &mut Query,
        catalog: &Catalog,
    ) -> bool {
        Optimize::new()
            .with_query_pass(pass)
            .verify_after_each_pass(true)
            .optimize_query(query, catalog)
            .expect("MX optimizer pass produced invalid IR")
    }

    fn run_optimizer(query: &mut Query, catalog: &Catalog) {
        MxOptimizer::default()
            .optimize(query, catalog)
            .expect("MX optimizer produced invalid IR");
    }

    fn simple_query() -> (Catalog, Query, ColumnId, ColumnId) {
        let id_field = FieldId(0);
        let age_field = FieldId(1);
        let table = TableDef {
            id: TableId(0),
            name: StringId(0),
            key_prefix: b"t/users/".to_vec(),
            fields: vec![
                FieldDef { id: id_field, name: StringId(1), ty: ScalarType::Number, nullable: false, ordinal: 0 },
                FieldDef { id: age_field, name: StringId(2), ty: ScalarType::Number, nullable: false, ordinal: 1 },
            ],
            primary_key: vec![id_field],
            indexes: vec![IndexId(0)],
            stats: TableStats { rows: 100_000, average_row_bytes: 64, distinct: HashMap::default() },
        };
        let index = IndexDef {
            id: IndexId(0),
            table: TableId(0),
            name: StringId(3),
            key_prefix: b"i/users/age/".to_vec(),
            columns: vec![IndexColumn { field: age_field, encoding: KeyEncoding::NumberOrdered, descending: false, nulls: NullOrder::First }],
            included_fields: vec![],
            unique: false,
        };
        let catalog = Catalog { tables: vec![table], indexes: vec![index] };
        let mut query = Query {
            id: QueryId(0),
            name: None,
            root: PlanId(0),
            params: vec![ParamDef { id: ParamId(0), name: None, ty: ScalarType::Number, nullable: false }],
            columns: vec![],
            exprs: vec![],
            plans: vec![],
            transaction: TransactionMode::Auto,
        };
        let id_column = query.add_column(
            Some(StringId(1)),
            ScalarType::Number,
            false,
            ColumnOrigin::Storage { table: TableId(0), field: id_field },
        );
        let age_column = query.add_column(
            Some(StringId(2)),
            ScalarType::Number,
            false,
            ColumnOrigin::Storage { table: TableId(0), field: age_field },
        );
        let scan = query.add_plan(PlanKind::TableScan {
            table: TableId(0),
            columns: vec![id_column, age_column],
        });
        query.root = scan;
        (catalog, query, id_column, age_column)
    }

    #[test]
    fn primary_key_equality_becomes_point_get() {
        let (catalog, mut query, id_field, _) = simple_query();
        let field = query.add_expr(ExprKind::Column(id_field), ScalarType::Number, false);
        let param = query.add_expr(ExprKind::Param(ParamId(0)), ScalarType::Number, false);
        let predicate = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Eq, left: field, right: param },
            ScalarType::Bool,
            false,
        );
        let filter = query.add_plan(PlanKind::Filter { input: query.root, predicate });
        query.root = filter;
        let _ = run_pass(AccessPathSelection, &mut query, &catalog);
        assert!(matches!(&query.plan(query.root).unwrap().kind, PlanKind::PointGet { .. }));
    }

    #[test]
    fn secondary_range_becomes_index_scan() {
        let (catalog, mut query, _, age_field) = simple_query();
        let field = query.add_expr(ExprKind::Column(age_field), ScalarType::Number, false);
        let low = query.add_expr(ExprKind::Const(number(18.0)), ScalarType::Number, false);
        let predicate = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Ge, left: field, right: low },
            ScalarType::Bool,
            false,
        );
        let filter = query.add_plan(PlanKind::Filter { input: query.root, predicate });
        query.root = filter;
        let _ = run_pass(AccessPathSelection, &mut query, &catalog);
        assert!(matches!(&query.plan(query.root).unwrap().kind, PlanKind::IndexRangeScan { index: IndexId(0), .. }));
    }
    #[test]
    fn unique_secondary_equality_is_bounded_to_one_entry() {
        let (mut catalog, mut query, _, age_column) = simple_query();
        catalog.indexes[0].unique = true;
        let column = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let param = query.add_expr(ExprKind::Param(ParamId(0)), ScalarType::Number, false);
        let predicate = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Eq, left: column, right: param },
            ScalarType::Bool,
            false,
        );
        let filter = query.add_plan(PlanKind::Filter { input: query.root, predicate });
        query.root = filter;

        let _ = run_pass(AccessPathSelection, &mut query, &catalog);
        assert!(matches!(
            &query.plan(query.root).unwrap().kind,
            PlanKind::IndexRangeScan { index: IndexId(0), limit: Some(1), .. }
        ));
    }

    #[test]
    fn distinct_on_row_with_primary_key_is_eliminated() {
        let (catalog, mut query, id_column, age_column) = simple_query();
        let scan = query.root;
        let distinct = query.add_plan(PlanKind::Distinct {
            input: scan,
            columns: vec![id_column, age_column],
        });
        query.root = distinct;

        let _ = run_pass(AccessPathSelection, &mut query, &catalog);
        let _ = run_pass(PropertyAnalysis, &mut query, &catalog);
        let _ = run_pass(DistinctElimination, &mut query, &catalog);

        assert!(!matches!(query.plan(query.root).map(|node| &node.kind), Some(PlanKind::Distinct { .. })));
    }

    #[test]
    fn exists_caps_residual_free_range_scan_at_one() {
        let (catalog, mut query, _, _) = simple_query();
        let _ = run_pass(AccessPathSelection, &mut query, &catalog);
        let scan = query.root;
        let output = query.add_column(None, ScalarType::Bool, false, ColumnOrigin::Derived);
        let exists = query.add_plan(PlanKind::Exists { input: scan, output });
        query.root = exists;

        let _ = run_pass(ExistencePushdown, &mut query, &catalog);
        assert!(matches!(
            &query.plan(scan).unwrap().kind,
            PlanKind::PrimaryRangeScan { limit: Some(1), .. }
        ));
    }

    #[test]
    fn sort_uses_index_order_after_pruned_equality_prefix() {
        let (mut catalog, mut query, _, age_column) = simple_query();
        let tenant_field = FieldId(2);
        catalog.tables[0].fields.push(FieldDef {
            id: tenant_field,
            name: StringId(4),
            ty: ScalarType::Number,
            nullable: false,
            ordinal: 2,
        });
        catalog.indexes[0].columns = vec![
            IndexColumn {
                field: tenant_field,
                encoding: KeyEncoding::NumberOrdered,
                descending: false,
                nulls: NullOrder::First,
            },
            IndexColumn {
                field: FieldId(1),
                encoding: KeyEncoding::NumberOrdered,
                descending: false,
                nulls: NullOrder::First,
            },
        ];
        let tenant_column = query.add_column(
            Some(StringId(4)),
            ScalarType::Number,
            false,
            ColumnOrigin::Storage { table: TableId(0), field: tenant_field },
        );
        let tenant_param = query.add_expr(ExprKind::Param(ParamId(0)), ScalarType::Number, false);
        let scan = query.add_plan(PlanKind::IndexRangeScan {
            table: TableId(0),
            index: IndexId(0),
            range: KeyRange {
                prefix: vec![tenant_param],
                ..Default::default()
            },
            // The equality-constrained tenant column has already been pruned.
            columns: vec![age_column],
            residual: None,
            direction: ScanDirection::Forward,
            skip: 0,
            limit: None,
            delivery: ScanDelivery::Pairs,
            covering: true,
        });
        let _ = tenant_column;
        let age_expr = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let sort = query.add_plan(PlanKind::Sort {
            input: scan,
            keys: vec![SortKey {
                expr: age_expr,
                direction: SortDirection::Asc,
                nulls: NullOrder::First,
            }],
        });
        query.root = sort;

        let _ = run_pass(SortElimination, &mut query, &catalog);
        assert_eq!(query.root, scan);
    }

    #[test]
    fn full_pipeline_keeps_the_query_verified() {
        let (catalog, mut query, _, age_column) = simple_query();
        let column = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let low = query.add_expr(ExprKind::Const(number(18.0)), ScalarType::Number, false);
        let predicate = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Ge, left: column, right: low },
            ScalarType::Bool,
            false,
        );
        let filter = query.add_plan(PlanKind::Filter { input: query.root, predicate });
        let sort_expr = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let sort = query.add_plan(PlanKind::Sort {
            input: filter,
            keys: vec![SortKey {
                expr: sort_expr,
                direction: SortDirection::Asc,
                nulls: NullOrder::First,
            }],
        });
        let limit = query.add_plan(PlanKind::Limit { input: sort, skip: 0, limit: Some(32) });
        query.root = limit;

        run_optimizer(&mut query, &catalog);
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn sql_unknown_predicate_folds_to_empty_relation() {
        let (catalog, mut query, _, _) = simple_query();
        let left = query.add_expr(ExprKind::Const(ScalarValue::Null), ScalarType::Number, true);
        let right = query.add_expr(ExprKind::Const(ScalarValue::Null), ScalarType::Number, true);
        let predicate = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Eq, left, right },
            ScalarType::Bool,
            true,
        );
        let filter = query.add_plan(PlanKind::Filter { input: query.root, predicate });
        query.root = filter;

        let _ = run_pass(ExpressionSimplify, &mut query, &catalog);
        let _ = run_pass(LogicalSimplify, &mut query, &catalog);

        assert!(matches!(
            &query.plan(query.root).unwrap().kind,
            PlanKind::Empty { columns } if columns.len() == 2
        ));
    }


    #[test]
    fn in_list_canonicalization_deduplicates_semantic_constants() {
        let (catalog, mut query, id_column, _) = simple_query();
        let column = query.add_expr(ExprKind::Column(id_column), ScalarType::Number, false);
        let one_a = query.add_expr(ExprKind::Const(number(1.0)), ScalarType::Number, false);
        let one_b = query.add_expr(ExprKind::Const(number(1.0)), ScalarType::Number, false);
        let two = query.add_expr(ExprKind::Const(number(2.0)), ScalarType::Number, false);
        let predicate = query.add_expr(
            ExprKind::InList {
                value: column,
                items: vec![one_a, one_b, two],
                negated: false,
            },
            ScalarType::Bool,
            false,
        );
        let filter = query.add_plan(PlanKind::Filter { input: query.root, predicate });
        query.root = filter;

        assert!(run_pass(InListCanonicalization, &mut query, &catalog));
        let ExprKind::InList { items, .. } = &query.expr(predicate).unwrap().kind else {
            panic!("IN predicate was not preserved");
        };
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn equality_constraint_propagation_exposes_primary_key_lookup() {
        let (catalog, mut query, id_column, age_column) = simple_query();
        let id = query.add_expr(ExprKind::Column(id_column), ScalarType::Number, false);
        let age = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let columns_equal = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Eq, left: id, right: age },
            ScalarType::Bool,
            false,
        );
        let age_again = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let parameter = query.add_expr(ExprKind::Param(ParamId(0)), ScalarType::Number, false);
        let age_bound = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Eq, left: age_again, right: parameter },
            ScalarType::Bool,
            false,
        );
        let predicate = query.add_expr(
            ExprKind::Binary { op: BinaryOp::And, left: columns_equal, right: age_bound },
            ScalarType::Bool,
            false,
        );
        let filter = query.add_plan(PlanKind::Filter { input: query.root, predicate });
        query.root = filter;

        assert!(run_pass(ConstraintPropagation, &mut query, &catalog));
        let _ = run_pass(AccessPathSelection, &mut query, &catalog);
        assert!(matches!(
            &query.plan(query.root).unwrap().kind,
            PlanKind::PointGet { key, .. } if key == &vec![parameter]
        ));
    }

    #[test]
    fn range_canonicalization_keeps_only_the_tightest_constant_lower_bound() {
        let (catalog, mut query, _, age_column) = simple_query();
        let age_a = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let ten = query.add_expr(ExprKind::Const(number(10.0)), ScalarType::Number, false);
        let weak = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Ge, left: age_a, right: ten },
            ScalarType::Bool,
            false,
        );
        let age_b = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let eighteen = query.add_expr(ExprKind::Const(number(18.0)), ScalarType::Number, false);
        let strong = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Ge, left: age_b, right: eighteen },
            ScalarType::Bool,
            false,
        );
        let predicate = query.add_expr(
            ExprKind::Binary { op: BinaryOp::And, left: weak, right: strong },
            ScalarType::Bool,
            false,
        );
        let filter = query.add_plan(PlanKind::Filter { input: query.root, predicate });
        query.root = filter;

        assert!(run_pass(RangeCanonicalization, &mut query, &catalog));
        let _ = run_pass(AccessPathSelection, &mut query, &catalog);
        let PlanKind::IndexRangeScan { range, .. } = &query.plan(query.root).unwrap().kind else {
            panic!("tightened range did not select the age index");
        };
        let BoundValue::Inclusive(bound) = &range.lower else {
            panic!("expected an inclusive lower bound");
        };
        assert_eq!(number_const(&query, *bound), Some(18.0));
    }

    #[test]
    fn contradictory_constant_range_becomes_empty() {
        let (catalog, mut query, _, age_column) = simple_query();
        let age_a = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let twenty_a = query.add_expr(ExprKind::Const(number(20.0)), ScalarType::Number, false);
        let lower = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Gt, left: age_a, right: twenty_a },
            ScalarType::Bool,
            false,
        );
        let age_b = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let twenty_b = query.add_expr(ExprKind::Const(number(20.0)), ScalarType::Number, false);
        let upper = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Le, left: age_b, right: twenty_b },
            ScalarType::Bool,
            false,
        );
        let predicate = query.add_expr(
            ExprKind::Binary { op: BinaryOp::And, left: lower, right: upper },
            ScalarType::Bool,
            false,
        );
        let filter = query.add_plan(PlanKind::Filter { input: query.root, predicate });
        query.root = filter;

        assert!(run_pass(RangeCanonicalization, &mut query, &catalog));
        assert!(matches!(
            &query.plan(query.root).unwrap().kind,
            PlanKind::Empty { columns } if columns.len() == 2
        ));
    }

    #[test]
    fn primary_key_in_list_becomes_union_of_point_gets() {
        let (catalog, mut query, id_column, _) = simple_query();
        let id = query.add_expr(ExprKind::Column(id_column), ScalarType::Number, false);
        let one = query.add_expr(ExprKind::Const(number(1.0)), ScalarType::Number, false);
        let two = query.add_expr(ExprKind::Const(number(2.0)), ScalarType::Number, false);
        let three = query.add_expr(ExprKind::Const(number(3.0)), ScalarType::Number, false);
        let predicate = query.add_expr(
            ExprKind::InList {
                value: id,
                items: vec![one, two, three],
                negated: false,
            },
            ScalarType::Bool,
            false,
        );
        let filter = query.add_plan(PlanKind::Filter { input: query.root, predicate });
        query.root = filter;

        assert!(run_pass(MultiPointAccessSelection, &mut query, &catalog));
        let PlanKind::UnionAll { inputs, .. } = &query.plan(query.root).unwrap().kind else {
            panic!("primary-key IN was not expanded to point probes");
        };
        assert_eq!(inputs.len(), 3);
        assert!(inputs.iter().all(|input| matches!(
            query.plan(*input).map(|node| &node.kind),
            Some(PlanKind::PointGet { .. })
        )));
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn secondary_in_list_becomes_union_of_index_ranges() {
        let (catalog, mut query, _, age_column) = simple_query();
        let age = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let eighteen = query.add_expr(ExprKind::Const(number(18.0)), ScalarType::Number, false);
        let twenty_one = query.add_expr(ExprKind::Const(number(21.0)), ScalarType::Number, false);
        let predicate = query.add_expr(
            ExprKind::InList {
                value: age,
                items: vec![eighteen, twenty_one],
                negated: false,
            },
            ScalarType::Bool,
            false,
        );
        let filter = query.add_plan(PlanKind::Filter { input: query.root, predicate });
        query.root = filter;

        assert!(run_pass(MultiPointAccessSelection, &mut query, &catalog));
        let PlanKind::UnionAll { inputs, .. } = &query.plan(query.root).unwrap().kind else {
            panic!("indexed IN was not expanded to range probes");
        };
        assert_eq!(inputs.len(), 2);
        assert!(inputs.iter().all(|input| matches!(
            query.plan(*input).map(|node| &node.kind),
            Some(PlanKind::IndexRangeScan { index: IndexId(0), .. })
        )));
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }



    fn multi_index_query() -> (Catalog, Query, ColumnId, ColumnId, ColumnId, ColumnId) {
        let id_field = FieldId(0);
        let age_field = FieldId(1);
        let status_field = FieldId(2);
        let score_field = FieldId(3);
        let mut distinct = HashMap::default();
        distinct.insert(id_field, 100_000);
        distinct.insert(age_field, 1_000);
        distinct.insert(status_field, 10);
        distinct.insert(score_field, 10_000);
        let table = TableDef {
            id: TableId(0),
            name: StringId(0),
            key_prefix: b"t/items/".to_vec(),
            fields: vec![
                FieldDef { id: id_field, name: StringId(1), ty: ScalarType::Number, nullable: false, ordinal: 0 },
                FieldDef { id: age_field, name: StringId(2), ty: ScalarType::Number, nullable: false, ordinal: 1 },
                FieldDef { id: status_field, name: StringId(3), ty: ScalarType::Number, nullable: false, ordinal: 2 },
                FieldDef { id: score_field, name: StringId(4), ty: ScalarType::Number, nullable: false, ordinal: 3 },
            ],
            primary_key: vec![id_field],
            indexes: vec![IndexId(0), IndexId(1)],
            stats: TableStats {
                rows: 100_000,
                average_row_bytes: 96,
                distinct,
            },
        };
        let age_index = IndexDef {
            id: IndexId(0),
            table: TableId(0),
            name: StringId(10),
            key_prefix: b"i/items/age/".to_vec(),
            columns: vec![IndexColumn {
                field: age_field,
                encoding: KeyEncoding::NumberOrdered,
                descending: false,
                nulls: NullOrder::First,
            }],
            included_fields: vec![],
            unique: false,
        };
        let status_index = IndexDef {
            id: IndexId(1),
            table: TableId(0),
            name: StringId(11),
            key_prefix: b"i/items/status/".to_vec(),
            columns: vec![IndexColumn {
                field: status_field,
                encoding: KeyEncoding::NumberOrdered,
                descending: false,
                nulls: NullOrder::First,
            }],
            included_fields: vec![],
            unique: false,
        };
        let catalog = Catalog {
            tables: vec![table],
            indexes: vec![age_index, status_index],
        };
        let mut query = Query {
            id: QueryId(0),
            name: None,
            root: PlanId(0),
            params: vec![],
            columns: vec![],
            exprs: vec![],
            plans: vec![],
            transaction: TransactionMode::Auto,
        };
        let id = query.add_column(
            Some(StringId(1)),
            ScalarType::Number,
            false,
            ColumnOrigin::Storage { table: TableId(0), field: id_field },
        );
        let age = query.add_column(
            Some(StringId(2)),
            ScalarType::Number,
            false,
            ColumnOrigin::Storage { table: TableId(0), field: age_field },
        );
        let status = query.add_column(
            Some(StringId(3)),
            ScalarType::Number,
            false,
            ColumnOrigin::Storage { table: TableId(0), field: status_field },
        );
        let score = query.add_column(
            Some(StringId(4)),
            ScalarType::Number,
            false,
            ColumnOrigin::Storage { table: TableId(0), field: score_field },
        );
        let scan = query.add_plan(PlanKind::TableScan {
            table: TableId(0),
            columns: vec![id, age, status, score],
        });
        query.root = scan;
        (catalog, query, id, age, status, score)
    }

    #[test]
    fn independent_or_predicates_become_primary_key_deduplicated_index_union() {
        let (catalog, mut query, id, age, status, _) = multi_index_query();
        let age_expr = query.add_expr(ExprKind::Column(age), ScalarType::Number, false);
        let twenty = query.add_expr(ExprKind::Const(number(20.0)), ScalarType::Number, false);
        let age_eq = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Eq, left: age_expr, right: twenty },
            ScalarType::Bool,
            false,
        );
        let status_expr = query.add_expr(ExprKind::Column(status), ScalarType::Number, false);
        let one = query.add_expr(ExprKind::Const(number(1.0)), ScalarType::Number, false);
        let status_eq = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Eq, left: status_expr, right: one },
            ScalarType::Bool,
            false,
        );
        let predicate = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Or, left: age_eq, right: status_eq },
            ScalarType::Bool,
            false,
        );
        let filter = query.add_plan(PlanKind::Filter { input: query.root, predicate });
        query.root = filter;

        assert!(run_pass(IndexUnionSelection, &mut query, &catalog));
        let PlanKind::FetchRows { input: distinct, .. } = &query.plan(query.root).unwrap().kind else {
            panic!("indexed OR did not postpone the primary fetch");
        };
        let PlanKind::Distinct { input, columns } = &query.plan(*distinct).unwrap().kind else {
            panic!("indexed OR did not de-duplicate its PK stream");
        };
        assert_eq!(columns, &vec![id]);
        let PlanKind::UnionAll { inputs, .. } = &query.plan(*input).unwrap().kind else {
            panic!("DISTINCT did not wrap an index union");
        };
        assert_eq!(inputs.len(), 2);
        assert!(inputs.iter().all(|input| matches!(
            query.plan(*input).map(|node| &node.kind),
            Some(PlanKind::IndexRangeScan { .. })
        )));
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn selective_and_predicates_intersect_secondary_indexes_before_fetching_rows() {
        let (catalog, mut query, _, age, status, _) = multi_index_query();
        let age_expr = query.add_expr(ExprKind::Column(age), ScalarType::Number, false);
        let twenty = query.add_expr(ExprKind::Const(number(20.0)), ScalarType::Number, false);
        let age_eq = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Eq, left: age_expr, right: twenty },
            ScalarType::Bool,
            false,
        );
        let status_expr = query.add_expr(ExprKind::Column(status), ScalarType::Number, false);
        let one = query.add_expr(ExprKind::Const(number(1.0)), ScalarType::Number, false);
        let status_eq = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Eq, left: status_expr, right: one },
            ScalarType::Bool,
            false,
        );
        let predicate = query.add_expr(
            ExprKind::Binary { op: BinaryOp::And, left: age_eq, right: status_eq },
            ScalarType::Bool,
            false,
        );
        let filter = query.add_plan(PlanKind::Filter { input: query.root, predicate });
        query.root = filter;

        assert!(run_pass(IndexIntersectionSelection, &mut query, &catalog));
        let PlanKind::FetchRows { input, .. } = &query.plan(query.root).unwrap().kind else {
            panic!("index intersection did not postpone the primary fetch");
        };
        let PlanKind::PhysicalJoin { left, right, algorithm: JoinAlgorithm::Hash, .. } =
            &query.plan(*input).unwrap().kind
        else {
            panic!("index intersection did not build a PK-stream join");
        };
        assert!(matches!(query.plan(*left).map(|node| &node.kind), Some(PlanKind::IndexRangeScan { covering: true, .. })));
        assert!(matches!(query.plan(*right).map(|node| &node.kind), Some(PlanKind::IndexRangeScan { covering: true, .. })));
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn non_covering_secondary_scan_fetches_primary_row_after_index_work() {
        let (catalog, mut query, _, age, _, _) = multi_index_query();
        let age_expr = query.add_expr(ExprKind::Column(age), ScalarType::Number, false);
        let eighteen = query.add_expr(ExprKind::Const(number(18.0)), ScalarType::Number, false);
        let predicate = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Ge, left: age_expr, right: eighteen },
            ScalarType::Bool,
            false,
        );
        let filter = query.add_plan(PlanKind::Filter { input: query.root, predicate });
        query.root = filter;

        assert!(run_pass(AccessPathSelection, &mut query, &catalog));
        assert!(matches!(
            query.plan(query.root).map(|node| &node.kind),
            Some(PlanKind::IndexRangeScan { index: IndexId(0), covering: false, .. })
        ));
        assert!(run_pass(LateRowFetch, &mut query, &catalog));
        let PlanKind::FetchRows { input, .. } = &query.plan(query.root).unwrap().kind else {
            panic!("non-covering index scan was not split into late row fetch");
        };
        assert!(matches!(
            query.plan(*input).map(|node| &node.kind),
            Some(PlanKind::IndexRangeScan { index: IndexId(0), covering: true, .. })
        ));
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn unique_index_full_equality_bounds_scan_to_one_and_skip_eliminates_it() {
        let (mut catalog, mut query, _, age_column) = simple_query();
        catalog.indexes[0].unique = true;
        let value = query.add_expr(ExprKind::Const(number(33.0)), ScalarType::Number, false);
        let scan = query.add_plan(PlanKind::IndexRangeScan {
            table: TableId(0),
            index: IndexId(0),
            range: KeyRange { prefix: vec![value], ..Default::default() },
            columns: vec![age_column],
            residual: None,
            direction: ScanDirection::Forward,
            skip: 0,
            limit: None,
            delivery: ScanDelivery::Pairs,
            covering: true,
        });
        query.root = scan;
        assert!(run_pass(UniqueIndexSpecialization, &mut query, &catalog));
        assert!(matches!(
            &query.plan(scan).unwrap().kind,
            PlanKind::IndexRangeScan { limit: Some(1), .. }
        ));

        if let PlanKind::IndexRangeScan { skip, limit, .. } = &mut query.plan_mut(scan).unwrap().kind {
            *skip = 1;
            *limit = None;
        }
        assert!(run_pass(UniqueIndexSpecialization, &mut query, &catalog));
        assert!(matches!(&query.plan(scan).unwrap().kind, PlanKind::Empty { .. }));
    }

    #[test]
    fn max_on_ordered_index_reverses_and_bounds_scan_but_keeps_aggregate() {
        let (catalog, mut query, id_column, age_column) = simple_query();
        let scan = query.add_plan(PlanKind::IndexRangeScan {
            table: TableId(0),
            index: IndexId(0),
            range: KeyRange::default(),
            columns: vec![id_column, age_column],
            residual: None,
            direction: ScanDirection::Forward,
            skip: 0,
            limit: None,
            delivery: ScanDelivery::Pairs,
            covering: true,
        });
        let argument = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let output = query.add_column(None, ScalarType::Number, true, ColumnOrigin::Derived);
        let aggregate = query.add_plan(PlanKind::Aggregate {
            input: scan,
            group_by: vec![],
            aggregates: vec![AggregateExpr {
                column: output,
                function: AggregateFn::Max,
                argument: Some(argument),
                distinct: false,
            }],
        });
        query.root = aggregate;
        let _ = run_pass(PropertyAnalysis, &mut query, &catalog);

        assert!(run_pass(MinMaxOptimization, &mut query, &catalog));
        assert!(matches!(&query.plan(query.root).unwrap().kind, PlanKind::Aggregate { .. }));
        assert!(matches!(
            &query.plan(scan).unwrap().kind,
            PlanKind::IndexRangeScan { direction: ScanDirection::Reverse, limit: Some(1), .. }
        ));
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn ordered_top_n_fuses_sort_and_limit_into_reverse_index_scan() {
        let (catalog, mut query, id_column, age_column) = simple_query();
        let scan = query.add_plan(PlanKind::IndexRangeScan {
            table: TableId(0),
            index: IndexId(0),
            range: KeyRange::default(),
            columns: vec![id_column, age_column],
            residual: None,
            direction: ScanDirection::Forward,
            skip: 0,
            limit: None,
            delivery: ScanDelivery::Pairs,
            covering: true,
        });
        let age = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let sort = query.add_plan(PlanKind::Sort {
            input: scan,
            keys: vec![SortKey {
                expr: age,
                direction: SortDirection::Desc,
                nulls: NullOrder::Last,
            }],
        });
        let limit = query.add_plan(PlanKind::Limit {
            input: sort,
            skip: 0,
            limit: Some(7),
        });
        query.root = limit;
        let _ = run_pass(PropertyAnalysis, &mut query, &catalog);

        assert!(run_pass(OrderedTopNPushdown, &mut query, &catalog));
        assert_eq!(query.root, scan);
        assert!(matches!(
            &query.plan(scan).unwrap().kind,
            PlanKind::IndexRangeScan { direction: ScanDirection::Reverse, limit: Some(7), .. }
        ));
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    fn two_table_join_query() -> (
        Catalog,
        Query,
        PlanId,
        PlanId,
        ColumnId,
        ColumnId,
        ColumnId,
        ColumnId,
        ColumnId,
    ) {
        let left_tenant_field = FieldId(0);
        let left_user_field = FieldId(1);
        let right_tenant_field = FieldId(2);
        let right_user_field = FieldId(3);
        let right_value_field = FieldId(4);
        let mut left_distinct = HashMap::default();
        left_distinct.insert(left_tenant_field, 100);
        left_distinct.insert(left_user_field, 100_000);
        let mut right_distinct = HashMap::default();
        right_distinct.insert(right_tenant_field, 100);
        right_distinct.insert(right_user_field, 1_000_000);

        let left_table = TableDef {
            id: TableId(0),
            name: StringId(20),
            key_prefix: b"t/left/".to_vec(),
            fields: vec![
                FieldDef { id: left_tenant_field, name: StringId(21), ty: ScalarType::Number, nullable: false, ordinal: 0 },
                FieldDef { id: left_user_field, name: StringId(22), ty: ScalarType::Number, nullable: false, ordinal: 1 },
            ],
            primary_key: vec![left_tenant_field, left_user_field],
            indexes: vec![],
            stats: TableStats {
                rows: 100_000,
                average_row_bytes: 32,
                distinct: left_distinct,
            },
        };
        let right_table = TableDef {
            id: TableId(1),
            name: StringId(23),
            key_prefix: b"t/right/".to_vec(),
            fields: vec![
                FieldDef { id: right_tenant_field, name: StringId(24), ty: ScalarType::Number, nullable: false, ordinal: 0 },
                FieldDef { id: right_user_field, name: StringId(25), ty: ScalarType::Number, nullable: false, ordinal: 1 },
                FieldDef { id: right_value_field, name: StringId(26), ty: ScalarType::Number, nullable: false, ordinal: 2 },
            ],
            primary_key: vec![right_tenant_field, right_user_field],
            indexes: vec![IndexId(0)],
            stats: TableStats {
                rows: 1_000_000,
                average_row_bytes: 48,
                distinct: right_distinct,
            },
        };
        let index = IndexDef {
            id: IndexId(0),
            table: TableId(1),
            name: StringId(27),
            key_prefix: b"i/right/tenant-user/".to_vec(),
            columns: vec![
                IndexColumn { field: right_tenant_field, encoding: KeyEncoding::NumberOrdered, descending: false, nulls: NullOrder::First },
                IndexColumn { field: right_user_field, encoding: KeyEncoding::NumberOrdered, descending: false, nulls: NullOrder::First },
            ],
            included_fields: vec![right_value_field],
            unique: true,
        };
        let catalog = Catalog {
            tables: vec![left_table, right_table],
            indexes: vec![index],
        };

        let mut query = Query {
            id: QueryId(0),
            name: None,
            root: PlanId(0),
            params: vec![ParamDef { id: ParamId(0), name: None, ty: ScalarType::Number, nullable: false }],
            columns: vec![],
            exprs: vec![],
            plans: vec![],
            transaction: TransactionMode::Auto,
        };
        let left_tenant = query.add_column(None, ScalarType::Number, false, ColumnOrigin::Storage { table: TableId(0), field: left_tenant_field });
        let left_user = query.add_column(None, ScalarType::Number, false, ColumnOrigin::Storage { table: TableId(0), field: left_user_field });
        let right_tenant = query.add_column(None, ScalarType::Number, false, ColumnOrigin::Storage { table: TableId(1), field: right_tenant_field });
        let right_user = query.add_column(None, ScalarType::Number, false, ColumnOrigin::Storage { table: TableId(1), field: right_user_field });
        let right_value = query.add_column(None, ScalarType::Number, false, ColumnOrigin::Storage { table: TableId(1), field: right_value_field });
        let left_scan = query.add_plan(PlanKind::TableScan { table: TableId(0), columns: vec![left_tenant, left_user] });
        let right_scan = query.add_plan(PlanKind::TableScan { table: TableId(1), columns: vec![right_tenant, right_user, right_value] });
        query.root = left_scan;
        (
            catalog,
            query,
            left_scan,
            right_scan,
            left_tenant,
            left_user,
            right_tenant,
            right_user,
            right_value,
        )
    }

    fn equality_expr(query: &mut Query, left: ColumnId, right: ColumnId) -> ExprId {
        let left_def = query.column(left).unwrap().clone();
        let right_def = query.column(right).unwrap().clone();
        let left_expr = query.add_expr(ExprKind::Column(left), left_def.ty, left_def.nullable);
        let right_expr = query.add_expr(ExprKind::Column(right), right_def.ty, right_def.nullable);
        query.add_expr(
            ExprKind::Binary { op: BinaryOp::Eq, left: left_expr, right: right_expr },
            ScalarType::Bool,
            left_def.nullable || right_def.nullable,
        )
    }

    #[test]
    fn left_join_with_null_rejecting_where_becomes_inner_join() {
        let (catalog, mut query, left, right, _, left_user, _, right_user, right_value) = two_table_join_query();
        let on = equality_expr(&mut query, left_user, right_user);
        let join = query.add_plan(PlanKind::Join { left, right, kind: JoinKind::Left, on });
        let value_expr = query.add_expr(ExprKind::Column(right_value), ScalarType::Number, false);
        let zero = query.add_expr(ExprKind::Const(number(0.0)), ScalarType::Number, false);
        let predicate = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Gt, left: value_expr, right: zero },
            ScalarType::Bool,
            false,
        );
        let filter = query.add_plan(PlanKind::Filter { input: join, predicate });
        query.root = filter;

        assert!(run_pass(OuterJoinReduction, &mut query, &catalog));
        assert!(matches!(
            query.plan(join).map(|node| &node.kind),
            Some(PlanKind::Join { kind: JoinKind::Inner, .. })
        ));
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn join_constraint_propagation_pushes_parameter_to_other_side() {
        let (catalog, mut query, left_scan, right_scan, _, left_user, _, right_user, _) = two_table_join_query();
        let left_expr = query.add_expr(ExprKind::Column(left_user), ScalarType::Number, false);
        let param = query.add_expr(ExprKind::Param(ParamId(0)), ScalarType::Number, false);
        let left_bound = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Eq, left: left_expr, right: param },
            ScalarType::Bool,
            false,
        );
        let left = query.add_plan(PlanKind::Filter { input: left_scan, predicate: left_bound });
        let on = equality_expr(&mut query, left_user, right_user);
        let join = query.add_plan(PlanKind::Join { left, right: right_scan, kind: JoinKind::Inner, on });
        query.root = join;

        assert!(run_pass(JoinConstraintPropagation, &mut query, &catalog));
        let PlanKind::Join { right, .. } = &query.plan(join).unwrap().kind else {
            panic!("join unexpectedly changed shape");
        };
        assert!(local_plan_comparisons(&query, *right).iter().any(|constraint| {
            constraint.column == right_user
                && constraint.op == BinaryOp::Eq
                && constraint.bound == param
        }));
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn composite_join_keys_choose_composite_index_nested_loop() {
        let (catalog, mut query, left, right, left_tenant, left_user, right_tenant, right_user, _) = two_table_join_query();
        let tenant_eq = equality_expr(&mut query, left_tenant, right_tenant);
        let user_eq = equality_expr(&mut query, left_user, right_user);
        let on = query.add_expr(
            ExprKind::Binary { op: BinaryOp::And, left: tenant_eq, right: user_eq },
            ScalarType::Bool,
            false,
        );
        let join = query.add_plan(PlanKind::Join { left, right, kind: JoinKind::Inner, on });
        query.root = join;
        let _ = run_pass(PropertyAnalysis, &mut query, &catalog);

        assert!(run_pass(JoinSelection, &mut query, &catalog));
        assert!(matches!(
            query.plan(join).map(|node| &node.kind),
            Some(PlanKind::PhysicalJoin { algorithm: JoinAlgorithm::IndexNestedLoop(IndexId(0)), .. })
        ));
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn semi_join_uses_index_probe_when_right_side_is_indexed() {
        let (catalog, mut query, left, right, left_tenant, left_user, right_tenant, right_user, _) = two_table_join_query();
        let tenant_eq = equality_expr(&mut query, left_tenant, right_tenant);
        let user_eq = equality_expr(&mut query, left_user, right_user);
        let on = query.add_expr(
            ExprKind::Binary { op: BinaryOp::And, left: tenant_eq, right: user_eq },
            ScalarType::Bool,
            false,
        );
        let join = query.add_plan(PlanKind::Join { left, right, kind: JoinKind::Semi, on });
        query.root = join;
        let _ = run_pass(PropertyAnalysis, &mut query, &catalog);

        assert!(run_pass(JoinSelection, &mut query, &catalog));
        assert!(matches!(
            query.plan(join).map(|node| &node.kind),
            Some(PlanKind::PhysicalJoin {
                kind: JoinKind::Semi,
                algorithm: JoinAlgorithm::IndexNestedLoop(IndexId(0)),
                ..
            })
        ));
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn distinct_aggregate_on_unique_column_drops_distinct_and_count_argument() {
        let (catalog, mut query, id_column, _) = simple_query();
        let input = query.root;
        let _ = run_pass(PropertyAnalysis, &mut query, &catalog);
        let argument = query.add_expr(ExprKind::Column(id_column), ScalarType::Number, false);
        let output = query.add_column(None, ScalarType::Number, false, ColumnOrigin::Derived);
        let aggregate = query.add_plan(PlanKind::Aggregate {
            input,
            group_by: vec![],
            aggregates: vec![AggregateExpr {
                column: output,
                function: AggregateFn::Count,
                argument: Some(argument),
                distinct: true,
            }],
        });
        query.root = aggregate;

        assert!(run_pass(AggregateDistinctElimination, &mut query, &catalog));
        assert!(run_pass(CountArgumentElimination, &mut query, &catalog));
        let PlanKind::Aggregate { aggregates, .. } = &query.plan(aggregate).unwrap().kind else {
            panic!("aggregate was unexpectedly lowered");
        };
        assert!(!aggregates[0].distinct);
        assert!(aggregates[0].argument.is_none());
    }

    #[test]
    fn group_by_unique_key_becomes_scalar_projection() {
        let (catalog, mut query, id_column, age_column) = simple_query();
        let input = query.root;
        let _ = run_pass(PropertyAnalysis, &mut query, &catalog);
        let id_expr = query.add_expr(ExprKind::Column(id_column), ScalarType::Number, false);
        let age_expr = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let group_output = query.add_column(None, ScalarType::Number, false, ColumnOrigin::Derived);
        let max_output = query.add_column(None, ScalarType::Number, true, ColumnOrigin::Derived);
        let count_output = query.add_column(None, ScalarType::Number, false, ColumnOrigin::Derived);
        let aggregate = query.add_plan(PlanKind::Aggregate {
            input,
            group_by: vec![ProjectionExpr { column: group_output, expr: id_expr }],
            aggregates: vec![
                AggregateExpr {
                    column: max_output,
                    function: AggregateFn::Max,
                    argument: Some(age_expr),
                    distinct: false,
                },
                AggregateExpr {
                    column: count_output,
                    function: AggregateFn::Count,
                    argument: None,
                    distinct: false,
                },
            ],
        });
        query.root = aggregate;

        assert!(run_pass(UniqueGroupAggregateElimination, &mut query, &catalog));
        assert!(matches!(
            query.plan(aggregate).map(|node| &node.kind),
            Some(PlanKind::Project { expressions, .. }) if expressions.len() == 3
        ));
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn global_aggregate_over_empty_relation_is_folded_to_one_values_row() {
        let (catalog, mut query, id_column, age_column) = simple_query();
        let empty = query.add_plan(PlanKind::Empty { columns: vec![id_column, age_column] });
        let count_output = query.add_column(None, ScalarType::Number, false, ColumnOrigin::Derived);
        let max_output = query.add_column(None, ScalarType::Number, true, ColumnOrigin::Derived);
        let age_expr = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let aggregate = query.add_plan(PlanKind::Aggregate {
            input: empty,
            group_by: vec![],
            aggregates: vec![
                AggregateExpr { column: count_output, function: AggregateFn::Count, argument: None, distinct: false },
                AggregateExpr { column: max_output, function: AggregateFn::Max, argument: Some(age_expr), distinct: false },
            ],
        });
        query.root = aggregate;

        assert!(run_pass(EmptyAggregateFolding, &mut query, &catalog));
        assert!(matches!(
            query.plan(aggregate).map(|node| &node.kind),
            Some(PlanKind::Values { rows, .. }) if rows.len() == 1 && rows[0].len() == 2
        ));
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn bounded_dp_reorders_three_way_inner_join_and_preserves_output_order() {
        let a_id_field = FieldId(10);
        let b_id_field = FieldId(11);
        let b_a_field = FieldId(12);
        let c_id_field = FieldId(13);
        let c_b_field = FieldId(14);
        let catalog = Catalog {
            tables: vec![
                TableDef {
                    id: TableId(0),
                    name: StringId(40),
                    key_prefix: b"t/a/".to_vec(),
                    fields: vec![FieldDef { id: a_id_field, name: StringId(41), ty: ScalarType::Number, nullable: false, ordinal: 0 }],
                    primary_key: vec![a_id_field],
                    indexes: vec![],
                    stats: TableStats { rows: 1_000_000, average_row_bytes: 16, distinct: HashMap::default() },
                },
                TableDef {
                    id: TableId(1),
                    name: StringId(42),
                    key_prefix: b"t/b/".to_vec(),
                    fields: vec![
                        FieldDef { id: b_id_field, name: StringId(43), ty: ScalarType::Number, nullable: false, ordinal: 0 },
                        FieldDef { id: b_a_field, name: StringId(44), ty: ScalarType::Number, nullable: false, ordinal: 1 },
                    ],
                    primary_key: vec![b_id_field],
                    indexes: vec![],
                    stats: TableStats { rows: 10_000, average_row_bytes: 24, distinct: HashMap::default() },
                },
                TableDef {
                    id: TableId(2),
                    name: StringId(45),
                    key_prefix: b"t/c/".to_vec(),
                    fields: vec![
                        FieldDef { id: c_id_field, name: StringId(46), ty: ScalarType::Number, nullable: false, ordinal: 0 },
                        FieldDef { id: c_b_field, name: StringId(47), ty: ScalarType::Number, nullable: false, ordinal: 1 },
                    ],
                    primary_key: vec![c_id_field],
                    indexes: vec![],
                    stats: TableStats { rows: 10, average_row_bytes: 24, distinct: HashMap::default() },
                },
            ],
            indexes: vec![],
        };
        let mut query = Query {
            id: QueryId(0),
            name: None,
            root: PlanId(0),
            params: vec![],
            columns: vec![],
            exprs: vec![],
            plans: vec![],
            transaction: TransactionMode::Auto,
        };
        let a_id = query.add_column(None, ScalarType::Number, false, ColumnOrigin::Storage { table: TableId(0), field: a_id_field });
        let b_id = query.add_column(None, ScalarType::Number, false, ColumnOrigin::Storage { table: TableId(1), field: b_id_field });
        let b_a = query.add_column(None, ScalarType::Number, false, ColumnOrigin::Storage { table: TableId(1), field: b_a_field });
        let c_id = query.add_column(None, ScalarType::Number, false, ColumnOrigin::Storage { table: TableId(2), field: c_id_field });
        let c_b = query.add_column(None, ScalarType::Number, false, ColumnOrigin::Storage { table: TableId(2), field: c_b_field });
        let a = query.add_plan(PlanKind::TableScan { table: TableId(0), columns: vec![a_id] });
        let b = query.add_plan(PlanKind::TableScan { table: TableId(1), columns: vec![b_id, b_a] });
        let c = query.add_plan(PlanKind::TableScan { table: TableId(2), columns: vec![c_id, c_b] });
        let ab_on = equality_expr(&mut query, a_id, b_a);
        let ab = query.add_plan(PlanKind::Join { left: a, right: b, kind: JoinKind::Inner, on: ab_on });
        let bc_on = equality_expr(&mut query, b_id, c_b);
        let abc = query.add_plan(PlanKind::Join { left: ab, right: c, kind: JoinKind::Inner, on: bc_on });
        query.root = abc;
        let original_columns = output_columns(&query, &catalog, abc);
        let _ = run_pass(PropertyAnalysis, &mut query, &catalog);

        assert!(run_pass(JoinReordering, &mut query, &catalog));
        let PlanKind::Project { input, expressions } = &query.plan(query.root).unwrap().kind else {
            panic!("reordered join did not restore the original output order");
        };
        assert_eq!(expressions.iter().map(|projection| projection.column).collect::<Vec<_>>(), original_columns);
        let mut leaves = Vec::new();
        let mut terms = Vec::new();
        flatten_inner_join_component(&query, *input, &mut leaves, &mut terms);
        assert_eq!(leaves.len(), 3);
        assert_ne!(leaves[0], a);
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }


    #[test]
    fn wave4_filter_is_fused_into_physical_scan_residual() {
        let (catalog, mut query, id_column, age_column) = simple_query();
        let scan = query.add_plan(PlanKind::PrimaryRangeScan {
            table: TableId(0),
            range: KeyRange::default(),
            columns: vec![id_column, age_column],
            residual: None,
            direction: ScanDirection::Forward,
            skip: 0,
            limit: None,
            delivery: ScanDelivery::Pairs,
        });
        let age = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let threshold = query.add_expr(ExprKind::Const(number(18.0)), ScalarType::Number, false);
        let predicate = query.add_expr(
            ExprKind::Binary { op: BinaryOp::Gt, left: age, right: threshold },
            ScalarType::Bool,
            false,
        );
        let filter = query.add_plan(PlanKind::Filter { input: scan, predicate });
        query.root = filter;

        assert!(run_pass(ScanPipelineFusion, &mut query, &catalog));
        assert_eq!(query.root, scan);
        assert!(matches!(
            query.plan(scan).map(|node| &node.kind),
            Some(PlanKind::PrimaryRangeScan { residual: Some(found), .. }) if *found == predicate
        ));
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn wave4_decode_pruning_shrinks_fetch_rows_after_late_materialization() {
        let (catalog, mut query, id_column, age_column) = simple_query();
        let scan = query.add_plan(PlanKind::IndexRangeScan {
            table: TableId(0),
            index: IndexId(0),
            range: KeyRange::default(),
            columns: vec![id_column],
            residual: None,
            direction: ScanDirection::Forward,
            skip: 0,
            limit: None,
            delivery: ScanDelivery::Keys,
            covering: true,
        });
        let fetch = query.add_plan(PlanKind::FetchRows {
            input: scan,
            table: TableId(0),
            columns: vec![id_column, age_column],
        });
        let id_expr = query.add_expr(ExprKind::Column(id_column), ScalarType::Number, false);
        let project = query.add_plan(PlanKind::Project {
            input: fetch,
            expressions: vec![ProjectionExpr { column: id_column, expr: id_expr }],
        });
        query.root = project;

        assert!(run_pass(PhysicalDecodePruning, &mut query, &catalog));
        assert!(matches!(
            query.plan(fetch).map(|node| &node.kind),
            Some(PlanKind::FetchRows { columns, .. }) if columns == &vec![id_column]
        ));
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn wave4_union_limit_caps_every_branch_without_removing_outer_limit() {
        let (catalog, mut query, id_column, age_column) = simple_query();
        let make_scan = |query: &mut Query| {
            query.add_plan(PlanKind::PrimaryRangeScan {
                table: TableId(0),
                range: KeyRange::default(),
                columns: vec![id_column, age_column],
                residual: None,
                direction: ScanDirection::Forward,
                skip: 0,
                limit: None,
                delivery: ScanDelivery::Pairs,
            })
        };
        let left = make_scan(&mut query);
        let right = make_scan(&mut query);
        let union = query.add_plan(PlanKind::UnionAll {
            inputs: vec![left, right],
            columns: vec![id_column, age_column],
        });
        let limit = query.add_plan(PlanKind::Limit {
            input: union,
            skip: 5,
            limit: Some(10),
        });
        query.root = limit;

        assert!(run_pass(UnionLimitPushdown, &mut query, &catalog));
        let PlanKind::UnionAll { inputs, .. } = &query.plan(union).unwrap().kind else {
            panic!("union disappeared");
        };
        assert_eq!(inputs.len(), 2);
        assert!(inputs.iter().all(|input| matches!(
            query.plan(*input).map(|node| &node.kind),
            Some(PlanKind::Limit { skip: 0, limit: Some(15), .. })
        )));
        assert_eq!(query.root, limit);
    }

    #[test]
    fn wave4_expensive_shared_subplan_gets_one_buffered_materialization() {
        let (catalog, mut query, id_column, age_column) = simple_query();
        let shared = query.root;
        let id_expr = query.add_expr(ExprKind::Column(id_column), ScalarType::Number, false);
        let age_expr = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let left = query.add_plan(PlanKind::Project {
            input: shared,
            expressions: vec![
                ProjectionExpr { column: id_column, expr: id_expr },
                ProjectionExpr { column: age_column, expr: age_expr },
            ],
        });
        let right = query.add_plan(PlanKind::Project {
            input: shared,
            expressions: vec![
                ProjectionExpr { column: id_column, expr: id_expr },
                ProjectionExpr { column: age_column, expr: age_expr },
            ],
        });
        let union = query.add_plan(PlanKind::UnionAll {
            inputs: vec![left, right],
            columns: vec![id_column, age_column],
        });
        query.root = union;
        let _ = run_pass(PropertyAnalysis, &mut query, &catalog);
        let _ = run_pass(MemoryCostAnalysis, &mut query, &catalog);

        assert!(run_pass(SharedSubplanMaterialization, &mut query, &catalog));
        let left_input = match &query.plan(left).unwrap().kind {
            PlanKind::Project { input, .. } => *input,
            _ => panic!("left projection changed unexpectedly"),
        };
        let right_input = match &query.plan(right).unwrap().kind {
            PlanKind::Project { input, .. } => *input,
            _ => panic!("right projection changed unexpectedly"),
        };
        assert_eq!(left_input, right_input);
        assert!(matches!(
            query.plan(left_input).map(|node| &node.kind),
            Some(PlanKind::Materialize { input, mode: Materialization::Buffered }) if *input == shared
        ));
    }

    #[test]
    fn wave4_identity_update_is_removed_completely() {
        let (catalog, mut query, _, age_column) = simple_query();
        let input = query.root;
        let age_expr = query.add_expr(ExprKind::Column(age_column), ScalarType::Number, false);
        let update = query.add_plan(PlanKind::Update {
            input,
            table: TableId(0),
            assignments: vec![(FieldId(1), age_expr)],
        });
        query.root = update;

        assert!(run_pass(WriteNoOpElimination, &mut query, &catalog));
        assert_eq!(query.root, input);
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn wave4_set_batch_keeps_only_last_write_for_same_runtime_key() {
        let (catalog, mut query, _, _) = simple_query();
        let key = query.add_expr(ExprKind::Param(ParamId(0)), ScalarType::Number, false);
        let first_age = query.add_expr(ExprKind::Const(number(10.0)), ScalarType::Number, false);
        let last_age = query.add_expr(ExprKind::Const(number(20.0)), ScalarType::Number, false);
        let batch = query.add_plan(PlanKind::WriteBatch {
            table: TableId(0),
            mode: MutationMode::Set,
            rows: vec![
                vec![(FieldId(0), key), (FieldId(1), first_age)],
                vec![(FieldId(0), key), (FieldId(1), last_age)],
            ],
        });
        query.root = batch;

        assert!(run_pass(WriteBatchCoalescing, &mut query, &catalog));
        let PlanKind::WriteBatch { rows, .. } = &query.plan(batch).unwrap().kind else {
            panic!("write batch disappeared");
        };
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].iter().find(|(field, _)| *field == FieldId(1)).map(|(_, expr)| *expr), Some(last_age));
    }

    #[test]
    fn wave4_constant_plan_needs_no_snapshot_transaction() {
        let catalog = Catalog::default();
        let mut query = Query {
            id: QueryId(0),
            name: None,
            root: PlanId(0),
            params: vec![],
            columns: vec![],
            exprs: vec![],
            plans: vec![],
            transaction: TransactionMode::ReadOnlySnapshot,
        };
        let output = query.add_column(None, ScalarType::Number, false, ColumnOrigin::Derived);
        let value = query.add_expr(ExprKind::Const(number(42.0)), ScalarType::Number, false);
        let values = query.add_plan(PlanKind::Values {
            columns: vec![output],
            rows: vec![vec![value]],
        });
        query.root = values;

        assert!(run_pass(TransactionSpecialization, &mut query, &catalog));
        assert_eq!(query.transaction, TransactionMode::Auto);
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn wave4_delivery_propagates_borrowing_through_distinct_consumer() {
        let (catalog, mut query, id_column, _) = simple_query();
        let scan = query.add_plan(PlanKind::PrimaryRangeScan {
            table: TableId(0),
            range: KeyRange::default(),
            columns: vec![id_column],
            residual: None,
            direction: ScanDirection::Forward,
            skip: 0,
            limit: None,
            delivery: ScanDelivery::Pairs,
        });
        let distinct = query.add_plan(PlanKind::Distinct {
            input: scan,
            columns: vec![id_column],
        });
        query.root = distinct;

        assert!(run_pass(DeliverySelection, &mut query, &catalog));
        assert!(matches!(
            query.plan(scan).map(|node| &node.kind),
            Some(PlanKind::PrimaryRangeScan { delivery: ScanDelivery::BorrowedKeys, .. })
        ));
    }

    #[test]
    fn wave4_memory_cost_prefers_key_only_borrowed_scan() {
        let (catalog, mut query, id_column, age_column) = simple_query();
        let pairs = query.add_plan(PlanKind::PrimaryRangeScan {
            table: TableId(0),
            range: KeyRange::default(),
            columns: vec![id_column, age_column],
            residual: None,
            direction: ScanDirection::Forward,
            skip: 0,
            limit: Some(1_000),
            delivery: ScanDelivery::Pairs,
        });
        let keys = query.add_plan(PlanKind::PrimaryRangeScan {
            table: TableId(0),
            range: KeyRange::default(),
            columns: vec![id_column],
            residual: None,
            direction: ScanDirection::Forward,
            skip: 0,
            limit: Some(1_000),
            delivery: ScanDelivery::BorrowedKeys,
        });
        query.plans[pairs.0 as usize].props.estimated_rows = Some(1_000);
        query.plans[keys.0 as usize].props.estimated_rows = Some(1_000);
        let pair_cost = memory_cost_for_plan(&query, &catalog, pairs);
        let key_cost = memory_cost_for_plan(&query, &catalog, keys);
        assert!(key_cost < pair_cost);
    }

}
