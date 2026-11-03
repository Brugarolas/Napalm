//! Extensible, verified optimizer for Napalm MX IR.
//!
//! This mirrors the WIR/Trace optimizer architecture: append-only arenas, stable
//! IDs, node-local worklist passes, whole-query transforms and whole-program
//! finalizers. The optimizer verifies the IR before optimization, optionally after
//! every stage, and once again at the end. Verification failures are returned with
//! the stage name instead of panicking inside the optimizer.

#![allow(dead_code)]

use std::any::Any;
use std::collections::VecDeque;
use std::mem::{discriminant, Discriminant};
use std::marker::PhantomData;

use smallbox::{smallbox, space::S1, SmallBox};
use smallvec::SmallVec;

use crate::napalm_collections::HashMap;
use crate::napalm_mx_ir::{
    verify_program, verify_query, Catalog, ExprId, ExprKind, PlanId, PlanKind, PlanNode, PlanProperties, Program,
    Query, ScalarType,
};

/// O(1) cursor into a stable plan-arena node.
pub struct PlanCursor<'q> {
    query: *mut Query,
    id: PlanId,
    _marker: PhantomData<&'q mut Query>,
}

impl<'q> PlanCursor<'q> {
    #[inline]
    pub fn new(query: &'q mut Query, id: PlanId) -> Self {
        assert!(query.plan(id).is_some(), "invalid PlanId {id:?}");
        Self {
            query: query as *mut Query,
            id,
            _marker: PhantomData,
        }
    }

    #[inline]
    unsafe fn query_ref(&self) -> &Query {
        unsafe { &*self.query }
    }

    #[inline]
    unsafe fn query_mut(&mut self) -> &mut Query {
        unsafe { &mut *self.query }
    }

    #[inline]
    pub fn id(&self) -> PlanId {
        self.id
    }

    #[inline]
    pub fn peek<R>(&self, f: impl FnOnce(&PlanNode) -> R) -> R {
        unsafe { f(self.query_ref().plan(self.id).expect("stale PlanCursor")) }
    }

    #[inline]
    pub fn peek_kind<R>(&self, f: impl FnOnce(&PlanKind) -> R) -> R {
        self.peek(|node| f(&node.kind))
    }

    #[inline]
    pub fn kind(&self) -> PlanKind {
        self.peek(|node| node.kind.clone())
    }

    #[inline]
    pub fn tag(&self) -> Discriminant<PlanKind> {
        self.peek(|node| discriminant(&node.kind))
    }

    #[inline]
    pub fn with_node<R>(&mut self, f: impl FnOnce(&mut PlanNode) -> R) -> R {
        let id = self.id;
        unsafe { f(self.query_mut().plan_mut(id).expect("stale PlanCursor")) }
    }

    #[inline]
    pub fn with_query<R>(&mut self, f: impl FnOnce(&mut Query) -> R) -> R {
        unsafe { f(self.query_mut()) }
    }

    #[inline]
    pub fn query_ref_public(&self) -> &Query {
        unsafe { self.query_ref() }
    }
}

#[derive(Clone, Debug)]
pub enum MxEdit {
    ReplaceKind {
        node: PlanId,
        kind: PlanKind,
    },
    ReplaceRoot {
        root: PlanId,
    },
    ReplaceChild {
        parent: PlanId,
        old_child: PlanId,
        new_child: PlanId,
    },
}

#[derive(Default)]
pub struct MxEditQueue {
    edits: SmallVec<[MxEdit; 8]>,
}

impl MxEditQueue {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.edits.clear();
    }

    #[inline]
    pub fn replace_kind(&mut self, node: PlanId, kind: PlanKind) {
        self.edits.push(MxEdit::ReplaceKind { node, kind });
    }

    #[inline]
    pub fn replace_root(&mut self, root: PlanId) {
        self.edits.push(MxEdit::ReplaceRoot { root });
    }

    #[inline]
    pub fn replace_child(&mut self, parent: PlanId, old_child: PlanId, new_child: PlanId) {
        self.edits.push(MxEdit::ReplaceChild {
            parent,
            old_child,
            new_child,
        });
    }

    /// Replace every live reference to `old` by `new`, including the query root.
    pub fn replace_all_uses(&mut self, query: &Query, old: PlanId, new: PlanId) {
        if query.root == old {
            self.replace_root(new);
        }
        for parent in query.live_plans_preorder() {
            let Some(node) = query.plan(parent) else { continue; };
            if node.kind.children().contains(&old) {
                self.replace_child(parent, old, new);
            }
        }
    }

    pub fn apply(&mut self, query: &mut Query) -> SmallVec<[PlanId; 16]> {
        let mut touched = SmallVec::<[PlanId; 16]>::new();
        for edit in self.edits.drain(..) {
            match edit {
                MxEdit::ReplaceKind { node, kind } => {
                    let target = query.plan_mut(node).expect("optimizer replaced invalid PlanId");
                    target.kind = kind;
                    target.props = PlanProperties::default();
                    touched.push(node);
                }
                MxEdit::ReplaceRoot { root } => {
                    assert!(query.plan(root).is_some(), "optimizer installed invalid root {root:?}");
                    query.root = root;
                    touched.push(root);
                }
                MxEdit::ReplaceChild {
                    parent,
                    old_child,
                    new_child,
                } => {
                    assert!(query.plan(new_child).is_some(), "optimizer installed invalid child");
                    let parent_node = query.plan_mut(parent).expect("optimizer edited invalid parent");
                    assert!(
                        replace_child_in_kind(&mut parent_node.kind, old_child, new_child),
                        "optimizer tried to replace non-child {old_child:?} of {parent:?}"
                    );
                    parent_node.props = PlanProperties::default();
                    touched.push(parent);
                    touched.push(new_child);
                }
            }
        }
        touched.sort_unstable_by_key(|id| id.0);
        touched.dedup();
        touched
    }
}

fn replace_child_in_kind(kind: &mut PlanKind, old_child: PlanId, new_child: PlanId) -> bool {
    let replace_one = |slot: &mut PlanId| {
        if *slot == old_child {
            *slot = new_child;
            true
        } else {
            false
        }
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
        | PlanKind::Materialize { input, .. } => replace_one(input),
        PlanKind::Join { left, right, .. } | PlanKind::PhysicalJoin { left, right, .. } => {
            replace_one(left) | replace_one(right)
        }
        PlanKind::UnionAll { inputs, .. } => {
            let mut changed = false;
            for input in inputs {
                changed |= replace_one(input);
            }
            changed
        }
        _ => false,
    }
}

/// Node-local plan peephole.
pub trait OptimizationMX: Any {
    fn optimize_node(&mut self, cursor: &mut PlanCursor<'_>, edits: &mut MxEditQueue) -> bool;

    fn maybe_interesting(&self, _tag: Discriminant<PlanKind>) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

pub struct OptimizationPass {
    optimizations: SmallVec<[SmallBox<dyn OptimizationMX, S1>; 8]>,
    by_tag: HashMap<Discriminant<PlanKind>, SmallVec<[usize; 8]>>,
}

impl Default for OptimizationPass {
    fn default() -> Self {
        Self {
            optimizations: SmallVec::new(),
            by_tag: HashMap::default(),
        }
    }
}

impl OptimizationPass {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, optimization: impl OptimizationMX + 'static) -> Self {
        self.optimizations.push(smallbox!(optimization));
        self.by_tag.clear();
        self
    }

    pub fn push(&mut self, optimization: impl OptimizationMX + 'static) {
        self.optimizations.push(smallbox!(optimization));
        self.by_tag.clear();
    }

    fn interested_for(&mut self, tag: Discriminant<PlanKind>) -> SmallVec<[usize; 8]> {
        if let Some(cached) = self.by_tag.get(&tag) {
            return cached.clone();
        }
        let mut out = SmallVec::new();
        for (index, optimization) in self.optimizations.iter().enumerate() {
            if optimization.maybe_interesting(tag) {
                out.push(index);
            }
        }
        self.by_tag.insert(tag, out.clone());
        out
    }

    /// Worklist optimizer. Rewrites revisit the touched node, its parents and children.
    pub fn pass_optimizations(&mut self, query: &mut Query) -> bool {
        let mut changed_any = false;
        let mut queued = vec![false; query.plans.len()];
        let mut live = vec![false; query.plans.len()];
        let mut worklist = VecDeque::new();

        for id in query.live_plans_postorder() {
            ensure_plan_slot(&mut queued, id);
            ensure_plan_slot(&mut live, id);
            queued[id.0 as usize] = true;
            live[id.0 as usize] = true;
            worklist.push_back(id);
        }

        let mut edits = MxEditQueue::new();
        while let Some(id) = worklist.pop_front() {
            ensure_plan_slot(&mut queued, id);
            queued[id.0 as usize] = false;
            if query.plan(id).is_none() || !live.get(id.0 as usize).copied().unwrap_or(false) {
                continue;
            }

            let mut cursor = PlanCursor::new(query, id);
            let interested = self.interested_for(cursor.tag());
            for optimization_index in interested {
                if self.optimizations[optimization_index].optimize_node(&mut cursor, &mut edits) {
                    break;
                }
            }

            if edits.is_empty() {
                continue;
            }
            changed_any = true;
            let old_parents = query.parents();
            let touched = edits.apply(query);
            let new_parents = query.parents();

            if queued.len() < query.plans.len() {
                queued.resize(query.plans.len(), false);
            }
            live.clear();
            live.resize(query.plans.len(), false);
            for live_id in query.live_plans_preorder() {
                live[live_id.0 as usize] = true;
            }
            for touched_id in touched {
                enqueue(touched_id, &mut queued, &mut worklist);
                if let Some(node) = query.plan(touched_id) {
                    for child in node.kind.children() {
                        enqueue(child, &mut queued, &mut worklist);
                    }
                }
                if let Some(parents) = old_parents.get(&touched_id) {
                    for parent in parents {
                        enqueue(*parent, &mut queued, &mut worklist);
                    }
                }
                if let Some(parents) = new_parents.get(&touched_id) {
                    for parent in parents {
                        enqueue(*parent, &mut queued, &mut worklist);
                    }
                }
            }
        }

        changed_any
    }
}

fn ensure_plan_slot(bits: &mut Vec<bool>, id: PlanId) {
    let needed = id.0 as usize + 1;
    if bits.len() < needed {
        bits.resize(needed, false);
    }
}

fn enqueue(id: PlanId, queued: &mut Vec<bool>, worklist: &mut VecDeque<PlanId>) {
    ensure_plan_slot(queued, id);
    if !queued[id.0 as usize] {
        queued[id.0 as usize] = true;
        worklist.push_back(id);
    }
}

/// Whole-query optimization stage.
pub trait QueryOptimization {
    fn optimize_query(&mut self, query: &mut Query, catalog: &Catalog) -> bool;

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// Whole-program finalization stage for cross-query/catalog work.
pub trait ProgramOptimization {
    fn optimize_program(&mut self, program: &mut Program) -> bool;

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

enum OptimizationStage {
    Nodes(OptimizationPass),
    Query(SmallBox<dyn QueryOptimization, S1>),
}

impl OptimizationStage {
    fn name(&self) -> &'static str {
        match self {
            Self::Nodes(_) => "napalm_mx_optimizer::OptimizationPass",
            Self::Query(pass) => pass.name(),
        }
    }
}

pub struct Optimize {
    stages: SmallVec<[OptimizationStage; 8]>,
    program_finalizers: SmallVec<[SmallBox<dyn ProgramOptimization, S1>; 2]>,
    verify_after_each_pass: bool,
}

impl Default for Optimize {
    fn default() -> Self {
        Self {
            stages: SmallVec::new(),
            program_finalizers: SmallVec::new(),
            verify_after_each_pass: cfg!(debug_assertions),
        }
    }
}

impl Optimize {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_pass(mut self, pass: OptimizationPass) -> Self {
        self.stages.push(OptimizationStage::Nodes(pass));
        self
    }

    pub fn with_query_pass(mut self, pass: impl QueryOptimization + 'static) -> Self {
        self.stages.push(OptimizationStage::Query(smallbox!(pass)));
        self
    }

    pub fn with_program_finalizer(mut self, pass: impl ProgramOptimization + 'static) -> Self {
        self.program_finalizers.push(smallbox!(pass));
        self
    }

    pub fn verify_after_each_pass(mut self, enabled: bool) -> Self {
        self.verify_after_each_pass = enabled;
        self
    }

    /// Optimize one query and report whether any stage changed it.
    ///
    /// The query is always verified before the first transform and after the final
    /// transform. In debug builds (or when explicitly enabled), it is also verified
    /// after every stage and errors are prefixed with that stage's concrete type.
    pub fn optimize_query(
        &mut self,
        query: &mut Query,
        catalog: &Catalog,
    ) -> Result<bool, Vec<String>> {
        verify_query(query, catalog).map_err(|errors| format_verify_errors("initial", errors))?;

        let mut changed_any = false;
        for stage in &mut self.stages {
            let stage_name = stage.name();
            let changed = match stage {
                OptimizationStage::Nodes(pass) => pass.pass_optimizations(query),
                OptimizationStage::Query(pass) => pass.optimize_query(query, catalog),
            };
            changed_any |= changed;

            if self.verify_after_each_pass {
                verify_query(query, catalog)
                    .map_err(|errors| format_verify_errors(stage_name, errors))?;
            }
        }

        verify_query(query, catalog).map_err(|errors| format_verify_errors("final", errors))?;
        Ok(changed_any)
    }

    /// Optimize every query and then run whole-program finalizers.
    pub fn optimize_program(&mut self, program: &mut Program) -> Result<bool, Vec<String>> {
        verify_program(program).map_err(|errors| format_verify_errors("initial", errors))?;

        let mut changed_any = false;
        {
            let catalog = &program.catalog;
            for query in &mut program.queries {
                changed_any |= self.optimize_query(query, catalog)?;
            }
        }

        changed_any |= self.optimize_program_finalizers(program)?;

        verify_program(program).map_err(|errors| format_verify_errors("final", errors))?;
        Ok(changed_any)
    }

    /// Run only whole-program finalizers. This is used after parallel query-local
    /// optimization so cross-query/catalog passes still execute exactly once.
    pub fn optimize_program_finalizers(
        &mut self,
        program: &mut Program,
    ) -> Result<bool, Vec<String>> {
        let mut changed_any = false;
        for finalizer in &mut self.program_finalizers {
            let name = finalizer.name();
            changed_any |= finalizer.optimize_program(program);
            if self.verify_after_each_pass {
                verify_program(program).map_err(|errors| format_verify_errors(name, errors))?;
            }
        }
        Ok(changed_any)
    }
}

fn format_verify_errors(
    stage: &str,
    errors: Vec<crate::napalm_mx_ir::VerifyError>,
) -> Vec<String> {
    errors
        .into_iter()
        .map(|error| format!("{stage}: {error:?}"))
        .collect()
}

/// Stable facade used by the compiler/runtime and by optimizer integration tests.
///
/// Pass composition remains in `napalm_mx_ir_optimizing`; this type deliberately
/// owns only orchestration, verification and the public entry points.
#[derive(Default, Clone, Copy)]
pub struct MxOptimizer;

impl MxOptimizer {
    pub fn optimize(&self, query: &mut Query, catalog: &Catalog) -> Result<(), Vec<String>> {
        crate::napalm_mx_ir_optimizing::optimization_passes()
            .optimize_query(query, catalog)
            .map(|_| ())
    }

    pub fn optimize_aggressive(
        &self,
        query: &mut Query,
        catalog: &Catalog,
    ) -> Result<(), Vec<String>> {
        crate::napalm_mx_ir_optimizing::aggressive_optimization_passes()
            .optimize_query(query, catalog)
            .map(|_| ())
    }

    pub fn optimize_program(&self, program: &mut Program) -> Result<(), Vec<String>> {
        crate::napalm_mx_ir_optimizing::optimization_passes()
            .optimize_program(program)
            .map(|_| ())
    }

    /// Optimize independent queries concurrently. The catalog is immutable and
    /// shared; each worker owns its own optimizer pipeline and one exclusive query.
    pub fn optimize_program_parallel(
        &self,
        program: &mut Program,
        worker_count: usize,
    ) -> Result<(), Vec<String>> {
        verify_program(program).map_err(|errors| format_verify_errors("initial", errors))?;

        let query_count = program.queries.len();
        if query_count <= 1 {
            return self.optimize_program(program);
        }

        let available = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1);
        let workers = if worker_count == 0 {
            available.min(query_count)
        } else {
            worker_count.max(1).min(query_count)
        };
        if workers == 1 {
            return self.optimize_program(program);
        }

        let catalog = &program.catalog;
        let errors = parking_lot::Mutex::new(Vec::<String>::new());
        let (tx, rx) = std::sync::mpsc::channel::<&mut Query>();
        let rx = parking_lot::Mutex::new(rx);

        std::thread::scope(|scope| {
            for _ in 0..workers {
                let rx = &rx;
                let errors = &errors;
                scope.spawn(move || {
                    let mut optimizer = crate::napalm_mx_ir_optimizing::optimization_passes();
                    loop {
                        let query = {
                            let receiver = rx.lock();
                            receiver.recv()
                        };
                        let Ok(query) = query else {
                            break;
                        };
                        if let Err(mut query_errors) = optimizer.optimize_query(query, catalog) {
                            errors.lock().append(&mut query_errors);
                        }
                    }
                });
            }

            for query in &mut program.queries {
                tx.send(query)
                    .expect("all MX optimizer workers stopped unexpectedly");
            }
            drop(tx);
        });

        let collected = errors.into_inner();
        if !collected.is_empty() {
            return Err(collected);
        }

        // Query-local stages ran in workers. Run any whole-program finalizers once
        // after all exclusive query borrows have been joined.
        let mut finalizer_pipeline = crate::napalm_mx_ir_optimizing::optimization_passes();
        let _ = finalizer_pipeline.optimize_program_finalizers(program)?;

        verify_program(program).map_err(|errors| format_verify_errors("final", errors))?;
        Ok(())
    }
}

/// Append-only plan factory for optimizer-created nodes.
pub struct PlanFactory<'q> {
    query: &'q mut Query,
}

impl<'q> PlanFactory<'q> {
    pub fn new(query: &'q mut Query) -> Self {
        Self { query }
    }

    pub fn create(&mut self, kind: PlanKind) -> PlanId {
        self.query.add_plan(kind)
    }
}

/// Append-only expression factory for optimizer-created expression nodes.
pub struct ExprFactory<'q> {
    query: &'q mut Query,
}

impl<'q> ExprFactory<'q> {
    pub fn new(query: &'q mut Query) -> Self {
        Self { query }
    }

    pub fn create(&mut self, kind: ExprKind, ty: ScalarType, nullable: bool) -> ExprId {
        self.query.add_expr(kind, ty, nullable)
    }
}

/// Replace all expression references in live plans and expression DAGs.
pub fn replace_expr_uses(query: &mut Query, old: ExprId, new: ExprId) -> bool {
    if old == new {
        return false;
    }
    let mut changed = false;

    for node in &mut query.exprs {
        changed |= replace_expr_in_kind(&mut node.kind, old, new);
    }
    for id in query.live_plans_preorder() {
        if let Some(node) = query.plan_mut(id) {
            let node_changed = replace_expr_in_plan(&mut node.kind, old, new);
            changed |= node_changed;
            if node_changed {
                node.props = PlanProperties::default();
            }
        }
    }
    changed
}

fn replace_expr_in_kind(kind: &mut ExprKind, old: ExprId, new: ExprId) -> bool {
    let replace = |slot: &mut ExprId| {
        if *slot == old {
            *slot = new;
            true
        } else {
            false
        }
    };
    match kind {
        ExprKind::Unary { value, .. } | ExprKind::IsNull { value, .. } => replace(value),
        ExprKind::Binary { left, right, .. } => replace(left) | replace(right),
        ExprKind::Between { value, low, high, .. } => replace(value) | replace(low) | replace(high),
        ExprKind::InList { value, items, .. } => {
            let mut changed = replace(value);
            for item in items {
                changed |= replace(item);
            }
            changed
        }
        ExprKind::Call { args, .. } => {
            let mut changed = false;
            for arg in args {
                changed |= replace(arg);
            }
            changed
        }
        ExprKind::Const(_) | ExprKind::Param(_) | ExprKind::Column(_) => false,
    }
}

fn replace_expr_in_plan(kind: &mut PlanKind, old: ExprId, new: ExprId) -> bool {
    let replace = |slot: &mut ExprId| {
        if *slot == old {
            *slot = new;
            true
        } else {
            false
        }
    };
    let mut changed = false;
    match kind {
        PlanKind::Values { rows, .. } => {
            for row in rows {
                for expr in row {
                    changed |= replace(expr);
                }
            }
        }
        PlanKind::Filter { predicate, .. } => changed |= replace(predicate),
        PlanKind::Project { expressions, .. } => {
            for projection in expressions {
                changed |= replace(&mut projection.expr);
            }
        }
        PlanKind::Sort { keys, .. } => {
            for key in keys {
                changed |= replace(&mut key.expr);
            }
        }
        PlanKind::Join { on, .. } | PlanKind::PhysicalJoin { on, .. } => changed |= replace(on),
        PlanKind::Aggregate { group_by, aggregates, .. } => {
            for projection in group_by {
                changed |= replace(&mut projection.expr);
            }
            for aggregate in aggregates {
                if let Some(argument) = &mut aggregate.argument {
                    changed |= replace(argument);
                }
            }
        }
        PlanKind::Insert { rows, .. } | PlanKind::WriteBatch { rows, .. } => {
            for row in rows {
                for (_, expr) in row {
                    changed |= replace(expr);
                }
            }
        }
        PlanKind::Update { assignments, .. } => {
            for (_, expr) in assignments {
                changed |= replace(expr);
            }
        }
        PlanKind::PointGet { key, residual, .. } => {
            for expr in key {
                changed |= replace(expr);
            }
            if let Some(predicate) = residual {
                changed |= replace(predicate);
            }
        }
        PlanKind::PrimaryRangeScan { range, residual, .. }
        | PlanKind::IndexRangeScan { range, residual, .. } => {
            for expr in &mut range.prefix {
                changed |= replace(expr);
            }
            match &mut range.lower {
                crate::napalm_mx_ir::BoundValue::Inclusive(expr)
                | crate::napalm_mx_ir::BoundValue::Exclusive(expr) => changed |= replace(expr),
                crate::napalm_mx_ir::BoundValue::Unbounded => {}
            }
            match &mut range.upper {
                crate::napalm_mx_ir::BoundValue::Inclusive(expr)
                | crate::napalm_mx_ir::BoundValue::Exclusive(expr) => changed |= replace(expr),
                crate::napalm_mx_ir::BoundValue::Unbounded => {}
            }
            if let Some(predicate) = residual {
                changed |= replace(predicate);
            }
        }
        PlanKind::RangeCount { range, .. } => {
            for expr in &mut range.prefix {
                changed |= replace(expr);
            }
            match &mut range.lower {
                crate::napalm_mx_ir::BoundValue::Inclusive(expr)
                | crate::napalm_mx_ir::BoundValue::Exclusive(expr) => changed |= replace(expr),
                crate::napalm_mx_ir::BoundValue::Unbounded => {}
            }
            match &mut range.upper {
                crate::napalm_mx_ir::BoundValue::Inclusive(expr)
                | crate::napalm_mx_ir::BoundValue::Exclusive(expr) => changed |= replace(expr),
                crate::napalm_mx_ir::BoundValue::Unbounded => {}
            }
        }
        _ => {}
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::napalm_mx_ir::{PlanKind, QueryId, TransactionMode};

    fn empty_query() -> Query {
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
        query.root = query.add_plan(PlanKind::Empty { columns: vec![] });
        query
    }

    #[test]
    fn edit_queue_replaces_root() {
        let mut query = empty_query();
        let a = query.root;
        let b = query.add_plan(PlanKind::Empty { columns: vec![] });
        let mut edits = MxEditQueue::new();
        edits.replace_all_uses(&query, a, b);
        edits.apply(&mut query);
        assert_eq!(query.root, b);
    }

    #[test]
    fn custom_pass_reports_invalid_ir_with_stage_name() {
        struct BrokenPass;

        impl QueryOptimization for BrokenPass {
            fn optimize_query(&mut self, query: &mut Query, _catalog: &Catalog) -> bool {
                query.root = PlanId(u32::MAX);
                true
            }
        }

        let catalog = Catalog::default();
        let mut query = empty_query();
        let errors = Optimize::new()
            .with_query_pass(BrokenPass)
            .verify_after_each_pass(true)
            .optimize_query(&mut query, &catalog)
            .unwrap_err();

        assert!(errors.iter().any(|error| {
            error.contains("BrokenPass") && error.contains("InvalidRoot")
        }));
    }

    #[test]
    fn mx_optimizer_facade_runs_default_pipeline() {
        let catalog = Catalog::default();
        let mut query = empty_query();
        MxOptimizer::default()
            .optimize(&mut query, &catalog)
            .expect("default MX optimizer rejected valid empty query");
        assert_eq!(verify_query(&query, &catalog), Ok(()));
    }

    #[test]
    fn optimizer_verifies_initial_ir_before_running_passes() {
        struct MustNotRun;

        impl QueryOptimization for MustNotRun {
            fn optimize_query(&mut self, _query: &mut Query, _catalog: &Catalog) -> bool {
                panic!("optimizer pass ran on invalid input IR");
            }
        }

        let catalog = Catalog::default();
        let mut query = empty_query();
        query.root = PlanId(u32::MAX);
        let errors = Optimize::new()
            .with_query_pass(MustNotRun)
            .optimize_query(&mut query, &catalog)
            .unwrap_err();

        assert!(errors.iter().all(|error| error.starts_with("initial:")));
    }
}
