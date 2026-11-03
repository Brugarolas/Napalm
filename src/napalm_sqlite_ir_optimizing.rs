//! Planned optimization surface for Napalm's typed SQLite IR.
//!
//! This file intentionally defines the optimizer API and the planned passes but
//! does not implement any transformation yet. SQLite remains responsible for its
//! own physical query planning (index selection, join algorithms, B-tree access,
//! etc.). These passes are for Napalm-specific compile-time normalization,
//! specialization and cheaper runtime materialization.

use crate::napalm_sqlite_ir::SqliteIr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteOptimizationStage {
    /// Canonical syntax-independent representation and expression cleanup.
    Normalize,
    /// Semantics-preserving logical SQL rewrites.
    Logical,
    /// DML/schema-specific simplification.
    DataModification,
    /// Typed SQLite <-> Napalm row/binding specialization.
    Materialization,
    /// Stable SQL/prepared-statement generation preparation.
    Lowering,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteOptimizationRisk {
    /// Local rewrite with no dependence on schema constraints or database state.
    Local,
    /// Requires compile-time schema facts such as unique keys or nullability.
    SchemaProven,
    /// Requires whole-plan reasoning but remains independent of runtime statistics.
    Global,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteOptimizationKind {
    ConstantFoldExpressions,
    SimplifyThreeValuedLogic,
    CanonicalizeComparisons,
    FlattenBooleanTrees,
    DeduplicateParameters,
    CanonicalizeParameters,
    PredicatePushdown,
    ProjectionPruning,
    RedundantProjectionElimination,
    SubqueryFlattening,
    EmptyQueryPropagation,
    LimitOffsetSimplification,
    DistinctElimination,
    RedundantOrderByElimination,
    CompoundQuerySimplification,
    PredicateInference,
    RedundantJoinElimination,
    DmlPredicateSimplification,
    InsertBatchFormation,
    UpsertSimplification,
    ReturningProjectionPruning,
    ResultShapeValidation,
    RowMaterializerSpecialization,
    ColumnCodecSpecialization,
    ShapeSlotPrebinding,
    ParameterCodecSpecialization,
    PlanCanonicalization,
    PreparedStatementHoisting,
    StaticSqlRendering,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SqliteOptimizationSpec {
    pub kind: SqliteOptimizationKind,
    pub stage: SqliteOptimizationStage,
    pub risk: SqliteOptimizationRisk,
    pub description: &'static str,
}

/// Future executable optimizer contract. Pass implementations should return true
/// when they mutate the IR. No concrete pass implements this trait yet.
pub trait SqliteOptimization {
    fn optimize(&mut self, ir: &mut SqliteIr) -> bool;

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    fn stage(&self) -> SqliteOptimizationStage;
}

/// Marker types reserve stable pass names without pretending the algorithms are
/// already implemented.
pub mod planned {
    #[derive(Default)]
    pub struct ConstantFoldExpressions;
    #[derive(Default)]
    pub struct SimplifyThreeValuedLogic;
    #[derive(Default)]
    pub struct CanonicalizeComparisons;
    #[derive(Default)]
    pub struct FlattenBooleanTrees;
    #[derive(Default)]
    pub struct DeduplicateParameters;
    #[derive(Default)]
    pub struct CanonicalizeParameters;
    #[derive(Default)]
    pub struct PredicatePushdown;
    #[derive(Default)]
    pub struct ProjectionPruning;
    #[derive(Default)]
    pub struct RedundantProjectionElimination;
    #[derive(Default)]
    pub struct SubqueryFlattening;
    #[derive(Default)]
    pub struct EmptyQueryPropagation;
    #[derive(Default)]
    pub struct LimitOffsetSimplification;
    #[derive(Default)]
    pub struct DistinctElimination;
    #[derive(Default)]
    pub struct RedundantOrderByElimination;
    #[derive(Default)]
    pub struct CompoundQuerySimplification;
    #[derive(Default)]
    pub struct PredicateInference;
    #[derive(Default)]
    pub struct RedundantJoinElimination;
    #[derive(Default)]
    pub struct DmlPredicateSimplification;
    #[derive(Default)]
    pub struct InsertBatchFormation;
    #[derive(Default)]
    pub struct UpsertSimplification;
    #[derive(Default)]
    pub struct ReturningProjectionPruning;
    #[derive(Default)]
    pub struct ResultShapeValidation;
    #[derive(Default)]
    pub struct RowMaterializerSpecialization;
    #[derive(Default)]
    pub struct ColumnCodecSpecialization;
    #[derive(Default)]
    pub struct ShapeSlotPrebinding;
    #[derive(Default)]
    pub struct ParameterCodecSpecialization;
    #[derive(Default)]
    pub struct PlanCanonicalization;
    #[derive(Default)]
    pub struct PreparedStatementHoisting;
    #[derive(Default)]
    pub struct StaticSqlRendering;
}

pub const PLANNED_SQLITE_OPTIMIZATIONS: &[SqliteOptimizationSpec] = &[
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::ConstantFoldExpressions,
        stage: SqliteOptimizationStage::Normalize,
        risk: SqliteOptimizationRisk::Local,
        description: "Fold deterministic literal-only SQL expressions while preserving SQLite coercion rules.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::SimplifyThreeValuedLogic,
        stage: SqliteOptimizationStage::Normalize,
        risk: SqliteOptimizationRisk::Local,
        description: "Simplify AND/OR/NOT/IS NULL using SQLite three-valued logic rather than Napalm boolean semantics.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::CanonicalizeComparisons,
        stage: SqliteOptimizationStage::Normalize,
        risk: SqliteOptimizationRisk::Local,
        description: "Canonicalize commutative comparisons and constant placement to improve plan reuse.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::FlattenBooleanTrees,
        stage: SqliteOptimizationStage::Normalize,
        risk: SqliteOptimizationRisk::Local,
        description: "Flatten nested conjunction/disjunction trees into a canonical order when reordering is safe.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::DeduplicateParameters,
        stage: SqliteOptimizationStage::Normalize,
        risk: SqliteOptimizationRisk::Global,
        description: "Reuse one bound parameter for provably identical host values and codecs.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::CanonicalizeParameters,
        stage: SqliteOptimizationStage::Normalize,
        risk: SqliteOptimizationRisk::Global,
        description: "Renumber parameters deterministically so equivalent queries render identical SQL.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::PredicatePushdown,
        stage: SqliteOptimizationStage::Logical,
        risk: SqliteOptimizationRisk::Global,
        description: "Push predicates into eligible subqueries/branches without crossing outer-join or aggregate semantic barriers.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::ProjectionPruning,
        stage: SqliteOptimizationStage::Logical,
        risk: SqliteOptimizationRisk::Global,
        description: "Remove unused intermediate/subquery columns while preserving every field required by the final typed materializer.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::RedundantProjectionElimination,
        stage: SqliteOptimizationStage::Logical,
        risk: SqliteOptimizationRisk::Local,
        description: "Eliminate identity projections and alias-only layers that do not change visible column identity.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::SubqueryFlattening,
        stage: SqliteOptimizationStage::Logical,
        risk: SqliteOptimizationRisk::Global,
        description: "Flatten simple SELECT subqueries when DISTINCT, LIMIT, aggregation and outer-join rules permit it.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::EmptyQueryPropagation,
        stage: SqliteOptimizationStage::Logical,
        risk: SqliteOptimizationRisk::Local,
        description: "Propagate statically empty VALUES/IN/false-predicate branches through compound queries.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::LimitOffsetSimplification,
        stage: SqliteOptimizationStage::Logical,
        risk: SqliteOptimizationRisk::Local,
        description: "Fold constant LIMIT/OFFSET forms and remove no-op offsets/limits.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::DistinctElimination,
        stage: SqliteOptimizationStage::Logical,
        risk: SqliteOptimizationRisk::SchemaProven,
        description: "Remove DISTINCT only when uniqueness follows from projected keys or a proven one-row source.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::RedundantOrderByElimination,
        stage: SqliteOptimizationStage::Logical,
        risk: SqliteOptimizationRisk::Global,
        description: "Drop ORDER BY clauses whose ordering is not observable by the parent query and is not tied to LIMIT/window semantics.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::CompoundQuerySimplification,
        stage: SqliteOptimizationStage::Logical,
        risk: SqliteOptimizationRisk::Global,
        description: "Simplify UNION/INTERSECT/EXCEPT against empty or identical proven branches.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::PredicateInference,
        stage: SqliteOptimizationStage::Logical,
        risk: SqliteOptimizationRisk::SchemaProven,
        description: "Infer safe equality/nullability predicates from schema constraints and existing conjunctions.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::RedundantJoinElimination,
        stage: SqliteOptimizationStage::Logical,
        risk: SqliteOptimizationRisk::SchemaProven,
        description: "Eliminate joins only when foreign-key/unique-key facts prove they cannot change row multiplicity or nullability.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::DmlPredicateSimplification,
        stage: SqliteOptimizationStage::DataModification,
        risk: SqliteOptimizationRisk::Local,
        description: "Simplify UPDATE/DELETE predicates and reject impossible writes before SQL generation.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::InsertBatchFormation,
        stage: SqliteOptimizationStage::DataModification,
        risk: SqliteOptimizationRisk::Global,
        description: "Combine compatible adjacent INSERT rows into one multi-row VALUES statement within SQLite variable limits.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::UpsertSimplification,
        stage: SqliteOptimizationStage::DataModification,
        risk: SqliteOptimizationRisk::SchemaProven,
        description: "Normalize conflict targets and remove unreachable UPSERT arms using unique-key metadata.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::ReturningProjectionPruning,
        stage: SqliteOptimizationStage::DataModification,
        risk: SqliteOptimizationRisk::Global,
        description: "Prune RETURNING expressions not consumed by the typed result materializer.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::ResultShapeValidation,
        stage: SqliteOptimizationStage::Materialization,
        risk: SqliteOptimizationRisk::SchemaProven,
        description: "Prove that every result column can be decoded into its interface property and that required fields are present exactly once.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::RowMaterializerSpecialization,
        stage: SqliteOptimizationStage::Materialization,
        risk: SqliteOptimizationRisk::SchemaProven,
        description: "Build the final column-ordinal -> PropertySlotId materializer for direct shaped-object allocation.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::ColumnCodecSpecialization,
        stage: SqliteOptimizationStage::Materialization,
        risk: SqliteOptimizationRisk::SchemaProven,
        description: "Replace dynamic SQLite value decoding with fixed Number/Bool/String/Blob codecs wherever the query/schema proves the type.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::ShapeSlotPrebinding,
        stage: SqliteOptimizationStage::Materialization,
        risk: SqliteOptimizationRisk::SchemaProven,
        description: "Pre-bind interface field names to structural property slots so row materialization performs no hash lookup.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::ParameterCodecSpecialization,
        stage: SqliteOptimizationStage::Materialization,
        risk: SqliteOptimizationRisk::SchemaProven,
        description: "Preselect rusqlite binding conversions from the Napalm parameter type instead of branching per execution.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::PlanCanonicalization,
        stage: SqliteOptimizationStage::Lowering,
        risk: SqliteOptimizationRisk::Global,
        description: "Produce a deterministic canonical plan identity suitable for SQL text and prepared-statement cache reuse.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::PreparedStatementHoisting,
        stage: SqliteOptimizationStage::Lowering,
        risk: SqliteOptimizationRisk::Global,
        description: "Mark static plans for connection-local prepare_cached reuse rather than preparing on every execution.",
    },
    SqliteOptimizationSpec {
        kind: SqliteOptimizationKind::StaticSqlRendering,
        stage: SqliteOptimizationStage::Lowering,
        risk: SqliteOptimizationRisk::Global,
        description: "Render immutable SQL text once after canonicalization; runtime execution only supplies bound parameters.",
    },
];

/// Passes intentionally delegated to SQLite's own physical planner.
pub const DELEGATED_TO_SQLITE: &[&str] = &[
    "index selection",
    "B-tree access strategy",
    "physical join algorithm",
    "physical join order when SQLite has better runtime statistics",
    "automatic indexes",
    "covering-index selection",
    "cost estimation",
];
