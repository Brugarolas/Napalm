//! Napalm MX IR — typed logical/physical query IR targeting SurrealMX.
//!
//! SurrealMX is an ordered in-memory KV engine rather than a SQL engine. This IR
//! therefore models relational/document semantics above the storage layer and is
//! progressively lowered to point lookups, ordered range scans, key-only scans,
//! zero-copy visitor scans, counts and transactional writes.
//!
//! Core design rules:
//! - `ExprId`, `PlanId`, `TableId`, `IndexId`, and related IDs are dense and stable.
//! - Expression and plan arenas are append-only; optimizers may detach old nodes.
//! - Logical and physical operators share one plan arena so lowering is incremental.
//! - Predicates remain typed expression DAGs until physical access-path selection.
//! - Nullable boolean expressions use `Const(Null)` as SQL-style UNKNOWN; filters
//!   retain only TRUE, so optimizer folds must preserve three-valued logic.
//! - Storage keys are never represented as formatted strings in the optimizer.
//! - Index order is explicit and can satisfy filtering, ordering and projection.
//! - A verifier checks every live plan after each optimizer stage in debug builds.

#![allow(dead_code)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::too_many_arguments)]

use crate::napalm_collections::{HashMap, HashSet};
use std::fmt;

macro_rules! id_newtype {
    ($name:ident, $repr:ty) => {
        #[repr(transparent)]
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        pub struct $name(pub $repr);

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}#{}", stringify!($name), self.0)
            }
        }
    };
}

id_newtype!(StringId, u32);
id_newtype!(BytesId, u32);
id_newtype!(TableId, u32);
/// Stable physical field identity in the storage catalog.
id_newtype!(FieldId, u32);
/// Query-local SSA-like column identity. Two scans of the same physical table get
/// distinct columns, which keeps self-joins and derived projections unambiguous.
id_newtype!(ColumnId, u32);
id_newtype!(IndexId, u32);
id_newtype!(QueryId, u32);
id_newtype!(ExprId, u32);
id_newtype!(PlanId, u32);
id_newtype!(ParamId, u16);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarType {
    Null,
    Bool,
    /// Napalm-visible Number. The runtime may specialize its representation.
    Number,
    String,
    Bytes,
    Any,
}

impl ScalarType {
    #[inline]
    pub const fn is_orderable(self) -> bool {
        matches!(self, Self::Bool | Self::Number | Self::String | Self::Bytes)
    }
}

/// Stable bitwise representation for a Napalm Number constant.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NumberBits(u64);

impl NumberBits {
    #[inline]
    pub fn from_f64(value: f64) -> Self {
        // Canonicalize every NaN so expression hashing/CSE stays deterministic.
        if value.is_nan() {
            Self(f64::NAN.to_bits())
        } else if value == 0.0 {
            // Napalm equality does not need -0 and +0 to produce distinct hash keys.
            Self(0.0f64.to_bits())
        } else {
            Self(value.to_bits())
        }
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        f64::from_bits(self.0)
    }
}

impl fmt::Debug for NumberBits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_f64().fmt(f)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScalarValue {
    Null,
    Bool(bool),
    Number(NumberBits),
    String(StringId),
    Bytes(BytesId),
}

impl ScalarValue {
    #[inline]
    pub const fn ty(&self) -> ScalarType {
        match self {
            Self::Null => ScalarType::Null,
            Self::Bool(_) => ScalarType::Bool,
            Self::Number(_) => ScalarType::Number,
            Self::String(_) => ScalarType::String,
            Self::Bytes(_) => ScalarType::Bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Not,
    Neg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

impl BinaryOp {
    #[inline]
    pub const fn is_comparison(self) -> bool {
        matches!(self, Self::Eq | Self::Ne | Self::Lt | Self::Le | Self::Gt | Self::Ge)
    }

    #[inline]
    pub const fn is_commutative(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::Eq | Self::Ne | Self::And | Self::Or)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinFn {
    Abs,
    Floor,
    Ceil,
    Round,
    LowerAscii,
    UpperAscii,
    StringLength,
    BytesLength,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExprKind {
    Const(ScalarValue),
    Param(ParamId),
    Column(ColumnId),
    Unary {
        op: UnaryOp,
        value: ExprId,
    },
    Binary {
        op: BinaryOp,
        left: ExprId,
        right: ExprId,
    },
    IsNull {
        value: ExprId,
        negated: bool,
    },
    Between {
        value: ExprId,
        low: ExprId,
        high: ExprId,
        negated: bool,
    },
    InList {
        value: ExprId,
        items: Vec<ExprId>,
        negated: bool,
    },
    Call {
        function: BuiltinFn,
        args: Vec<ExprId>,
    },
}

impl ExprKind {
    pub fn operands(&self) -> Vec<ExprId> {
        match self {
            Self::Const(_) | Self::Param(_) | Self::Column(_) => Vec::new(),
            Self::Unary { value, .. } | Self::IsNull { value, .. } => vec![*value],
            Self::Binary { left, right, .. } => vec![*left, *right],
            Self::Between { value, low, high, .. } => vec![*value, *low, *high],
            Self::InList { value, items, .. } => {
                let mut out = Vec::with_capacity(items.len() + 1);
                out.push(*value);
                out.extend(items.iter().copied());
                out
            }
            Self::Call { args, .. } => args.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExprNode {
    pub id: ExprId,
    pub kind: ExprKind,
    pub ty: ScalarType,
    pub nullable: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyEncoding {
    Bool,
    NumberOrdered,
    StringEscaped,
    BytesEscaped,
}

impl KeyEncoding {
    #[inline]
    pub const fn supports(self, ty: ScalarType) -> bool {
        matches!(
            (self, ty),
            (Self::Bool, ScalarType::Bool)
                | (Self::NumberOrdered, ScalarType::Number)
                | (Self::StringEscaped, ScalarType::String)
                | (Self::BytesEscaped, ScalarType::Bytes)
        )
    }

    #[inline]
    pub const fn for_type(ty: ScalarType) -> Option<Self> {
        match ty {
            ScalarType::Bool => Some(Self::Bool),
            ScalarType::Number => Some(Self::NumberOrdered),
            ScalarType::String => Some(Self::StringEscaped),
            ScalarType::Bytes => Some(Self::BytesEscaped),
            ScalarType::Null | ScalarType::Any => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FieldDef {
    pub id: FieldId,
    pub name: StringId,
    pub ty: ScalarType,
    pub nullable: bool,
    /// Stable ordinal in the encoded row/document representation.
    pub ordinal: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColumnOrigin {
    /// A query-local view of one physical storage field. Multiple columns may point
    /// at the same `(table, field)` pair when the table is scanned more than once.
    Storage { table: TableId, field: FieldId },
    /// Produced by a projection, aggregate, VALUES row, EXISTS, or another derived
    /// relational operator.
    Derived,
}

#[derive(Clone, Debug)]
pub struct ColumnDef {
    pub id: ColumnId,
    pub name: Option<StringId>,
    pub ty: ScalarType,
    pub nullable: bool,
    pub origin: ColumnOrigin,
}

#[derive(Clone, Debug)]
pub struct IndexColumn {
    pub field: FieldId,
    pub encoding: KeyEncoding,
    pub descending: bool,
    /// Null tag ordering encoded into the index key.
    pub nulls: NullOrder,
}

/// Napalm-managed secondary index metadata. SurrealMX itself exposes ordered KV
/// primitives rather than a catalog/index DDL. The backend owns a deterministic
/// keyspace convention: non-unique entries normally append the primary key to the
/// encoded secondary key, while unique entries map the secondary key to the primary
/// key. `included_fields` live in the index value for covering scans.
#[derive(Clone, Debug)]
pub struct IndexDef {
    pub id: IndexId,
    pub table: TableId,
    pub name: StringId,
    /// Dedicated binary namespace prefix for index entries.
    pub key_prefix: Vec<u8>,
    pub columns: Vec<IndexColumn>,
    /// Fields copied into the index value. Primary key fields are implicitly available.
    pub included_fields: Vec<FieldId>,
    pub unique: bool,
}

impl IndexDef {
    pub fn covers(&self, fields: &[FieldId], table: &TableDef) -> bool {
        fields.iter().all(|field| {
            table.primary_key.contains(field)
                || self.columns.iter().any(|column| column.field == *field)
                || self.included_fields.contains(field)
        })
    }
}

#[derive(Clone, Debug)]
pub struct TableStats {
    pub rows: u64,
    pub average_row_bytes: u32,
    /// Per-field approximate distinct count; absent means unknown.
    pub distinct: HashMap<FieldId, u64>,
}

impl Default for TableStats {
    fn default() -> Self {
        Self {
            rows: 0,
            average_row_bytes: 64,
            distinct: HashMap::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TableDef {
    pub id: TableId,
    pub name: StringId,
    /// Namespace prefix for primary row keys.
    pub key_prefix: Vec<u8>,
    pub fields: Vec<FieldDef>,
    pub primary_key: Vec<FieldId>,
    pub indexes: Vec<IndexId>,
    pub stats: TableStats,
}

impl TableDef {
    pub fn field(&self, field: FieldId) -> Option<&FieldDef> {
        self.fields.iter().find(|candidate| candidate.id == field)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Catalog {
    pub tables: Vec<TableDef>,
    pub indexes: Vec<IndexDef>,
}

impl Catalog {
    pub fn table(&self, id: TableId) -> Option<&TableDef> {
        self.tables.get(id.0 as usize).filter(|table| table.id == id)
    }

    pub fn index(&self, id: IndexId) -> Option<&IndexDef> {
        self.indexes.get(id.0 as usize).filter(|index| index.id == id)
    }

    pub fn indexes_for(&self, table: TableId) -> impl Iterator<Item = &IndexDef> {
        self.indexes.iter().filter(move |index| index.table == table)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SortDirection {
    Asc,
    Desc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NullOrder {
    First,
    Last,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SortKey {
    pub expr: ExprId,
    pub direction: SortDirection,
    pub nulls: NullOrder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JoinKind {
    Inner,
    Left,
    Semi,
    Anti,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AggregateFn {
    Count,
    Sum,
    Min,
    Max,
    Avg,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AggregateExpr {
    pub column: ColumnId,
    pub function: AggregateFn,
    pub argument: Option<ExprId>,
    pub distinct: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProjectionExpr {
    pub column: ColumnId,
    pub expr: ExprId,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoundValue {
    Unbounded,
    Inclusive(ExprId),
    Exclusive(ExprId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyRange {
    /// Equality-constrained leading key components.
    pub prefix: Vec<ExprId>,
    /// Optional range on the immediately following component.
    pub lower: BoundValue,
    pub upper: BoundValue,
}

impl Default for KeyRange {
    fn default() -> Self {
        Self {
            prefix: Vec::new(),
            lower: BoundValue::Unbounded,
            upper: BoundValue::Unbounded,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScanDirection {
    Forward,
    Reverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScanDelivery {
    /// Materialize `(key,value)` pairs.
    Pairs,
    /// Materialize only keys.
    Keys,
    /// Use `scan_with`/equivalent and inspect borrowed bytes in a visitor.
    BorrowedPairs,
    /// Use `keys_for_each`/equivalent and inspect borrowed keys in a visitor.
    BorrowedKeys,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointDelivery {
    Owned,
    /// Prefer SurrealMX `with_value` on direct non-transactional point reads.
    Borrowed,
    ExistsOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MutationMode {
    /// Insert only; fail when the key already exists (`put`).
    Insert,
    /// Insert or replace (`set`).
    Set,
    /// Update existing logical row; the executor may need a read-before-write.
    Update,
    Delete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JoinAlgorithm {
    Hash,
    Merge,
    IndexNestedLoop(IndexId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Materialization {
    Streaming,
    Buffered,
}

#[derive(Clone, Debug)]
pub enum PlanKind {
    // ----- Logical operators -----
    Empty {
        /// Empty relation with a preserved output schema.
        columns: Vec<ColumnId>,
    },
    Values {
        columns: Vec<ColumnId>,
        rows: Vec<Vec<ExprId>>,
    },
    TableScan {
        table: TableId,
        /// Query-local output columns backed by physical fields of `table`. Column
        /// pruning may shrink this to an empty key-only scan.
        columns: Vec<ColumnId>,
    },
    Filter {
        input: PlanId,
        predicate: ExprId,
    },
    Project {
        input: PlanId,
        expressions: Vec<ProjectionExpr>,
    },
    Sort {
        input: PlanId,
        keys: Vec<SortKey>,
    },
    Limit {
        input: PlanId,
        skip: u64,
        limit: Option<u64>,
    },
    Distinct {
        input: PlanId,
        columns: Vec<ColumnId>,
    },
    Join {
        left: PlanId,
        right: PlanId,
        kind: JoinKind,
        on: ExprId,
    },
    Aggregate {
        input: PlanId,
        group_by: Vec<ProjectionExpr>,
        aggregates: Vec<AggregateExpr>,
    },
    UnionAll {
        inputs: Vec<PlanId>,
        /// Positional output columns owned by the union node. Child column IDs may
        /// differ; the executor/optimizer maps by ordinal.
        columns: Vec<ColumnId>,
    },
    Exists {
        input: PlanId,
        output: ColumnId,
    },
    Insert {
        table: TableId,
        rows: Vec<Vec<(FieldId, ExprId)>>,
    },
    Update {
        input: PlanId,
        table: TableId,
        assignments: Vec<(FieldId, ExprId)>,
    },
    Delete {
        input: PlanId,
        table: TableId,
    },

    // ----- Physical SurrealMX-oriented access operators -----
    PointGet {
        table: TableId,
        key: Vec<ExprId>,
        columns: Vec<ColumnId>,
        /// Predicate terms not proven by the primary-key lookup.
        residual: Option<ExprId>,
        delivery: PointDelivery,
    },
    PrimaryRangeScan {
        table: TableId,
        range: KeyRange,
        columns: Vec<ColumnId>,
        /// Predicate terms not represented by the selected key range.
        residual: Option<ExprId>,
        direction: ScanDirection,
        skip: u64,
        limit: Option<u64>,
        delivery: ScanDelivery,
    },
    IndexRangeScan {
        table: TableId,
        index: IndexId,
        range: KeyRange,
        columns: Vec<ColumnId>,
        /// Predicate terms not represented by the selected index range.
        residual: Option<ExprId>,
        direction: ScanDirection,
        skip: u64,
        limit: Option<u64>,
        delivery: ScanDelivery,
        /// True when every requested field can be decoded from the index entry.
        covering: bool,
    },
    FetchRows {
        /// Key-producing child (normally an index scan).
        input: PlanId,
        table: TableId,
        /// Final query-local columns decoded from the primary rows.
        columns: Vec<ColumnId>,
    },
    RangeCount {
        output: ColumnId,
        table: TableId,
        index: Option<IndexId>,
        range: KeyRange,
        skip: u64,
        limit: Option<u64>,
    },
    PhysicalJoin {
        left: PlanId,
        right: PlanId,
        kind: JoinKind,
        on: ExprId,
        algorithm: JoinAlgorithm,
    },
    WriteBatch {
        table: TableId,
        mode: MutationMode,
        rows: Vec<Vec<(FieldId, ExprId)>>,
    },
    Materialize {
        input: PlanId,
        mode: Materialization,
    },
}

impl PlanKind {
    pub fn children(&self) -> Vec<PlanId> {
        match self {
            Self::Filter { input, .. }
            | Self::Project { input, .. }
            | Self::Sort { input, .. }
            | Self::Limit { input, .. }
            | Self::Distinct { input, .. }
            | Self::Aggregate { input, .. }
            | Self::Exists { input, .. }
            | Self::Update { input, .. }
            | Self::Delete { input, .. }
            | Self::FetchRows { input, .. }
            | Self::Materialize { input, .. } => vec![*input],
            Self::Join { left, right, .. } | Self::PhysicalJoin { left, right, .. } => {
                vec![*left, *right]
            }
            Self::UnionAll { inputs, .. } => inputs.clone(),
            Self::Empty { .. }
            | Self::Values { .. }
            | Self::TableScan { .. }
            | Self::Insert { .. }
            | Self::PointGet { .. }
            | Self::PrimaryRangeScan { .. }
            | Self::IndexRangeScan { .. }
            | Self::RangeCount { .. }
            | Self::WriteBatch { .. } => Vec::new(),
        }
    }

    pub fn expressions(&self) -> Vec<ExprId> {
        match self {
            Self::Filter { predicate, .. } => vec![*predicate],
            Self::Project { expressions, .. } => expressions.iter().map(|p| p.expr).collect(),
            Self::Sort { keys, .. } => keys.iter().map(|key| key.expr).collect(),
            Self::Join { on, .. } | Self::PhysicalJoin { on, .. } => vec![*on],
            Self::Aggregate { group_by, aggregates, .. } => {
                let mut out: Vec<_> = group_by.iter().map(|projection| projection.expr).collect();
                out.extend(aggregates.iter().filter_map(|agg| agg.argument));
                out
            }
            Self::Values { rows, .. } => rows.iter().flatten().copied().collect(),
            Self::Insert { rows, .. } | Self::WriteBatch { rows, .. } => rows
                .iter()
                .flat_map(|row| row.iter().map(|(_, expr)| *expr))
                .collect(),
            Self::Update { assignments, .. } => assignments.iter().map(|(_, expr)| *expr).collect(),
            Self::PointGet { key, residual, .. } => {
                let mut out = key.clone();
                if let Some(predicate) = residual { out.push(*predicate); }
                out
            }
            Self::PrimaryRangeScan { range, residual, .. }
            | Self::IndexRangeScan { range, residual, .. } => {
                let mut out = range_exprs(range);
                if let Some(predicate) = residual { out.push(*predicate); }
                out
            }
            Self::RangeCount { range, .. } => range_exprs(range),
            _ => Vec::new(),
        }
    }

    pub fn is_physical(&self) -> bool {
        matches!(
            self,
            Self::Empty { .. }
                | Self::Values { .. }
                | Self::Filter { .. }
                | Self::Project { .. }
                | Self::Sort { .. }
                | Self::Limit { .. }
                | Self::Distinct { .. }
                | Self::Aggregate { .. }
                | Self::UnionAll { .. }
                | Self::Exists { .. }
                | Self::PointGet { .. }
                | Self::PrimaryRangeScan { .. }
                | Self::IndexRangeScan { .. }
                | Self::FetchRows { .. }
                | Self::RangeCount { .. }
                | Self::PhysicalJoin { .. }
                | Self::WriteBatch { .. }
                | Self::Materialize { .. }
        )
    }
}

fn range_exprs(range: &KeyRange) -> Vec<ExprId> {
    let mut out = range.prefix.clone();
    if let BoundValue::Inclusive(expr) | BoundValue::Exclusive(expr) = &range.lower {
        out.push(*expr);
    }
    if let BoundValue::Inclusive(expr) | BoundValue::Exclusive(expr) = &range.upper {
        out.push(*expr);
    }
    out
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RowShape {
    pub columns: Vec<ColumnId>,
}

impl RowShape {
    pub fn contains(&self, column: ColumnId) -> bool {
        self.columns.contains(&column)
    }

    pub fn dedup(&mut self) {
        let mut seen = HashSet::default();
        self.columns.retain(|column| seen.insert(*column));
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OrderingProperty {
    pub keys: Vec<(ColumnId, SortDirection, NullOrder)>,
}

#[derive(Clone, Debug, Default)]
pub struct PlanProperties {
    pub row_shape: RowShape,
    pub ordering: OrderingProperty,
    pub unique_on: Vec<Vec<ColumnId>>,
    pub estimated_rows: Option<u64>,
    pub estimated_cost: f64,
}

#[derive(Clone, Debug)]
pub struct PlanNode {
    pub id: PlanId,
    pub kind: PlanKind,
    pub props: PlanProperties,
}

#[derive(Clone, Debug)]
pub struct ParamDef {
    pub id: ParamId,
    pub name: Option<StringId>,
    pub ty: ScalarType,
    pub nullable: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransactionMode {
    /// Eligible for direct `Database` methods when the plan permits it.
    Auto,
    /// One read-only snapshot transaction for the complete query.
    ReadOnlySnapshot,
    /// One writable serializable transaction for the complete query.
    ReadWrite,
}

#[derive(Clone, Debug)]
pub struct Query {
    pub id: QueryId,
    pub name: Option<StringId>,
    pub root: PlanId,
    pub params: Vec<ParamDef>,
    pub columns: Vec<ColumnDef>,
    pub exprs: Vec<ExprNode>,
    pub plans: Vec<PlanNode>,
    pub transaction: TransactionMode,
}

impl Query {
    pub fn column(&self, id: ColumnId) -> Option<&ColumnDef> {
        self.columns.get(id.0 as usize).filter(|column| column.id == id)
    }

    pub fn column_mut(&mut self, id: ColumnId) -> Option<&mut ColumnDef> {
        self.columns.get_mut(id.0 as usize).filter(|column| column.id == id)
    }

    pub fn add_column(
        &mut self,
        name: Option<StringId>,
        ty: ScalarType,
        nullable: bool,
        origin: ColumnOrigin,
    ) -> ColumnId {
        let id = ColumnId(self.columns.len() as u32);
        self.columns.push(ColumnDef { id, name, ty, nullable, origin });
        id
    }

    pub fn storage_origin(&self, column: ColumnId) -> Option<(TableId, FieldId)> {
        match self.column(column)?.origin {
            ColumnOrigin::Storage { table, field } => Some((table, field)),
            ColumnOrigin::Derived => None,
        }
    }

    pub fn expr(&self, id: ExprId) -> Option<&ExprNode> {
        self.exprs.get(id.0 as usize).filter(|expr| expr.id == id)
    }

    pub fn expr_mut(&mut self, id: ExprId) -> Option<&mut ExprNode> {
        self.exprs.get_mut(id.0 as usize).filter(|expr| expr.id == id)
    }

    pub fn plan(&self, id: PlanId) -> Option<&PlanNode> {
        self.plans.get(id.0 as usize).filter(|plan| plan.id == id)
    }

    pub fn plan_mut(&mut self, id: PlanId) -> Option<&mut PlanNode> {
        self.plans.get_mut(id.0 as usize).filter(|plan| plan.id == id)
    }

    pub fn add_expr(&mut self, kind: ExprKind, ty: ScalarType, nullable: bool) -> ExprId {
        let id = ExprId(self.exprs.len() as u32);
        self.exprs.push(ExprNode { id, kind, ty, nullable });
        id
    }

    pub fn add_plan(&mut self, kind: PlanKind) -> PlanId {
        let id = PlanId(self.plans.len() as u32);
        self.plans.push(PlanNode {
            id,
            kind,
            props: PlanProperties::default(),
        });
        id
    }

    /// Plan nodes reachable from the query root in post-order (children first).
    pub fn live_plans_postorder(&self) -> Vec<PlanId> {
        fn visit(query: &Query, id: PlanId, seen: &mut HashSet<PlanId>, out: &mut Vec<PlanId>) {
            if !seen.insert(id) {
                return;
            }
            let Some(node) = query.plan(id) else {
                return;
            };
            for child in node.kind.children() {
                visit(query, child, seen, out);
            }
            out.push(id);
        }

        let mut seen = HashSet::default();
        let mut out = Vec::new();
        visit(self, self.root, &mut seen, &mut out);
        out
    }

    pub fn live_plans_preorder(&self) -> Vec<PlanId> {
        let mut out = self.live_plans_postorder();
        out.reverse();
        out
    }

    pub fn parents(&self) -> HashMap<PlanId, Vec<PlanId>> {
        let mut parents: HashMap<PlanId, Vec<PlanId>> = HashMap::default();
        for parent in self.live_plans_preorder() {
            if let Some(node) = self.plan(parent) {
                for child in node.kind.children() {
                    parents.entry(child).or_default().push(parent);
                }
            }
        }
        parents
    }
}

#[derive(Clone, Debug, Default)]
pub struct Program {
    pub strings: Vec<String>,
    pub bytes: Vec<Vec<u8>>,
    pub catalog: Catalog,
    pub queries: Vec<Query>,
}

impl Program {
    pub fn string(&self, id: StringId) -> Option<&str> {
        self.strings.get(id.0 as usize).map(String::as_str)
    }

    pub fn bytes(&self, id: BytesId) -> Option<&[u8]> {
        self.bytes.get(id.0 as usize).map(Vec::as_slice)
    }

    pub fn query(&self, id: QueryId) -> Option<&Query> {
        self.queries.get(id.0 as usize).filter(|query| query.id == id)
    }

    pub fn query_mut(&mut self, id: QueryId) -> Option<&mut Query> {
        self.queries.get_mut(id.0 as usize).filter(|query| query.id == id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StorageNamespace {
    Table(TableId),
    Index(IndexId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    InvalidRoot(PlanId),
    InvalidPlanId(PlanId),
    InvalidExprId(ExprId),
    InvalidTable(TableId),
    InvalidIndex(IndexId),
    InvalidField(FieldId),
    InvalidColumn(ColumnId),
    ColumnTypeMismatch { column: ColumnId, expected: ScalarType, actual: ScalarType },
    InvalidKeyEncoding { field: FieldId, encoding: KeyEncoding, ty: ScalarType },
    InvalidValueWidth { plan: PlanId, expected: usize, actual: usize },
    InvalidPredicateType { expr: ExprId, actual: ScalarType },
    InvalidSortType { expr: ExprId, actual: ScalarType },
    InvalidColumnNullability { column: ColumnId, expr: ExprId },
    InvalidExpressionNullability(ExprId),
    IndexTableMismatch { index: IndexId, table: TableId },
    CyclicPlan(PlanId),
    CyclicExpr(ExprId),
    InvalidParam(ParamId),
    TypeMismatch { expr: ExprId, expected: ScalarType, actual: ScalarType },
    InvalidOperandType { expr: ExprId, operand: ExprId, expected: ScalarType, actual: ScalarType },
    InvalidPrimaryKeyArity { table: TableId, actual: usize, expected: usize },
    MissingPrimaryKey(TableId),
    EmptyKeyNamespace(StorageNamespace),
    OverlappingKeyNamespace { first: StorageNamespace, second: StorageNamespace },
    NullablePrimaryKey(FieldId),
    InvalidPrimaryKeyType { field: FieldId, ty: ScalarType },
    DuplicatePrimaryKeyField(FieldId),
    InvalidRangePrefix { index: Option<IndexId>, components: usize, max: usize },
    WriteInReadOnlyQuery(PlanId),
    LogicalNodeAfterPhysicalLowering(PlanId),
}

pub fn verify_program(program: &Program) -> Result<(), Vec<VerifyError>> {
    let mut errors = Vec::new();
    verify_catalog(&program.catalog, &mut errors);
    for query in &program.queries {
        if let Err(mut query_errors) = verify_query(query, &program.catalog) {
            errors.append(&mut query_errors);
        }
    }
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

pub fn verify_query(query: &Query, catalog: &Catalog) -> Result<(), Vec<VerifyError>> {
    let mut errors = Vec::new();
    if query.plan(query.root).is_none() {
        errors.push(VerifyError::InvalidRoot(query.root));
        return Err(errors);
    }

    verify_query_arenas(query, catalog, &mut errors);
    verify_plan_graph(query, catalog, &mut errors);
    verify_expr_graph(query, &mut errors);

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

fn verify_catalog(catalog: &Catalog, errors: &mut Vec<VerifyError>) {
    let mut physical_fields = HashSet::default();
    let mut namespaces: Vec<(StorageNamespace, &[u8])> = Vec::new();
    for (table_ordinal, table) in catalog.tables.iter().enumerate() {
        if table.id.0 as usize != table_ordinal {
            errors.push(VerifyError::InvalidTable(table.id));
        }
        if table.key_prefix.is_empty() {
            errors.push(VerifyError::EmptyKeyNamespace(StorageNamespace::Table(table.id)));
        } else {
            namespaces.push((StorageNamespace::Table(table.id), table.key_prefix.as_slice()));
        }
        if table.primary_key.is_empty() {
            errors.push(VerifyError::MissingPrimaryKey(table.id));
        }
        for field in &table.fields {
            // Field IDs are global catalog identities, not per-table ordinals.
            if !physical_fields.insert(field.id) {
                errors.push(VerifyError::InvalidField(field.id));
            }
        }
        let mut primary_fields = HashSet::default();
        for field in &table.primary_key {
            if !primary_fields.insert(*field) {
                errors.push(VerifyError::DuplicatePrimaryKeyField(*field));
            }
            let Some(field_def) = table.field(*field) else {
                errors.push(VerifyError::InvalidField(*field));
                continue;
            };
            if field_def.nullable {
                errors.push(VerifyError::NullablePrimaryKey(*field));
            }
            if KeyEncoding::for_type(field_def.ty).is_none() {
                errors.push(VerifyError::InvalidPrimaryKeyType {
                    field: *field,
                    ty: field_def.ty,
                });
            }
        }
        for index in &table.indexes {
            match catalog.index(*index) {
                Some(index_def) if index_def.table == table.id => {}
                Some(_) => errors.push(VerifyError::IndexTableMismatch { index: *index, table: table.id }),
                None => errors.push(VerifyError::InvalidIndex(*index)),
            }
        }
    }

    for (ordinal, index) in catalog.indexes.iter().enumerate() {
        if index.id.0 as usize != ordinal {
            errors.push(VerifyError::InvalidIndex(index.id));
        }
        let Some(table) = catalog.table(index.table) else {
            errors.push(VerifyError::InvalidTable(index.table));
            continue;
        };
        if index.key_prefix.is_empty() {
            errors.push(VerifyError::EmptyKeyNamespace(StorageNamespace::Index(index.id)));
        } else {
            namespaces.push((StorageNamespace::Index(index.id), index.key_prefix.as_slice()));
        }
        for column in &index.columns {
            let Some(field) = table.field(column.field) else {
                errors.push(VerifyError::InvalidField(column.field));
                continue;
            };
            if !column.encoding.supports(field.ty) {
                errors.push(VerifyError::InvalidKeyEncoding {
                    field: column.field,
                    encoding: column.encoding,
                    ty: field.ty,
                });
            }
        }
        for field in &index.included_fields {
            if table.field(*field).is_none() {
                errors.push(VerifyError::InvalidField(*field));
            }
        }
    }

    // Namespace prefixes must be prefix-disjoint. Otherwise a range over one table
    // or index can accidentally walk entries belonging to another logical object.
    for left in 0..namespaces.len() {
        for right in (left + 1)..namespaces.len() {
            let (left_id, left_prefix) = namespaces[left];
            let (right_id, right_prefix) = namespaces[right];
            if left_prefix.starts_with(right_prefix) || right_prefix.starts_with(left_prefix) {
                errors.push(VerifyError::OverlappingKeyNamespace {
                    first: left_id,
                    second: right_id,
                });
            }
        }
    }
}

fn verify_query_arenas(query: &Query, catalog: &Catalog, errors: &mut Vec<VerifyError>) {
    for (ordinal, param) in query.params.iter().enumerate() {
        if param.id.0 as usize != ordinal {
            errors.push(VerifyError::InvalidParam(param.id));
        }
    }
    for (ordinal, column) in query.columns.iter().enumerate() {
        if column.id.0 as usize != ordinal {
            errors.push(VerifyError::InvalidColumn(column.id));
        }
        if let ColumnOrigin::Storage { table, field } = column.origin {
            let Some(table_def) = catalog.table(table) else {
                errors.push(VerifyError::InvalidTable(table));
                continue;
            };
            let Some(field_def) = table_def.field(field) else {
                errors.push(VerifyError::InvalidField(field));
                continue;
            };
            if column.ty != field_def.ty {
                errors.push(VerifyError::ColumnTypeMismatch {
                    column: column.id,
                    expected: field_def.ty,
                    actual: column.ty,
                });
            }
        }
    }
    for (ordinal, expr) in query.exprs.iter().enumerate() {
        if expr.id.0 as usize != ordinal {
            errors.push(VerifyError::InvalidExprId(expr.id));
        }
    }
    for (ordinal, plan) in query.plans.iter().enumerate() {
        if plan.id.0 as usize != ordinal {
            errors.push(VerifyError::InvalidPlanId(plan.id));
        }
    }
}

fn verify_plan_graph(query: &Query, catalog: &Catalog, errors: &mut Vec<VerifyError>) {
    fn visit(
        query: &Query,
        catalog: &Catalog,
        id: PlanId,
        visiting: &mut HashSet<PlanId>,
        done: &mut HashSet<PlanId>,
        errors: &mut Vec<VerifyError>,
    ) {
        if done.contains(&id) {
            return;
        }
        if !visiting.insert(id) {
            errors.push(VerifyError::CyclicPlan(id));
            return;
        }
        let Some(node) = query.plan(id) else {
            errors.push(VerifyError::InvalidPlanId(id));
            visiting.remove(&id);
            return;
        };

        verify_plan_node(query, catalog, node, errors);
        for child in node.kind.children() {
            visit(query, catalog, child, visiting, done, errors);
        }
        visiting.remove(&id);
        done.insert(id);
    }

    let mut visiting = HashSet::default();
    let mut done = HashSet::default();
    visit(query, catalog, query.root, &mut visiting, &mut done, errors);
}

fn verify_plan_node(query: &Query, catalog: &Catalog, node: &PlanNode, errors: &mut Vec<VerifyError>) {
    for expr in node.kind.expressions() {
        if query.expr(expr).is_none() {
            errors.push(VerifyError::InvalidExprId(expr));
        }
    }

    let check_table = |table: TableId, errors: &mut Vec<VerifyError>| {
        if catalog.table(table).is_none() {
            errors.push(VerifyError::InvalidTable(table));
        }
    };
    let check_column = |column: ColumnId, errors: &mut Vec<VerifyError>| {
        if query.column(column).is_none() {
            errors.push(VerifyError::InvalidColumn(column));
        }
    };
    let check_scan_columns = |table: TableId, columns: &[ColumnId], errors: &mut Vec<VerifyError>| {
        for &column in columns {
            check_column(column, errors);
            if let Some(column_def) = query.column(column) {
                if !matches!(column_def.origin, ColumnOrigin::Storage { table: origin_table, .. } if origin_table == table) {
                    errors.push(VerifyError::InvalidColumn(column));
                }
            }
        }
    };

    match &node.kind {
        PlanKind::TableScan { table, .. }
        | PlanKind::Insert { table, .. }
        | PlanKind::Update { table, .. }
        | PlanKind::Delete { table, .. }
        | PlanKind::PointGet { table, .. }
        | PlanKind::PrimaryRangeScan { table, .. }
        | PlanKind::IndexRangeScan { table, .. }
        | PlanKind::FetchRows { table, .. }
        | PlanKind::RangeCount { table, .. }
        | PlanKind::WriteBatch { table, .. } => check_table(*table, errors),
        _ => {}
    }

    match &node.kind {
        PlanKind::TableScan { table, columns }
        | PlanKind::PointGet { table, columns, .. }
        | PlanKind::PrimaryRangeScan { table, columns, .. }
        | PlanKind::IndexRangeScan { table, columns, .. }
        | PlanKind::FetchRows { table, columns, .. } => {
            check_scan_columns(*table, columns, errors);
        }
        _ => {}
    }

    if query.transaction == TransactionMode::ReadOnlySnapshot
        && matches!(
            &node.kind,
            PlanKind::Insert { .. }
                | PlanKind::Update { .. }
                | PlanKind::Delete { .. }
                | PlanKind::WriteBatch { .. }
        )
    {
        errors.push(VerifyError::WriteInReadOnlyQuery(node.id));
    }

    match &node.kind {
        PlanKind::Values { columns, rows } => {
            for &column in columns { check_column(column, errors); }
            for row in rows {
                if row.len() != columns.len() {
                    errors.push(VerifyError::InvalidValueWidth {
                        plan: node.id,
                        expected: columns.len(),
                        actual: row.len(),
                    });
                }
            }
        }
        PlanKind::Filter { predicate, .. }
        | PlanKind::Join { on: predicate, .. }
        | PlanKind::PhysicalJoin { on: predicate, .. } => {
            if let Some(expr) = query.expr(*predicate) {
                if expr.ty != ScalarType::Bool && expr.ty != ScalarType::Any {
                    errors.push(VerifyError::InvalidPredicateType { expr: *predicate, actual: expr.ty });
                }
            }
        }
        PlanKind::PointGet { residual: Some(predicate), .. }
        | PlanKind::PrimaryRangeScan { residual: Some(predicate), .. }
        | PlanKind::IndexRangeScan { residual: Some(predicate), .. } => {
            if let Some(expr) = query.expr(*predicate) {
                if expr.ty != ScalarType::Bool && expr.ty != ScalarType::Any {
                    errors.push(VerifyError::InvalidPredicateType { expr: *predicate, actual: expr.ty });
                }
            }
        }
        PlanKind::Project { expressions, .. } => {
            for projection in expressions {
                check_column(projection.column, errors);
                let (Some(column), Some(expr)) = (
                    query.column(projection.column),
                    query.expr(projection.expr),
                ) else { continue; };
                if column.ty != expr.ty
                    && column.ty != ScalarType::Any
                    && expr.ty != ScalarType::Any
                    && !(expr.ty == ScalarType::Null && column.nullable)
                {
                    errors.push(VerifyError::TypeMismatch {
                        expr: projection.expr,
                        expected: column.ty,
                        actual: expr.ty,
                    });
                }
                if expr.nullable && !column.nullable {
                    errors.push(VerifyError::InvalidColumnNullability {
                        column: projection.column,
                        expr: projection.expr,
                    });
                }
            }
        }
        PlanKind::Sort { keys, .. } => {
            for key in keys {
                if let Some(expr) = query.expr(key.expr) {
                    if expr.ty != ScalarType::Any && expr.ty != ScalarType::Null && !expr.ty.is_orderable() {
                        errors.push(VerifyError::InvalidSortType { expr: key.expr, actual: expr.ty });
                    }
                }
            }
        }
        PlanKind::Distinct { columns, .. } => {
            for &column in columns { check_column(column, errors); }
        }
        PlanKind::Aggregate { group_by, aggregates, .. } => {
            for projection in group_by { check_column(projection.column, errors); }
            for aggregate in aggregates { check_column(aggregate.column, errors); }
        }
        PlanKind::UnionAll { inputs, columns } => {
            for &column in columns { check_column(column, errors); }
            for &input in inputs {
                if let Some(actual) = verifier_output_width(query, input) {
                    if actual != columns.len() {
                        errors.push(VerifyError::InvalidValueWidth {
                            plan: node.id,
                            expected: columns.len(),
                            actual,
                        });
                    }
                }
            }
        }
        PlanKind::Exists { output, .. } | PlanKind::RangeCount { output, .. } => {
            check_column(*output, errors);
        }
        PlanKind::PointGet { table, key, .. } => {
            if let Some(table_def) = catalog.table(*table) {
                if key.len() != table_def.primary_key.len() {
                    errors.push(VerifyError::InvalidPrimaryKeyArity {
                        table: *table,
                        actual: key.len(),
                        expected: table_def.primary_key.len(),
                    });
                }
            }
        }
        PlanKind::IndexRangeScan { table, index, range, .. } => {
            match catalog.index(*index) {
                None => errors.push(VerifyError::InvalidIndex(*index)),
                Some(index_def) => {
                    if index_def.table != *table {
                        errors.push(VerifyError::IndexTableMismatch { index: *index, table: *table });
                    }
                    if range.prefix.len() > index_def.columns.len() {
                        errors.push(VerifyError::InvalidRangePrefix {
                            index: Some(*index),
                            components: range.prefix.len(),
                            max: index_def.columns.len(),
                        });
                    }
                }
            }
        }
        PlanKind::PrimaryRangeScan { table, range, .. }
        | PlanKind::RangeCount { table, index: None, range, .. } => {
            if let Some(table_def) = catalog.table(*table) {
                if range.prefix.len() > table_def.primary_key.len() {
                    errors.push(VerifyError::InvalidRangePrefix {
                        index: None,
                        components: range.prefix.len(),
                        max: table_def.primary_key.len(),
                    });
                }
            }
        }
        PlanKind::RangeCount { table, index: Some(index), range, .. } => {
            match catalog.index(*index) {
                None => errors.push(VerifyError::InvalidIndex(*index)),
                Some(index_def) => {
                    if index_def.table != *table {
                        errors.push(VerifyError::IndexTableMismatch { index: *index, table: *table });
                    }
                    if range.prefix.len() > index_def.columns.len() {
                        errors.push(VerifyError::InvalidRangePrefix {
                            index: Some(*index),
                            components: range.prefix.len(),
                            max: index_def.columns.len(),
                        });
                    }
                }
            }
        }
        PlanKind::Insert { table, rows } | PlanKind::WriteBatch { table, rows, .. } => {
            if let Some(table_def) = catalog.table(*table) {
                for row in rows {
                    for (field, _) in row {
                        if table_def.field(*field).is_none() { errors.push(VerifyError::InvalidField(*field)); }
                    }
                }
            }
        }
        PlanKind::Update { table, assignments, .. } => {
            if let Some(table_def) = catalog.table(*table) {
                for (field, _) in assignments {
                    if table_def.field(*field).is_none() { errors.push(VerifyError::InvalidField(*field)); }
                }
            }
        }
        _ => {}
    }
}

fn verifier_output_width(query: &Query, id: PlanId) -> Option<usize> {
    fn visit(query: &Query, id: PlanId, visiting: &mut HashSet<PlanId>) -> Option<usize> {
        if !visiting.insert(id) {
            return None;
        }
        let node = query.plan(id)?;
        let width = match &node.kind {
            PlanKind::Empty { columns }
            | PlanKind::TableScan { columns, .. }
            | PlanKind::PointGet { columns, .. }
            | PlanKind::PrimaryRangeScan { columns, .. }
            | PlanKind::IndexRangeScan { columns, .. }
            | PlanKind::FetchRows { columns, .. }
            | PlanKind::Values { columns, .. }
            | PlanKind::UnionAll { columns, .. } => columns.len(),
            PlanKind::Filter { input, .. }
            | PlanKind::Sort { input, .. }
            | PlanKind::Limit { input, .. }
            | PlanKind::Distinct { input, .. }
            | PlanKind::Update { input, .. }
            | PlanKind::Delete { input, .. }
            | PlanKind::Materialize { input, .. } => visit(query, *input, visiting)?,
            PlanKind::Project { expressions, .. } => expressions.len(),
            PlanKind::Join { left, right, kind, .. }
            | PlanKind::PhysicalJoin { left, right, kind, .. } => {
                let left_width = visit(query, *left, visiting)?;
                if matches!(kind, JoinKind::Semi | JoinKind::Anti) {
                    left_width
                } else {
                    left_width.saturating_add(visit(query, *right, visiting)?)
                }
            }
            PlanKind::Aggregate { group_by, aggregates, .. } => group_by.len() + aggregates.len(),
            PlanKind::Exists { .. } | PlanKind::RangeCount { .. } => 1,
            PlanKind::Insert { .. } | PlanKind::WriteBatch { .. } => 0,
        };
        visiting.remove(&id);
        Some(width)
    }

    visit(query, id, &mut HashSet::default())
}

fn verify_expr_graph(query: &Query, errors: &mut Vec<VerifyError>) {
    fn visit(
        query: &Query,
        id: ExprId,
        visiting: &mut HashSet<ExprId>,
        done: &mut HashSet<ExprId>,
        errors: &mut Vec<VerifyError>,
    ) {
        if done.contains(&id) {
            return;
        }
        if !visiting.insert(id) {
            errors.push(VerifyError::CyclicExpr(id));
            return;
        }
        let Some(expr) = query.expr(id) else {
            errors.push(VerifyError::InvalidExprId(id));
            visiting.remove(&id);
            return;
        };
        match &expr.kind {
            ExprKind::Param(param) => {
                if query.params.get(param.0 as usize).is_none_or(|candidate| candidate.id != *param) {
                    errors.push(VerifyError::InvalidParam(*param));
                }
            }
            ExprKind::Column(column) => {
                let Some(column_def) = query.column(*column) else {
                    errors.push(VerifyError::InvalidColumn(*column));
                    visiting.remove(&id);
                    return;
                };
                if expr.ty != column_def.ty {
                    errors.push(VerifyError::TypeMismatch {
                        expr: id,
                        expected: column_def.ty,
                        actual: expr.ty,
                    });
                }
            }
            _ => {}
        }
        verify_expr_semantics(query, expr, errors);
        for operand in expr.kind.operands() {
            visit(query, operand, visiting, done, errors);
        }
        visiting.remove(&id);
        done.insert(id);
    }

    let mut roots = Vec::new();
    for plan in query.live_plans_preorder() {
        if let Some(node) = query.plan(plan) {
            roots.extend(node.kind.expressions());
        }
    }

    let mut visiting = HashSet::default();
    let mut done = HashSet::default();
    for root in roots {
        visit(query, root, &mut visiting, &mut done, errors);
    }
}

fn verify_expr_semantics(query: &Query, expr: &ExprNode, errors: &mut Vec<VerifyError>) {
    let require_result = |expected: ScalarType, errors: &mut Vec<VerifyError>| {
        if expr.ty != expected && expr.ty != ScalarType::Any {
            errors.push(VerifyError::TypeMismatch {
                expr: expr.id,
                expected,
                actual: expr.ty,
            });
        }
    };
    let require_operand = |operand: ExprId, expected: ScalarType, errors: &mut Vec<VerifyError>| {
        let Some(actual) = query.expr(operand).map(|node| node.ty) else { return; };
        if actual != expected && actual != ScalarType::Any && actual != ScalarType::Null {
            errors.push(VerifyError::InvalidOperandType {
                expr: expr.id,
                operand,
                expected,
                actual,
            });
        }
    };
    let compatible = |left: ExprId, right: ExprId, errors: &mut Vec<VerifyError>| {
        let (Some(left_ty), Some(right_ty)) = (
            query.expr(left).map(|node| node.ty),
            query.expr(right).map(|node| node.ty),
        ) else { return; };
        if left_ty != right_ty
            && left_ty != ScalarType::Any
            && right_ty != ScalarType::Any
            && left_ty != ScalarType::Null
            && right_ty != ScalarType::Null
        {
            errors.push(VerifyError::InvalidOperandType {
                expr: expr.id,
                operand: right,
                expected: left_ty,
                actual: right_ty,
            });
        }
    };

    match &expr.kind {
        ExprKind::Const(value) => {
            if matches!(value, ScalarValue::Null) && !expr.nullable {
                errors.push(VerifyError::InvalidExpressionNullability(expr.id));
            }
            if !matches!(value, ScalarValue::Null)
                && expr.ty != value.ty()
                && expr.ty != ScalarType::Any
            {
                errors.push(VerifyError::TypeMismatch {
                    expr: expr.id,
                    expected: value.ty(),
                    actual: expr.ty,
                });
            }
        }
        ExprKind::Param(param) => {
            if let Some(param) = query.params.get(param.0 as usize) {
                if expr.ty != param.ty && expr.ty != ScalarType::Any && param.ty != ScalarType::Any {
                    errors.push(VerifyError::TypeMismatch {
                        expr: expr.id,
                        expected: param.ty,
                        actual: expr.ty,
                    });
                }
            }
        }
        ExprKind::Column(_) => {}
        ExprKind::Unary { op: UnaryOp::Not, value } => {
            require_operand(*value, ScalarType::Bool, errors);
            require_result(ScalarType::Bool, errors);
        }
        ExprKind::Unary { op: UnaryOp::Neg, value } => {
            require_operand(*value, ScalarType::Number, errors);
            require_result(ScalarType::Number, errors);
        }
        ExprKind::Binary { op, left, right } => {
            match op {
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem => {
                    require_operand(*left, ScalarType::Number, errors);
                    require_operand(*right, ScalarType::Number, errors);
                    require_result(ScalarType::Number, errors);
                }
                BinaryOp::And | BinaryOp::Or => {
                    require_operand(*left, ScalarType::Bool, errors);
                    require_operand(*right, ScalarType::Bool, errors);
                    require_result(ScalarType::Bool, errors);
                }
                BinaryOp::Eq | BinaryOp::Ne => {
                    compatible(*left, *right, errors);
                    require_result(ScalarType::Bool, errors);
                }
                BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                    compatible(*left, *right, errors);
                    for operand in [*left, *right] {
                        if let Some(ty) = query.expr(operand).map(|node| node.ty) {
                            if ty != ScalarType::Any && ty != ScalarType::Null && !ty.is_orderable() {
                                errors.push(VerifyError::InvalidOperandType {
                                    expr: expr.id,
                                    operand,
                                    expected: ScalarType::Any,
                                    actual: ty,
                                });
                            }
                        }
                    }
                    require_result(ScalarType::Bool, errors);
                }
            }
        }
        ExprKind::IsNull { .. } => require_result(ScalarType::Bool, errors),
        ExprKind::Between { value, low, high, .. } => {
            compatible(*value, *low, errors);
            compatible(*value, *high, errors);
            require_result(ScalarType::Bool, errors);
        }
        ExprKind::InList { value, items, .. } => {
            for item in items {
                compatible(*value, *item, errors);
            }
            require_result(ScalarType::Bool, errors);
        }
        ExprKind::Call { function, args } => {
            let (argument, result) = match function {
                BuiltinFn::Abs | BuiltinFn::Floor | BuiltinFn::Ceil | BuiltinFn::Round => {
                    (ScalarType::Number, ScalarType::Number)
                }
                BuiltinFn::LowerAscii | BuiltinFn::UpperAscii => {
                    (ScalarType::String, ScalarType::String)
                }
                BuiltinFn::StringLength => (ScalarType::String, ScalarType::Number),
                BuiltinFn::BytesLength => (ScalarType::Bytes, ScalarType::Number),
            };
            if args.len() == 1 {
                require_operand(args[0], argument, errors);
            }
            require_result(result, errors);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_program() -> Program {
        let mut program = Program::default();
        program.strings.push("users".into());
        program.strings.push("id".into());
        program.strings.push("age".into());

        let id = FieldId(0);
        let age = FieldId(1);
        program.catalog.tables.push(TableDef {
            id: TableId(0),
            name: StringId(0),
            key_prefix: b"t/users/".to_vec(),
            fields: vec![
                FieldDef { id, name: StringId(1), ty: ScalarType::Number, nullable: false, ordinal: 0 },
                FieldDef { id: age, name: StringId(2), ty: ScalarType::Number, nullable: false, ordinal: 1 },
            ],
            primary_key: vec![id],
            indexes: vec![],
            stats: TableStats { rows: 100, ..Default::default() },
        });

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
        let id_column = query.add_column(
            Some(StringId(1)),
            ScalarType::Number,
            false,
            ColumnOrigin::Storage { table: TableId(0), field: id },
        );
        let age_column = query.add_column(
            Some(StringId(2)),
            ScalarType::Number,
            false,
            ColumnOrigin::Storage { table: TableId(0), field: age },
        );
        let scan = query.add_plan(PlanKind::TableScan {
            table: TableId(0),
            columns: vec![id_column, age_column],
        });
        query.root = scan;
        program.queries.push(query);
        program
    }

    #[test]
    fn verifies_simple_query() {
        let program = tiny_program();
        assert_eq!(verify_program(&program), Ok(()));
    }
}
