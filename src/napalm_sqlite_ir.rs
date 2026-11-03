//! Typed SQLite IR for Napalm Reloaded.
//!
//! This module deliberately models SQL semantics instead of mirroring rusqlite's
//! API. The parser/lowerer can build this IR, optimization passes can normalize it,
//! and the SQLite backend can finally render SQL and bind values through rusqlite.
//!
//! The most important invariant is typed row materialization: query results carry
//! a canonical Napalm structural layout and a pre-bound mapping from result column
//! ordinals to `PropertySlotId`s. The runtime can therefore allocate the final
//! structural object directly, attach the interface contract once, and populate
//! fixed slots without transient string-keyed Maps.

#![allow(clippy::large_enum_variant)]
#![allow(clippy::too_many_arguments)]

use crate::napalm_ir::{
    PropertySlotId, StringId, StructuralLayoutId, Type, ValueId,
};
use std::fmt;

macro_rules! sqlite_id_newtype {
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

sqlite_id_newtype!(SqliteTableId, u32);
sqlite_id_newtype!(SqliteColumnId, u16);
sqlite_id_newtype!(SqliteIndexId, u32);
sqlite_id_newtype!(SqliteExprId, u32);
sqlite_id_newtype!(SqliteQueryId, u32);
sqlite_id_newtype!(SqliteStatementId, u32);
sqlite_id_newtype!(SqliteParamId, u16);
sqlite_id_newtype!(SqliteRelationId, u16);
sqlite_id_newtype!(SqliteConstraintId, u16);

/// SQLite storage classes as observed at runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteStorageClass {
    Null,
    Integer,
    Real,
    Text,
    Blob,
}

/// SQLite column affinity used when emitting schemas and casts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteAffinity {
    Integer,
    Real,
    Text,
    Blob,
    Numeric,
}

/// Stable f64 representation suitable for Eq/Hash based canonicalization.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SqliteReal(pub u64);

impl SqliteReal {
    #[inline]
    pub fn from_f64(value: f64) -> Self {
        Self(value.to_bits())
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        f64::from_bits(self.0)
    }
}

impl fmt::Debug for SqliteReal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_f64().fmt(f)
    }
}

/// Literal values embedded in a plan. Runtime values use `SqliteParamId` instead.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteLiteral {
    Null,
    Integer(i64),
    Real(SqliteReal),
    Text(StringId),
    Blob(Vec<u8>),
    Bool(bool),
}

/// Logical scalar type used by the SQL expression verifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteScalarType {
    Null,
    Integer,
    Real,
    Text,
    Blob,
    Bool,
    Numeric,
    Dynamic,
}

impl SqliteScalarType {
    #[inline]
    pub fn nullable_join(&self, other: &Self) -> Self {
        if self == other {
            return self.clone();
        }
        match (self, other) {
            (Self::Integer, Self::Real) | (Self::Real, Self::Integer) => Self::Numeric,
            (Self::Null, ty) | (ty, Self::Null) => ty.clone(),
            _ => Self::Dynamic,
        }
    }
}

/// Conversion performed when a SQLite value is materialized into a Napalm value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteDecodeCodec {
    /// SQLite INTEGER/REAL -> Napalm Number.
    Number,
    /// SQLite INTEGER 0/1 -> Napalm Bool.
    Bool,
    /// SQLite TEXT -> Napalm String.
    String,
    /// SQLite BLOB -> backend-defined byte container.
    Blob,
    /// SQLite TEXT/BLOB JSON -> the declared Napalm structural/container type.
    Json(Type),
    /// Runtime codec registered under a compiler-interned name.
    Custom(StringId),
    /// Preserve SQLite's dynamic storage class and let the runtime box it.
    Dynamic,
}

/// Conversion performed when a Napalm value is bound to a SQLite parameter.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteEncodeCodec {
    Number,
    Bool,
    String,
    Blob,
    Json(Type),
    Custom(StringId),
    Dynamic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteSortOrder {
    Asc,
    Desc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteNullsOrder {
    Default,
    First,
    Last,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteConflictAction {
    Rollback,
    Abort,
    Fail,
    Ignore,
    Replace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteForeignKeyAction {
    NoAction,
    Restrict,
    SetNull,
    SetDefault,
    Cascade,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteForeignKeyMatch {
    Simple,
    Full,
    Partial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteGeneratedKind {
    Virtual,
    Stored,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteTransactionMode {
    Deferred,
    Immediate,
    Exclusive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteJoinKind {
    Inner,
    Left,
    Cross,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteCompoundKind {
    Union,
    Intersect,
    Except,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteUnaryOp {
    Positive,
    Negative,
    BitNot,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    ShiftLeft,
    ShiftRight,
    Concat,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteBuiltinFunction {
    Abs,
    Coalesce,
    IfNull,
    NullIf,
    Lower,
    Upper,
    Length,
    Trim,
    LTrim,
    RTrim,
    Replace,
    Substr,
    Round,
    Min,
    Max,
    Sum,
    Avg,
    Count,
    Total,
    GroupConcat,
    Json,
    JsonExtract,
    Date,
    Time,
    DateTime,
    UnixEpoch,
    Strftime,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteFunctionRef {
    Builtin(SqliteBuiltinFunction),
    Named(StringId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteDatabaseName {
    Main,
    Temp,
    Attached(StringId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqlitePrimaryKey {
    pub order: SqliteSortOrder,
    pub conflict: Option<SqliteConflictAction>,
    pub autoincrement: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteForeignKey {
    pub table: SqliteTableId,
    pub column: SqliteColumnId,
    pub on_delete: SqliteForeignKeyAction,
    pub on_update: SqliteForeignKeyAction,
    pub match_kind: SqliteForeignKeyMatch,
    pub deferrable: bool,
    pub initially_deferred: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteColumnDefault {
    Literal(SqliteLiteral),
    CurrentTime,
    CurrentDate,
    CurrentTimestamp,
    Expression(SqliteExprId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteGeneratedColumn {
    pub expression: SqliteExprId,
    pub kind: SqliteGeneratedKind,
}

/// One table column derived from a typed Napalm interface property.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteColumn {
    pub id: SqliteColumnId,
    pub name: StringId,
    /// Property slot used by typed row materialization.
    pub property_slot: PropertySlotId,
    pub napalm_type: Type,
    pub scalar_type: SqliteScalarType,
    pub affinity: SqliteAffinity,
    pub encode: SqliteEncodeCodec,
    pub decode: SqliteDecodeCodec,
    pub nullable: bool,
    pub primary_key: Option<SqlitePrimaryKey>,
    pub unique: bool,
    pub unique_conflict: Option<SqliteConflictAction>,
    pub default: Option<SqliteColumnDefault>,
    pub check: Option<SqliteExprId>,
    pub references: Option<SqliteForeignKey>,
    pub collate: Option<StringId>,
    pub generated: Option<SqliteGeneratedColumn>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteIndexedColumn {
    pub column: SqliteColumnId,
    pub collate: Option<StringId>,
    pub order: SqliteSortOrder,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteTableConstraintKind {
    PrimaryKey {
        columns: Vec<SqliteIndexedColumn>,
        conflict: Option<SqliteConflictAction>,
    },
    Unique {
        columns: Vec<SqliteIndexedColumn>,
        conflict: Option<SqliteConflictAction>,
    },
    Check(SqliteExprId),
    ForeignKey {
        columns: Vec<SqliteColumnId>,
        foreign_table: SqliteTableId,
        foreign_columns: Vec<SqliteColumnId>,
        on_delete: SqliteForeignKeyAction,
        on_update: SqliteForeignKeyAction,
        match_kind: SqliteForeignKeyMatch,
        deferrable: bool,
        initially_deferred: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteTableConstraint {
    pub id: SqliteConstraintId,
    pub name: Option<StringId>,
    pub kind: SqliteTableConstraintKind,
}

/// Compile-time table schema. `row_type` is normally `Type::Interface` and
/// `layout` is the canonical `StructuralLayoutId` already owned by Napalm WIR.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteTable {
    pub id: SqliteTableId,
    pub database: SqliteDatabaseName,
    pub name: StringId,
    pub row_type: Type,
    pub layout: StructuralLayoutId,
    pub columns: Vec<SqliteColumn>,
    pub constraints: Vec<SqliteTableConstraint>,
    pub strict: bool,
    pub without_rowid: bool,
}

impl SqliteTable {
    #[inline]
    pub fn column(&self, id: SqliteColumnId) -> Option<&SqliteColumn> {
        self.columns
            .get(id.0 as usize)
            .filter(|column| column.id == id)
    }

    #[inline]
    pub fn column_by_property_slot(&self, slot: PropertySlotId) -> Option<&SqliteColumn> {
        self.columns.iter().find(|column| column.property_slot == slot)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteIndex {
    pub id: SqliteIndexId,
    pub database: SqliteDatabaseName,
    pub name: StringId,
    pub table: SqliteTableId,
    pub unique: bool,
    pub columns: Vec<SqliteIndexedColumn>,
    pub predicate: Option<SqliteExprId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteColumnSelector {
    TableColumn(SqliteColumnId),
    /// Column ordinal exposed by a subquery relation.
    QueryOutput(u16),
    RowId,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteColumnRef {
    pub relation: SqliteRelationId,
    pub column: SqliteColumnSelector,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteWhenThen {
    pub when: SqliteExprId,
    pub then: SqliteExprId,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteExprKind {
    Literal(SqliteLiteral),
    Parameter(SqliteParamId),
    Column(SqliteColumnRef),
    Unary {
        op: SqliteUnaryOp,
        value: SqliteExprId,
    },
    Binary {
        op: SqliteBinaryOp,
        left: SqliteExprId,
        right: SqliteExprId,
    },
    Function {
        function: SqliteFunctionRef,
        args: Vec<SqliteExprId>,
        distinct: bool,
        filter: Option<SqliteExprId>,
    },
    Cast {
        value: SqliteExprId,
        affinity: SqliteAffinity,
    },
    Case {
        operand: Option<SqliteExprId>,
        arms: Vec<SqliteWhenThen>,
        else_value: Option<SqliteExprId>,
    },
    Between {
        value: SqliteExprId,
        low: SqliteExprId,
        high: SqliteExprId,
        negated: bool,
    },
    InList {
        value: SqliteExprId,
        values: Vec<SqliteExprId>,
        negated: bool,
    },
    InQuery {
        value: SqliteExprId,
        query: SqliteQueryId,
        negated: bool,
    },
    IsNull {
        value: SqliteExprId,
        negated: bool,
    },
    Is {
        left: SqliteExprId,
        right: SqliteExprId,
        negated: bool,
    },
    Like {
        value: SqliteExprId,
        pattern: SqliteExprId,
        escape: Option<SqliteExprId>,
        negated: bool,
    },
    Glob {
        value: SqliteExprId,
        pattern: SqliteExprId,
        negated: bool,
    },
    Collate {
        value: SqliteExprId,
        collation: StringId,
    },
    Exists(SqliteQueryId),
    ScalarSubquery(SqliteQueryId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteExpr {
    pub id: SqliteExprId,
    pub ty: SqliteScalarType,
    pub nullable: bool,
    /// False for volatile/custom functions unless the frontend can prove otherwise.
    pub deterministic: bool,
    pub kind: SqliteExprKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteOrderKey {
    pub expression: SqliteExprId,
    pub order: SqliteSortOrder,
    pub nulls: SqliteNullsOrder,
    pub collate: Option<StringId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteProjection {
    pub expression: SqliteExprId,
    pub alias: Option<StringId>,
    pub ty: SqliteScalarType,
    pub nullable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteFromSource {
    Table {
        table: SqliteTableId,
        relation: SqliteRelationId,
        alias: Option<StringId>,
    },
    Subquery {
        query: SqliteQueryId,
        relation: SqliteRelationId,
        alias: Option<StringId>,
    },
}

impl SqliteFromSource {
    #[inline]
    pub fn relation(&self) -> SqliteRelationId {
        match self {
            Self::Table { relation, .. } | Self::Subquery { relation, .. } => *relation,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteJoin {
    pub kind: SqliteJoinKind,
    pub source: SqliteFromSource,
    pub on: Option<SqliteExprId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteSelect {
    pub distinct: bool,
    pub projection: Vec<SqliteProjection>,
    pub from: Option<SqliteFromSource>,
    pub joins: Vec<SqliteJoin>,
    pub predicate: Option<SqliteExprId>,
    pub group_by: Vec<SqliteExprId>,
    pub having: Option<SqliteExprId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteValues {
    pub rows: Vec<Vec<SqliteExprId>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteCompound {
    pub kind: SqliteCompoundKind,
    pub all: bool,
    pub left: SqliteQueryId,
    pub right: SqliteQueryId,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteQueryBody {
    Select(SqliteSelect),
    Values(SqliteValues),
    Compound(SqliteCompound),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteQuery {
    pub id: SqliteQueryId,
    pub body: SqliteQueryBody,
    pub order_by: Vec<SqliteOrderKey>,
    pub limit: Option<SqliteExprId>,
    pub offset: Option<SqliteExprId>,
    /// Filled by validation/lowering and kept stable through later optimization.
    pub output: Vec<SqliteProjection>,
}

/// One result column is decoded directly into a pre-bound structural property slot.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteMaterializedField {
    pub result_column: u16,
    pub property_slot: PropertySlotId,
    pub property_name: StringId,
    pub napalm_type: Type,
    pub decode: SqliteDecodeCodec,
    pub nullable: bool,
}

/// Typed result object layout for a query/RETURNING clause.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteRowMaterializer {
    pub row_type: Type,
    pub layout: StructuralLayoutId,
    pub fields: Vec<SqliteMaterializedField>,
}

impl SqliteRowMaterializer {
    #[inline]
    pub fn field_for_result_column(&self, index: u16) -> Option<&SqliteMaterializedField> {
        self.fields.iter().find(|field| field.result_column == index)
    }

    #[inline]
    pub fn field_for_property_slot(&self, slot: PropertySlotId) -> Option<&SqliteMaterializedField> {
        self.fields.iter().find(|field| field.property_slot == slot)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteResultShape {
    None,
    AffectedRows,
    LastInsertRowId,
    Scalar {
        napalm_type: Type,
        decode: SqliteDecodeCodec,
        nullable: bool,
    },
    Rows(SqliteRowMaterializer),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteReturning {
    pub projection: Vec<SqliteProjection>,
    pub result: SqliteResultShape,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteAssignment {
    pub column: SqliteColumnId,
    pub value: SqliteExprId,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteInsertSource {
    DefaultValues,
    Values(Vec<Vec<SqliteExprId>>),
    Query(SqliteQueryId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteUpsertTarget {
    Any,
    Columns {
        columns: Vec<SqliteIndexedColumn>,
        predicate: Option<SqliteExprId>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteUpsertAction {
    DoNothing,
    DoUpdate {
        assignments: Vec<SqliteAssignment>,
        predicate: Option<SqliteExprId>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteUpsert {
    pub target: SqliteUpsertTarget,
    pub action: SqliteUpsertAction,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteInsert {
    pub table: SqliteTableId,
    pub columns: Vec<SqliteColumnId>,
    pub source: SqliteInsertSource,
    pub or_conflict: Option<SqliteConflictAction>,
    pub upsert: Vec<SqliteUpsert>,
    pub returning: Option<SqliteReturning>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteUpdate {
    pub table: SqliteTableId,
    pub assignments: Vec<SqliteAssignment>,
    pub predicate: Option<SqliteExprId>,
    pub order_by: Vec<SqliteOrderKey>,
    pub limit: Option<SqliteExprId>,
    pub returning: Option<SqliteReturning>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteDelete {
    pub table: SqliteTableId,
    pub predicate: Option<SqliteExprId>,
    pub order_by: Vec<SqliteOrderKey>,
    pub limit: Option<SqliteExprId>,
    pub returning: Option<SqliteReturning>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteAlterTable {
    RenameTable {
        table: SqliteTableId,
        new_name: StringId,
    },
    RenameColumn {
        table: SqliteTableId,
        column: SqliteColumnId,
        new_name: StringId,
    },
    AddColumn {
        table: SqliteTableId,
        column: SqliteColumn,
    },
    DropColumn {
        table: SqliteTableId,
        column: SqliteColumnId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqlitePragmaValue {
    Integer(i64),
    Text(StringId),
    Bool(bool),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteRawStatement {
    pub sql: StringId,
    pub result: SqliteResultShape,
}

/// Schema, DML, query and transaction operations accepted by the SQLite backend.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteStatementKind {
    Query {
        query: SqliteQueryId,
        result: SqliteResultShape,
    },
    CreateTable {
        table: SqliteTableId,
        if_not_exists: bool,
    },
    CreateTables {
        tables: Vec<SqliteTableId>,
        if_not_exists: bool,
    },
    DropTable {
        table: SqliteTableId,
        if_exists: bool,
    },
    DropTables {
        tables: Vec<SqliteTableId>,
        if_exists: bool,
    },
    CreateIndex {
        index: SqliteIndexId,
        if_not_exists: bool,
    },
    DropIndex {
        index: SqliteIndexId,
        if_exists: bool,
    },
    AlterTable(SqliteAlterTable),
    Insert(SqliteInsert),
    Update(SqliteUpdate),
    Delete(SqliteDelete),
    Begin {
        mode: SqliteTransactionMode,
    },
    Commit,
    Rollback,
    Savepoint {
        name: StringId,
    },
    ReleaseSavepoint {
        name: StringId,
    },
    RollbackToSavepoint {
        name: StringId,
    },
    Attach {
        path: SqliteExprId,
        schema: StringId,
    },
    Detach {
        schema: StringId,
    },
    Pragma {
        database: SqliteDatabaseName,
        name: StringId,
        value: Option<SqlitePragmaValue>,
    },
    Vacuum {
        database: Option<SqliteDatabaseName>,
    },
    Analyze {
        database: Option<SqliteDatabaseName>,
        target: Option<StringId>,
    },
    Raw(SqliteRawStatement),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteStatement {
    pub id: SqliteStatementId,
    pub kind: SqliteStatementKind,
}

/// Compile-time parameter declaration. The runtime binding is a WIR `ValueId`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteParameter {
    pub id: SqliteParamId,
    pub napalm_type: Type,
    pub scalar_type: SqliteScalarType,
    pub encode: SqliteEncodeCodec,
    pub nullable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteHostBinding {
    pub parameter: SqliteParamId,
    pub value: ValueId,
}

/// Bridge object used by the later Napalm lowerer/WIR integration.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteInvocation {
    pub statement: SqliteStatementId,
    pub bindings: Vec<SqliteHostBinding>,
    pub result: SqliteResultShape,
}

/// Logical connection configuration. The runtime backend maps this to
/// `rusqlite::Connection` and configures the statement cache.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SqliteOpenSource {
    Memory,
    File(StringId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteJournalMode {
    Delete,
    Truncate,
    Persist,
    Memory,
    Wal,
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SqliteSynchronous {
    Off,
    Normal,
    Full,
    Extra,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteConnectionOptions {
    pub read_only: bool,
    pub create: bool,
    pub uri: bool,
    pub foreign_keys: bool,
    pub busy_timeout_ms: Option<u64>,
    pub statement_cache_capacity: usize,
    pub journal_mode: Option<SqliteJournalMode>,
    pub synchronous: Option<SqliteSynchronous>,
}

impl Default for SqliteConnectionOptions {
    fn default() -> Self {
        Self {
            read_only: false,
            create: true,
            uri: false,
            foreign_keys: true,
            busy_timeout_ms: Some(5_000),
            statement_cache_capacity: 128,
            journal_mode: None,
            synchronous: Some(SqliteSynchronous::Normal),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SqliteConnectionPlan {
    pub source: SqliteOpenSource,
    pub options: SqliteConnectionOptions,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SqliteIrError {
    InvalidTableId(SqliteTableId),
    InvalidIndexId(SqliteIndexId),
    InvalidExprId(SqliteExprId),
    InvalidQueryId(SqliteQueryId),
    InvalidStatementId(SqliteStatementId),
    InvalidColumnId {
        table: SqliteTableId,
        column: SqliteColumnId,
    },
    DuplicateRelation(SqliteRelationId),
    DuplicateParameter(SqliteParamId),
    DuplicateMaterializedColumn(u16),
    DuplicateMaterializedProperty(PropertySlotId),
    InvalidMaterializedColumn {
        column: u16,
        output_len: usize,
    },
    InvalidMaterializedLayout,
    InvalidCreateTablesEntry(SqliteTableId),
    InvalidDropTablesEntry(SqliteTableId),
}

/// Arena-backed SQLite program owned by a compiled Napalm module/program.
///
/// IDs are stable and index directly into their arenas, matching Napalm WIR's
/// append-only ID style.
#[derive(Clone, Debug, Default)]
pub struct SqliteIr {
    pub tables: Vec<SqliteTable>,
    pub indexes: Vec<SqliteIndex>,
    pub expressions: Vec<SqliteExpr>,
    pub queries: Vec<SqliteQuery>,
    pub statements: Vec<SqliteStatement>,
    pub parameters: Vec<SqliteParameter>,
}

impl SqliteIr {
    #[inline]
    pub fn table(&self, id: SqliteTableId) -> Option<&SqliteTable> {
        self.tables.get(id.0 as usize).filter(|table| table.id == id)
    }

    #[inline]
    pub fn index(&self, id: SqliteIndexId) -> Option<&SqliteIndex> {
        self.indexes.get(id.0 as usize).filter(|index| index.id == id)
    }

    #[inline]
    pub fn expression(&self, id: SqliteExprId) -> Option<&SqliteExpr> {
        self.expressions
            .get(id.0 as usize)
            .filter(|expression| expression.id == id)
    }

    #[inline]
    pub fn query(&self, id: SqliteQueryId) -> Option<&SqliteQuery> {
        self.queries.get(id.0 as usize).filter(|query| query.id == id)
    }

    #[inline]
    pub fn statement(&self, id: SqliteStatementId) -> Option<&SqliteStatement> {
        self.statements
            .get(id.0 as usize)
            .filter(|statement| statement.id == id)
    }

    #[inline]
    pub fn parameter(&self, id: SqliteParamId) -> Option<&SqliteParameter> {
        self.parameters
            .get(id.0 as usize)
            .filter(|parameter| parameter.id == id)
    }

    pub fn push_table(&mut self, mut table: SqliteTable) -> SqliteTableId {
        let id = SqliteTableId(self.tables.len() as u32);
        table.id = id;
        for (index, column) in table.columns.iter_mut().enumerate() {
            column.id = SqliteColumnId(index as u16);
        }
        for (index, constraint) in table.constraints.iter_mut().enumerate() {
            constraint.id = SqliteConstraintId(index as u16);
        }
        self.tables.push(table);
        id
    }

    pub fn push_index(&mut self, mut index: SqliteIndex) -> SqliteIndexId {
        let id = SqliteIndexId(self.indexes.len() as u32);
        index.id = id;
        self.indexes.push(index);
        id
    }

    pub fn push_expression(&mut self, mut expression: SqliteExpr) -> SqliteExprId {
        let id = SqliteExprId(self.expressions.len() as u32);
        expression.id = id;
        self.expressions.push(expression);
        id
    }

    pub fn push_query(&mut self, mut query: SqliteQuery) -> SqliteQueryId {
        let id = SqliteQueryId(self.queries.len() as u32);
        query.id = id;
        self.queries.push(query);
        id
    }

    pub fn push_statement(&mut self, mut statement: SqliteStatement) -> SqliteStatementId {
        let id = SqliteStatementId(self.statements.len() as u32);
        statement.id = id;
        self.statements.push(statement);
        id
    }

    pub fn push_parameter(&mut self, mut parameter: SqliteParameter) -> SqliteParamId {
        let id = SqliteParamId(self.parameters.len() as u16);
        parameter.id = id;
        self.parameters.push(parameter);
        id
    }

    /// Basic structural verification. Semantic SQL validation remains a separate
    /// pass because it needs scope/type information and should produce richer
    /// frontend diagnostics.
    pub fn verify(&self) -> Result<(), Vec<SqliteIrError>> {
        let mut errors = Vec::new();

        for (index, table) in self.tables.iter().enumerate() {
            if table.id.0 as usize != index {
                errors.push(SqliteIrError::InvalidTableId(table.id));
            }
            for (column_index, column) in table.columns.iter().enumerate() {
                if column.id.0 as usize != column_index {
                    errors.push(SqliteIrError::InvalidColumnId {
                        table: table.id,
                        column: column.id,
                    });
                }
                if let Some(foreign) = &column.references {
                    if self.table(foreign.table).is_none() {
                        errors.push(SqliteIrError::InvalidTableId(foreign.table));
                    }
                }
            }
        }

        for (index, sqlite_index) in self.indexes.iter().enumerate() {
            if sqlite_index.id.0 as usize != index {
                errors.push(SqliteIrError::InvalidIndexId(sqlite_index.id));
            }
            if self.table(sqlite_index.table).is_none() {
                errors.push(SqliteIrError::InvalidTableId(sqlite_index.table));
            }
        }

        for (index, expression) in self.expressions.iter().enumerate() {
            if expression.id.0 as usize != index {
                errors.push(SqliteIrError::InvalidExprId(expression.id));
            }
            self.verify_expression_refs(expression, &mut errors);
        }

        for (index, query) in self.queries.iter().enumerate() {
            if query.id.0 as usize != index {
                errors.push(SqliteIrError::InvalidQueryId(query.id));
            }
            self.verify_query_refs(query, &mut errors);
        }

        for (index, statement) in self.statements.iter().enumerate() {
            if statement.id.0 as usize != index {
                errors.push(SqliteIrError::InvalidStatementId(statement.id));
            }
            self.verify_statement_refs(statement, &mut errors);
        }

        for (index, parameter) in self.parameters.iter().enumerate() {
            if parameter.id.0 as usize != index {
                errors.push(SqliteIrError::DuplicateParameter(parameter.id));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn verify_expression_refs(&self, expression: &SqliteExpr, errors: &mut Vec<SqliteIrError>) {
        fn check_expr(ir: &SqliteIr, id: SqliteExprId, errors: &mut Vec<SqliteIrError>) {
            if ir.expression(id).is_none() {
                errors.push(SqliteIrError::InvalidExprId(id));
            }
        }

        fn check_query(ir: &SqliteIr, id: SqliteQueryId, errors: &mut Vec<SqliteIrError>) {
            if ir.query(id).is_none() {
                errors.push(SqliteIrError::InvalidQueryId(id));
            }
        }

        match &expression.kind {
            SqliteExprKind::Literal(_) | SqliteExprKind::Column(_) => {}
            SqliteExprKind::Parameter(id) => {
                if self.parameter(*id).is_none() {
                    errors.push(SqliteIrError::DuplicateParameter(*id));
                }
            }
            SqliteExprKind::Unary { value, .. } => check_expr(self, *value, errors),
            SqliteExprKind::Binary { left, right, .. }
            | SqliteExprKind::Is { left, right, .. } => {
                check_expr(self, *left, errors);
                check_expr(self, *right, errors);
            }
            SqliteExprKind::Function { args, filter, .. } => {
                for arg in args {
                    check_expr(self, *arg, errors);
                }
                if let Some(filter) = filter {
                    check_expr(self, *filter, errors);
                }
            }
            SqliteExprKind::Cast { value, .. }
            | SqliteExprKind::IsNull { value, .. }
            | SqliteExprKind::Collate { value, .. } => check_expr(self, *value, errors),
            SqliteExprKind::Case {
                operand,
                arms,
                else_value,
            } => {
                if let Some(operand) = operand {
                    check_expr(self, *operand, errors);
                }
                for arm in arms {
                    check_expr(self, arm.when, errors);
                    check_expr(self, arm.then, errors);
                }
                if let Some(else_value) = else_value {
                    check_expr(self, *else_value, errors);
                }
            }
            SqliteExprKind::Between {
                value, low, high, ..
            } => {
                check_expr(self, *value, errors);
                check_expr(self, *low, errors);
                check_expr(self, *high, errors);
            }
            SqliteExprKind::InList { value, values, .. } => {
                check_expr(self, *value, errors);
                for candidate in values {
                    check_expr(self, *candidate, errors);
                }
            }
            SqliteExprKind::InQuery { value, query, .. } => {
                check_expr(self, *value, errors);
                check_query(self, *query, errors);
            }
            SqliteExprKind::Like {
                value,
                pattern,
                escape,
                ..
            } => {
                check_expr(self, *value, errors);
                check_expr(self, *pattern, errors);
                if let Some(escape) = escape {
                    check_expr(self, *escape, errors);
                }
            }
            SqliteExprKind::Glob { value, pattern, .. } => {
                check_expr(self, *value, errors);
                check_expr(self, *pattern, errors);
            }
            SqliteExprKind::Exists(query) | SqliteExprKind::ScalarSubquery(query) => {
                check_query(self, *query, errors)
            }
        }
    }

    fn verify_query_refs(&self, query: &SqliteQuery, errors: &mut Vec<SqliteIrError>) {
        let check_expr = |id: SqliteExprId, errors: &mut Vec<SqliteIrError>| {
            if self.expression(id).is_none() {
                errors.push(SqliteIrError::InvalidExprId(id));
            }
        };

        match &query.body {
            SqliteQueryBody::Select(select) => {
                let mut relations = Vec::new();
                if let Some(from) = &select.from {
                    self.verify_source(from, &mut relations, errors);
                }
                for join in &select.joins {
                    self.verify_source(&join.source, &mut relations, errors);
                    if let Some(on) = join.on {
                        check_expr(on, errors);
                    }
                }
                for projection in &select.projection {
                    check_expr(projection.expression, errors);
                }
                if let Some(predicate) = select.predicate {
                    check_expr(predicate, errors);
                }
                for group in &select.group_by {
                    check_expr(*group, errors);
                }
                if let Some(having) = select.having {
                    check_expr(having, errors);
                }
            }
            SqliteQueryBody::Values(values) => {
                for row in &values.rows {
                    for expression in row {
                        check_expr(*expression, errors);
                    }
                }
            }
            SqliteQueryBody::Compound(compound) => {
                if self.query(compound.left).is_none() {
                    errors.push(SqliteIrError::InvalidQueryId(compound.left));
                }
                if self.query(compound.right).is_none() {
                    errors.push(SqliteIrError::InvalidQueryId(compound.right));
                }
            }
        }

        for key in &query.order_by {
            check_expr(key.expression, errors);
        }
        if let Some(limit) = query.limit {
            check_expr(limit, errors);
        }
        if let Some(offset) = query.offset {
            check_expr(offset, errors);
        }
        for projection in &query.output {
            check_expr(projection.expression, errors);
        }
    }

    fn verify_source(
        &self,
        source: &SqliteFromSource,
        relations: &mut Vec<SqliteRelationId>,
        errors: &mut Vec<SqliteIrError>,
    ) {
        let relation = source.relation();
        if relations.contains(&relation) {
            errors.push(SqliteIrError::DuplicateRelation(relation));
        } else {
            relations.push(relation);
        }

        match source {
            SqliteFromSource::Table { table, .. } => {
                if self.table(*table).is_none() {
                    errors.push(SqliteIrError::InvalidTableId(*table));
                }
            }
            SqliteFromSource::Subquery { query, .. } => {
                if self.query(*query).is_none() {
                    errors.push(SqliteIrError::InvalidQueryId(*query));
                }
            }
        }
    }

    fn verify_statement_refs(
        &self,
        statement: &SqliteStatement,
        errors: &mut Vec<SqliteIrError>,
    ) {
        match &statement.kind {
            SqliteStatementKind::Query { query, result } => {
                if self.query(*query).is_none() {
                    errors.push(SqliteIrError::InvalidQueryId(*query));
                } else {
                    self.verify_result_shape(*query, result, errors);
                }
            }
            SqliteStatementKind::CreateTable { table, .. }
            | SqliteStatementKind::DropTable { table, .. } => {
                if self.table(*table).is_none() {
                    errors.push(SqliteIrError::InvalidTableId(*table));
                }
            }
            SqliteStatementKind::CreateTables { tables, .. } => {
                for table in tables {
                    if self.table(*table).is_none() {
                        errors.push(SqliteIrError::InvalidCreateTablesEntry(*table));
                    }
                }
            }
            SqliteStatementKind::DropTables { tables, .. } => {
                for table in tables {
                    if self.table(*table).is_none() {
                        errors.push(SqliteIrError::InvalidDropTablesEntry(*table));
                    }
                }
            }
            SqliteStatementKind::CreateIndex { index, .. }
            | SqliteStatementKind::DropIndex { index, .. } => {
                if self.index(*index).is_none() {
                    errors.push(SqliteIrError::InvalidIndexId(*index));
                }
            }
            SqliteStatementKind::AlterTable(alter) => match alter {
                SqliteAlterTable::RenameTable { table, .. }
                | SqliteAlterTable::RenameColumn { table, .. }
                | SqliteAlterTable::AddColumn { table, .. }
                | SqliteAlterTable::DropColumn { table, .. } => {
                    if self.table(*table).is_none() {
                        errors.push(SqliteIrError::InvalidTableId(*table));
                    }
                }
            },
            SqliteStatementKind::Insert(insert) => {
                if self.table(insert.table).is_none() {
                    errors.push(SqliteIrError::InvalidTableId(insert.table));
                }
            }
            SqliteStatementKind::Update(update) => {
                if self.table(update.table).is_none() {
                    errors.push(SqliteIrError::InvalidTableId(update.table));
                }
            }
            SqliteStatementKind::Delete(delete) => {
                if self.table(delete.table).is_none() {
                    errors.push(SqliteIrError::InvalidTableId(delete.table));
                }
            }
            SqliteStatementKind::Attach { path, .. } => {
                if self.expression(*path).is_none() {
                    errors.push(SqliteIrError::InvalidExprId(*path));
                }
            }
            SqliteStatementKind::Begin { .. }
            | SqliteStatementKind::Commit
            | SqliteStatementKind::Rollback
            | SqliteStatementKind::Savepoint { .. }
            | SqliteStatementKind::ReleaseSavepoint { .. }
            | SqliteStatementKind::RollbackToSavepoint { .. }
            | SqliteStatementKind::Detach { .. }
            | SqliteStatementKind::Pragma { .. }
            | SqliteStatementKind::Vacuum { .. }
            | SqliteStatementKind::Analyze { .. }
            | SqliteStatementKind::Raw(_) => {}
        }
    }

    fn verify_result_shape(
        &self,
        query: SqliteQueryId,
        result: &SqliteResultShape,
        errors: &mut Vec<SqliteIrError>,
    ) {
        let Some(query) = self.query(query) else {
            return;
        };
        let SqliteResultShape::Rows(materializer) = result else {
            return;
        };

        if !matches!(materializer.row_type, Type::Interface(_) | Type::Record(_)) {
            errors.push(SqliteIrError::InvalidMaterializedLayout);
        }

        let mut result_columns = Vec::with_capacity(materializer.fields.len());
        let mut property_slots = Vec::with_capacity(materializer.fields.len());
        for field in &materializer.fields {
            if field.result_column as usize >= query.output.len() {
                errors.push(SqliteIrError::InvalidMaterializedColumn {
                    column: field.result_column,
                    output_len: query.output.len(),
                });
            }
            if result_columns.contains(&field.result_column) {
                errors.push(SqliteIrError::DuplicateMaterializedColumn(field.result_column));
            } else {
                result_columns.push(field.result_column);
            }
            if property_slots.contains(&field.property_slot) {
                errors.push(SqliteIrError::DuplicateMaterializedProperty(field.property_slot));
            } else {
                property_slots.push(field.property_slot);
            }
        }
    }
}
