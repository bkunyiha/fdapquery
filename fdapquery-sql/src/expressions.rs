//! The SQL **AST** produced by the parser. This is distinct from the
//! logical-plan `LogicalExpr`: `SqlPlanner` translates this untyped, pre-binding
//! syntax tree into the schema-aware logical expressions.
//!
//! ## Notes
//! - The AST is modeled as a single `SqlExpr` enum (interface-hierarchy → enum,
//!   §3.1). `SqlSelect` is a struct surfaced as the `SqlExpr::Select` variant.
//! - `SqlAlias` and `SqlCast` carry their alias / data-type as a plain `String`
//!   rather than a dedicated identifier wrapper — the planner only ever reads
//!   the string content.

/// A SQL abstract-syntax-tree node.
#[derive(Debug, Clone, PartialEq)]
pub enum SqlExpr {
    /// Identifier — table/column name.
    Identifier(String),
    /// Binary expression.
    BinaryExpr {
        l: Box<SqlExpr>,
        op: String,
        r: Box<SqlExpr>,
    },
    /// String literal.
    String(String),
    /// Long literal.
    Long(i64),
    /// Double literal.
    Double(f64),
    /// Date literal.
    Date(String),
    /// Interval literal.
    Interval(String),
    /// Function call.
    Function { id: String, args: Vec<SqlExpr> },
    /// Aliased expression `expr AS alias`. `alias` is stored as plain text.
    Alias { expr: Box<SqlExpr>, alias: String },
    /// `CAST(expr AS type)`. `data_type` is stored as plain text.
    Cast {
        expr: Box<SqlExpr>,
        data_type: String,
    },
    /// Sort key `expr ASC|DESC`.
    Sort { expr: Box<SqlExpr>, asc: bool },
    /// A SELECT statement. Boxed to break the size cycle
    /// `SqlExpr → SqlSelect → Option<SqlExpr>`; a Rust value type needs
    /// explicit indirection to be sized.
    Select(Box<SqlSelect>),
}

/// A parsed `SELECT` statement.
#[derive(Debug, Clone, PartialEq)]
pub struct SqlSelect {
    pub projection: Vec<SqlExpr>,
    pub selection: Option<SqlExpr>,
    pub group_by: Vec<SqlExpr>,
    pub order_by: Vec<SqlExpr>,
    pub having: Option<SqlExpr>,
    pub limit: Option<i32>,
    pub table_name: String,
}
