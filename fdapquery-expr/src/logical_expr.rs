//! # Why the `Expr` variants live here, and not in `expressions.rs`
//!
//! `Expr` is a sum type (an "or"): a logical expression is a `Column`
//! OR a `LiteralString` OR an `Eq` OR … . In Rust a sum type *is* the list of
//! its summands — writing a variant is how you define part of the type — so
//! every variant must sit inside this one `enum` declaration; a summand cannot
//! be written in another file. That is why all of the comparison, arithmetic,
//! literal, and structural cases land in the `Expr` enum below, and why
//! this file — named for the declaration of the *type* — is where they belong.
//!
//! The **aggregate functions are the one exception**, and they reveal the
//! limit of "collapse everything into one enum." An aggregate has *two*
//! memberships at once: it is an `AggregateExpr` (the narrow family that the
//! `Aggregate` plan ranges over with a typed `Vec<AggregateExpr>`) **and** a
//! `Expr` (so it can appear inside any expression, e.g. the
//! `HAVING MAX(salary) > 10` predicate, where the aggregate is an operand of a
//! comparison). A flat enum can express only one of those memberships.
//!
//! This module preserves both, the way DataFusion's `Expr::AggregateFunction`
//! does: `AggregateExpr` stays its own enum (in `expressions.rs`), so the
//! narrow family keeps a name and the `Aggregate` plan keeps a typed
//! `Vec<AggregateExpr>`; and a single bridge variant,
//! `Expr::AggregateExpr(Box<AggregateExpr>)`, injects an aggregate into
//! the broad family so it can nest inside any expression. (The `Box` breaks
//! the `Expr` ↔ `AggregateExpr` size cycle.) The convenience
//! constructors `sum`/`min`/… return `AggregateExpr`, and
//! `impl From<AggregateExpr> for Expr` performs the bridge for nesting.
//!
//! `to_field` and `Display` are implemented as `match`es (defining behaviour
//! on a sum means answering for every summand, exhaustively checked); the
//! bridge variant simply delegates to the inner `AggregateExpr`.
//!
//! What is NOT a summand of `Expr` stays in `expressions.rs`: the
//! separate `AggregateExpr` sum type and the convenience constructors (`col`,
//! `lit_*`, `cast`, the `eq`/`add`/… builder methods, and the `sum`/`min`/…
//! aggregate constructors) — the functions that *build* a `Expr`. See
//! that file's header for why those are freely separable from this type's
//! definition.

use crate::expressions::AggregateExpr;
use crate::logical_plan::LogicalPlan;
use arrow_schema::DataType;
use fdapquery_datatypes::{FdapQueryError, Field, Result};
use std::fmt;

/// A logical expression used in logical query plans. It provides the planning-
/// phase metadata (name and data type) of the value it will produce.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Reference to a column by name.
    Column(String),
    /// Reference to a column by index.
    ColumnIndex(usize),

    LiteralString(String),
    LiteralLong(i64),
    LiteralFloat(f32),
    LiteralDouble(f64),
    /// Date literal, stored as `chrono::NaiveDate` (the workspace's date type
    /// for `Date32` columns).
    LiteralDate(chrono::NaiveDate),
    LiteralIntervalDays(i64),

    DateSubtractInterval {
        date: Box<Expr>,
        interval: Box<Expr>,
    },
    DateAddInterval {
        date: Box<Expr>,
        interval: Box<Expr>,
    },

    Cast {
        expr: Box<Expr>,
        data_type: DataType,
    },

    /// Logical negation — the only unary boolean expression.
    Not(Box<Expr>),

    // Boolean binary expressions.
    Eq {
        l: Box<Expr>,
        r: Box<Expr>,
    },
    Neq {
        l: Box<Expr>,
        r: Box<Expr>,
    },
    Gt {
        l: Box<Expr>,
        r: Box<Expr>,
    },
    GtEq {
        l: Box<Expr>,
        r: Box<Expr>,
    },
    Lt {
        l: Box<Expr>,
        r: Box<Expr>,
    },
    LtEq {
        l: Box<Expr>,
        r: Box<Expr>,
    },
    And {
        l: Box<Expr>,
        r: Box<Expr>,
    },
    Or {
        l: Box<Expr>,
        r: Box<Expr>,
    },

    // Math binary expressions.
    Add {
        l: Box<Expr>,
        r: Box<Expr>,
    },
    Subtract {
        l: Box<Expr>,
        r: Box<Expr>,
    },
    Multiply {
        l: Box<Expr>,
        r: Box<Expr>,
    },
    Divide {
        l: Box<Expr>,
        r: Box<Expr>,
    },
    Modulus {
        l: Box<Expr>,
        r: Box<Expr>,
    },

    /// `expr AS alias`.
    Alias {
        expr: Box<Expr>,
        alias: String,
    },

    /// Scalar function call (`name(args) -> return_type`).
    ScalarFunction {
        name: String,
        args: Vec<Expr>,
        return_type: DataType,
    },

    /// An aggregate function used as a logical expression. Bridges the
    /// separate [`AggregateExpr`] family (the narrow type the `Aggregate` plan
    /// ranges over) into `Expr` (the broad family), so an aggregate can
    /// nest inside any expression — e.g. the `HAVING MAX(salary) > 10`
    /// predicate. Mirrors DataFusion's `Expr::AggregateFunction`. Boxed to
    /// break the `Expr` ↔ `AggregateExpr` size cycle.
    AggregateExpr(Box<AggregateExpr>),
}

impl Expr {
    /// Metadata about the value this expression produces against `input`.
    pub fn to_field(&self, input: &LogicalPlan) -> Result<Field> {
        match self {
            Expr::Column(name) => {
                let schema = input.schema()?;
                schema
                    .fields()
                    .iter()
                    .find(|f| f.name() == name)
                    .map(|f| f.as_ref().clone())
                    .ok_or_else(|| {
                        let names: Vec<String> =
                            schema.fields().iter().map(|f| f.name().clone()).collect();
                        FdapQueryError::SchemaError(format!(
                            "Expr::to_field: no column named '{name}' in {names:?}"
                        ))
                    })
            }
            Expr::ColumnIndex(i) => {
                let schema = input.schema()?;
                schema
                    .fields()
                    .get(*i)
                    .map(|f| f.as_ref().clone())
                    .ok_or_else(|| {
                        FdapQueryError::Internal(format!(
                            "Expr::to_field: column index {i} out of bounds \
                             (schema has {} fields)",
                            schema.fields().len()
                        ))
                    })
            }
            Expr::LiteralString(s) => Ok(Field::new(s.clone(), arrow_schema::DataType::Utf8, true)),
            Expr::LiteralLong(n) => Ok(Field::new(
                n.to_string(),
                arrow_schema::DataType::Int64,
                true,
            )),
            Expr::LiteralFloat(n) => Ok(Field::new(
                n.to_string(),
                arrow_schema::DataType::Float32,
                true,
            )),
            Expr::LiteralDouble(n) => Ok(Field::new(
                n.to_string(),
                arrow_schema::DataType::Float64,
                true,
            )),
            // `NaiveDate`'s `Display` emits the ISO-8601 form ("YYYY-MM-DD").
            Expr::LiteralDate(d) => Ok(Field::new(
                d.to_string(),
                arrow_schema::DataType::Date32,
                true,
            )),
            Expr::LiteralIntervalDays(days) => Ok(Field::new(
                format!("{days} days"),
                arrow_schema::DataType::Interval(arrow_schema::IntervalUnit::DayTime),
                true,
            )),
            Expr::DateSubtractInterval { .. } => Ok(Field::new(
                "date_subtract",
                arrow_schema::DataType::Date32,
                true,
            )),
            Expr::DateAddInterval { .. } => {
                Ok(Field::new("date_add", arrow_schema::DataType::Date32, true))
            }
            Expr::Cast { expr, data_type } => Ok(Field::new(
                expr.to_field(input)?.name().clone(),
                data_type.clone(),
                true,
            )),
            Expr::Not(_) => Ok(Field::new("not", arrow_schema::DataType::Boolean, true)),
            Expr::Eq { .. } => Ok(Field::new("eq", arrow_schema::DataType::Boolean, true)),
            Expr::Neq { .. } => Ok(Field::new("neq", arrow_schema::DataType::Boolean, true)),
            Expr::Gt { .. } => Ok(Field::new("gt", arrow_schema::DataType::Boolean, true)),
            Expr::GtEq { .. } => Ok(Field::new("gteq", arrow_schema::DataType::Boolean, true)),
            Expr::Lt { .. } => Ok(Field::new("lt", arrow_schema::DataType::Boolean, true)),
            Expr::LtEq { .. } => Ok(Field::new("lteq", arrow_schema::DataType::Boolean, true)),
            Expr::And { .. } => Ok(Field::new("and", arrow_schema::DataType::Boolean, true)),
            Expr::Or { .. } => Ok(Field::new("or", arrow_schema::DataType::Boolean, true)),
            Expr::Add { l, .. } => Ok(Field::new(
                "add",
                l.to_field(input)?.data_type().clone(),
                true,
            )),
            Expr::Subtract { l, .. } => Ok(Field::new(
                "subtract",
                l.to_field(input)?.data_type().clone(),
                true,
            )),
            Expr::Multiply { l, .. } => Ok(Field::new(
                "mult",
                l.to_field(input)?.data_type().clone(),
                true,
            )),
            Expr::Divide { l, .. } => Ok(Field::new(
                "div",
                l.to_field(input)?.data_type().clone(),
                true,
            )),
            Expr::Modulus { l, .. } => Ok(Field::new(
                "mod",
                l.to_field(input)?.data_type().clone(),
                true,
            )),
            Expr::Alias { expr, alias } => Ok(Field::new(
                alias.clone(),
                expr.to_field(input)?.data_type().clone(),
                true,
            )),
            Expr::ScalarFunction {
                name, return_type, ..
            } => Ok(Field::new(name.clone(), return_type.clone(), true)),
            // An aggregate used as an expression delegates to the inner
            // `AggregateExpr` for its field metadata.
            Expr::AggregateExpr(agg) => agg.to_field(input),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Column(name) => write!(f, "#{name}"),
            Expr::ColumnIndex(i) => write!(f, "#{i}"),
            Expr::LiteralString(s) => write!(f, "'{s}'"),
            Expr::LiteralLong(n) => write!(f, "{n}"),
            Expr::LiteralFloat(n) => write!(f, "{n}"),
            Expr::LiteralDouble(n) => write!(f, "{n}"),
            Expr::LiteralDate(d) => write!(f, "DATE '{d}'"),
            Expr::LiteralIntervalDays(days) => write!(f, "INTERVAL '{days} days'"),
            Expr::DateSubtractInterval { date, interval } => {
                write!(f, "{date} - {interval}")
            }
            Expr::DateAddInterval { date, interval } => write!(f, "{date} + {interval}"),
            Expr::Cast { expr, data_type } => write!(f, "CAST({expr} AS {data_type:?})"),
            Expr::Not(e) => write!(f, "NOT {e}"),
            Expr::Eq { l, r } => write!(f, "{l} = {r}"),
            Expr::Neq { l, r } => write!(f, "{l} != {r}"),
            Expr::Gt { l, r } => write!(f, "{l} > {r}"),
            Expr::GtEq { l, r } => write!(f, "{l} >= {r}"),
            Expr::Lt { l, r } => write!(f, "{l} < {r}"),
            Expr::LtEq { l, r } => write!(f, "{l} <= {r}"),
            Expr::And { l, r } => write!(f, "{l} AND {r}"),
            Expr::Or { l, r } => write!(f, "{l} OR {r}"),
            Expr::Add { l, r } => write!(f, "{l} + {r}"),
            Expr::Subtract { l, r } => write!(f, "{l} - {r}"),
            Expr::Multiply { l, r } => write!(f, "{l} * {r}"),
            Expr::Divide { l, r } => write!(f, "{l} / {r}"),
            Expr::Modulus { l, r } => write!(f, "{l} % {r}"),
            Expr::Alias { expr, alias } => write!(f, "{expr} as {alias}"),
            Expr::ScalarFunction { name, args, .. } => {
                let args_str: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "{name}([{}])", args_str.join(", "))
            }
            Expr::AggregateExpr(agg) => write!(f, "{agg}"),
        }
    }
}
