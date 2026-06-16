//! # Why the `LogicalExpr` variants live here, and not in `expressions.rs`
//!
//! `LogicalExpr` is a sum type (an "or"): a logical expression is a `Column`
//! OR a `LiteralString` OR an `Eq` OR … . In Rust a sum type *is* the list of
//! its summands — writing a variant is how you define part of the type — so
//! every variant must sit inside this one `enum` declaration; a summand cannot
//! be written in another file. That is why all of the comparison, arithmetic,
//! literal, and structural cases land in the `LogicalExpr` enum below, and why
//! this file — named for the declaration of the *type* — is where they belong.
//!
//! The **aggregate functions are the one exception**, and they reveal the
//! limit of "collapse everything into one enum." An aggregate has *two*
//! memberships at once: it is an `AggregateExpr` (the narrow family that the
//! `Aggregate` plan ranges over with a typed `Vec<AggregateExpr>`) **and** a
//! `LogicalExpr` (so it can appear inside any expression, e.g. the
//! `HAVING MAX(salary) > 10` predicate, where the aggregate is an operand of a
//! comparison). A flat enum can express only one of those memberships.
//!
//! This module preserves both, the way DataFusion's `Expr::AggregateFunction`
//! does: `AggregateExpr` stays its own enum (in `expressions.rs`), so the
//! narrow family keeps a name and the `Aggregate` plan keeps a typed
//! `Vec<AggregateExpr>`; and a single bridge variant,
//! `LogicalExpr::AggregateExpr(Box<AggregateExpr>)`, injects an aggregate into
//! the broad family so it can nest inside any expression. (The `Box` breaks
//! the `LogicalExpr` ↔ `AggregateExpr` size cycle.) The convenience
//! constructors `sum`/`min`/… return `AggregateExpr`, and
//! `impl From<AggregateExpr> for LogicalExpr` performs the bridge for nesting.
//!
//! `to_field` and `Display` are implemented as `match`es (defining behaviour
//! on a sum means answering for every summand, exhaustively checked); the
//! bridge variant simply delegates to the inner `AggregateExpr`.
//!
//! What is NOT a summand of `LogicalExpr` stays in `expressions.rs`: the
//! separate `AggregateExpr` sum type and the convenience constructors (`col`,
//! `lit_*`, `cast`, the `eq`/`add`/… builder methods, and the `sum`/`min`/…
//! aggregate constructors) — the functions that *build* a `LogicalExpr`. See
//! that file's header for why those are freely separable from this type's
//! definition.

use crate::expressions::AggregateExpr;
use crate::logical_plan::LogicalPlan;
use arrow_schema::DataType;
use datatypes::Field;
use datatypes::arrow_types::{
    BOOLEAN_TYPE, DATE_DAY_TYPE, DOUBLE_TYPE, FLOAT_TYPE, INT64_TYPE, INTERVAL_DAY_TIME_TYPE,
    STRING_TYPE,
};
use std::fmt;

/// A logical expression used in logical query plans. It provides the planning-
/// phase metadata (name and data type) of the value it will produce.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalExpr {
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
        date: Box<LogicalExpr>,
        interval: Box<LogicalExpr>,
    },
    DateAddInterval {
        date: Box<LogicalExpr>,
        interval: Box<LogicalExpr>,
    },

    Cast {
        expr: Box<LogicalExpr>,
        data_type: DataType,
    },

    /// Logical negation — the only unary boolean expression.
    Not(Box<LogicalExpr>),

    // Boolean binary expressions.
    Eq {
        l: Box<LogicalExpr>,
        r: Box<LogicalExpr>,
    },
    Neq {
        l: Box<LogicalExpr>,
        r: Box<LogicalExpr>,
    },
    Gt {
        l: Box<LogicalExpr>,
        r: Box<LogicalExpr>,
    },
    GtEq {
        l: Box<LogicalExpr>,
        r: Box<LogicalExpr>,
    },
    Lt {
        l: Box<LogicalExpr>,
        r: Box<LogicalExpr>,
    },
    LtEq {
        l: Box<LogicalExpr>,
        r: Box<LogicalExpr>,
    },
    And {
        l: Box<LogicalExpr>,
        r: Box<LogicalExpr>,
    },
    Or {
        l: Box<LogicalExpr>,
        r: Box<LogicalExpr>,
    },

    // Math binary expressions.
    Add {
        l: Box<LogicalExpr>,
        r: Box<LogicalExpr>,
    },
    Subtract {
        l: Box<LogicalExpr>,
        r: Box<LogicalExpr>,
    },
    Multiply {
        l: Box<LogicalExpr>,
        r: Box<LogicalExpr>,
    },
    Divide {
        l: Box<LogicalExpr>,
        r: Box<LogicalExpr>,
    },
    Modulus {
        l: Box<LogicalExpr>,
        r: Box<LogicalExpr>,
    },

    /// `expr AS alias`.
    Alias {
        expr: Box<LogicalExpr>,
        alias: String,
    },

    /// Scalar function call (`name(args) -> return_type`).
    ScalarFunction {
        name: String,
        args: Vec<LogicalExpr>,
        return_type: DataType,
    },

    /// An aggregate function used as a logical expression. Bridges the
    /// separate [`AggregateExpr`] family (the narrow type the `Aggregate` plan
    /// ranges over) into `LogicalExpr` (the broad family), so an aggregate can
    /// nest inside any expression — e.g. the `HAVING MAX(salary) > 10`
    /// predicate. Mirrors DataFusion's `Expr::AggregateFunction`. Boxed to
    /// break the `LogicalExpr` ↔ `AggregateExpr` size cycle.
    AggregateExpr(Box<AggregateExpr>),
}

impl LogicalExpr {
    /// Metadata about the value this expression produces against `input`.
    pub fn to_field(&self, input: &LogicalPlan) -> Field {
        match self {
            LogicalExpr::Column(name) => {
                let schema = input.schema();
                schema
                    .fields
                    .iter()
                    .find(|f| &f.name == name)
                    .cloned()
                    .unwrap_or_else(|| {
                        let names: Vec<String> =
                            schema.fields.iter().map(|f| f.name.clone()).collect();
                        panic!("No column named '{}' in {:?}", name, names)
                    })
            }
            LogicalExpr::ColumnIndex(i) => input.schema().fields[*i].clone(),
            LogicalExpr::LiteralString(s) => Field::new(s.clone(), STRING_TYPE),
            LogicalExpr::LiteralLong(n) => Field::new(n.to_string(), INT64_TYPE),
            LogicalExpr::LiteralFloat(n) => Field::new(n.to_string(), FLOAT_TYPE),
            LogicalExpr::LiteralDouble(n) => Field::new(n.to_string(), DOUBLE_TYPE),
            // `NaiveDate`'s `Display` emits the ISO-8601 form ("YYYY-MM-DD").
            LogicalExpr::LiteralDate(d) => Field::new(d.to_string(), DATE_DAY_TYPE),
            LogicalExpr::LiteralIntervalDays(days) => {
                Field::new(format!("{days} days"), INTERVAL_DAY_TIME_TYPE)
            }
            LogicalExpr::DateSubtractInterval { .. } => Field::new("date_subtract", DATE_DAY_TYPE),
            LogicalExpr::DateAddInterval { .. } => Field::new("date_add", DATE_DAY_TYPE),
            LogicalExpr::Cast { expr, data_type } => {
                Field::new(expr.to_field(input).name, data_type.clone())
            }
            LogicalExpr::Not(_) => Field::new("not", BOOLEAN_TYPE),
            LogicalExpr::Eq { .. } => Field::new("eq", BOOLEAN_TYPE),
            LogicalExpr::Neq { .. } => Field::new("neq", BOOLEAN_TYPE),
            LogicalExpr::Gt { .. } => Field::new("gt", BOOLEAN_TYPE),
            LogicalExpr::GtEq { .. } => Field::new("gteq", BOOLEAN_TYPE),
            LogicalExpr::Lt { .. } => Field::new("lt", BOOLEAN_TYPE),
            LogicalExpr::LtEq { .. } => Field::new("lteq", BOOLEAN_TYPE),
            LogicalExpr::And { .. } => Field::new("and", BOOLEAN_TYPE),
            LogicalExpr::Or { .. } => Field::new("or", BOOLEAN_TYPE),
            LogicalExpr::Add { l, .. } => Field::new("add", l.to_field(input).data_type),
            LogicalExpr::Subtract { l, .. } => Field::new("subtract", l.to_field(input).data_type),
            LogicalExpr::Multiply { l, .. } => Field::new("mult", l.to_field(input).data_type),
            LogicalExpr::Divide { l, .. } => Field::new("div", l.to_field(input).data_type),
            LogicalExpr::Modulus { l, .. } => Field::new("mod", l.to_field(input).data_type),
            LogicalExpr::Alias { expr, alias } => {
                Field::new(alias.clone(), expr.to_field(input).data_type)
            }
            LogicalExpr::ScalarFunction {
                name, return_type, ..
            } => Field::new(name.clone(), return_type.clone()),
            // An aggregate used as an expression delegates to the inner
            // `AggregateExpr` for its field metadata.
            LogicalExpr::AggregateExpr(agg) => agg.to_field(input),
        }
    }
}

impl fmt::Display for LogicalExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogicalExpr::Column(name) => write!(f, "#{name}"),
            LogicalExpr::ColumnIndex(i) => write!(f, "#{i}"),
            LogicalExpr::LiteralString(s) => write!(f, "'{s}'"),
            LogicalExpr::LiteralLong(n) => write!(f, "{n}"),
            LogicalExpr::LiteralFloat(n) => write!(f, "{n}"),
            LogicalExpr::LiteralDouble(n) => write!(f, "{n}"),
            LogicalExpr::LiteralDate(d) => write!(f, "DATE '{d}'"),
            LogicalExpr::LiteralIntervalDays(days) => write!(f, "INTERVAL '{days} days'"),
            LogicalExpr::DateSubtractInterval { date, interval } => {
                write!(f, "{date} - {interval}")
            }
            LogicalExpr::DateAddInterval { date, interval } => write!(f, "{date} + {interval}"),
            LogicalExpr::Cast { expr, data_type } => write!(f, "CAST({expr} AS {data_type:?})"),
            LogicalExpr::Not(e) => write!(f, "NOT {e}"),
            LogicalExpr::Eq { l, r } => write!(f, "{l} = {r}"),
            LogicalExpr::Neq { l, r } => write!(f, "{l} != {r}"),
            LogicalExpr::Gt { l, r } => write!(f, "{l} > {r}"),
            LogicalExpr::GtEq { l, r } => write!(f, "{l} >= {r}"),
            LogicalExpr::Lt { l, r } => write!(f, "{l} < {r}"),
            LogicalExpr::LtEq { l, r } => write!(f, "{l} <= {r}"),
            LogicalExpr::And { l, r } => write!(f, "{l} AND {r}"),
            LogicalExpr::Or { l, r } => write!(f, "{l} OR {r}"),
            LogicalExpr::Add { l, r } => write!(f, "{l} + {r}"),
            LogicalExpr::Subtract { l, r } => write!(f, "{l} - {r}"),
            LogicalExpr::Multiply { l, r } => write!(f, "{l} * {r}"),
            LogicalExpr::Divide { l, r } => write!(f, "{l} / {r}"),
            LogicalExpr::Modulus { l, r } => write!(f, "{l} % {r}"),
            LogicalExpr::Alias { expr, alias } => write!(f, "{expr} as {alias}"),
            LogicalExpr::ScalarFunction { name, args, .. } => {
                let args_str: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "{name}([{}])", args_str.join(", "))
            }
            LogicalExpr::AggregateExpr(agg) => write!(f, "{agg}"),
        }
    }
}
