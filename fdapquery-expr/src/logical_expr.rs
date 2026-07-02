//! # Why the `Expr` variants live here, and not in `expressions.rs`
//!
//! `Expr` is a sum type (an "or"): a logical expression is a `Column`
//! OR a `Literal` OR a `BinaryExpr` OR … . In Rust a sum type *is* the
//! list of its summands — writing a variant is how you define part of the
//! type — so every variant must sit inside this one `enum` declaration; a
//! summand cannot be written in another file. That is why all of the
//! literal, structural, binary, and aggregate cases land in the `Expr`
//! enum below, and why this file — named for the declaration of the *type*
//! — is where they belong.
//!
//! ## aggregates fold into `Expr::AggregateFunction`
//!
//! Aggregate functions ride on a single
//! `Expr::AggregateFunction(AggregateFunction)` variant — byte-for-byte
//! the same shape DataFusion uses on the
//! `datafusion_expr::Expr::AggregateFunction` variant and the
//! `datafusion_expr::expr::AggregateFunction` struct.
//!
//! The `Aggregate` plan keeps a typed `Vec<Expr>` (every element is
//! `Expr::AggregateFunction(...)` by construction; the SQL planner
//! enforces this), exactly as DataFusion's `LogicalPlan::Aggregate`
//! does. DISTINCT is a struct field (`AggregateFunctionParams::distinct`)
//! rather than a separate enum variant — also matching DataFusion.
//!
//! ## `Expr::BinaryExpr { left, op, right }`
//!
//! A single `Expr::BinaryExpr(BinaryExpr)` variant parameterised by the
//! `Operator` enum — byte-for-byte the same shape DataFusion uses.

use crate::aggregate_function::AggregateFunction;
use crate::logical_plan::LogicalPlan;
use crate::operator::Operator;
use arrow_schema::DataType;
use fdapquery_common::ScalarValue;
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

    /// A constant value. Mirrors DataFusion's `Expr::Literal(ScalarValue)`:
    /// a single enum carrying a typed `ScalarValue` rather than one variant
    /// per Arrow primitive. Aligns fdapquery's logical `Expr`
    /// byte-for-byte with DataFusion. Typed branching is recovered by
    /// matching on the inner `ScalarValue` variant (see
    /// `physical_planner.rs::create_physical_expr`).
    ///
    /// Date literals lower to `ScalarValue::Date32(days_since_unix_epoch)`;
    /// interval-days literals lower to `ScalarValue::Int64(days)` (same
    /// representation the old `LiteralIntervalDays(i64)` used).
    Literal(ScalarValue),

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

    /// A binary expression `left op right`. Collapsed the
    /// 13 sibling binary variants (`Eq`, `Neq`, `Gt`, `GtEq`, `Lt`,
    /// `LtEq`, `And`, `Or`, `Add`, `Subtract`, `Multiply`, `Divide`,
    /// `Modulus`) into this single shape, parameterised by [`Operator`].
    /// Mirrors DataFusion's `Expr::BinaryExpr(BinaryExpr)`.
    BinaryExpr {
        left: Box<Expr>,
        op: Operator,
        right: Box<Expr>,
    },

    /// `expr AS alias`.
    Alias { expr: Box<Expr>, alias: String },

    /// Scalar function call (`name(args) -> return_type`).
    ScalarFunction {
        name: String,
        args: Vec<Expr>,
        return_type: DataType,
    },

    /// An aggregate function call. Byte-for-byte mirror of DataFusion's
    /// `datafusion_expr::Expr::AggregateFunction(AggregateFunction)`
    /// variant. The inner struct carries the
    /// function kind, arguments, DISTINCT flag, optional FILTER, optional
    /// ORDER BY, and optional null treatment — see
    /// [`crate::aggregate_function::AggregateFunctionParams`].
    AggregateFunction(AggregateFunction),
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
            // A literal's field name is its `ScalarValue` `Display`
            // (mirroring DataFusion's `Expr::Literal(scalar)` schema
            // derivation), its data type comes from `ScalarValue::data_type()`,
            // and it is nullable iff the value is the `Null` variant.
            Expr::Literal(scalar) => Ok(Field::new(
                scalar.to_string(),
                scalar.data_type(),
                scalar.is_null(),
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
            // Comparison and logical operators produce `Boolean`;
            // arithmetic operators inherit the left operand's data type.
            // The field name is the short name for the operator ("eq",
            // "add", …), matching what downstream schema-inspecting
            // consumers expect.
            Expr::BinaryExpr { left, op, .. } => {
                let name = binary_field_name(*op);
                let data_type = if op.is_comparison_operator() || op.is_logic_operator() {
                    arrow_schema::DataType::Boolean
                } else {
                    left.to_field(input)?.data_type().clone()
                };
                Ok(Field::new(name, data_type, true))
            }
            Expr::Alias { expr, alias } => Ok(Field::new(
                alias.clone(),
                expr.to_field(input)?.data_type().clone(),
                true,
            )),
            Expr::ScalarFunction {
                name, return_type, ..
            } => Ok(Field::new(name.clone(), return_type.clone(), true)),
            // An aggregate's output field:
            //  - SUM/MIN/MAX/AVG carry the data type of their first
            //    argument expression.
            //  - COUNT (with or without DISTINCT) is an integer count;
            //    DISTINCT widens the type to UInt32.
            // Field names match what downstream schema-inspecting
            // consumers expect.
            Expr::AggregateFunction(agg) => {
                use crate::aggregate_function::AggregateFunctionKind;
                let first = agg.params.args.first().ok_or_else(|| {
                    FdapQueryError::Internal(
                        "Expr::AggregateFunction has no argument expressions".into(),
                    )
                })?;
                match agg.func {
                    AggregateFunctionKind::Sum => Ok(Field::new(
                        "SUM",
                        first.to_field(input)?.data_type().clone(),
                        true,
                    )),
                    AggregateFunctionKind::Min => Ok(Field::new(
                        "MIN",
                        first.to_field(input)?.data_type().clone(),
                        true,
                    )),
                    AggregateFunctionKind::Max => Ok(Field::new(
                        "MAX",
                        first.to_field(input)?.data_type().clone(),
                        true,
                    )),
                    AggregateFunctionKind::Avg => Ok(Field::new(
                        "AVG",
                        first.to_field(input)?.data_type().clone(),
                        true,
                    )),
                    AggregateFunctionKind::Count if agg.params.distinct => Ok(Field::new(
                        "COUNT_DISTINCT",
                        arrow_schema::DataType::UInt32,
                        true,
                    )),
                    AggregateFunctionKind::Count => {
                        Ok(Field::new("COUNT", arrow_schema::DataType::Int32, true))
                    }
                }
            }
        }
    }
}

/// Short field name for each operator. DataFusion uses
/// `format!("{left} {op} {right}")` here; fdapquery uses the short
/// operator name because the optimizer's existing test fixtures and a
/// few downstream consumers depend on the short form.
fn binary_field_name(op: Operator) -> &'static str {
    match op {
        Operator::Eq => "eq",
        Operator::NotEq => "neq",
        Operator::Gt => "gt",
        Operator::GtEq => "gteq",
        Operator::Lt => "lt",
        Operator::LtEq => "lteq",
        Operator::And => "and",
        Operator::Or => "or",
        Operator::Plus => "add",
        Operator::Minus => "subtract",
        Operator::Multiply => "mult",
        Operator::Divide => "div",
        Operator::Modulo => "mod",
        // Operators the engine doesn't emit today fall back to a short
        // identifier — kept only for forward parity with DataFusion's
        // `Operator` set.
        Operator::IsDistinctFrom => "is_distinct_from",
        Operator::IsNotDistinctFrom => "is_not_distinct_from",
        Operator::LikeMatch => "like",
        Operator::ILikeMatch => "ilike",
        Operator::NotLikeMatch => "not_like",
        Operator::NotILikeMatch => "not_ilike",
        Operator::BitwiseAnd => "bitand",
        Operator::BitwiseOr => "bitor",
        Operator::BitwiseXor => "bitxor",
        Operator::BitwiseShiftRight => "shr",
        Operator::BitwiseShiftLeft => "shl",
        Operator::StringConcat => "concat",
        Operator::AtArrow => "at_arrow",
        Operator::ArrowAt => "arrow_at",
        Operator::RegexMatch => "regex_match",
        Operator::RegexIMatch => "regex_imatch",
        Operator::RegexNotMatch => "regex_not_match",
        Operator::RegexNotIMatch => "regex_not_imatch",
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Column(name) => write!(f, "#{name}"),
            Expr::ColumnIndex(i) => write!(f, "#{i}"),
            // Byte-for-byte mirror of DataFusion's
            // `Expr::Literal(scalar)` arm:
            //     Literal(scalar, _) => write!(f, "{scalar}")
            // Delegates to `ScalarValue::Display`, whose strict-mirror
            // implementation is documented in
            // `fdapquery_common::scalar_value`. Strings are bare (no
            // surrounding quotes), Date32 prints as `YYYY-MM-DD`, integers
            // and floats print bare via their `{}` formatter.
            Expr::Literal(scalar) => write!(f, "{scalar}"),
            Expr::DateSubtractInterval { date, interval } => {
                write!(f, "{date} - {interval}")
            }
            Expr::DateAddInterval { date, interval } => write!(f, "{date} + {interval}"),
            Expr::Cast { expr, data_type } => write!(f, "CAST({expr} AS {data_type:?})"),
            Expr::Not(e) => write!(f, "NOT {e}"),
            // Single binary arm delegates to
            // `Operator::Display`. Byte-for-byte mirror of DataFusion's
            // `Expr::BinaryExpr(BinaryExpr { left, op, right }) =>
            // write!(f, "{left} {op} {right}")`. No parenthesisation at
            // this layer — the physical-side `BinaryExpr::Display` adds
            // parens via the operator precedence rules.
            Expr::BinaryExpr { left, op, right } => write!(f, "{left} {op} {right}"),
            Expr::Alias { expr, alias } => write!(f, "{expr} as {alias}"),
            Expr::ScalarFunction { name, args, .. } => {
                let args_str: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "{name}([{}])", args_str.join(", "))
            }
            // Aggregate Display matches DataFusion's `display_name`
            // output for the corresponding built-in aggregates:
            //   SUM(expr) / MIN(expr) / MAX(expr) / AVG(expr) /
            //   COUNT(expr) / COUNT(DISTINCT expr).
            // DataFusion's `impl Display for datafusion_expr::Expr`
            // arm for `Expr::AggregateFunction` delegates to
            // `func.display_name(params)`, which for these built-in
            // aggregates produces exactly this format — byte-for-byte
            // the same Aggregate-plan and HAVING-expression output that
            // downstream tests check.
            Expr::AggregateFunction(agg) => {
                let args: Vec<String> = agg.params.args.iter().map(|a| a.to_string()).collect();
                if agg.params.distinct {
                    write!(f, "{}(DISTINCT {})", agg.func, args.join(", "))
                } else {
                    write!(f, "{}({})", agg.func, args.join(", "))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Byte-for-byte verification that `Expr::Literal(ScalarValue)` and the
    //! generic `lit<T: Literal>` factory match DataFusion's surface.
    //! The expected strings here are the same ones DataFusion's
    //! `Expr::Display` and `ScalarValue::Display` produce in
    //! `datafusion/expr/src/expr.rs` and
    //! `datafusion/common/src/scalar/mod.rs`.
    use super::*;
    use crate::expr_fn::lit;
    use fdapquery_common::ScalarValue;

    #[test]
    fn expr_literal_display_matches_datafusion() {
        assert_eq!(format!("{}", Expr::Literal(ScalarValue::Int64(42))), "42");
        // String literals are bare — no surrounding quotes. Matches
        // DataFusion's `Utf8(Some(s)) => write!(f, "{s}")`.
        assert_eq!(
            format!("{}", Expr::Literal(ScalarValue::Utf8("CO".into()))),
            "CO"
        );
        assert_eq!(
            format!("{}", Expr::Literal(ScalarValue::Float64(1.5))),
            "1.5"
        );
        assert_eq!(
            format!("{}", Expr::Literal(ScalarValue::Boolean(true))),
            "true"
        );
        // Date32: 18750 days since the Unix epoch = 2021-05-03.
        assert_eq!(
            format!("{}", Expr::Literal(ScalarValue::Date32(18750))),
            "2021-05-03"
        );
    }

    #[test]
    fn lit_factory_matches_datafusion_signatures() {
        assert_eq!(format!("{}", lit(42_i64)), "42");
        assert_eq!(format!("{}", lit("CO")), "CO");
        assert_eq!(format!("{}", lit(1.5_f64)), "1.5");
        assert_eq!(format!("{}", lit(true)), "true");
    }

    /// `Expr::BinaryExpr { left, op, right }` Display
    /// is `"{left} {op} {right}"`, matching DataFusion byte-for-byte.
    #[test]
    fn binary_expr_display_matches_datafusion() {
        use crate::expressions::col;

        // `#state = CO`
        let e = col("state").eq(lit("CO"));
        assert_eq!(format!("{e}"), "#state = CO");

        // `#salary + 10`
        let e = col("salary").add(lit(10_i64));
        assert_eq!(format!("{e}"), "#salary + 10");

        // `#a AND #b`
        let e = col("a").and(col("b"));
        assert_eq!(format!("{e}"), "#a AND #b");
    }

    /// `Expr::AggregateFunction(...)` Display mirrors
    /// DataFusion's `display_name` output for the corresponding built-in
    /// UDAFs (see DataFusion's `impl Display for datafusion_expr::Expr`
    /// arm for `Expr::AggregateFunction`, and each function's
    /// `display_name` in `datafusion/functions-aggregate/src/`).
    ///
    /// Asserts byte-for-byte:
    ///   - `MIN(#salary)`, `MAX(#salary)`, `SUM(#salary)`,
    ///     `AVG(#salary)`, `COUNT(#salary)` for the non-DISTINCT forms.
    ///   - `COUNT(DISTINCT #salary)` for the DISTINCT form, where
    ///     DISTINCT now rides on
    ///     `AggregateFunctionParams::distinct: bool` (not a separate
    ///     enum variant) — same shape DataFusion uses.
    #[test]
    fn aggregate_function_display_matches_datafusion() {
        use crate::expressions::{avg, col, count, count_distinct, max, min, sum};

        assert_eq!(format!("{}", min(col("salary"))), "MIN(#salary)");
        assert_eq!(format!("{}", max(col("salary"))), "MAX(#salary)");
        assert_eq!(format!("{}", sum(col("salary"))), "SUM(#salary)");
        assert_eq!(format!("{}", avg(col("salary"))), "AVG(#salary)");
        assert_eq!(format!("{}", count(col("salary"))), "COUNT(#salary)");
        assert_eq!(
            format!("{}", count_distinct(col("salary"))),
            "COUNT(DISTINCT #salary)"
        );
    }
}
