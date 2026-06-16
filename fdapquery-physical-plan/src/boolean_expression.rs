//!
//! Comparison and logical operators that always produce a `Boolean` column:
//! `AND`, `OR`, `=`, `!=`, `<`, `<=`, `>`, `>=`.
//!
//! ## Trait with a default method
//! As with [`crate::binary_expression::BinaryExpression`], `BooleanExpression`
//! is a trait with a default method. Unlike the math family, `BooleanExpression`
//! extends [`Expression`] directly (not `BinaryExpression`): comparison does
//! *not* coerce numeric types — it requires the two sides to already share a
//! type and panics otherwise. The default
//! [`BooleanExpression::evaluate_boolean`] holds the shared "evaluate both
//! sides, build a Boolean column cell-by-cell" logic; each concrete operator
//! supplies only its per-cell predicate via `compare_value`.
//!
//! ## The `compare_typed!` macro
//! The six comparison operators (`==`, `!=`, `<`, `<=`, `>`, `>=`) each need a
//! near-identical per-type match arm. [`compare_typed!`] is a small
//! `macro_rules!` macro parameterised by the operator token:
//! `compare_typed!(l, r, t, >=)` expands to the full per-type match applying
//! `>=` to the typed values pulled out of each `ScalarValue`. Using the real
//! Rust operator per type also gives the correct `NaN` behaviour for free
//! (`NaN >= x` is `false`).
//!
//! ## `Option<bool>` three-valued logic
//! arrow-rs typed inference reads an empty CSV field as a genuine
//! `ScalarValue::Null`, so the comparison must answer for nulls. SQL's
//! three-valued logic is modelled directly: a comparison returns
//! `Option<bool>`, where `None` is SQL `UNKNOWN` (the result of comparing
//! against `NULL`). `None` propagates through every operator the way `NULL`
//! does in SQL, `AND`/`OR` use the Kleene truth tables ([`and3`]/[`or3`]),
//! and `evaluate_boolean` writes an `UNKNOWN` out as a `ScalarValue::Null`
//! cell — so the output is a *nullable* Boolean column, exactly what a real
//! SQL engine produces. `SelectionExec` keeps only `Some(true)` rows, so both
//! `FALSE` and `UNKNOWN` correctly drop a row from a `WHERE` clause.

use crate::expressions::Expression;
use arrow_schema::DataType;
use datatypes::arrow_types::BOOLEAN_TYPE;
use datatypes::{ArrowVectorBuilder, ColumnVector, RecordBatch, ScalarValue};
use std::sync::Arc;

/// A boolean (comparison or logical) binary expression.
pub trait BooleanExpression: Expression {
    /// The left operand expression.
    fn left(&self) -> &Arc<dyn Expression>;
    /// The right operand expression.
    fn right(&self) -> &Arc<dyn Expression>;

    /// The per-cell predicate. Returns `Option<bool>`: `Some(b)` for a definite
    /// result, `None` for SQL `UNKNOWN` (produced whenever an operand is
    /// `NULL`). See the module note on three-valued logic.
    fn compare_value(
        &self,
        l: &ScalarValue,
        r: &ScalarValue,
        arrow_type: &DataType,
    ) -> Option<bool>;

    /// Wire-format operator name (`"eq"`, `"and"`, …). Used by
    /// `protobuf::serialize_physical_expr` to serialise this expression as a
    /// `pb::PhysicalBinaryExprNode` with the matching `op` string.
    fn op_name(&self) -> &'static str;

    /// Template method: evaluate both sides, require equal lengths and
    /// identical types, then build a *nullable* `Boolean` column by applying
    /// [`compare_value`](Self::compare_value) cell-by-cell. A `None` (SQL
    /// `UNKNOWN`) result is written as a null cell.
    fn evaluate_boolean(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        let ll = self.left().evaluate(input);
        let rr = self.right().evaluate(input);
        assert_eq!(ll.size(), rr.size());
        if ll.get_type() != rr.get_type() {
            panic!(
                "Cannot compare values of different type: {:?} != {:?}",
                ll.get_type(),
                rr.get_type()
            );
        }
        let arrow_type = ll.get_type();
        let mut builder = ArrowVectorBuilder::new(&BOOLEAN_TYPE, ll.size());
        for i in 0..ll.size() {
            match self.compare_value(&ll.get_value(i), &rr.get_value(i), &arrow_type) {
                Some(b) => builder.append_value(&ScalarValue::Boolean(b)),
                None => builder.append_value(&ScalarValue::Null), // SQL UNKNOWN → null cell
            }
        }
        builder.set_value_count(ll.size());
        Box::new(builder.build())
    }
}

/// Expand to a per-type `match` that applies the comparison operator `$op` to the
/// two operands, pulling the typed value out of each `ScalarValue`. Each arm
/// goes through
/// [`cmp_opt`], so a `NULL` operand yields `None` (SQL `UNKNOWN`) instead of
/// panicking; the result type is `Option<bool>`.
macro_rules! compare_typed {
    ($l:expr, $r:expr, $t:expr, $op:tt) => {
        match $t {
            DataType::Int8    => cmp_opt(as_opt_i8($l),   as_opt_i8($r),   |a, b| a $op b),
            DataType::Int16   => cmp_opt(as_opt_i16($l),  as_opt_i16($r),  |a, b| a $op b),
            DataType::Int32   => cmp_opt(as_opt_i32($l),  as_opt_i32($r),  |a, b| a $op b),
            DataType::Int64   => cmp_opt(as_opt_i64($l),  as_opt_i64($r),  |a, b| a $op b),
            DataType::Float32 => cmp_opt(as_opt_f32($l),  as_opt_f32($r),  |a, b| a $op b),
            DataType::Float64 => cmp_opt(as_opt_f64($l),  as_opt_f64($r),  |a, b| a $op b),
            DataType::Utf8    => cmp_opt(as_opt_str($l),  as_opt_str($r),  |a, b| a $op b),
            DataType::Date32  => cmp_opt(as_opt_date($l), as_opt_date($r), |a, b| a $op b),
            other => panic!("Unsupported data type in comparison expression: {other:?}"),
        }
    };
}

/// Apply a comparison only when both operands are present. A `NULL` (absent)
/// operand makes the whole comparison `None` — SQL `UNKNOWN`. This is the single
/// chokepoint that turns Rust `Option` handling into SQL null-propagation.
fn cmp_opt<T>(l: Option<T>, r: Option<T>, f: impl FnOnce(T, T) -> bool) -> Option<bool> {
    match (l, r) {
        (Some(l), Some(r)) => Some(f(l, r)),
        _ => None, // any NULL operand → UNKNOWN
    }
}

// Null-aware extractors for the comparison family. Unlike the panicking `as_i32`
// etc. in `expressions.rs` (shared with the math family, which has no null
// semantics), these return `None` for `ScalarValue::Null` so the comparison can
// propagate SQL UNKNOWN. A wrong *non-null* variant still panics, matching the
// "types already coerced equal" contract of `evaluate_boolean`.

fn as_opt_i8(v: &ScalarValue) -> Option<i8> {
    match v {
        ScalarValue::Null => None,
        ScalarValue::Int8(x) => Some(*x),
        other => panic!("expected Int8, got {other:?}"),
    }
}
fn as_opt_i16(v: &ScalarValue) -> Option<i16> {
    match v {
        ScalarValue::Null => None,
        ScalarValue::Int16(x) => Some(*x),
        other => panic!("expected Int16, got {other:?}"),
    }
}
fn as_opt_i32(v: &ScalarValue) -> Option<i32> {
    match v {
        ScalarValue::Null => None,
        ScalarValue::Int32(x) => Some(*x),
        other => panic!("expected Int32, got {other:?}"),
    }
}
fn as_opt_i64(v: &ScalarValue) -> Option<i64> {
    match v {
        ScalarValue::Null => None,
        ScalarValue::Int64(x) => Some(*x),
        other => panic!("expected Int64, got {other:?}"),
    }
}
fn as_opt_f32(v: &ScalarValue) -> Option<f32> {
    match v {
        ScalarValue::Null => None,
        ScalarValue::Float32(x) => Some(*x),
        other => panic!("expected Float32, got {other:?}"),
    }
}
fn as_opt_f64(v: &ScalarValue) -> Option<f64> {
    match v {
        ScalarValue::Null => None,
        ScalarValue::Float64(x) => Some(*x),
        other => panic!("expected Float64, got {other:?}"),
    }
}
fn as_opt_date(v: &ScalarValue) -> Option<i32> {
    match v {
        ScalarValue::Null => None,
        ScalarValue::Date32(x) => Some(*x),
        other => panic!("expected Date32, got {other:?}"),
    }
}

/// Borrow a string-typed cell as `&str`, or `None` if the cell is null. No
/// allocation: `Utf8` borrows its `String`, `Binary` is validated in place
/// (invalid UTF-8 → `None`).
fn as_opt_str(v: &ScalarValue) -> Option<&str> {
    match v {
        ScalarValue::Null => None,
        ScalarValue::Utf8(s) => Some(s.as_str()),
        ScalarValue::Binary(b) => std::str::from_utf8(b).ok(),
        other => panic!("expected Utf8/Binary, got {other:?}"),
    }
}

/// Read a cell as a three-valued boolean: `None` for `NULL`, `Some(b)` for a
/// boolean, and `Some(n == 1)` for an integer. Used by the logical `AND`/`OR`
/// operators. In practice their operands are already Boolean columns (the
/// results of comparisons), which may now be null.
fn as_opt_bool(v: &ScalarValue) -> Option<bool> {
    match v {
        ScalarValue::Null => None,
        ScalarValue::Boolean(b) => Some(*b),
        ScalarValue::Int8(n) => Some(*n == 1),
        ScalarValue::Int16(n) => Some(*n == 1),
        ScalarValue::Int32(n) => Some(*n == 1),
        ScalarValue::Int64(n) => Some(*n == 1),
        ScalarValue::UInt8(n) => Some(*n == 1),
        ScalarValue::UInt16(n) => Some(*n == 1),
        ScalarValue::UInt32(n) => Some(*n == 1),
        ScalarValue::UInt64(n) => Some(*n == 1),
        other => panic!("Cannot convert {other:?} to bool"),
    }
}

/// SQL Kleene `AND`: `FALSE` dominates (so `FALSE AND NULL = FALSE`), `TRUE AND
/// TRUE = TRUE`, and any other combination involving `NULL` is `UNKNOWN`.
fn and3(l: Option<bool>, r: Option<bool>) -> Option<bool> {
    match (l, r) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

/// SQL Kleene `OR`: `TRUE` dominates (so `TRUE OR NULL = TRUE`), `FALSE OR FALSE
/// = FALSE`, and any other combination involving `NULL` is `UNKNOWN`.
fn or3(l: Option<bool>, r: Option<bool>) -> Option<bool> {
    match (l, r) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

/// Generate a concrete boolean operator: a struct holding `l`/`r`, its
/// `BooleanExpression` impl (`compare_value` is the per-cell body `$body`), the
/// trivial `Expression` delegate to `evaluate_boolean`, and a `Display` impl
/// rendering `"l <sym> r"`. Each operator becomes a tiny struct overriding one
/// method.
macro_rules! boolean_op {
    // `$proto_op` is the wire-format operator name used by
    // `protobuf::serialize_physical_expr` (e.g. "eq", "neq", "and"). It is
    // distinct from `$sym` (the Display symbol like "=", "!=", "AND").
    ($name:ident, $sym:literal, $proto_op:literal,
     |$l:ident, $r:ident, $t:ident| $body:expr) => {
        #[doc = concat!("`l ", $sym, " r`.")]
        pub struct $name {
            l: Arc<dyn Expression>,
            r: Arc<dyn Expression>,
        }

        impl $name {
            pub fn new(l: Arc<dyn Expression>, r: Arc<dyn Expression>) -> Self {
                Self { l, r }
            }
        }

        impl BooleanExpression for $name {
            fn left(&self) -> &Arc<dyn Expression> {
                &self.l
            }
            fn right(&self) -> &Arc<dyn Expression> {
                &self.r
            }
            fn compare_value(
                &self,
                $l: &ScalarValue,
                $r: &ScalarValue,
                $t: &DataType,
            ) -> Option<bool> {
                $body
            }
            fn op_name(&self) -> &'static str {
                $proto_op
            }
        }

        impl Expression for $name {
            fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
                self.evaluate_boolean(input)
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn as_boolean_expression(&self) -> Option<&dyn BooleanExpression> {
                Some(self)
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{} {} {}", self.l, $sym, self.r)
            }
        }
    };
}

// AND / OR ignore the Arrow type and operate on the truthiness of each side,
// using SQL Kleene three-valued logic so a NULL operand propagates correctly.
boolean_op!(AndExpression, "AND", "and", |l, r, _t| and3(
    as_opt_bool(l),
    as_opt_bool(r)
));
boolean_op!(OrExpression, "OR", "or", |l, r, _t| or3(
    as_opt_bool(l),
    as_opt_bool(r)
));

// Comparisons dispatch on the (shared) Arrow type via `compare_typed!`.
boolean_op!(
    EqExpression,
    "=",
    "eq",
    |l, r, t| compare_typed!(l, r, t, ==)
);
boolean_op!(
    NeqExpression,
    "!=",
    "neq",
    |l, r, t| compare_typed!(l, r, t, !=)
);
boolean_op!(
    LtExpression,
    "<",
    "lt",
    |l, r, t| compare_typed!(l, r, t, <)
);
boolean_op!(
    LtEqExpression,
    "<=",
    "lteq",
    |l, r, t| compare_typed!(l, r, t, <=)
);
boolean_op!(
    GtExpression,
    ">",
    "gt",
    |l, r, t| compare_typed!(l, r, t, >)
);
boolean_op!(
    GtEqExpression,
    ">=",
    "gteq",
    |l, r, t| compare_typed!(l, r, t, >=)
);

#[cfg(test)]
mod tests {
    //!
    //! These tests build the `RecordBatch` directly from typed arrow arrays.
    //! The `fuzzer` crate covered in module 9 is not yet implemented; once it
    //! is, the same tests will be rewritten to use it. Each test compares the
    //! operator's output against Rust's own `>=` over the same values, so the
    //! assertion is self-consistent regardless of the concrete numbers chosen.
    use super::*;
    use crate::column_expression::ColumnExpression;
    use arrow_array::{
        ArrayRef, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, StringArray,
    };
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use datatypes::RecordBatch;
    use std::sync::Arc;

    /// Build a two-column batch ("a", "b") of the same type.
    fn batch2(t: DataType, a: ArrayRef, b: ArrayRef) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", t.clone(), true),
            ArrowField::new("b", t, true),
        ]));
        RecordBatch::try_new(schema, vec![a, b]).unwrap()
    }

    fn gteq(batch: &RecordBatch) -> Box<dyn ColumnVector> {
        let expr = GtEqExpression::new(
            Arc::new(ColumnExpression::new(0)),
            Arc::new(ColumnExpression::new(1)),
        );
        expr.evaluate(batch)
    }

    #[test]
    fn gteq_bytes() {
        let a: Vec<i8> = vec![10, 20, 30, i8::MIN, i8::MAX];
        let b: Vec<i8> = vec![10, 30, 20, i8::MAX, i8::MIN];
        let batch = batch2(
            DataType::Int8,
            Arc::new(Int8Array::from(a.clone())),
            Arc::new(Int8Array::from(b.clone())),
        );
        let result = gteq(&batch);
        assert_eq!(result.size(), a.len());
        for i in 0..result.size() {
            assert_eq!(result.get_value(i), ScalarValue::Boolean(a[i] >= b[i]));
        }
    }

    #[test]
    fn gteq_shorts() {
        let a: Vec<i16> = vec![111, 222, 333, i16::MIN, i16::MAX];
        let b: Vec<i16> = vec![111, 333, 222, i16::MAX, i16::MIN];
        let batch = batch2(
            DataType::Int16,
            Arc::new(Int16Array::from(a.clone())),
            Arc::new(Int16Array::from(b.clone())),
        );
        let result = gteq(&batch);
        assert_eq!(result.size(), a.len());
        for i in 0..result.size() {
            assert_eq!(result.get_value(i), ScalarValue::Boolean(a[i] >= b[i]));
        }
    }

    #[test]
    fn gteq_ints() {
        let a: Vec<i32> = vec![111, 222, 333, i32::MIN, i32::MAX];
        let b: Vec<i32> = vec![111, 333, 222, i32::MAX, i32::MIN];
        let batch = batch2(
            DataType::Int32,
            Arc::new(Int32Array::from(a.clone())),
            Arc::new(Int32Array::from(b.clone())),
        );
        let result = gteq(&batch);
        assert_eq!(result.size(), a.len());
        for i in 0..result.size() {
            assert_eq!(result.get_value(i), ScalarValue::Boolean(a[i] >= b[i]));
        }
    }

    #[test]
    fn gteq_longs() {
        let a: Vec<i64> = vec![111, 222, 333, i64::MIN, i64::MAX];
        let b: Vec<i64> = vec![111, 333, 222, i64::MAX, i64::MIN];
        let batch = batch2(
            DataType::Int64,
            Arc::new(Int64Array::from(a.clone())),
            Arc::new(Int64Array::from(b.clone())),
        );
        let result = gteq(&batch);
        assert_eq!(result.size(), a.len());
        for i in 0..result.size() {
            assert_eq!(result.get_value(i), ScalarValue::Boolean(a[i] >= b[i]));
        }
    }

    #[test]
    fn gteq_doubles() {
        // Includes NaN: `NaN >= x` is false on both sides, so the operator's
        // output and Rust's own `>=` agree.
        let a: Vec<f64> = vec![0.0, 1.0, f64::MIN_POSITIVE, f64::MAX, f64::NAN];
        let b: Vec<f64> = a.iter().copied().rev().collect();
        let batch = batch2(
            DataType::Float64,
            Arc::new(Float64Array::from(a.clone())),
            Arc::new(Float64Array::from(b.clone())),
        );
        let result = gteq(&batch);
        assert_eq!(result.size(), a.len());
        for i in 0..result.size() {
            assert_eq!(result.get_value(i), ScalarValue::Boolean(a[i] >= b[i]));
        }
    }

    #[test]
    fn eq_with_null_string_is_unknown_not_panic() {
        // Regression for the execution-module integration: arrow reads an empty CSV
        // field as a null cell, so `state = 'CO'` compares a `ScalarValue::Null`.
        // SQL three-valued logic: NULL = 'CO' is UNKNOWN, written as a null cell —
        // not a panic, and not `false`. The WHERE filter drops it either way.
        use crate::expressions::LiteralStringExpression;
        let a: Vec<Option<&str>> = vec![Some("CO"), None, Some("CA")];
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "state",
            DataType::Utf8,
            true,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(a)) as ArrayRef]).unwrap();
        let expr = EqExpression::new(
            Arc::new(ColumnExpression::new(0)),
            Arc::new(LiteralStringExpression::new("CO".to_string())),
        );
        let result = expr.evaluate(&batch);
        assert_eq!(result.get_value(0), ScalarValue::Boolean(true)); // "CO" == "CO"
        assert_eq!(result.get_value(1), ScalarValue::Null); // NULL = 'CO' -> UNKNOWN
        assert_eq!(result.get_value(2), ScalarValue::Boolean(false)); // "CA" != "CO"
    }

    #[test]
    fn neq_with_null_string_is_unknown_not_true() {
        // SQL says NULL != 'CO' is UNKNOWN (null cell) -> drop. A previous
        // implementation that stringified both sides treated `null != 'CO'` as
        // `"null" != "CO"` == true, which wrongly KEPT a null row in
        // `WHERE state != 'CO'`.
        use crate::expressions::LiteralStringExpression;
        let a: Vec<Option<&str>> = vec![Some("CO"), None, Some("CA")];
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "state",
            DataType::Utf8,
            true,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(a)) as ArrayRef]).unwrap();
        let expr = NeqExpression::new(
            Arc::new(ColumnExpression::new(0)),
            Arc::new(LiteralStringExpression::new("CO".to_string())),
        );
        let result = expr.evaluate(&batch);
        assert_eq!(result.get_value(0), ScalarValue::Boolean(false)); // "CO" != "CO"
        assert_eq!(result.get_value(1), ScalarValue::Null); // NULL != 'CO' -> UNKNOWN
        assert_eq!(result.get_value(2), ScalarValue::Boolean(true)); // "CA" != "CO"
    }

    #[test]
    fn numeric_compare_with_null_is_unknown_not_panic() {
        // The numeric path used to panic on a null. Under three-valued logic
        // it must yield UNKNOWN instead.
        let a: Vec<Option<i32>> = vec![Some(5), None, Some(20)];
        let b: Vec<Option<i32>> = vec![Some(10), Some(10), Some(10)];
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("b", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(a)) as ArrayRef,
                Arc::new(Int32Array::from(b)) as ArrayRef,
            ],
        )
        .unwrap();
        let result = GtExpression::new(
            Arc::new(ColumnExpression::new(0)),
            Arc::new(ColumnExpression::new(1)),
        )
        .evaluate(&batch);
        assert_eq!(result.get_value(0), ScalarValue::Boolean(false)); // 5 > 10
        assert_eq!(result.get_value(1), ScalarValue::Null); // NULL > 10 -> UNKNOWN
        assert_eq!(result.get_value(2), ScalarValue::Boolean(true)); // 20 > 10
    }

    #[test]
    fn and_or_kleene_three_valued_logic() {
        // Verify the Kleene truth tables directly against the helpers.
        let (t, f, n) = (Some(true), Some(false), None);
        // AND: FALSE dominates; NULL only when no FALSE and at least one NULL.
        assert_eq!(and3(f, n), Some(false)); // FALSE AND NULL = FALSE
        assert_eq!(and3(t, n), None); //         TRUE  AND NULL = UNKNOWN
        assert_eq!(and3(n, n), None); //         NULL  AND NULL = UNKNOWN
        assert_eq!(and3(t, t), Some(true));
        // OR: TRUE dominates; NULL only when no TRUE and at least one NULL.
        assert_eq!(or3(t, n), Some(true)); //    TRUE  OR  NULL = TRUE
        assert_eq!(or3(f, n), None); //          FALSE OR  NULL = UNKNOWN
        assert_eq!(or3(f, f), Some(false));
    }

    #[test]
    fn gteq_strings() {
        let a = vec!["aaa", "bbb", "ccc"];
        let b = vec!["aaa", "ccc", "bbb"];
        let batch = batch2(
            DataType::Utf8,
            Arc::new(StringArray::from(a.clone())),
            Arc::new(StringArray::from(b.clone())),
        );
        let result = gteq(&batch);
        assert_eq!(result.size(), a.len());
        for i in 0..result.size() {
            assert_eq!(result.get_value(i), ScalarValue::Boolean(a[i] >= b[i]));
        }
    }
}
