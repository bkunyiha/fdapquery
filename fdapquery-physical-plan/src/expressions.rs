//!
//! Home of the root [`Expression`] trait, the literal expressions, and the
//! [`Accumulator`] trait. Like [`crate::physical_plan::PhysicalPlan`], `Expression`
//! is a **trait** referenced through `Arc<dyn Expression>` rather than an enum
//! (the expression set is large and open in spirit).
//!
//! A physical expression evaluates against an input [`RecordBatch`] and produces
//! a whole column of output ([`ColumnVector`]) — it is the runtime counterpart of
//! a `logical_plan::LogicalExpr`.
//!
//! ## Typed values via `ScalarValue`
//! `Accumulator` and related traits exchange typed values via the
//! [`ScalarValue`] enum (with its own `Null` variant), rather than an
//! untyped boxed-`Any`.

use datatypes::arrow_types::{DATE_DAY_TYPE, DOUBLE_TYPE, INT64_TYPE, STRING_TYPE};
use datatypes::{ColumnVector, LiteralValueVector, RecordBatch, ScalarValue, record_batch};
use std::fmt;

/// Physical representation of an expression.
///
/// `Expression: fmt::Display` so that composite expressions (binary, cast) and
/// operators can print their operands. `Send + Sync` lets `Arc<dyn Expression>`
/// be shared with rayon workers in `ParallelContext` (see the `PhysicalPlan`
/// module note). It holds: every concrete expression stores only
/// `Arc<dyn Expression>` operands plus plain data.
pub trait Expression: fmt::Display + Send + Sync {
    /// Evaluate against an input record batch and produce a column of output.
    ///
    /// boxed trait object `Box<dyn ColumnVector>`.
    fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector>;

    /// Type-erased self-reference for runtime downcasting (see
    /// `PhysicalPlan::as_any` for the rationale). The protobuf serializer
    /// — the only caller that needs to branch on concrete expression type —
    /// uses `expr.as_any().downcast_ref::<ColumnExpression>()` etc., the same
    /// pattern DataFusion uses for `PhysicalExpr`. Each leaf `impl Expression`
    /// (column, the four literals, cast) overrides with
    /// `fn as_any(&self) -> &dyn Any { self }`.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Family-narrowing accessor for the boolean expression family. Returns
    /// `&dyn BooleanExpression` so the caller can read `op_name()`/`left()`/
    /// `right()` without further per-operator dispatch. Each of the eight
    /// boolean operators overrides this (via the `bool_expr!` macro). Kept
    /// even after the `as_any` migration because there is no concrete type
    /// that represents "any boolean op" — the dispatch is genuinely 1-to-N.
    fn as_boolean_expression(&self) -> Option<&dyn crate::BooleanExpression> {
        None
    }

    /// Family-narrowing accessor for the math expression family (Add/Sub/Mul/
    /// Div/Mod). Same rationale as `as_boolean_expression`.
    fn as_math_expression(&self) -> Option<&dyn crate::MathExpression> {
        None
    }
}

// ---------------------------------------------------------------------------
// Literal expressions — each evaluates to a LiteralValueVector that repeats the
// same constant for every row of the input batch.
// ---------------------------------------------------------------------------

/// A literal `i64`.
pub struct LiteralLongExpression {
    pub value: i64,
}

impl LiteralLongExpression {
    pub fn new(value: i64) -> Self {
        Self { value }
    }
}

impl Expression for LiteralLongExpression {
    fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        Box::new(LiteralValueVector::new(
            INT64_TYPE,
            ScalarValue::Int64(self.value),
            record_batch::row_count(input),
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for LiteralLongExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

/// A literal `f64`.
pub struct LiteralDoubleExpression {
    pub value: f64,
}

impl LiteralDoubleExpression {
    pub fn new(value: f64) -> Self {
        Self { value }
    }
}

impl Expression for LiteralDoubleExpression {
    fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        Box::new(LiteralValueVector::new(
            DOUBLE_TYPE,
            ScalarValue::Float64(self.value),
            record_batch::row_count(input),
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for LiteralDoubleExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

/// A literal string. Stored as `ScalarValue::Utf8(String)` under `STRING_TYPE`
/// so it compares directly against string columns read from a scan, which
/// also surface as `Utf8`.
pub struct LiteralStringExpression {
    pub value: String,
}

impl LiteralStringExpression {
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
        }
    }
}

impl Expression for LiteralStringExpression {
    fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        Box::new(LiteralValueVector::new(
            STRING_TYPE,
            ScalarValue::Utf8(self.value.clone()),
            record_batch::row_count(input),
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for LiteralStringExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "'{}'", self.value)
    }
}

/// A literal date stored as days since the Unix epoch
///.
pub struct LiteralDateExpression {
    pub days_since_epoch: i32,
}

impl LiteralDateExpression {
    pub fn new(days_since_epoch: i32) -> Self {
        Self { days_since_epoch }
    }
}

impl Expression for LiteralDateExpression {
    fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        Box::new(LiteralValueVector::new(
            DATE_DAY_TYPE,
            ScalarValue::Date32(self.days_since_epoch),
            record_batch::row_count(input),
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for LiteralDateExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.days_since_epoch)
    }
}

/// A literal interval expressed as a whole number of days, stored under
/// `INT64_TYPE` / `Int64`.
pub struct LiteralIntervalDaysExpression {
    pub days: i64,
}

impl LiteralIntervalDaysExpression {
    pub fn new(days: i64) -> Self {
        Self { days }
    }
}

impl Expression for LiteralIntervalDaysExpression {
    fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        Box::new(LiteralValueVector::new(
            INT64_TYPE,
            ScalarValue::Int64(self.days),
            record_batch::row_count(input),
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for LiteralIntervalDaysExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.days)
    }
}

// ---------------------------------------------------------------------------
// Accumulator — the per-key state object used by aggregate expressions
// (MIN/MAX/SUM/…). Concrete implementations live with each aggregate
// expression file.
// ---------------------------------------------------------------------------

/// The value an [`Accumulator`] exchanges during *partial* (distributed,
/// two-stage) aggregation. Almost every accumulator's intermediate value is a
/// single scalar (MIN/MAX/SUM keep their running value, COUNT keeps a running
/// count), but AVG must carry **both** a running sum and a count so the two
/// can be merged correctly in the final stage. This enum unions those two
/// shapes into a typed value flowing through `intermediate_value()` /
/// `merge()`.
///
/// Single-node (`AggregateMode::Complete`) aggregation never uses this type — it
/// only calls `accumulate` + `final_value`. It exists for the distributed path
/// (module 15), which is why the variants beyond `Scalar` aren't exercised yet.
#[derive(Debug, Clone, PartialEq)]
pub enum AccumulatorValue {
    /// A plain scalar — MIN/MAX/SUM/COUNT partial state.
    Scalar(ScalarValue),
    /// AVG's partial state.
    AvgState { sum: f64, count: i32 },
}

/// Running aggregation state.
///
/// `accumulate`/`merge` mutate the state, so they take `&mut self`;
/// `final_value`/`intermediate_value` only read it. The per-row input
/// (`accumulate`) and the final output (`final_value`) are always a single
/// [`ScalarValue`] (its `Null` variant stands in for the "no value yet" /
/// "result is null" case). The distributed-only `intermediate_value`/`merge`
/// traffic in [`AccumulatorValue`], which can also carry AVG's compound
/// (sum, count) state — the one place a scalar is insufficient.
pub trait Accumulator {
    /// Fold one input value into the running state.
    fn accumulate(&mut self, value: &ScalarValue);

    /// The final aggregate result.
    fn final_value(&self) -> ScalarValue;

    /// Intermediate state for partial (distributed) aggregation. Defaults to the
    /// final value wrapped as a scalar; only AVG (with its compound running
    /// sum + count state) overrides this.
    fn intermediate_value(&self) -> AccumulatorValue {
        AccumulatorValue::Scalar(self.final_value())
    }

    /// Merge another accumulator's intermediate value into this one — used in the
    /// final stage of distributed aggregation.
    fn merge(&mut self, other: &AccumulatorValue);
}

/// Coerce any numeric (or date) [`ScalarValue`] to `i64`, truncating floats.
/// Panics on a non-numeric value.
pub(crate) fn number_to_i64(v: &ScalarValue) -> i64 {
    match v {
        ScalarValue::Int8(n) => *n as i64,
        ScalarValue::Int16(n) => *n as i64,
        ScalarValue::Int32(n) => *n as i64,
        ScalarValue::Int64(n) => *n,
        ScalarValue::UInt8(n) => *n as i64,
        ScalarValue::UInt16(n) => *n as i64,
        ScalarValue::UInt32(n) => *n as i64,
        ScalarValue::UInt64(n) => *n as i64,
        ScalarValue::Float32(f) => *f as i64,
        ScalarValue::Float64(f) => *f as i64,
        ScalarValue::Date32(d) => *d as i64,
        other => panic!("expected a number, got {other:?}"),
    }
}

/// Coerce any numeric [`ScalarValue`] to `f64`. Panics on a non-numeric value.
pub(crate) fn number_to_f64(v: &ScalarValue) -> f64 {
    match v {
        ScalarValue::Int8(n) => *n as f64,
        ScalarValue::Int16(n) => *n as f64,
        ScalarValue::Int32(n) => *n as f64,
        ScalarValue::Int64(n) => *n as f64,
        ScalarValue::UInt8(n) => *n as f64,
        ScalarValue::UInt16(n) => *n as f64,
        ScalarValue::UInt32(n) => *n as f64,
        ScalarValue::UInt64(n) => *n as f64,
        ScalarValue::Float32(f) => *f as f64,
        ScalarValue::Float64(f) => *f,
        other => panic!("expected a number, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Shared ScalarValue extractors. Used by the math and boolean expression
// families to pull a typed value out of a `ScalarValue` after dispatching on
// the Arrow type. A wrong variant panics.
// ---------------------------------------------------------------------------

pub(crate) fn as_i8(v: &ScalarValue) -> i8 {
    match v {
        ScalarValue::Int8(x) => *x,
        other => panic!("expected Int8, got {other:?}"),
    }
}

pub(crate) fn as_i16(v: &ScalarValue) -> i16 {
    match v {
        ScalarValue::Int16(x) => *x,
        other => panic!("expected Int16, got {other:?}"),
    }
}

pub(crate) fn as_i32(v: &ScalarValue) -> i32 {
    match v {
        ScalarValue::Int32(x) => *x,
        other => panic!("expected Int32, got {other:?}"),
    }
}

pub(crate) fn as_i64(v: &ScalarValue) -> i64 {
    match v {
        ScalarValue::Int64(x) => *x,
        other => panic!("expected Int64, got {other:?}"),
    }
}

pub(crate) fn as_f32(v: &ScalarValue) -> f32 {
    match v {
        ScalarValue::Float32(x) => *x,
        other => panic!("expected Float32, got {other:?}"),
    }
}

pub(crate) fn as_f64(v: &ScalarValue) -> f64 {
    match v {
        ScalarValue::Float64(x) => *x,
        other => panic!("expected Float64, got {other:?}"),
    }
}

// Note: there is no `as_date` here. The boolean comparison family's Date32 arm uses
// the null-aware `as_opt_date` in `boolean_expression.rs`; the math family has no
// date arm, so a panicking `as_date` extractor has no remaining caller.
