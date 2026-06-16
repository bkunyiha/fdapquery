//!
//! `SUM(expr)` — running total of non-null values.
//!
//! ## Output type stability
//! Addition preserves the input type (`Int8 + Int8 → Int8`, wrapping on overflow),
//! which matches arrow's typed columns and avoids a per-row type change.
//! `Int32`/`Int64`/`Float64` — the common cases for summation — are unaffected.

use crate::aggregate_expression::AggregateExpression;
use crate::expressions::{Accumulator, AccumulatorValue, Expression};
use datatypes::ScalarValue;
use std::fmt;
use std::sync::Arc;

/// `SUM(expr)`.
pub struct SumExpression {
    expr: Arc<dyn Expression>,
}

impl SumExpression {
    pub fn new(expr: Arc<dyn Expression>) -> Self {
        Self { expr }
    }
}

impl AggregateExpression for SumExpression {
    fn input_expression(&self) -> Arc<dyn Expression> {
        Arc::clone(&self.expr)
    }
    fn create_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(SumAccumulator::new())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for SumExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SUM({})", self.expr)
    }
}

/// Keeps the running sum.
pub struct SumAccumulator {
    value: ScalarValue,
}

impl SumAccumulator {
    pub fn new() -> Self {
        Self {
            value: ScalarValue::Null,
        }
    }
}

impl Default for SumAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accumulator for SumAccumulator {
    fn accumulate(&mut self, value: &ScalarValue) {
        if value.is_null() {
            return;
        }
        if self.value.is_null() {
            self.value = value.clone();
        } else {
            self.value = scalar_add(&self.value, value);
        }
    }

    fn final_value(&self) -> ScalarValue {
        self.value.clone()
    }

    fn merge(&mut self, other: &AccumulatorValue) {
        // For SUM, merging a partial state is the same as accumulating it.
        if let AccumulatorValue::Scalar(v) = other {
            self.accumulate(v);
        }
    }
}

/// Add two same-typed numeric scalars (integers wrap on overflow). Panics on a
/// type SUM doesn't support.
fn scalar_add(a: &ScalarValue, b: &ScalarValue) -> ScalarValue {
    use ScalarValue::*;
    match (a, b) {
        (Int8(x), Int8(y)) => Int8(x.wrapping_add(*y)),
        (Int16(x), Int16(y)) => Int16(x.wrapping_add(*y)),
        (Int32(x), Int32(y)) => Int32(x.wrapping_add(*y)),
        (Int64(x), Int64(y)) => Int64(x.wrapping_add(*y)),
        (UInt8(x), UInt8(y)) => UInt8(x.wrapping_add(*y)),
        (UInt16(x), UInt16(y)) => UInt16(x.wrapping_add(*y)),
        (UInt32(x), UInt32(y)) => UInt32(x.wrapping_add(*y)),
        (UInt64(x), UInt64(y)) => UInt64(x.wrapping_add(*y)),
        (Float32(x), Float32(y)) => Float32(x + y),
        (Float64(x), Float64(y)) => Float64(x + y),
        _ => panic!("SUM is not implemented for type: {a:?}"),
    }
}
