//!
//! `SUM(expr)` — running total of non-null values.
//!
//! ## Output type stability
//! Addition preserves the input type (`Int8 + Int8 → Int8`, wrapping on overflow),
//! which matches arrow's typed columns and avoids a per-row type change.
//! `Int32`/`Int64`/`Float64` — the common cases for summation — are unaffected.

use crate::aggregate_expression::AggregateExpr;
use crate::expressions::{Accumulator, AccumulatorValue, PhysicalExpr};
use fdapquery_common::{FdapQueryError, Result, ScalarValue};
use std::fmt;
use std::sync::Arc;

/// `SUM(expr)`.
#[derive(Debug)]
pub struct SumExpr {
    expr: Arc<dyn PhysicalExpr>,
}

impl SumExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self { expr }
    }
}

impl AggregateExpr for SumExpr {
    fn input_expression(&self) -> Arc<dyn PhysicalExpr> {
        Arc::clone(&self.expr)
    }
    fn create_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(SumAccumulator::new())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for SumExpr {
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
    fn accumulate(&mut self, value: &ScalarValue) -> Result<()> {
        if value.is_null() {
            return Ok(());
        }
        if self.value.is_null() {
            self.value = value.clone();
        } else {
            self.value = scalar_add(&self.value, value)?;
        }
        Ok(())
    }

    fn final_value(&self) -> Result<ScalarValue> {
        Ok(self.value.clone())
    }

    fn merge(&mut self, other: &AccumulatorValue) -> Result<()> {
        // For SUM, merging a partial state is the same as accumulating it.
        if let AccumulatorValue::Scalar(v) = other {
            self.accumulate(v)?;
        }
        Ok(())
    }
}

/// Add two same-typed numeric scalars (integers wrap on overflow). A type SUM
/// doesn't support surfaces as `Err(NotImplemented(_))`.
fn scalar_add(a: &ScalarValue, b: &ScalarValue) -> Result<ScalarValue> {
    use ScalarValue::{Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64};
    Ok(match (a, b) {
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
        _ => {
            return Err(FdapQueryError::NotImplemented(format!(
                "SUM is not implemented for type: {a:?}"
            )));
        }
    })
}
