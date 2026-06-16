//!
//! `COUNT(expr)` — number of non-null values. Always returns an `Int32` (the
//! count is `0` for an empty/all-null group, never null).

use crate::aggregate_expression::AggregateExpression;
use crate::expressions::{Accumulator, AccumulatorValue, Expression, number_to_i64};
use datatypes::ScalarValue;
use std::fmt;
use std::sync::Arc;

/// `COUNT(expr)`.
pub struct CountExpression {
    expr: Arc<dyn Expression>,
}

impl CountExpression {
    pub fn new(expr: Arc<dyn Expression>) -> Self {
        Self { expr }
    }
}

impl AggregateExpression for CountExpression {
    fn input_expression(&self) -> Arc<dyn Expression> {
        Arc::clone(&self.expr)
    }
    fn create_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(CountAccumulator::new())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for CountExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "COUNT({})", self.expr)
    }
}

/// Counts non-null values.
pub struct CountAccumulator {
    count: i32,
}

impl CountAccumulator {
    pub fn new() -> Self {
        Self { count: 0 }
    }
}

impl Default for CountAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accumulator for CountAccumulator {
    fn accumulate(&mut self, value: &ScalarValue) {
        if !value.is_null() {
            self.count += 1;
        }
    }

    fn final_value(&self) -> ScalarValue {
        ScalarValue::Int32(self.count)
    }

    fn merge(&mut self, other: &AccumulatorValue) {
        // COUNT merges by adding the partial counts together.
        if let AccumulatorValue::Scalar(s) = other {
            if !s.is_null() {
                self.count += number_to_i64(s) as i32;
            }
        }
    }
}
