//!
//! `AVG(expr)` — mean of non-null values, returned as `Float64`. Unlike the other
//! aggregates, AVG's *intermediate* state is compound (a running sum **and** a
//! count) so partial averages can be merged correctly in distributed execution —
//! this is the one accumulator that overrides `intermediate_value` to return an
//! [`AccumulatorValue::AvgState`] rather than a scalar.

use crate::aggregate_expression::AggregateExpression;
use crate::expressions::{Accumulator, AccumulatorValue, Expression, number_to_f64};
use datatypes::ScalarValue;
use std::fmt;
use std::sync::Arc;

/// `AVG(expr)`.
pub struct AvgExpression {
    expr: Arc<dyn Expression>,
}

impl AvgExpression {
    pub fn new(expr: Arc<dyn Expression>) -> Self {
        Self { expr }
    }
}

impl AggregateExpression for AvgExpression {
    fn input_expression(&self) -> Arc<dyn Expression> {
        Arc::clone(&self.expr)
    }
    fn create_accumulator(&self) -> Box<dyn Accumulator> {
        Box::new(AvgAccumulator::new())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for AvgExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AVG({})", self.expr)
    }
}

/// Tracks running `sum` and `count`.
pub struct AvgAccumulator {
    sum: f64,
    count: i32,
}

impl AvgAccumulator {
    pub fn new() -> Self {
        Self { sum: 0.0, count: 0 }
    }
}

impl Default for AvgAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accumulator for AvgAccumulator {
    fn accumulate(&mut self, value: &ScalarValue) {
        if !value.is_null() {
            self.count += 1;
            self.sum += number_to_f64(value);
        }
    }

    fn final_value(&self) -> ScalarValue {
        // Empty group: null. Otherwise: sum / count.
        if self.count == 0 {
            ScalarValue::Null
        } else {
            ScalarValue::Float64(self.sum / self.count as f64)
        }
    }

    fn intermediate_value(&self) -> AccumulatorValue {
        // `AccumulatorValue` has no null variant; an empty group is represented
        // as a null scalar — the same observable "no partial state".
        if self.count == 0 {
            AccumulatorValue::Scalar(ScalarValue::Null)
        } else {
            AccumulatorValue::AvgState {
                sum: self.sum,
                count: self.count,
            }
        }
    }

    fn merge(&mut self, other: &AccumulatorValue) {
        // Merge sum and count separately from an `AvgState`.
        match other {
            AccumulatorValue::AvgState { sum, count } => {
                self.sum += sum;
                self.count += count;
            }
            // A null partial (empty group) contributes nothing.
            AccumulatorValue::Scalar(ScalarValue::Null) => {}
            other => panic!("Cannot merge AVG with: {other:?}"),
        }
    }
}
