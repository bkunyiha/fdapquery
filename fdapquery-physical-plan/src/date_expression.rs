//!
//! Add or subtract an interval (a whole number of days) to/from a date,
//! producing a date. Dates and day-intervals are both stored as integers (days
//! since the Unix epoch / a count of days), so the arithmetic is plain integer
//! add/subtract on the day counts, with a null in either operand yielding null.

use crate::expressions::{Expression, number_to_i64};
use datatypes::arrow_types::DATE_DAY_TYPE;
use datatypes::{ArrowVectorBuilder, ColumnVector, RecordBatch, ScalarValue};
use std::fmt;
use std::sync::Arc;

/// `date - interval` → date.
pub struct DateSubtractIntervalExpression {
    pub date_expr: Arc<dyn Expression>,
    pub interval_expr: Arc<dyn Expression>,
}

impl DateSubtractIntervalExpression {
    pub fn new(date_expr: Arc<dyn Expression>, interval_expr: Arc<dyn Expression>) -> Self {
        Self {
            date_expr,
            interval_expr,
        }
    }
}

impl Expression for DateSubtractIntervalExpression {
    fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        date_interval(&self.date_expr, &self.interval_expr, input, |d, i| d - i)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for DateSubtractIntervalExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} - {}", self.date_expr, self.interval_expr)
    }
}

/// `date + interval` → date.
pub struct DateAddIntervalExpression {
    pub date_expr: Arc<dyn Expression>,
    pub interval_expr: Arc<dyn Expression>,
}

impl DateAddIntervalExpression {
    pub fn new(date_expr: Arc<dyn Expression>, interval_expr: Arc<dyn Expression>) -> Self {
        Self {
            date_expr,
            interval_expr,
        }
    }
}

impl Expression for DateAddIntervalExpression {
    fn evaluate(&self, input: &RecordBatch) -> Box<dyn ColumnVector> {
        date_interval(&self.date_expr, &self.interval_expr, input, |d, i| d + i)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for DateAddIntervalExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}", self.date_expr, self.interval_expr)
    }
}

/// Shared evaluation: evaluate both sides, then apply `op` to the (i32) day counts
/// row by row, propagating nulls. Result is a `Date32` column.
fn date_interval(
    date_expr: &Arc<dyn Expression>,
    interval_expr: &Arc<dyn Expression>,
    input: &RecordBatch,
    op: impl Fn(i32, i32) -> i32,
) -> Box<dyn ColumnVector> {
    let date_col: Box<dyn ColumnVector> = date_expr.evaluate(input);
    let interval_col: Box<dyn ColumnVector> = interval_expr.evaluate(input);
    let mut builder = ArrowVectorBuilder::new(&DATE_DAY_TYPE, date_col.size());
    for i in 0..date_col.size() {
        let date_value = date_col.get_value(i);
        let interval_value = interval_col.get_value(i);
        if date_value.is_null() || interval_value.is_null() {
            builder.append_null();
        } else {
            let date_days = number_to_i64(&date_value) as i32;
            let interval_days = number_to_i64(&interval_value) as i32;
            builder.append_value(&ScalarValue::Date32(op(date_days, interval_days)));
        }
    }
    builder.set_value_count(date_col.size());
    Box::new(builder.build())
}
