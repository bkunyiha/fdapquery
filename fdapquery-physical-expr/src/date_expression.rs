//!
//! Add or subtract an interval (a whole number of days) to/from a date,
//! producing a date. Dates and day-intervals are both stored as integers (days
//! since the Unix epoch / a count of days), so the arithmetic is plain integer
//! add/subtract on the day counts, with a null in either operand yielding null.

use crate::columnar_value::ColumnarValue;
use crate::expressions::{PhysicalExpr, number_to_i64};
use arrow_schema::{DataType, Schema};
use fdapquery_common::{ArrowVectorBuilder, Result, ScalarValue};
use fdapquery_datatypes::{RecordBatch, record_batch};
use std::fmt;
use std::sync::Arc;

/// `date - interval` → date.
#[derive(Debug)]
pub struct DateSubtractIntervalExpr {
    pub date_expr: Arc<dyn PhysicalExpr>,
    pub interval_expr: Arc<dyn PhysicalExpr>,
}

impl DateSubtractIntervalExpr {
    pub fn new(date_expr: Arc<dyn PhysicalExpr>, interval_expr: Arc<dyn PhysicalExpr>) -> Self {
        Self {
            date_expr,
            interval_expr,
        }
    }
}

impl PhysicalExpr for DateSubtractIntervalExpr {
    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        date_interval(&self.date_expr, &self.interval_expr, batch, |d, i| d - i)
    }

    /// `date - interval(days)` → `Date32`. fdapquery only models the
    /// `Date32 - Days → Date32` form (the runtime builds a `Date32` result
    /// column in `date_interval`), so the data type is always `Date32`
    /// regardless of the input schema. DataFusion's general date-arithmetic
    /// path runs through `BinaryExpr` + `BinaryTypeCoercer`; the
    /// fdapquery-specific `DateSubtractIntervalExpr` is the narrower
    /// strict-mirror shape, and `Date32` matches DataFusion's result type
    /// for the `Date32 - Interval(YearMonth/DayTime) → Date32` arm.
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Date32)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for DateSubtractIntervalExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} - {}", self.date_expr, self.interval_expr)
    }
}

/// `date + interval` → date.
#[derive(Debug)]
pub struct DateAddIntervalExpr {
    pub date_expr: Arc<dyn PhysicalExpr>,
    pub interval_expr: Arc<dyn PhysicalExpr>,
}

impl DateAddIntervalExpr {
    pub fn new(date_expr: Arc<dyn PhysicalExpr>, interval_expr: Arc<dyn PhysicalExpr>) -> Self {
        Self {
            date_expr,
            interval_expr,
        }
    }
}

impl PhysicalExpr for DateAddIntervalExpr {
    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        date_interval(&self.date_expr, &self.interval_expr, batch, |d, i| d + i)
    }

    /// `date + interval(days)` → `Date32`. Symmetric counterpart of
    /// [`DateSubtractIntervalExpr::data_type`] above; the runtime builds a
    /// `Date32` result column in `date_interval`.
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Date32)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl fmt::Display for DateAddIntervalExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}", self.date_expr, self.interval_expr)
    }
}

/// Shared evaluation: evaluate both sides, then apply `op` to the (i32) day counts
/// row by row, propagating nulls. Result is a `Date32` column.
fn date_interval(
    date_expr: &Arc<dyn PhysicalExpr>,
    interval_expr: &Arc<dyn PhysicalExpr>,
    batch: &RecordBatch,
    op: impl Fn(i32, i32) -> i32,
) -> Result<ColumnarValue> {
    let num_rows = record_batch::row_count(batch);
    let date_col = date_expr.evaluate(batch)?.into_array(num_rows)?;
    let interval_col = interval_expr.evaluate(batch)?.into_array(num_rows)?;
    let mut builder = ArrowVectorBuilder::new(&arrow_schema::DataType::Date32, date_col.len());
    for i in 0..date_col.len() {
        let date_value = ScalarValue::try_from_array(&date_col, i)?;
        let interval_value = ScalarValue::try_from_array(&interval_col, i)?;
        if date_value.is_null() || interval_value.is_null() {
            builder.append_null();
        } else {
            let date_days = number_to_i64(&date_value)? as i32;
            let interval_days = number_to_i64(&interval_value)? as i32;
            builder.append_value(&ScalarValue::Date32(op(date_days, interval_days)));
        }
    }
    Ok(ColumnarValue::Array(builder.build()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expressions::Literal;
    use fdapquery_common::ScalarValue;

    /// Both date-arithmetic expressions produce a `Date32` column —
    /// `data_type` returns that unconditionally (no schema dependency).
    #[test]
    fn data_type_is_date32() {
        let schema = arrow_schema::Schema::empty();
        let date = Arc::new(Literal::new(ScalarValue::Date32(0))) as Arc<dyn PhysicalExpr>;
        let interval = Arc::new(Literal::new(ScalarValue::Int32(1))) as Arc<dyn PhysicalExpr>;

        let sub = DateSubtractIntervalExpr::new(date.clone(), interval.clone());
        assert_eq!(sub.data_type(&schema).unwrap(), DataType::Date32);

        let add = DateAddIntervalExpr::new(date, interval);
        assert_eq!(add.data_type(&schema).unwrap(), DataType::Date32);
    }
}
