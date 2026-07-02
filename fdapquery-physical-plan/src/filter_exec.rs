//!
//! Filters rows: evaluates a boolean predicate against each batch and keeps only
//! the rows where it is true. The schema is unchanged.
//!
//! ## Reading the filter vector
//! Filter stays at the `ColumnVector` abstraction: the predicate evaluates to
//! a boolean column, and each row is read as `ScalarValue::Boolean(true)`. No
//! downcast is needed, and it keeps the operator working against any
//! `ColumnVector` implementation.
//!
//! ## Strict mirror of DataFusion's `FilterExec`
//! Struct field order, constructor signature, and `DisplayAs::fmt_as` output
//! match `datafusion::physical_plan::filter::FilterExec` byte-for-byte. The
//! omitted fields (`metrics`, `cache` → `properties`, `default_selectivity`,
//! `projection`, `batch_size`, `fetch`) reflect infrastructure not yet built
//! in fdapquery — they are planned follow-ups, not divergences.

use crate::PhysicalExpr;
use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use arrow_array::ArrayRef;
use fdapquery_common::{ArrowVectorBuilder, FdapQueryError, Result, ScalarValue};
use fdapquery_datatypes::{Schema, record_batch};
use fdapquery_execution::TaskContext;
use futures::StreamExt;
use std::sync::Arc;

/// `FilterExec` evaluates a boolean predicate against all input batches to
/// determine which rows to include in its output batches. Strict mirror of
/// `datafusion::physical_plan::filter::FilterExec`.
#[derive(Debug)]
pub struct FilterExec {
    /// The expression to filter on. This expression must evaluate to a
    /// boolean value.
    predicate: Arc<dyn PhysicalExpr>,
    /// The input plan
    input: Arc<dyn ExecutionPlan>,
    properties: PlanProperties,
}

impl FilterExec {
    /// Create a `FilterExec` on an input. Argument order matches
    /// DataFusion's `FilterExec::try_new(predicate, input)` — predicate
    /// first, input second.
    pub fn new(predicate: Arc<dyn PhysicalExpr>, input: Arc<dyn ExecutionPlan>) -> Self {
        let properties = PlanProperties::single_partition_unknown();
        Self {
            predicate,
            input,
            properties,
        }
    }

    /// The expression to filter on. This expression must evaluate to a
    /// boolean value.
    pub fn predicate(&self) -> &Arc<dyn PhysicalExpr> {
        &self.predicate
    }

    /// The input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
}

impl ExecutionPlan for FilterExec {
    fn name(&self) -> &'static str {
        "FilterExec"
    }

    fn schema(&self) -> Schema {
        self.input.schema()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn execute(
        &self,
        partition: usize,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(FdapQueryError::Internal(format!(
                "FilterExec has 1 output partition; partition {partition} is out of range"
            )));
        }
        // Filter just filters per batch — no context use; pass through.
        let schema = self.input.schema();
        let arrow_schema = Arc::new(schema.clone());
        let predicate = Arc::clone(&self.predicate);
        let input_stream = self.input.execute(0, Arc::clone(&ctx))?;
        // `map(move |...| ...)` builds a lazy stream adapter: it does not drain
        // `input_stream` here. Each time the output stream is polled, one
        // `RecordBatch` is pulled from the input, moved through this closure,
        // filtered, and yielded downstream.
        // Each `Vec<ArrayRef>` below contains only the filtered columns for one batch;
        // the input stream is still processed lazily, one batch at a time.
        let filtered = input_stream.map(move |batch_res| {
            let batch = batch_res?;
            // Variable was `selection` pre-15d-1 #89 (Selection→Filter);
            // renamed to `mask` here to avoid shadowing the file-local
            // `fn filter(...)` helper. `mask` matches DataFusion's
            // `filter_exec.rs` convention for the evaluated predicate result.
            let mask = predicate.evaluate(&batch)?.into_array(batch.num_rows())?;
            let columns: Vec<ArrayRef> = (0..batch.num_columns())
                .map(|i| filter(&batch.column(i).clone(), &mask))
                .collect::<Result<Vec<_>>>()?;
            record_batch::create(&schema, columns)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema,
            filtered,
        )))
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Rebuild this filter with a new input child. Arity 1: a filter has
    /// one input (the relation being filtered). The predicate `expr` is
    /// reused — it doesn't depend on which concrete input feeds the
    /// filter.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(FdapQueryError::Internal(format!(
                "FilterExec::with_new_children expected 1 child, got {}",
                children.len()
            )));
        }
        Ok(Arc::new(FilterExec::new(
            Arc::clone(&self.predicate),
            children.into_iter().next().unwrap(),
        )))
    }
}

impl crate::display::DisplayAs for FilterExec {
    fn fmt_as(
        &self,
        t: crate::display::DisplayFormatType,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match t {
            crate::display::DisplayFormatType::Default
            | crate::display::DisplayFormatType::Verbose => {
                // fdapquery doesn't yet carry `projection` or `fetch` on
                // `FilterExec`, so DataFusion's `{display_projections}{fetch}`
                // suffix is always empty here. The visible format is the
                // base-case byte-equivalent: `FilterExec: {predicate}`.
                write!(f, "FilterExec: {}", self.predicate)
            }
            crate::display::DisplayFormatType::TreeRender => {
                write!(f, "predicate={}", self.predicate)
            }
        }
    }
}

impl std::fmt::Display for FilterExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as crate::display::DisplayAs>::fmt_as(
            self,
            crate::display::DisplayFormatType::Default,
            f,
        )
    }
}

/// Keep the cells of `v` whose corresponding row in the boolean `filter`
/// column is true, returning a new (shorter) column of the same type.
///
/// Walking cell-by-cell via `ScalarValue::try_from_array` keeps the operator
/// uniform across every supported arrow type without per-type compute-kernel
/// dispatch. A future optimisation could swap this for
/// `arrow::compute::filter` once we want kernel-speed filtering — DataFusion
/// uses that path.
fn filter(v: &ArrayRef, filter: &ArrayRef) -> Result<ArrayRef> {
    // Count selected rows first, to size the builder.
    let mut count = 0usize;
    for i in 0..filter.len() {
        let sel = ScalarValue::try_from_array(filter, i)?;
        if matches!(sel, ScalarValue::Boolean(true)) {
            count += 1;
        }
    }

    let mut builder = ArrowVectorBuilder::new(v.data_type(), count);
    for i in 0..filter.len() {
        let sel = ScalarValue::try_from_array(filter, i)?;
        if matches!(sel, ScalarValue::Boolean(true)) {
            let value = ScalarValue::try_from_array(v, i)?;
            builder.append_value(&value);
        }
    }
    Ok(builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BinaryExpr;
    use crate::Column;
    use crate::Literal;
    use crate::test_util::employee_source;
    use fdapquery_common::ScalarValue;
    use fdapquery_expr::Operator;

    /// Byte-for-byte mirror of DataFusion's `DisplayFormatType::Default`
    /// output for `FilterExec`: `"FilterExec: {predicate}{display_projections}{fetch}"`.
    /// With `projection=None` and `fetch=None` (the omitted-infrastructure
    /// fields in fdapquery), the format collapses to `"FilterExec: {predicate}"`.
    /// Source: `datafusion::physical_plan::filter::FilterExec::fmt_as`.
    #[test]
    fn display_default_matches_datafusion() {
        let predicate = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("id", 0)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int64(2))),
        ));
        let predicate_str = predicate.to_string();
        let filter = FilterExec::new(predicate, employee_source());
        assert_eq!(format!("{filter}"), format!("FilterExec: {predicate_str}"));
    }

    /// Drive the full `displayable(plan).indent(false)` pipeline — the
    /// path EXPLAIN uses. The operator's first line through the tree
    /// walker must match DataFusion's exact string.
    #[test]
    fn displayable_indent_default_first_line() {
        let predicate = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("id", 0)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int64(2))),
        ));
        let predicate_str = predicate.to_string();
        let plan: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::new(predicate, employee_source()));
        let rendered = format!(
            "{}",
            crate::display::displayable(plan.as_ref()).indent(false)
        );
        let first_line = rendered.lines().next().unwrap();
        assert_eq!(first_line, format!("FilterExec: {predicate_str}"));
    }

    #[test]
    fn predicate_and_input_accessors() {
        let predicate = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("id", 0)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int64(2))),
        ));
        let filter = FilterExec::new(predicate.clone(), employee_source());
        // Accessor names match DataFusion's: `predicate()` and `input()`.
        assert_eq!(filter.predicate().to_string(), predicate.to_string());
        assert_eq!(filter.input().name(), "TestSourceExec");
    }
}
