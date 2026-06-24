//!
//! Filters rows: evaluates a boolean predicate against each batch and keeps only
//! the rows where it is true. The schema is unchanged.
//!
//! ## Reading the selection vector
//! Selection stays at the `ColumnVector` abstraction: the predicate evaluates to
//! a boolean column, and each row is read as `ScalarValue::Boolean(true)`. No
//! downcast is needed, and it keeps the operator working against any
//! `ColumnVector` implementation.

use crate::expressions::Expression;
use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use crate::task_context::TaskContext;
use fdapquery_datatypes::{
    ArrowVectorBuilder, ColumnVector, FdapQueryError, Result, ScalarValue, Schema, record_batch,
};
use futures::StreamExt;
use std::sync::Arc;

/// Execute a selection (row filter).
pub struct SelectionExec {
    pub input: Arc<dyn ExecutionPlan>,
    pub expr: Arc<dyn Expression>,
    properties: PlanProperties,
}

impl SelectionExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, expr: Arc<dyn Expression>) -> Self {
        let properties = PlanProperties::single_partition_unknown();
        Self {
            input,
            expr,
            properties,
        }
    }
}

impl ExecutionPlan for SelectionExec {
    fn name(&self) -> &str {
        "SelectionExec"
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
                "SelectionExec has 1 output partition; partition {partition} is out of range"
            )));
        }
        // Selection just filters per batch — no context use; pass through.
        let schema = self.input.schema();
        let arrow_schema = Arc::new(schema.to_arrow());
        let expr = Arc::clone(&self.expr);
        let input_stream = self.input.execute(0, Arc::clone(&ctx))?;
        let filtered = input_stream.map(move |batch_res| {
            let batch = batch_res?;
            let selection = expr.evaluate(&batch)?;
            let columns: Vec<Box<dyn ColumnVector>> = (0..batch.num_columns())
                .map(|i| filter(&record_batch::field(&batch, i), selection.as_ref()))
                .collect::<Result<Vec<_>>>()?;
            record_batch::create(&schema, columns)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(arrow_schema, filtered)))
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Rebuild this selection with a new input child. Arity 1: a selection has
    /// one input (the relation being filtered). The predicate `expr` is
    /// reused — it doesn't depend on which concrete input feeds the
    /// selection.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(FdapQueryError::Internal(format!(
                "SelectionExec::with_new_children expected 1 child, got {}",
                children.len()
            )));
        }
        Ok(Arc::new(SelectionExec::new(
            children.into_iter().next().unwrap(),
            Arc::clone(&self.expr),
        )))
    }
}

impl std::fmt::Display for SelectionExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SelectionExec: {}", self.expr)
    }
}

/// Keep the cells of `v` whose corresponding row in the boolean `selection`
/// column is true, returning a new (shorter) column of the same type.
fn filter(v: &dyn ColumnVector, selection: &dyn ColumnVector) -> Result<Box<dyn ColumnVector>> {
    // Count selected rows first, to size the builder.
    let mut count = 0usize;
    for i in 0..selection.size() {
        let sel = selection.get_value(i)?;
        if matches!(sel, ScalarValue::Boolean(true)) {
            count += 1;
        }
    }

    let mut builder = ArrowVectorBuilder::new(&v.get_type(), count);
    for i in 0..selection.size() {
        let sel = selection.get_value(i)?;
        if matches!(sel, ScalarValue::Boolean(true)) {
            let value = v.get_value(i)?;
            builder.append_value(&value);
        }
    }
    builder.set_value_count(count);
    Ok(Box::new(builder.build()))
}
