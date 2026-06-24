//!
//! Stops emitting rows once `limit` rows have been produced. Full batches pass
//! through untouched until the running budget would be exceeded; the boundary
//! batch is truncated to exactly the remaining count and the stream then ends.
//!
//! ## Implementation — `try_stream!` for the budget tracker
//! The async-stream version uses `async_stream::try_stream!` to express the
//! "consume input, track remaining, stop early" loop as a sequential body.
//! The generator macro `.await`s on the input stream's `.next()` calls and
//! `yield`s output batches; when the budget hits zero, the loop just `break`s
//! and the generator ends. Same shape as DataFusion's `LimitStream` (which is
//! a hand-rolled `Stream` impl carrying a remaining counter — the generator
//! macro is more readable for the simple case).

use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use crate::task_context::TaskContext;
use async_stream::try_stream;
use fdapquery_datatypes::{
    ArrowVectorBuilder, ColumnVector, FdapQueryError, RecordBatch, Result, Schema, record_batch,
};
use futures::StreamExt;
use std::sync::Arc;

/// Execute a limit. `limit` is a row count.
pub struct LimitExec {
    pub input: Arc<dyn ExecutionPlan>,
    pub limit: usize,
    properties: PlanProperties,
}

impl LimitExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, limit: usize) -> Self {
        let properties = PlanProperties::single_partition_unknown();
        Self {
            input,
            limit,
            properties,
        }
    }
}

impl ExecutionPlan for LimitExec {
    fn name(&self) -> &str {
        "LimitExec"
    }

    fn schema(&self) -> Schema {
        self.input.schema()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn execute(
        &self,
        partition: usize,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(FdapQueryError::Internal(format!(
                "LimitExec has 1 output partition; partition {partition} is out of range"
            )));
        }
        let schema = self.input.schema();
        let arrow_schema = Arc::new(schema.to_arrow());
        let limit = self.limit;
        let input_stream = self.input.execute(0, Arc::clone(&ctx))?;
        let stream = try_stream! {
            let mut remaining = limit;
            let mut input = std::pin::pin!(input_stream);
            while remaining > 0 {
                match input.next().await {
                    Some(Ok(batch)) => {
                        let rows = batch.num_rows();
                        if rows <= remaining {
                            remaining -= rows;
                            yield batch;
                        } else {
                            let take = remaining;
                            remaining = 0;
                            yield truncate(&batch, take, &schema)?;
                        }
                    }
                    Some(Err(e)) => Err(e)?,
                    None => break,
                }
            }
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(arrow_schema, stream)))
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// Rebuild this limit with a new input child. Arity 1: a limit has one
    /// input. The `limit` budget is reused — it's part of this operator's
    /// definition, not the child's.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(FdapQueryError::Internal(format!(
                "LimitExec::with_new_children expected 1 child, got {}",
                children.len()
            )));
        }
        Ok(Arc::new(LimitExec::new(
            children.into_iter().next().unwrap(),
            self.limit,
        )))
    }
}

impl std::fmt::Display for LimitExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LimitExec: limit={}", self.limit)
    }
}

/// Build a new batch containing only the first `n` rows of `batch`, copying
/// cell-by-cell.
fn truncate(batch: &RecordBatch, n: usize, schema: &Schema) -> Result<RecordBatch> {
    let columns: Vec<Box<dyn ColumnVector>> = (0..batch.num_columns())
        .map(|i| -> Result<Box<dyn ColumnVector>> {
            let source = record_batch::field(batch, i);
            let mut builder = ArrowVectorBuilder::new(&source.get_type(), n);
            for row in 0..n {
                let value = source.get_value(row)?;
                builder.append_value(&value);
            }
            builder.set_value_count(n);
            Ok(Box::new(builder.build()) as Box<dyn ColumnVector>)
        })
        .collect::<Result<Vec<_>>>()?;
    record_batch::create(schema, columns)
}

#[cfg(test)]
mod tests {
    //! End-to-end pipeline verification: drives `ScanExec` →
    //! `Projection`/`Selection`/`LimitExec` over the `employee.csv` fixture and
    //! checks row/column counts. Uses `CsvDataSource` directly.
    use super::*;
    use crate::boolean_expression::GtExpression;
    use crate::column_expression::ColumnExpression;
    use crate::expressions::LiteralLongExpression;
    use crate::projection_exec::ProjectionExec;
    use crate::scan_exec::ScanExec;
    use crate::selection_exec::SelectionExec;
    use futures::TryStreamExt;
    use fdapquery_datasource::{CsvDataSource, DataSource};
    use std::sync::Arc;

    fn employee_ds() -> Arc<dyn DataSource> {
        Arc::new(CsvDataSource::new(
            "../testdata/employee.csv",
            None,
            true,
            1024,
        ))
    }

    /// All column names, in schema order: id, first_name, last_name, state,
    /// job_title, salary.
    fn all_columns(ds: &Arc<dyn DataSource>) -> Vec<String> {
        ds.schema().fields.iter().map(|f| f.name.clone()).collect()
    }

    /// Single-node test context fixture.
    fn test_ctx() -> Arc<TaskContext> {
        Arc::new(TaskContext::default_test())
    }

    async fn total_rows(plan: Arc<dyn ExecutionPlan>) -> usize {
        let batches = plan
            .execute(0, test_ctx())
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        batches.iter().map(|b| b.num_rows()).sum()
    }

    #[tokio::test]
    async fn scan_reads_all_rows() {
        let ds = employee_ds();
        let scan: Arc<dyn ExecutionPlan> =
            Arc::new(ScanExec::new(Arc::clone(&ds), all_columns(&ds)).unwrap());
        assert_eq!(total_rows(Arc::clone(&scan)).await, 4);
        // ScanExec is a leaf.
        assert!(scan.children().is_empty());
    }

    #[tokio::test]
    async fn limit_truncates_to_budget() {
        let ds = employee_ds();
        let scan = ScanExec::new(Arc::clone(&ds), all_columns(&ds)).unwrap();
        let limited: Arc<dyn ExecutionPlan> = Arc::new(LimitExec::new(Arc::new(scan), 3));
        assert_eq!(total_rows(limited).await, 3);
    }

    #[tokio::test]
    async fn limit_above_total_keeps_everything() {
        let ds = employee_ds();
        let scan = ScanExec::new(Arc::clone(&ds), all_columns(&ds)).unwrap();
        let limited: Arc<dyn ExecutionPlan> = Arc::new(LimitExec::new(Arc::new(scan), 100));
        assert_eq!(total_rows(limited).await, 4);
    }

    #[tokio::test]
    async fn projection_keeps_one_column() {
        let ds = employee_ds();
        let scan = ScanExec::new(Arc::clone(&ds), all_columns(&ds)).unwrap();
        // Output schema is just the first column (id).
        let schema = scan.schema().project(&[0]);
        let proj = ProjectionExec::new(
            Arc::new(scan),
            schema,
            vec![Arc::new(ColumnExpression::new(0))],
        );
        let batches = proj
            .execute(0, test_ctx())
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 4);
        assert!(batches.iter().all(|b| b.num_columns() == 1));
    }

    #[tokio::test]
    async fn selection_filters_rows() {
        let ds = employee_ds();
        let scan = ScanExec::new(Arc::clone(&ds), all_columns(&ds)).unwrap();
        // WHERE id > 2  →  ids 3 and 4  →  2 rows.
        let predicate = GtExpression::new(
            Arc::new(ColumnExpression::new(0)),
            Arc::new(LiteralLongExpression::new(2)),
        );
        let selection: Arc<dyn ExecutionPlan> =
            Arc::new(SelectionExec::new(Arc::new(scan), Arc::new(predicate)));
        assert_eq!(total_rows(Arc::clone(&selection)).await, 2);
        // Selection preserves the schema (all six columns).
        assert_eq!(selection.schema().fields.len(), 6);
    }

    #[tokio::test]
    async fn pipeline_scan_select_project_limit() {
        // End-to-end: scan → WHERE id > 2 → SELECT id → LIMIT 1.
        let ds = employee_ds();
        let scan = ScanExec::new(Arc::clone(&ds), all_columns(&ds)).unwrap();
        let selection = SelectionExec::new(
            Arc::new(scan),
            Arc::new(GtExpression::new(
                Arc::new(ColumnExpression::new(0)),
                Arc::new(LiteralLongExpression::new(2)),
            )),
        );
        let project_schema = selection.schema().project(&[0]);
        let projection = ProjectionExec::new(
            Arc::new(selection),
            project_schema,
            vec![Arc::new(ColumnExpression::new(0))],
        );
        let limited = LimitExec::new(Arc::new(projection), 1);

        let batches = limited
            .execute(0, test_ctx())
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 1);
        assert!(batches.iter().all(|b| b.num_columns() == 1));
    }
}
