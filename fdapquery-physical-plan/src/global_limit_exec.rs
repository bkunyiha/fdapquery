//!
//! Skips `skip` rows, then emits at most `fetch` rows (or all remaining rows
//! when `fetch` is `None`). Full batches pass through untouched until the
//! running budget would be exceeded; the boundary batch is sliced to exactly
//! the remaining count and the stream then ends. This is a strict mirror of
//! DataFusion's `GlobalLimitExec` (struct fields, constructor signature,
//! `DisplayAs::fmt_as` output) — the only divergence is the error type
//! (`FdapQueryError` instead of `DataFusionError`).
//!
//! ## Implementation — `try_stream!` for the skip+fetch tracker
//! The async-stream version uses `async_stream::try_stream!` to express the
//! "consume input, drop `skip` rows, then emit up to `fetch` rows" loop as a
//! sequential body. The generator macro `.await`s on the input stream's
//! `.next()` calls and `yield`s output batches; once the fetch budget hits
//! zero the loop `break`s and the generator ends. Same shape as DataFusion's
//! `LimitStream` (a hand-rolled `Stream` impl carrying skip + remaining
//! counters — the generator macro is more readable for the simple case).

use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use async_stream::try_stream;
use fdapquery_common::{FdapQueryError, Result};
use fdapquery_datatypes::Schema;
use fdapquery_execution::TaskContext;
use futures::StreamExt;
use std::sync::Arc;

/// Limit execution plan: skip `skip` rows then emit up to `fetch` rows
/// (or all remaining when `fetch.is_none()`). Strict mirror of DataFusion's
/// `datafusion::physical_plan::limit::GlobalLimitExec`.
#[derive(Debug)]
pub struct GlobalLimitExec {
    /// Input execution plan
    input: Arc<dyn ExecutionPlan>,
    /// Number of rows to skip before fetch
    skip: usize,
    /// Maximum number of rows to fetch,
    /// `None` means fetching all rows
    fetch: Option<usize>,
    properties: PlanProperties,
}

impl GlobalLimitExec {
    /// Create a new `GlobalLimitExec`.
    pub fn new(input: Arc<dyn ExecutionPlan>, skip: usize, fetch: Option<usize>) -> Self {
        let properties = PlanProperties::single_partition_unknown();
        Self {
            input,
            skip,
            fetch,
            properties,
        }
    }

    /// Input execution plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Number of rows to skip before fetch
    pub fn skip(&self) -> usize {
        self.skip
    }

    /// Maximum number of rows to fetch
    pub fn fetch(&self) -> Option<usize> {
        self.fetch
    }
}

impl ExecutionPlan for GlobalLimitExec {
    fn name(&self) -> &'static str {
        "GlobalLimitExec"
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
                "GlobalLimitExec has 1 output partition; partition {partition} is out of range"
            )));
        }
        let arrow_schema = Arc::new(self.input.schema());
        // Mirror DataFusion's `LimitStream`: `None` fetch is treated as
        // `usize::MAX` rows remaining, so the same "drop the boundary slice
        // and stop" logic handles both bounded and unbounded fetch.
        let mut skip = self.skip;
        let mut remaining = self.fetch.unwrap_or(usize::MAX);
        let input_stream = self.input.execute(0, Arc::clone(&ctx))?;
        let stream = try_stream! {
            let mut input = std::pin::pin!(input_stream);
            while remaining > 0 {
                match input.next().await {
                    Some(Ok(batch)) => {
                        let rows = batch.num_rows();
                        // First drop up to `skip` rows from the head of the batch.
                        let (batch, rows) = if skip == 0 {
                            (batch, rows)
                        } else if skip >= rows {
                            skip -= rows;
                            continue;
                        } else {
                            let kept = rows - skip;
                            let sliced = batch.slice(skip, kept);
                            skip = 0;
                            (sliced, kept)
                        };
                        // Then emit at most `remaining` rows from what's left.
                        if rows <= remaining {
                            remaining -= rows;
                            yield batch;
                        } else {
                            let take = remaining;
                            remaining = 0;
                            yield batch.slice(0, take);
                        }
                    }
                    Some(Err(e)) => Err(e)?,
                    None => break,
                }
            }
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema,
            stream,
        )))
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// Rebuild this limit with a new input child. Arity 1: a limit has one
    /// input. The `skip` / `fetch` budget is reused — it's part of this
    /// operator's definition, not the child's.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(FdapQueryError::Internal(format!(
                "GlobalLimitExec::with_new_children expected 1 child, got {}",
                children.len()
            )));
        }
        Ok(Arc::new(GlobalLimitExec::new(
            children.into_iter().next().unwrap(),
            self.skip,
            self.fetch,
        )))
    }
}

impl crate::display::DisplayAs for GlobalLimitExec {
    fn fmt_as(
        &self,
        t: crate::display::DisplayFormatType,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match t {
            crate::display::DisplayFormatType::Default
            | crate::display::DisplayFormatType::Verbose => {
                write!(
                    f,
                    "GlobalLimitExec: skip={}, fetch={}",
                    self.skip,
                    self.fetch
                        .map_or_else(|| "None".to_string(), |x| x.to_string())
                )
            }
            crate::display::DisplayFormatType::TreeRender => {
                if let Some(fetch) = self.fetch {
                    writeln!(f, "limit={fetch}")?;
                }
                write!(f, "skip={}", self.skip)
            }
        }
    }
}

impl std::fmt::Display for GlobalLimitExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as crate::display::DisplayAs>::fmt_as(
            self,
            crate::display::DisplayFormatType::Default,
            f,
        )
    }
}

// The old cell-by-cell `slice` helper was replaced by
// `RecordBatch::slice(offset, len)` (arrow-rs's native zero-copy slice).
// Now that the `ColumnVector` trait is gone, we can use arrow's own slicing
// directly. The helper module-comment §6 line about "Cell-by-cell copy"
// becomes obsolete: arrow's `slice` is a buffer-level view that costs
// O(1) per call.

#[cfg(test)]
mod tests {
    //! End-to-end pipeline verification: drives `TestSourceExec` →
    //! `Projection`/`Filter`/`GlobalLimitExec` over the in-memory
    //! employee fixture and checks row/column counts.
    use super::*;
    use crate::BinaryExpr;
    use crate::Column;
    use crate::Literal;
    use crate::filter_exec::FilterExec;
    use crate::projection_exec::ProjectionExec;
    use crate::test_util::employee_source;
    use fdapquery_common::ScalarValue;
    use fdapquery_expr::Operator;
    use futures::TryStreamExt;
    use std::sync::Arc;

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
        let scan = employee_source();
        assert_eq!(total_rows(Arc::clone(&scan)).await, 4);
        // The test source is a leaf.
        assert!(scan.children().is_empty());
    }

    #[tokio::test]
    async fn limit_truncates_to_budget() {
        let limited: Arc<dyn ExecutionPlan> =
            Arc::new(GlobalLimitExec::new(employee_source(), 0, Some(3)));
        assert_eq!(total_rows(limited).await, 3);
    }

    #[tokio::test]
    async fn limit_above_total_keeps_everything() {
        let limited: Arc<dyn ExecutionPlan> =
            Arc::new(GlobalLimitExec::new(employee_source(), 0, Some(100)));
        assert_eq!(total_rows(limited).await, 4);
    }

    #[tokio::test]
    async fn skip_drops_leading_rows() {
        // employee fixture has 4 rows; skip=2, fetch=None should emit
        // the trailing 2 rows.
        let limited: Arc<dyn ExecutionPlan> =
            Arc::new(GlobalLimitExec::new(employee_source(), 2, None));
        assert_eq!(total_rows(limited).await, 2);
    }

    #[tokio::test]
    async fn skip_plus_fetch_window() {
        // skip=1, fetch=2 over 4 rows → 2 rows.
        let limited: Arc<dyn ExecutionPlan> =
            Arc::new(GlobalLimitExec::new(employee_source(), 1, Some(2)));
        assert_eq!(total_rows(limited).await, 2);
    }

    #[tokio::test]
    async fn skip_past_end_returns_empty() {
        // skip=10 on 4 rows → 0 rows regardless of fetch.
        let limited: Arc<dyn ExecutionPlan> =
            Arc::new(GlobalLimitExec::new(employee_source(), 10, Some(5)));
        assert_eq!(total_rows(limited).await, 0);
    }

    /// Byte-for-byte mirror of DataFusion's `DisplayFormatType::Default`
    /// output: `"GlobalLimitExec: skip={skip}, fetch={fetch|None}"`.
    /// Source: `datafusion::physical_plan::limit::GlobalLimitExec::fmt_as`.
    #[test]
    fn display_default_matches_datafusion() {
        let plan = GlobalLimitExec::new(employee_source(), 0, Some(10));
        assert_eq!(format!("{plan}"), "GlobalLimitExec: skip=0, fetch=10");

        let plan = GlobalLimitExec::new(employee_source(), 5, None);
        assert_eq!(format!("{plan}"), "GlobalLimitExec: skip=5, fetch=None");

        let plan = GlobalLimitExec::new(employee_source(), 3, Some(7));
        assert_eq!(format!("{plan}"), "GlobalLimitExec: skip=3, fetch=7");
    }

    /// Exercise the full `displayable(plan).indent(false)` pipeline — the
    /// path EXPLAIN uses. Confirms that the operator's first line through
    /// the tree walker matches DataFusion's exact string, including the
    /// trailing newline emitted by the walker.
    #[test]
    fn displayable_indent_default_first_line() {
        let plan: Arc<dyn ExecutionPlan> =
            Arc::new(GlobalLimitExec::new(employee_source(), 0, Some(10)));
        let rendered = format!(
            "{}",
            crate::display::displayable(plan.as_ref()).indent(false)
        );
        let first_line = rendered.lines().next().unwrap();
        assert_eq!(first_line, "GlobalLimitExec: skip=0, fetch=10");
    }

    #[tokio::test]
    async fn projection_keeps_one_column() {
        let source = employee_source();
        // Output schema is just the first column (id).
        let schema = source.schema().project(&[0]).unwrap();
        let proj = ProjectionExec::new(vec![Arc::new(Column::new("id", 0))], source, schema);
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
    async fn filter_drops_non_matching_rows() {
        // WHERE id > 2  →  ids 3 and 4  →  2 rows.
        let predicate = BinaryExpr::new(
            Arc::new(Column::new("id", 0)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int64(2))),
        );
        let filter: Arc<dyn ExecutionPlan> =
            Arc::new(FilterExec::new(Arc::new(predicate), employee_source()));
        assert_eq!(total_rows(Arc::clone(&filter)).await, 2);
        // Filter preserves the schema (all six columns).
        assert_eq!(filter.schema().fields().len(), 6);
    }

    #[tokio::test]
    async fn pipeline_scan_select_project_limit() {
        // End-to-end: scan → WHERE id > 2 → SELECT id → LIMIT 1.
        let filter = FilterExec::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new("id", 0)),
                Operator::Gt,
                Arc::new(Literal::new(ScalarValue::Int64(2))),
            )),
            employee_source(),
        );
        let project_schema = filter.schema().project(&[0]).unwrap();
        let projection = ProjectionExec::new(
            vec![Arc::new(Column::new("id", 0))],
            Arc::new(filter),
            project_schema,
        );
        let limited = GlobalLimitExec::new(Arc::new(projection), 0, Some(1));

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
