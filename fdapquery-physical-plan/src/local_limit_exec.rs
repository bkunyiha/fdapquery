//! `LocalLimitExec` — per-partition early-termination limit.
//!
//! In a parallel plan, `LocalLimitExec(fetch=N)` is applied to each
//! input partition before they're merged. The `GlobalLimitExec` at the
//! root then applies the final skip/fetch over the merged result.
//! Same shape as DataFusion's `LocalLimitExec`.
//!
//! ## Status
//! Scaffolded — no optimizer rule currently
//! emits `LocalLimitExec` because fdapquery is single-partition, so
//! the global limit at the root is sufficient. This type comes into
//! play once `RepartitionExec` lands and an `EnforceDistribution`-class
//! optimizer rule starts pushing per-partition early termination
//! underneath repartitions. Until then it's reachable only through
//! direct construction; the test below proves its semantics.
//!
//! ## Implementation
//! Identical body to `GlobalLimitExec`'s `execute()` — both express
//! the "consume input, track remaining, stop early" loop via
//! `async_stream::try_stream!`. They diverge only when `skip` is
//! introduced on `GlobalLimitExec` (DataFusion's
//! `GlobalLimitExec::fetch + skip` semantics); `LocalLimitExec`
//! intentionally has no skip — that's the global responsibility.

use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use async_stream::try_stream;
use fdapquery_common::{FdapQueryError, Result};
use fdapquery_datatypes::Schema;
use fdapquery_execution::TaskContext;
use futures::StreamExt;
use std::sync::Arc;

/// Per-partition early-termination limit. Emits at most `fetch` rows
/// from one input partition. Strict mirror of
/// `datafusion::physical_plan::limit::LocalLimitExec`.
#[derive(Debug)]
pub struct LocalLimitExec {
    /// Input execution plan
    input: Arc<dyn ExecutionPlan>,
    /// Maximum number of rows to return
    fetch: usize,
    properties: PlanProperties,
}

impl LocalLimitExec {
    /// Create a new `LocalLimitExec` partition.
    pub fn new(input: Arc<dyn ExecutionPlan>, fetch: usize) -> Self {
        let properties = PlanProperties::single_partition_unknown();
        Self {
            input,
            fetch,
            properties,
        }
    }

    /// Input execution plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Maximum number of rows to fetch
    pub fn fetch(&self) -> usize {
        self.fetch
    }
}

impl ExecutionPlan for LocalLimitExec {
    fn name(&self) -> &'static str {
        "LocalLimitExec"
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
                "LocalLimitExec has 1 output partition; partition {partition} is out of range"
            )));
        }
        let arrow_schema = Arc::new(self.input.schema());
        let fetch = self.fetch;
        let input_stream = self.input.execute(0, Arc::clone(&ctx))?;
        let stream = try_stream! {
            let mut remaining = fetch;
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

    /// Rebuild this local limit with a new input child. Arity 1.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(FdapQueryError::Internal(format!(
                "LocalLimitExec::with_new_children expected 1 child, got {}",
                children.len()
            )));
        }
        Ok(Arc::new(LocalLimitExec::new(
            children.into_iter().next().unwrap(),
            self.fetch,
        )))
    }
}

impl crate::display::DisplayAs for LocalLimitExec {
    fn fmt_as(
        &self,
        t: crate::display::DisplayFormatType,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match t {
            crate::display::DisplayFormatType::Default
            | crate::display::DisplayFormatType::Verbose => {
                write!(f, "LocalLimitExec: fetch={}", self.fetch)
            }
            crate::display::DisplayFormatType::TreeRender => {
                write!(f, "limit={}", self.fetch)
            }
        }
    }
}

impl std::fmt::Display for LocalLimitExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as crate::display::DisplayAs>::fmt_as(
            self,
            crate::display::DisplayFormatType::Default,
            f,
        )
    }
}

// The old cell-by-cell `truncate` helper was replaced
// by `RecordBatch::slice(0, n)` (arrow-rs's native zero-copy slice). The
// `Schema` argument is no longer needed at the call site.

#[cfg(test)]
mod tests {
    //! Exercises `LocalLimitExec` directly (no planner emits it yet).
    //! Same shape as `global_limit_exec.rs`'s `limit_truncates_to_budget`.
    use super::*;
    use crate::test_util::employee_source;
    use futures::TryStreamExt;

    fn test_ctx() -> Arc<TaskContext> {
        Arc::new(TaskContext::default_test())
    }

    #[tokio::test]
    async fn local_limit_truncates_to_fetch() {
        let limited: Arc<dyn ExecutionPlan> = Arc::new(LocalLimitExec::new(employee_source(), 2));
        let batches = limited
            .execute(0, test_ctx())
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 2);
    }

    /// Byte-for-byte mirror of DataFusion's `DisplayFormatType::Default`
    /// output: `"LocalLimitExec: fetch={fetch}"`. Source:
    /// `datafusion::physical_plan::limit::LocalLimitExec::fmt_as`.
    #[test]
    fn display_default_matches_datafusion() {
        let plan = LocalLimitExec::new(employee_source(), 7);
        assert_eq!(format!("{plan}"), "LocalLimitExec: fetch=7");

        let plan = LocalLimitExec::new(employee_source(), 0);
        assert_eq!(format!("{plan}"), "LocalLimitExec: fetch=0");
    }

    /// Drive the full `displayable(plan).indent(false)` pipeline — the
    /// path EXPLAIN uses.
    #[test]
    fn displayable_indent_default_first_line() {
        let plan: Arc<dyn ExecutionPlan> = Arc::new(LocalLimitExec::new(employee_source(), 3));
        let rendered = format!(
            "{}",
            crate::display::displayable(plan.as_ref()).indent(false)
        );
        let first_line = rendered.lines().next().unwrap();
        assert_eq!(first_line, "LocalLimitExec: fetch=3");
    }

    #[test]
    fn input_and_fetch_accessors() {
        let plan = LocalLimitExec::new(employee_source(), 5);
        // Accessor names match DataFusion's: `input()` and `fetch()`.
        assert_eq!(plan.input().name(), "TestSourceExec");
        assert_eq!(plan.fetch(), 5);
    }
}
