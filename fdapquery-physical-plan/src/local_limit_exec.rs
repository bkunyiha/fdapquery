//! `LocalLimitExec` — per-partition early-termination limit.
//!
//! In a parallel plan, `LocalLimitExec(fetch=N)` is applied to each
//! input partition before they're merged. The `GlobalLimitExec` at the
//! root then applies the final skip/fetch over the merged result.
//! Same shape as DataFusion's `LocalLimitExec`.
//!
//! ## Status
//! Scaffolded in Session 15d-1 #105 — no optimizer rule currently
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
use fdapquery_datatypes::{
    ArrowVectorBuilder, ColumnVector, FdapQueryError, RecordBatch, Result, Schema, record_batch,
};
use fdapquery_execution::TaskContext;
use futures::StreamExt;
use std::sync::Arc;

/// Per-partition early-termination limit. Emits at most `fetch` rows
/// from one input partition.
pub struct LocalLimitExec {
    pub input: Arc<dyn ExecutionPlan>,
    pub fetch: usize,
    properties: PlanProperties,
}

impl LocalLimitExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, fetch: usize) -> Self {
        let properties = PlanProperties::single_partition_unknown();
        Self {
            input,
            fetch,
            properties,
        }
    }
}

impl ExecutionPlan for LocalLimitExec {
    fn name(&self) -> &str {
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
        let schema = self.input.schema();
        let arrow_schema = Arc::new(schema.clone());
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
                            yield truncate(&batch, take, &schema)?;
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

impl std::fmt::Display for LocalLimitExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LocalLimitExec: fetch={}", self.fetch)
    }
}

/// Build a new batch containing only the first `n` rows of `batch`,
/// copying cell-by-cell. Duplicated from `global_limit_exec.rs`;
/// when Session 15d-2 #106/#107 drop the `Schema`/`ColumnVector`
/// wrappers, this collapses to `batch.slice(0, n)` and the duplicate
/// can be hoisted to a shared helper.
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
    //! Exercises `LocalLimitExec` directly (no planner emits it yet).
    //! Same shape as `global_limit_exec.rs`'s `limit_truncates_to_budget`.
    use super::*;
    use crate::scan_exec::ScanExec;
    use fdapquery_catalog::CsvDataSource;
    use fdapquery_catalog::TableProvider;
    use futures::TryStreamExt;

    fn employee_ds() -> Arc<dyn TableProvider> {
        Arc::new(CsvDataSource::new(
            "../testdata/employee.csv",
            None,
            true,
            1024,
        ))
    }

    fn all_columns(ds: &Arc<dyn TableProvider>) -> Vec<String> {
        ds.schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }

    fn test_ctx() -> Arc<TaskContext> {
        Arc::new(TaskContext::default_test())
    }

    #[tokio::test]
    async fn local_limit_truncates_to_fetch() {
        let ds = employee_ds();
        let scan = ScanExec::new(Arc::clone(&ds), all_columns(&ds)).unwrap();
        let limited: Arc<dyn ExecutionPlan> = Arc::new(LocalLimitExec::new(Arc::new(scan), 2));
        let batches = limited
            .execute(0, test_ctx())
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 2);
    }
}
