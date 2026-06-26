//!
//! Reads shuffle output from one or more [`ShuffleLocation`]s at the start
//! of a stage that consumes a previous stage's output (local files for data
//! on this executor, Arrow Flight for remote executors).
//!
//! ## Local vs remote
//! For each `shuffle_locations[i]`, the reader compares
//! `location.executor_id` against `ctx.executor_id`:
//! - **Local** — this executor wrote the file; reads via
//!   `ctx.runtime.shuffle_manager.read_partition(...)`.
//! - **Remote** — another executor wrote it; would need an Arrow Flight
//!   client (not yet wired into `RuntimeEnv`). Surfaces as
//!   `Err(NotImplemented(_))`.

use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::shuffle_location::ShuffleLocation;
use crate::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use crate::task_context::TaskContext;
use fdapquery_datatypes::{FdapQueryError, Result, Schema};
use futures::StreamExt;
use std::sync::Arc;

/// Reads shuffle data from a set of locations.
pub struct ShuffleReaderExec {
    pub shuffle_schema: Schema,
    pub shuffle_locations: Vec<ShuffleLocation>,
    properties: PlanProperties,
}

impl ShuffleReaderExec {
    pub fn new(shuffle_schema: Schema, shuffle_locations: Vec<ShuffleLocation>) -> Self {
        let properties = PlanProperties::single_partition_unknown();
        Self {
            shuffle_schema,
            shuffle_locations,
            properties,
        }
    }
}

impl ExecutionPlan for ShuffleReaderExec {
    fn name(&self) -> &str {
        "ShuffleReaderExec"
    }

    fn schema(&self) -> Schema {
        self.shuffle_schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // A shuffle read is a leaf — its input is the previous stage's output.
        vec![]
    }

    /// Rebuild this shuffle reader with new children. Arity 0 (leaf): a
    /// shuffle reader has no input plan — its data comes from
    /// `shuffle_locations`.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(FdapQueryError::Internal(format!(
                "ShuffleReaderExec is a leaf and expects no children, got {}",
                children.len()
            )));
        }
        Ok(self)
    }

    /// Read every shuffle location in order and yield the resulting
    /// `RecordBatch`es as a single stream via `flat_map` over per-location
    /// async streams.
    ///
    /// **Local reads only.** A location whose `executor_id` doesn't match
    /// `ctx.executor_id` surfaces as `Err(NotImplemented(_))`. Remote reads
    /// would require a Flight client field on `RuntimeEnv`; not currently
    /// implemented.
    fn execute(
        &self,
        partition: usize,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(FdapQueryError::Internal(format!(
                "ShuffleReaderExec has 1 output partition; partition {partition} is out of range"
            )));
        }
        // Validate all locations are local up front so the error — if one
        // belongs to another executor — fires before any disk I/O.
        for loc in &self.shuffle_locations {
            if loc.executor_id != ctx.executor_id {
                return Err(FdapQueryError::NotImplemented(format!(
                    "ShuffleReaderExec: remote shuffle reads require an Arrow Flight \
                     client. Location belongs to executor '{}' but this executor is \
                     '{}'. Remote reads need a Flight client field on RuntimeEnv.",
                    loc.executor_id, ctx.executor_id
                )));
            }
        }

        // Open every partition up-front so any I/O failures surface from
        // `execute()` itself (the outer Result). Each per-partition iterator
        // becomes a sync stream via `futures::stream::iter`; we chain them
        // with `flatten()`.
        let mut per_location_streams: Vec<
            futures::stream::Iter<
                Box<dyn Iterator<Item = Result<fdapquery_datatypes::RecordBatch>> + Send>,
            >,
        > = Vec::new();
        for location in &self.shuffle_locations {
            let iter = ctx.runtime.shuffle_manager.read_partition(
                &location.job_uuid,
                location.stage_id,
                location.partition_id,
            )?;
            per_location_streams.push(futures::stream::iter(iter));
        }
        let flattened = futures::stream::iter(per_location_streams).flatten();
        let arrow_schema = Arc::new(self.shuffle_schema.to_arrow());
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema,
            flattened,
        )))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl std::fmt::Display for ShuffleReaderExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ShuffleReaderExec: schema={:?}, locations={}",
            self.shuffle_schema,
            self.shuffle_locations.len()
        )
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the async `execute(partition, ctx)` surface — full writer →
    //! reader round-trip via the new trait method.

    use super::*;
    use crate::ColumnExpression;
    use crate::scan_exec::ScanExec;
    use crate::shuffle_manager::ShuffleManager;
    use crate::shuffle_writer_exec::ShuffleWriterExec;
    use crate::task_context::{RuntimeEnv, SessionConfig, TaskContext};
    use fdapquery_catalog::CsvDataSource;
    use fdapquery_catalog::TableProvider;
    use futures::TryStreamExt;

    /// Build a `RuntimeEnv` with a specific shuffle base directory — lets
    /// the round-trip tests isolate per-test on-disk state.
    fn default_runtime_for(base: &str) -> RuntimeEnv {
        RuntimeEnv::new(Arc::new(ShuffleManager::new(base.to_string())))
    }

    const EMPLOYEE_CSV: &str = "../testdata/employee.csv";

    fn temp_dir(tag: &str) -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("/tmp/rquery-shuffle-test-{tag}-{nanos}")
    }

    fn employee_ds() -> Arc<dyn TableProvider> {
        Arc::new(CsvDataSource::new(EMPLOYEE_CSV, None, true, 1024))
    }

    fn employee_columns(ds: &Arc<dyn TableProvider>) -> Vec<String> {
        ds.schema().fields.iter().map(|f| f.name.clone()).collect()
    }

    /// Build a `TaskContext` with a specific shuffle base dir and executor
    /// identity (so the round-trip tests can isolate per-test on-disk state).
    fn make_ctx(executor_id: &str, host: &str, port: u16, base: &str) -> Arc<TaskContext> {
        let runtime = Arc::new(default_runtime_for(base));
        Arc::new(TaskContext::new(
            executor_id,
            host,
            port,
            SessionConfig::new(),
            runtime,
        ))
    }

    async fn write_employee_shuffle(
        ctx: Arc<TaskContext>,
        job_uuid: &str,
        partition_count: i32,
    ) -> (usize, Vec<ShuffleLocation>, Schema) {
        let ds = employee_ds();
        let schema = ds.schema();
        let scan: Arc<dyn ExecutionPlan> =
            Arc::new(ScanExec::new(Arc::clone(&ds), employee_columns(&ds)).unwrap());
        let input_batches = scan
            .execute(0, Arc::clone(&ctx))
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let input_row_count: usize = input_batches.iter().map(|b| b.num_rows()).sum();
        let writer = ShuffleWriterExec::new(
            Arc::new(ScanExec::new(Arc::clone(&ds), employee_columns(&ds)).unwrap()),
            vec![Arc::new(ColumnExpression::new(0))],
            job_uuid,
            0,
            partition_count,
        );
        // ShuffleWriterExec's real entry point is `write_shuffle(ctx)`; the
        // trait `execute()` is a NotImplemented stub.
        let locations = writer.write_shuffle(Arc::clone(&ctx)).unwrap();
        (input_row_count, locations, schema)
    }

    #[tokio::test]
    async fn writer_then_reader_round_trips_full_row_count() {
        let base = temp_dir("reader-roundtrip");
        let ctx = make_ctx("exec-test", "127.0.0.1", 50099, &base);

        let (input_rows, locations, schema) =
            write_employee_shuffle(Arc::clone(&ctx), "test-job-reader-roundtrip", 3).await;

        let reader = ShuffleReaderExec::new(schema, locations);
        let read_batches = reader
            .execute(0, Arc::clone(&ctx))
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let read_rows: usize = read_batches.iter().map(|b| b.num_rows()).sum();

        assert_eq!(
            read_rows, input_rows,
            "writer→reader must preserve all rows"
        );
        ctx.runtime.shuffle_manager.cleanup_all();
    }

    #[tokio::test]
    async fn empty_locations_yields_empty_iterator() {
        let base = temp_dir("reader-empty");
        let ctx = make_ctx("exec-test", "127.0.0.1", 50099, &base);

        let ds = employee_ds();
        let reader = ShuffleReaderExec::new(ds.schema(), vec![]);
        let batches = reader
            .execute(0, Arc::clone(&ctx))
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert!(batches.is_empty());
        ctx.runtime.shuffle_manager.cleanup_all();
    }

    #[tokio::test]
    async fn single_partition_round_trip_reads_all_rows_from_one_file() {
        let base = temp_dir("reader-single");
        let ctx = make_ctx("exec-test", "127.0.0.1", 50099, &base);

        let (input_rows, locations, schema) =
            write_employee_shuffle(Arc::clone(&ctx), "test-job-reader-single", 1).await;
        assert_eq!(
            locations.len(),
            1,
            "single partition writer emits 1 location"
        );

        let reader = ShuffleReaderExec::new(schema, locations);
        let read_batches = reader
            .execute(0, Arc::clone(&ctx))
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let read_rows: usize = read_batches.iter().map(|b| b.num_rows()).sum();

        assert_eq!(read_rows, input_rows);
        ctx.runtime.shuffle_manager.cleanup_all();
    }

    #[tokio::test]
    async fn remote_location_errors_until_flight_client_lands() {
        let base = temp_dir("reader-remote");
        let ctx = make_ctx("exec-A", "127.0.0.1", 50099, &base);

        let remote_loc = ShuffleLocation::new("test-job-remote", 0, 0, "exec-B", "10.0.0.2", 50099);
        let reader = ShuffleReaderExec::new(employee_ds().schema(), vec![remote_loc]);
        let err = reader
            .execute(0, Arc::clone(&ctx))
            .map(|_| ())
            .expect_err("remote shuffle reads must error until the Flight client is wired in");
        assert!(
            matches!(err, FdapQueryError::NotImplemented(_)),
            "expected NotImplemented, got {err:?}"
        );
        assert!(
            err.to_string()
                .contains("remote shuffle reads require an Arrow Flight client")
        );
    }
}
