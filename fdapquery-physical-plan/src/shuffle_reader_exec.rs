//!
//! Reads shuffle output from one or more [`ShuffleLocation`]s at the start
//! of a stage that consumes a previous stage's output (local files for data
//! on this executor, Arrow Flight for remote executors).
//!
//! ## Single-surface execute(ctx)
//!
//! Every `PhysicalPlan` operator takes the executor context as a parameter on
//! the trait method itself — the compiler refuses to compile a call site that
//! doesn't supply one. There is no `execute_with_context` sibling; the trait
//! `execute(ctx)` *is* the context-aware entry point.
//!
//! ## Local vs remote
//! For each `shuffle_locations[i]`, the reader compares
//! `location.executor_id` against `ctx.executor_id`:
//! - **Local** — this executor wrote the file; reads via
//!   `ctx.shuffle_manager.read_partition(...)`.
//! - **Remote** — another executor wrote it; would need an Arrow Flight
//!   client (not yet wired into `ExecutorContext`). Triggers
//!   `unimplemented!()` for now.

use crate::executor_context::ExecutorContext;
use crate::physical_plan::PhysicalPlan;
use crate::shuffle_location::ShuffleLocation;
use datatypes::{RecordBatch, Schema};
use std::sync::Arc;

/// Reads shuffle data from a set of locations.
pub struct ShuffleReaderExec {
    pub shuffle_schema: Schema,
    pub shuffle_locations: Vec<ShuffleLocation>,
}

impl ShuffleReaderExec {
    pub fn new(shuffle_schema: Schema, shuffle_locations: Vec<ShuffleLocation>) -> Self {
        Self {
            shuffle_schema,
            shuffle_locations,
        }
    }
}

impl PhysicalPlan for ShuffleReaderExec {
    fn schema(&self) -> Schema {
        self.shuffle_schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalPlan>> {
        // A shuffle read is a leaf — its input is the previous stage's output.
        vec![]
    }

    /// Rebuild this shuffle reader with new children. See the trait-level
    /// `PhysicalPlan::with_new_children` doc for the general rewrite pattern.
    ///
    /// Arity 0 (leaf): a shuffle reader has no input plan — its data comes
    /// from `shuffle_locations`. The incoming `children` vec is always
    /// empty; we hand back `self` unchanged.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalPlan>>,
    ) -> Arc<dyn PhysicalPlan> {
        assert!(
            children.is_empty(),
            "ShuffleReaderExec is a leaf and expects no children"
        );
        self
    }

    /// Read every shuffle location in order and yield the resulting
    /// `RecordBatch`es as a single iterator.
    ///
    /// **Local reads only.** A location whose `executor_id` doesn't match
    /// `ctx.executor_id` triggers `unimplemented!()`. Remote reads would
    /// require a Flight client field on `ExecutorContext`; not currently
    /// implemented.
    fn execute(&self, ctx: &ExecutorContext) -> Box<dyn Iterator<Item = RecordBatch>> {
        // Validate all locations are local up front so the panic — if one
        // belongs to another executor — fires before any disk I/O.
        for loc in &self.shuffle_locations {
            if loc.executor_id != ctx.executor_id {
                unimplemented!(
                    "ShuffleReaderExec: remote shuffle reads require an Arrow Flight \
                     client. Location belongs to executor '{}' but this executor is \
                     '{}'. Remote reads need a Flight client field on ExecutorContext.",
                    loc.executor_id,
                    ctx.executor_id
                );
            }
        }

        // Move owned copies into the closure: cheap Arc bump on the manager
        // (one atomic op) and a `Vec<ShuffleLocation>` clone (small pure data).
        let locations = self.shuffle_locations.clone();
        let shuffle_manager = Arc::clone(&ctx.shuffle_manager);

        Box::new(locations.into_iter().flat_map(move |location| {
            shuffle_manager.read_partition(
                &location.job_uuid,
                location.stage_id,
                location.partition_id,
            )
        }))
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
    //! Tests for `execute(ctx)` — full writer → reader round-trip via the
    //! unified context-aware trait method.

    use super::*;
    use crate::column_expression::ColumnExpression;
    use crate::scan_exec::ScanExec;
    use crate::shuffle_writer_exec::ShuffleWriterExec;
    use datasource::{CsvDataSource, DataSource};

    const EMPLOYEE_CSV: &str = "../testdata/employee.csv";

    fn temp_dir(tag: &str) -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("/tmp/rquery-shuffle-test-{tag}-{nanos}")
    }

    fn employee_ds() -> Arc<dyn DataSource> {
        Arc::new(CsvDataSource::new(EMPLOYEE_CSV, None, true, 1024))
    }

    fn employee_columns(ds: &Arc<dyn DataSource>) -> Vec<String> {
        ds.schema().fields.iter().map(|f| f.name.clone()).collect()
    }

    fn write_employee_shuffle(
        ctx: &ExecutorContext,
        job_uuid: &str,
        partition_count: i32,
    ) -> (usize, Vec<ShuffleLocation>, Schema) {
        let ds = employee_ds();
        let schema = ds.schema();
        let scan = Arc::new(ScanExec::new(Arc::clone(&ds), employee_columns(&ds)));
        let input_row_count: usize = scan.execute(ctx).map(|b| b.num_rows()).sum();
        let writer = ShuffleWriterExec::new(
            Arc::new(ScanExec::new(Arc::clone(&ds), employee_columns(&ds))),
            vec![Arc::new(ColumnExpression::new(0))],
            job_uuid,
            0,
            partition_count,
        );
        // Note: ShuffleWriterExec has a separate `write_shuffle(ctx) -> locations` method
        // because writers don't fit the "execute returns iterator" shape — they have a
        // side effect (write files) and return a location list, not record batches.
        let locations = writer.write_shuffle(ctx);
        (input_row_count, locations, schema)
    }

    #[test]
    fn writer_then_reader_round_trips_full_row_count() {
        let base = temp_dir("reader-roundtrip");
        let ctx = ExecutorContext::new("exec-test", "127.0.0.1", 50099, &base);

        let (input_rows, locations, schema) =
            write_employee_shuffle(&ctx, "test-job-reader-roundtrip", 3);

        let reader = ShuffleReaderExec::new(schema, locations);
        let read_rows: usize = reader.execute(&ctx).map(|b| b.num_rows()).sum();

        assert_eq!(
            read_rows, input_rows,
            "writer→reader must preserve all rows"
        );
        ctx.shuffle_manager.cleanup_all();
    }

    #[test]
    fn empty_locations_yields_empty_iterator() {
        let base = temp_dir("reader-empty");
        let ctx = ExecutorContext::new("exec-test", "127.0.0.1", 50099, &base);

        let ds = employee_ds();
        let reader = ShuffleReaderExec::new(ds.schema(), vec![]);
        let batches: Vec<RecordBatch> = reader.execute(&ctx).collect();

        assert!(batches.is_empty());
        ctx.shuffle_manager.cleanup_all();
    }

    #[test]
    fn single_partition_round_trip_reads_all_rows_from_one_file() {
        let base = temp_dir("reader-single");
        let ctx = ExecutorContext::new("exec-test", "127.0.0.1", 50099, &base);

        let (input_rows, locations, schema) =
            write_employee_shuffle(&ctx, "test-job-reader-single", 1);
        assert_eq!(
            locations.len(),
            1,
            "single partition writer emits 1 location"
        );

        let reader = ShuffleReaderExec::new(schema, locations);
        let read_rows: usize = reader.execute(&ctx).map(|b| b.num_rows()).sum();

        assert_eq!(read_rows, input_rows);
        ctx.shuffle_manager.cleanup_all();
    }

    #[test]
    #[should_panic(expected = "remote shuffle reads require an Arrow Flight client")]
    fn remote_location_panics_until_flight_client_lands() {
        let base = temp_dir("reader-remote");
        let ctx = ExecutorContext::new("exec-A", "127.0.0.1", 50099, &base);

        let remote_loc = ShuffleLocation::new("test-job-remote", 0, 0, "exec-B", "10.0.0.2", 50099);
        let reader = ShuffleReaderExec::new(employee_ds().schema(), vec![remote_loc]);
        let _ = reader.execute(&ctx);
    }
}
