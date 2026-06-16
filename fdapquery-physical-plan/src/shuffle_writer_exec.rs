//!
//! Executes its input and writes the output to local shuffle files, partitioned by
//! the hash of a set of partition expressions. Used at shuffle boundaries in
//! distributed execution.
//!
//! ## Two execution surfaces
//! - The `PhysicalPlan::execute()` method panics. A writer can't run
//!   without knowing which executor it lives on (the `ShuffleLocation`s it
//!   reports embed the executor id/host/port) and where on local disk the
//!   shuffle storage is. There is no sensible value `execute()` could return
//!   without that information.
//! - [`Self::write_shuffle`] is the real entry point. It takes an
//!   [`ExecutorContext`] (built once per executor binary in `flight-server`)
//!   and returns the [`ShuffleLocation`]s the upstream stage can read from.
//!
//! ## Hash-partition algorithm
//! For each input batch, evaluate the partition expressions row-by-row, hash
//! the resulting tuple via [`crate::row_key::RowKey`] (the same float-aware
//! hasher `HashJoinExec`/`HashAggregateExec` use for join/group keys), take
//! modulo `partition_count` to pick a target partition, then filter the batch
//! into per-partition sub-batches. After all input is consumed, every
//! non-empty partition's sub-batches are written via
//! `ShuffleManager::write_partition` and a `ShuffleLocation` is added to the
//! returned vec.
//!
//! ## Empty-partition policy
//! Empty partitions get **no file** and **no `ShuffleLocation`**. This
//! matches `ShuffleManager::write_partition`'s no-op-on-empty contract. The
//! downstream reader sees only the locations that actually contain data; the
//! scheduler's location list is always the union across all tasks, so a
//! per-task gap is fine.

use crate::executor_context::ExecutorContext;
use crate::expressions::Expression;
use crate::physical_plan::PhysicalPlan;
use crate::row_key::RowKey;
use crate::shuffle_location::ShuffleLocation;
use datatypes::{ArrowVectorBuilder, ColumnVector, RecordBatch, ScalarValue, Schema, record_batch};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Partitions input by hash and writes shuffle output.
pub struct ShuffleWriterExec {
    pub input: Arc<dyn PhysicalPlan>,
    pub partition_expr: Vec<Arc<dyn Expression>>,
    pub job_uuid: String,
    pub stage_id: i32,
    pub partition_count: i32,
}

impl ShuffleWriterExec {
    pub fn new(
        input: Arc<dyn PhysicalPlan>,
        partition_expr: Vec<Arc<dyn Expression>>,
        job_uuid: impl Into<String>,
        stage_id: i32,
        partition_count: i32,
    ) -> Self {
        Self {
            input,
            partition_expr,
            job_uuid: job_uuid.into(),
            stage_id,
            partition_count,
        }
    }

    /// Execute the input plan, hash-partition the resulting rows by
    /// `partition_expr`, write each non-empty partition's batches to local
    /// shuffle storage via `ctx.shuffle_manager`, and return a
    /// [`ShuffleLocation`] tagged with this executor's identity for every
    /// partition that received at least one row.
    ///
    /// ## Why this isn't `PhysicalPlan::execute(ctx)`
    ///
    /// The trait `execute(ctx)` returns `Box<dyn Iterator<Item = RecordBatch>>`
    /// — an operator that *produces* batches. A shuffle writer
    /// *consumes* batches and produces a `Vec<ShuffleLocation>` instead
    /// (the input batches are written to disk, not streamed onward). This
    /// is a fundamental shape mismatch, not a context-missing problem. The
    /// trait `execute(ctx)` on `ShuffleWriterExec` panics with a message
    /// pointing here. The `do_action("execute_task")` handler in
    /// `flight-server` downcasts to `ShuffleWriterExec` and calls this
    /// method directly.
    ///
    pub fn write_shuffle(&self, ctx: &ExecutorContext) -> Vec<ShuffleLocation> {
        let partition_count = self.partition_count as usize;
        // Captured once — output schema equals input schema (a shuffle preserves
        // the columns; it only re-distributes the rows).
        let schema = self.input.schema();

        // Per-partition accumulators. Index by partition_id directly.
        let mut buffers: Vec<Vec<RecordBatch>> = (0..partition_count).map(|_| Vec::new()).collect();

        // Pull each input batch, decide each row's target partition, push the
        // filtered sub-batch into that partition's accumulator.
        for batch in self.input.execute(ctx) {
            let key_columns: Vec<Box<dyn ColumnVector>> = self
                .partition_expr
                .iter()
                .map(|e| e.evaluate(&batch))
                .collect();

            let row_count = batch.num_rows();
            let targets = compute_targets(&key_columns, row_count, partition_count);

            for (partition_id, buffer) in buffers.iter_mut().enumerate() {
                let take: Vec<bool> = targets.iter().map(|&t| t == partition_id).collect();
                if take.iter().any(|&b| b) {
                    buffer.push(select_rows(&batch, &schema, &take));
                }
            }
        }

        // Write non-empty partitions and emit their locations.
        // The on-disk shuffle layout is:
        // {base_dir}/
        //   {job_uuid}/
        //     {stage_id}/
        //       partition_{partition_id}.arrow
        //  So for example, with the default shuffle base dir:
        // /tmp/rquery-shuffle/
        //   550e8400-e29b-41d4-a716-446655440000/
        //     0/
        //       partition_0.arrow
        //       partition_1.arrow
        //       partition_2.arrow
        let mut locations = Vec::new();
        for (partition_id, batches) in buffers.into_iter().enumerate() {
            if batches.is_empty() {
                continue;
            }
            ctx.shuffle_manager.write_partition(
                &self.job_uuid,
                self.stage_id,
                partition_id as i32,
                &batches,
            );
            locations.push(ShuffleLocation::new(
                &self.job_uuid,
                self.stage_id,
                partition_id as i32,
                &ctx.executor_id,
                &ctx.executor_host,
                ctx.executor_port,
            ));
        }
        locations
    }
}

/// Compute the target partition for every row of an input batch by hashing
/// the row's partition-key tuple. Floats hash by bit pattern — same shape as
/// [`crate::row_key::RowKey`]. The use of `DefaultHasher::new()` (zero seed)
/// makes the partition assignment deterministic across runs, which keeps
/// tests reproducible.
fn compute_targets(
    key_columns: &[Box<dyn ColumnVector>],
    row_count: usize,
    partition_count: usize,
) -> Vec<usize> {
    let mut targets = Vec::with_capacity(row_count);
    for row in 0..row_count {
        let key: Vec<ScalarValue> = key_columns.iter().map(|c| c.get_value(row)).collect();
        let mut hasher = DefaultHasher::new();
        RowKey(key).hash(&mut hasher);
        targets.push((hasher.finish() % partition_count as u64) as usize);
    }
    targets
}

/// Build a new `RecordBatch` containing only the rows of `batch` where
/// `take[i]` is true. The output schema equals the supplied `schema`. Same
/// row-by-row construction shape as `SelectionExec::filter`, generalised to
/// a boolean selection slice instead of a `ColumnVector`.
fn select_rows(batch: &RecordBatch, schema: &Schema, take: &[bool]) -> RecordBatch {
    let count = take.iter().filter(|&&b| b).count();
    let columns: Vec<Box<dyn ColumnVector>> = (0..batch.num_columns())
        .map(|col_idx| {
            let source = record_batch::field(batch, col_idx);
            let mut builder = ArrowVectorBuilder::new(&source.get_type(), count);
            for (row, &t) in take.iter().enumerate() {
                if t {
                    builder.append_value(&source.get_value(row));
                }
            }
            builder.set_value_count(count);
            Box::new(builder.build()) as Box<dyn ColumnVector>
        })
        .collect();
    record_batch::create(schema, columns)
}

impl PhysicalPlan for ShuffleWriterExec {
    fn schema(&self) -> Schema {
        self.input.schema()
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalPlan>> {
        vec![&self.input]
    }

    /// Rebuild this shuffle writer with a new input child. See the trait-level
    /// `PhysicalPlan::with_new_children` doc for the general rewrite pattern.
    ///
    /// Arity 1: a shuffle writer wraps exactly one input — the operator whose
    /// output will be hash-partitioned and written to local shuffle files.
    /// `into_iter().next().unwrap()` consumes the length-1 children vec and
    /// takes ownership of that single Arc.
    ///
    /// The shuffle-identifying fields (`partition_expr`, `job_uuid`,
    /// `stage_id`, `partition_count`) are reused — they describe where this
    /// stage's output goes, which is independent of which concrete input
    /// produces the rows.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalPlan>>,
    ) -> Arc<dyn PhysicalPlan> {
        assert_eq!(
            children.len(),
            1,
            "ShuffleWriterExec expects exactly 1 child"
        );
        Arc::new(ShuffleWriterExec::new(
            children.into_iter().next().unwrap(),
            self.partition_expr.clone(),
            self.job_uuid.clone(),
            self.stage_id,
            self.partition_count,
        ))
    }

    fn execute(&self, _ctx: &ExecutorContext) -> Box<dyn Iterator<Item = RecordBatch>> {
        // Shape mismatch: shuffle writers consume batches and produce
        // `Vec<ShuffleLocation>`, not iterators of batches. The trait's
        // `execute(ctx) -> Iterator<RecordBatch>` shape doesn't fit. Use
        // `Self::write_shuffle(&ctx)` instead — the `do_action("execute_task")`
        // handler in `flight-server` downcasts to `ShuffleWriterExec` and
        // calls it directly.
        unimplemented!(
            "ShuffleWriterExec::execute() doesn't fit the trait's batch-yielding \
             shape — use write_shuffle(&ExecutorContext) which returns Vec<ShuffleLocation>. \
             flight-server's do_action handler does this via downcast."
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl std::fmt::Display for ShuffleWriterExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let exprs: Vec<String> = self.partition_expr.iter().map(|e| e.to_string()).collect();
        write!(
            f,
            "ShuffleWriterExec: jobUuid={}, stageId={}, partitionCount={}, partitionExpr=[{}]",
            self.job_uuid,
            self.stage_id,
            self.partition_count,
            exprs.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    //! Tests for `write_shuffle`. Each test uses a per-test tempdir keyed by
    //! nanoseconds so parallel `cargo test` runs don't collide on disk;
    //! `cleanup_all` runs at the end of each test to keep `/tmp/` tidy.

    use super::*;
    use crate::column_expression::ColumnExpression;
    use crate::scan_exec::ScanExec;
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

    #[test]
    fn writes_partitions_and_reports_locations_tagged_with_executor() {
        // 4-row employee.csv → partition by `id` into 3 buckets.
        let ds = employee_ds();
        let scan = Arc::new(ScanExec::new(Arc::clone(&ds), employee_columns(&ds)));
        let writer = ShuffleWriterExec::new(
            scan,
            vec![Arc::new(ColumnExpression::new(0))], // partition by `id`
            "test-job-shuffle-writer",
            0, // stage_id
            3, // partition_count
        );

        let base = temp_dir("writer-happy");
        let ctx = ExecutorContext::new("exec-test", "127.0.0.1", 50099, &base);

        let locations = writer.write_shuffle(&ctx);

        // At least one partition must be non-empty (4 rows can't all hash to a
        // disjoint bucket-set), and the count never exceeds partition_count.
        assert!(!locations.is_empty());
        assert!(locations.len() <= 3);

        // Every reported location is tagged with this executor's identity and
        // the same job/stage we configured. partition_id is in [0, 3).
        for loc in &locations {
            assert_eq!(loc.job_uuid, "test-job-shuffle-writer");
            assert_eq!(loc.stage_id, 0);
            assert!(loc.partition_id >= 0 && loc.partition_id < 3);
            assert_eq!(loc.executor_id, "exec-test");
            assert_eq!(loc.executor_host, "127.0.0.1");
            assert_eq!(loc.executor_port, 50099);
        }

        // Round-trip: the union of rows across the written partition files
        // equals the input row count (4).
        let mut total_rows = 0;
        for loc in &locations {
            let batches: Vec<_> = ctx
                .shuffle_manager
                .read_partition(&loc.job_uuid, loc.stage_id, loc.partition_id)
                .collect();
            total_rows += batches.iter().map(|b| b.num_rows()).sum::<usize>();
        }
        assert_eq!(total_rows, 4, "round-trip row count must match input");

        ctx.shuffle_manager.cleanup_all();
    }

    #[test]
    fn empty_input_produces_no_locations_and_no_files() {
        // Use a 1-column projection over an empty filter result. Cheapest way
        // to get a real ScanExec → empty stream is to project a column that
        // exists, then never read any batches… actually, ScanExec over the
        // CSV always yields rows. Easier: wire a SelectionExec with a literal
        // false predicate? Even simpler: just give the scan an empty
        // projection. CsvDataSource's iterator still yields batches, so the
        // cleanest test is to construct a never-yielding source. Instead, we
        // build a manual zero-row helper inline.
        struct EmptyInput {
            schema: Schema,
        }
        impl PhysicalPlan for EmptyInput {
            fn schema(&self) -> Schema {
                self.schema.clone()
            }
            fn execute(&self, _ctx: &ExecutorContext) -> Box<dyn Iterator<Item = RecordBatch>> {
                Box::new(std::iter::empty())
            }
            fn children(&self) -> Vec<&Arc<dyn PhysicalPlan>> {
                vec![]
            }
            fn with_new_children(
                self: Arc<Self>,
                _children: Vec<Arc<dyn PhysicalPlan>>,
            ) -> Arc<dyn PhysicalPlan> {
                self
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        impl std::fmt::Display for EmptyInput {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "EmptyInput")
            }
        }

        let ds = employee_ds();
        let writer = ShuffleWriterExec::new(
            Arc::new(EmptyInput {
                schema: ds.schema(),
            }),
            vec![Arc::new(ColumnExpression::new(0))],
            "test-job-shuffle-writer-empty",
            0,
            3,
        );

        let base = temp_dir("writer-empty");
        let ctx = ExecutorContext::new("exec-test", "127.0.0.1", 50099, &base);

        let locations = writer.write_shuffle(&ctx);

        assert!(
            locations.is_empty(),
            "empty input must produce no locations"
        );

        // No files should have been created (the no-op-on-empty contract of
        // ShuffleManager::write_partition isn't even reached — we don't call
        // it for empty buffers).
        for partition_id in 0..3 {
            let path = ctx.shuffle_manager.get_partition_file(
                "test-job-shuffle-writer-empty",
                0,
                partition_id,
            );
            assert!(
                !path.exists(),
                "no file should exist for empty partition {partition_id}: {}",
                path.display()
            );
        }

        ctx.shuffle_manager.cleanup_all();
    }

    #[test]
    fn single_partition_collects_all_rows_into_one_bucket() {
        // partition_count = 1 → every row must land in partition 0,
        // regardless of how the partition_expr hashes.
        let ds = employee_ds();
        let scan = Arc::new(ScanExec::new(Arc::clone(&ds), employee_columns(&ds)));
        let writer = ShuffleWriterExec::new(
            scan,
            vec![Arc::new(ColumnExpression::new(0))],
            "test-job-shuffle-writer-one",
            0,
            1, // single partition
        );

        let base = temp_dir("writer-one");
        let ctx = ExecutorContext::new("exec-test", "127.0.0.1", 50099, &base);

        let locations = writer.write_shuffle(&ctx);

        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0].partition_id, 0);

        let batches: Vec<_> = ctx
            .shuffle_manager
            .read_partition("test-job-shuffle-writer-one", 0, 0)
            .collect();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 4);

        ctx.shuffle_manager.cleanup_all();
    }
}
