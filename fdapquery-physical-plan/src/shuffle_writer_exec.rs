//!
//! Executes its input and writes the output to local shuffle files, partitioned by
//! the hash of a set of partition expressions. Used at shuffle boundaries in
//! distributed execution.
//!
//! ## Two execution surfaces
//! - The `ExecutionPlan::execute()` method returns
//!   `Err(NotImplemented(_))`. A writer can't run without knowing which
//!   executor it lives on (the `ShuffleLocation`s it reports embed the
//!   executor id/host/port) and where on local disk the shuffle storage is.
//!   The "produces RecordBatches" shape doesn't fit either — writers produce
//!   `Vec<ShuffleLocation>`.
//! - [`ShuffleWriterExec::write_shuffle`] is the real entry point. It takes an
//!   `Arc<TaskContext>` (built once per executor binary in `flight-server`)
//!   and returns the [`ShuffleLocation`]s the upstream stage can read from.
//!   This stays sync because the caller (`flight-server::do_action`) runs
//!   it on a `spawn_blocking` thread.
//!
//! ## Hash-partition algorithm
//! For each input batch, evaluate the partition expressions row-by-row, hash
//! the resulting tuple via `crate::row_key::RowKey` (the same float-aware
//! hasher `HashJoinExec`/`AggregateExec` use for join/group keys), take
//! modulo `partition_count` to pick a target partition, then filter the batch
//! into per-partition sub-batches. After all input is consumed, every
//! non-empty partition's sub-batches are written via
//! `ShuffleManager::write_partition` and a `ShuffleLocation` is added to the
//! returned vec.
//!
//! ## Empty-partition policy
//! Empty partitions get **no file** and **no `ShuffleLocation`**. This
//! matches `ShuffleManager::write_partition`'s no-op-on-empty contract.

use crate::PhysicalExpr;
use crate::physical_plan::ExecutionPlan;
use crate::plan_properties::PlanProperties;
use crate::row_key::RowKey;
use crate::stream::SendableRecordBatchStream;
use arrow_array::ArrayRef;
use fdapquery_common::{ArrowVectorBuilder, FdapQueryError, Result, ScalarValue};
use fdapquery_datatypes::{RecordBatch, Schema, record_batch};
use fdapquery_execution::ShuffleLocation;
use fdapquery_execution::TaskContext;
use futures::TryStreamExt;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Partitions input by hash and writes shuffle output.
#[derive(Debug)]
pub struct ShuffleWriterExec {
    pub input: Arc<dyn ExecutionPlan>,
    pub partition_expr: Vec<Arc<dyn PhysicalExpr>>,
    pub job_uuid: String,
    pub stage_id: i32,
    pub partition_count: i32,
    properties: PlanProperties,
}

impl ShuffleWriterExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        partition_expr: Vec<Arc<dyn PhysicalExpr>>,
        job_uuid: impl Into<String>,
        stage_id: i32,
        partition_count: i32,
    ) -> Self {
        // The writer's "output" (logically) is the shuffle locations, not
        // record batches — but the `ExecutionPlan::execute()` shape demands a
        // partitioning descriptor. Single-partition unknown is the closest
        // honest answer; the trait's `execute()` returns `NotImplemented`
        // anyway.
        let properties = PlanProperties::single_partition_unknown();
        Self {
            input,
            partition_expr,
            job_uuid: job_uuid.into(),
            stage_id,
            partition_count,
            properties,
        }
    }

    /// Execute the input plan, hash-partition the resulting rows by
    /// `partition_expr`, write each non-empty partition's batches to local
    /// shuffle storage via `ctx.runtime.shuffle_manager`, and return a
    /// [`ShuffleLocation`] tagged with this executor's identity for every
    /// partition that received at least one row.
    ///
    /// ## Why this isn't `ExecutionPlan::execute(partition, ctx)`
    ///
    /// The trait `execute(partition, ctx)` returns a `SendableRecordBatchStream`
    /// — an operator that *produces* batches. A shuffle writer *consumes*
    /// batches and produces a `Vec<ShuffleLocation>` instead. This is a
    /// fundamental shape mismatch, not a context-missing problem.
    ///
    /// ## Why this stays sync
    ///
    /// The caller (`flight-server::do_action("execute_task")`) already runs
    /// this method on a `spawn_blocking` thread, so blocking on the input
    /// stream is fine. We use `futures::executor::block_on` to drain the
    /// async input stream synchronously. Future work may revisit if the
    /// distributed module's call sites benefit from an async flavour.
    pub fn write_shuffle(&self, ctx: &Arc<TaskContext>) -> Result<Vec<ShuffleLocation>> {
        let partition_count = self.partition_count as usize;
        // Captured once — output schema equals input schema.
        let schema = self.input.schema();

        // Per-partition accumulators.
        let mut buffers: Vec<Vec<RecordBatch>> = (0..partition_count).map(|_| Vec::new()).collect();

        // Drain the input stream synchronously via `block_on`.
        let input_stream = self.input.execute(0, Arc::clone(ctx))?;
        let batches: Vec<RecordBatch> = futures::executor::block_on(input_stream.try_collect())?;

        for batch in batches {
            let row_count = batch.num_rows();
            let key_columns: Vec<ArrayRef> = self
                .partition_expr
                .iter()
                .map(|e| e.evaluate(&batch)?.into_array(row_count))
                .collect::<Result<Vec<_>>>()?;

            let targets = compute_targets(&key_columns, row_count, partition_count)?;

            for (partition_id, buffer) in buffers.iter_mut().enumerate() {
                let take: Vec<bool> = targets.iter().map(|&t| t == partition_id).collect();
                if take.iter().any(|&b| b) {
                    buffer.push(select_rows(&batch, &schema, &take)?);
                }
            }
        }

        // Write non-empty partitions and emit their locations.
        let mut locations = Vec::new();
        for (partition_id, batches) in buffers.into_iter().enumerate() {
            if batches.is_empty() {
                continue;
            }
            ctx.runtime.shuffle_manager.write_partition(
                &self.job_uuid,
                self.stage_id,
                partition_id as i32,
                &batches,
            )?;
            locations.push(ShuffleLocation::new(
                &self.job_uuid,
                self.stage_id,
                partition_id as i32,
                &ctx.executor_id,
                &ctx.executor_host,
                ctx.executor_port.into(),
            ));
        }
        Ok(locations)
    }
}

/// Compute the target partition for every row of an input batch by hashing
/// the row's partition-key tuple. Floats hash by bit pattern — same shape as
/// [`crate::row_key::RowKey`].
fn compute_targets(
    key_columns: &[ArrayRef],
    row_count: usize,
    partition_count: usize,
) -> Result<Vec<usize>> {
    let mut targets = Vec::with_capacity(row_count);
    for row in 0..row_count {
        let key: Vec<ScalarValue> = key_columns
            .iter()
            .map(|c| ScalarValue::try_from_array(c, row))
            .collect::<Result<Vec<_>>>()?;
        let mut hasher = DefaultHasher::new();
        RowKey(key).hash(&mut hasher);
        targets.push((hasher.finish() % partition_count as u64) as usize);
    }
    Ok(targets)
}

/// Build a new `RecordBatch` containing only the rows of `batch` where
/// `take[i]` is true.
fn select_rows(batch: &RecordBatch, schema: &Schema, take: &[bool]) -> Result<RecordBatch> {
    let count = take.iter().filter(|&&b| b).count();
    let columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|col_idx| -> Result<ArrayRef> {
            let source = batch.column(col_idx).clone();
            let mut builder = ArrowVectorBuilder::new(source.data_type(), count);
            for (row, &t) in take.iter().enumerate() {
                if t {
                    let value = ScalarValue::try_from_array(&source, row)?;
                    builder.append_value(&value);
                }
            }
            Ok(builder.build())
        })
        .collect::<Result<Vec<_>>>()?;
    record_batch::create(schema, columns)
}

impl ExecutionPlan for ShuffleWriterExec {
    fn name(&self) -> &'static str {
        "ShuffleWriterExec"
    }

    fn schema(&self) -> Schema {
        self.input.schema()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// Rebuild this shuffle writer with a new input child. Arity 1.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(FdapQueryError::Internal(format!(
                "ShuffleWriterExec::with_new_children expected 1 child, got {}",
                children.len()
            )));
        }
        Ok(Arc::new(ShuffleWriterExec::new(
            children.into_iter().next().unwrap(),
            self.partition_expr.clone(),
            self.job_uuid.clone(),
            self.stage_id,
            self.partition_count,
        )))
    }

    fn execute(
        &self,
        _partition: usize,
        _ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Shape mismatch: writers consume batches and produce
        // `Vec<ShuffleLocation>`, not a stream of batches. Use
        // `Self::write_shuffle(ctx)` instead — the
        // `do_action("execute_task")` handler in `flight-server`
        // downcasts to `ShuffleWriterExec` and calls it directly.
        Err(FdapQueryError::NotImplemented(
            "ShuffleWriterExec::execute() doesn't fit the trait's batch-yielding shape \
             — use write_shuffle(ctx) which returns Vec<ShuffleLocation>. \
             flight-server's do_action handler does this via downcast."
                .into(),
        ))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl crate::display::DisplayAs for ShuffleWriterExec {
    fn fmt_as(
        &self,
        _t: crate::display::DisplayFormatType,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
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

impl std::fmt::Display for ShuffleWriterExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as crate::display::DisplayAs>::fmt_as(
            self,
            crate::display::DisplayFormatType::Default,
            f,
        )
    }
}

#[cfg(test)]
mod tests {
    //! Tests for `write_shuffle`. Each test uses a per-test tempdir keyed by
    //! nanoseconds so parallel `cargo test` runs don't collide on disk.

    use super::*;
    use crate::Column;
    use crate::test_util::employee_source;
    use fdapquery_execution::ShuffleManager;
    use fdapquery_execution::{RuntimeEnv, SessionConfig};

    fn temp_dir(tag: &str) -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("/tmp/fdapquery-shuffle-test-{tag}-{nanos}")
    }

    fn make_ctx(executor_id: &str, host: &str, port: u16, base: &str) -> Arc<TaskContext> {
        let runtime = Arc::new(RuntimeEnv::new(Arc::new(ShuffleManager::new(
            base.to_string(),
        ))));
        Arc::new(TaskContext::new(
            executor_id,
            host,
            port,
            SessionConfig::new(),
            runtime,
        ))
    }

    #[test]
    fn writes_partitions_and_reports_locations_tagged_with_executor() {
        // 4-row employee fixture → partition by `id` into 3 buckets.
        let writer = ShuffleWriterExec::new(
            employee_source(),
            vec![Arc::new(Column::new("id", 0))], // partition by `id`
            "test-job-shuffle-writer",
            0, // stage_id
            3, // partition_count
        );

        let base = temp_dir("writer-happy");
        let ctx = make_ctx("exec-test", "127.0.0.1", 50099, &base);

        let locations = writer.write_shuffle(&ctx).unwrap();

        assert!(!locations.is_empty());
        assert!(locations.len() <= 3);

        for loc in &locations {
            assert_eq!(loc.job_uuid, "test-job-shuffle-writer");
            assert_eq!(loc.stage_id, 0);
            assert!(loc.partition_id >= 0 && loc.partition_id < 3);
            assert_eq!(loc.executor_id, "exec-test");
            assert_eq!(loc.executor_host, "127.0.0.1");
            assert_eq!(loc.executor_port, 50099);
        }

        // Round-trip via the shuffle manager's sync read API.
        let mut total_rows = 0;
        for loc in &locations {
            let batches: Vec<_> = ctx
                .runtime
                .shuffle_manager
                .read_partition(&loc.job_uuid, loc.stage_id, loc.partition_id)
                .unwrap()
                .collect::<Result<Vec<_>>>()
                .unwrap();
            total_rows += batches.iter().map(|b| b.num_rows()).sum::<usize>();
        }
        assert_eq!(total_rows, 4, "round-trip row count must match input");

        ctx.runtime.shuffle_manager.cleanup_all();
    }

    #[test]
    fn empty_input_produces_no_locations_and_no_files() {
        // A tiny stub operator that yields no batches.
        #[derive(Debug)]
        struct EmptyInput {
            schema: Schema,
            properties: PlanProperties,
        }
        impl EmptyInput {
            fn new(schema: Schema) -> Self {
                Self {
                    schema,
                    properties: PlanProperties::single_partition_unknown(),
                }
            }
        }
        impl ExecutionPlan for EmptyInput {
            fn name(&self) -> &'static str {
                "EmptyInput"
            }
            fn schema(&self) -> Schema {
                self.schema.clone()
            }
            fn properties(&self) -> &PlanProperties {
                &self.properties
            }
            fn execute(
                &self,
                _partition: usize,
                _ctx: Arc<TaskContext>,
            ) -> Result<SendableRecordBatchStream> {
                use crate::stream::RecordBatchStreamAdapter;
                let arrow_schema = Arc::new(self.schema.clone());
                let inner = futures::stream::empty::<Result<RecordBatch>>();
                Ok(Box::pin(RecordBatchStreamAdapter::new(arrow_schema, inner)))
            }
            fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
                vec![]
            }
            fn with_new_children(
                self: Arc<Self>,
                _children: Vec<Arc<dyn ExecutionPlan>>,
            ) -> Result<Arc<dyn ExecutionPlan>> {
                Ok(self)
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        impl crate::display::DisplayAs for EmptyInput {
            fn fmt_as(
                &self,
                _t: crate::display::DisplayFormatType,
                f: &mut std::fmt::Formatter<'_>,
            ) -> std::fmt::Result {
                write!(f, "EmptyInput")
            }
        }
        impl std::fmt::Display for EmptyInput {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                <Self as crate::display::DisplayAs>::fmt_as(
                    self,
                    crate::display::DisplayFormatType::Default,
                    f,
                )
            }
        }

        let writer = ShuffleWriterExec::new(
            Arc::new(EmptyInput::new(crate::test_util::employee_schema())),
            vec![Arc::new(Column::new("id", 0))],
            "test-job-shuffle-writer-empty",
            0,
            3,
        );

        let base = temp_dir("writer-empty");
        let ctx = make_ctx("exec-test", "127.0.0.1", 50099, &base);

        let locations = writer.write_shuffle(&ctx).unwrap();

        assert!(
            locations.is_empty(),
            "empty input must produce no locations"
        );

        for partition_id in 0..3 {
            let path = ctx.runtime.shuffle_manager.get_partition_file(
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

        ctx.runtime.shuffle_manager.cleanup_all();
    }

    #[test]
    fn single_partition_collects_all_rows_into_one_bucket() {
        // partition_count = 1 → every row must land in partition 0.
        let writer = ShuffleWriterExec::new(
            employee_source(),
            vec![Arc::new(Column::new("id", 0))],
            "test-job-shuffle-writer-one",
            0,
            1, // single partition
        );

        let base = temp_dir("writer-one");
        let ctx = make_ctx("exec-test", "127.0.0.1", 50099, &base);

        let locations = writer.write_shuffle(&ctx).unwrap();

        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0].partition_id, 0);

        let batches: Vec<_> = ctx
            .runtime
            .shuffle_manager
            .read_partition("test-job-shuffle-writer-one", 0, 0)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 4);

        ctx.runtime.shuffle_manager.cleanup_all();
    }
}
