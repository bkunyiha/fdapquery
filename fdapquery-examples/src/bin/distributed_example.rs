//! Single-process distributed-shape demo using a `LocalExecutorClient`.
//!
//! ## What this shows
//!
//! How a query is split into stages and how tasks are distributed across
//! executors, *without* requiring a real flight-server. Everything runs in
//! one process. The `LocalExecutorClient` plays the role of the transport:
//! it satisfies the `ExecutorClient` trait by running each task directly
//! against a shared in-process `ExecutorContext`. The same `ShuffleManager`
//! is used for writes and reads, so stage 0 writes shuffle files and
//! stage 1 reads them out of the same directory.
//!
//! The cluster config lists three executor entries — these are *descriptors*
//! (id/host/port) that the scheduler uses to round-robin tasks. Because
//! every call lands on the same in-process `LocalExecutorClient`, the
//! descriptors are purely cosmetic for this demo. The sibling
//! `distributed_flight_example` binary spawns a real flight-server and uses
//! the descriptors as real network addresses.
//!
//! ## Why this works end-to-end
//!
//! `PhysicalPlan::execute` takes `&ExecutorContext` as a trait-method
//! parameter, so `task.plan.execute(&ctx)` works for any plan tree
//! including one with a `ShuffleReaderExec`: the context flows through
//! every operator including the reader.
//!
//! ## How to run
//!
//! ```text
//! cd examples && cargo run --bin distributed_example
//! ```
//!
//! The `testdata/employee.csv` path is relative to the `examples/` crate
//! directory, matching the convention used by the other example binaries
//! in this crate.

use std::sync::Arc;
use std::time::Instant;

use datatypes::{ArrowFieldVector, ColumnVector, RecordBatch, ScalarValue};
use distributed::{DistributedConfig, DistributedContext, ExecutorClient, ExecutorConfig};
use physical_plan::{ExecutorContext, ShuffleLocation, ShuffleManager, ShuffleWriterExec, Task};

const EMPLOYEE_CSV: &str = "../testdata/employee.csv";
const SQL: &str = "SELECT state, SUM(salary) FROM employee GROUP BY state";

fn main() {
    env_logger::init();

    println!("=== Distributed Query Execution Example (Local) ===\n");
    println!("Query: {SQL}\n");

    // Configure a cluster with 3 executors. In this demo every "executor"
    // routes to the same in-process LocalExecutorClient; the IDs and ports
    // are purely cosmetic for the round-robin dispatch.
    let config = DistributedConfig::new(vec![
        ExecutorConfig::new("exec-1", "localhost", 50051),
        ExecutorConfig::new("exec-2", "localhost", 50052),
        ExecutorConfig::new("exec-3", "localhost", 50053),
    ])
    .with_default_partitions(3);

    println!(
        "Configured cluster with {} executors:",
        config.executors.len()
    );
    for e in &config.executors {
        println!("  - {} at {}:{}", e.id, e.host, e.port);
    }
    println!();

    // Fresh shuffle directory per run — nanosecond-keyed so parallel runs
    // don't collide on disk.
    let shuffle_dir = unique_shuffle_dir();
    let executor_client = LocalExecutorClient::new(&shuffle_dir);

    // Build the context and register the test data.
    let mut ctx = DistributedContext::new(config, executor_client);
    ctx.register_csv("employee", EMPLOYEE_CSV, true);

    // Execute the query.
    println!("Executing query (stage 0 → 3 shuffle-writer tasks, stage 1 → 1 final task):");
    let start = Instant::now();
    let results: Vec<RecordBatch> = ctx.sql(SQL).collect();
    let elapsed = start.elapsed().as_millis();
    println!("\nExecution completed in {elapsed}ms\n");

    println!("Results:");
    print_results(&results);

    // Clean up shuffle files left by stage 0.
    ShuffleManager::new(shuffle_dir).cleanup_all();

    println!("\n=== Example Complete ===");
}

/// An `ExecutorClient` that runs every task in the current process against a
/// single shared `ExecutorContext`. The `ExecutorConfig` descriptor passed by
/// the scheduler is logged but otherwise ignored — every task runs in-process
/// against the same shuffle manager.
///
/// Works end-to-end for aggregate queries because `PhysicalPlan::execute(&ctx)`
/// flows the context through `ShuffleReaderExec`.
struct LocalExecutorClient {
    /// Single shared executor context. All tasks see the same `executor_id`
    /// (`"local-executor"`) and the same `Arc<ShuffleManager>`, so every
    /// `ShuffleLocation` written in stage 0 is tagged with this id and every
    /// stage-1 read finds the matching `executor_id == ctx.executor_id` and
    /// reads via `ctx.shuffle_manager` (the local-path branch of
    /// `ShuffleReaderExec::execute`).
    ctx: Arc<ExecutorContext>,
}

impl LocalExecutorClient {
    fn new(shuffle_dir: &str) -> Self {
        Self {
            ctx: Arc::new(ExecutorContext::new(
                "local-executor",
                "localhost",
                0,
                shuffle_dir,
            )),
        }
    }
}

impl ExecutorClient for LocalExecutorClient {
    fn execute_task(&self, executor: &ExecutorConfig, task: Task) -> Vec<ShuffleLocation> {
        println!(
            "  [{}] execute_task stage={} task={} partition={}",
            executor.id, task.stage_id, task.task_id, task.partition_id,
        );

        // Stage 0 tasks ship a `ShuffleWriterExec`. We downcast to call the
        // sibling `write_shuffle(&ctx)` directly — the writer's trait
        // `execute()` deliberately panics because its return shape
        // (`Iterator<RecordBatch>`) doesn't fit "produce shuffle locations."
        if let Some(writer) = task.plan.as_any().downcast_ref::<ShuffleWriterExec>() {
            writer.write_shuffle(&self.ctx)
        } else {
            // Non-shuffle intermediate stage — drain and return no locations.
            task.plan.execute(&self.ctx).for_each(|_| {});
            Vec::new()
        }
    }

    fn execute_final_task(
        &self,
        executor: &ExecutorConfig,
        task: Task,
    ) -> Box<dyn Iterator<Item = RecordBatch>> {
        println!(
            "  [{}] execute_final_task stage={} task={} partition={}",
            executor.id, task.stage_id, task.task_id, task.partition_id,
        );

        // The final-stage plan is `HashAggregateExec(Final)` wrapping a
        // `ShuffleReaderExec` whose `shuffle_locations` were populated by
        // `DistributedPlanner::update_shuffle_locations`. `execute(&ctx)`
        // flows the context through the aggregate to the reader, which
        // reads via `ctx.shuffle_manager.read_partition(...)`.
        task.plan.execute(&self.ctx)
    }

    fn fetch_shuffle(
        &self,
        _executor: &ExecutorConfig,
        location: &ShuffleLocation,
    ) -> Box<dyn Iterator<Item = RecordBatch>> {
        // Single-process demo: `fetch_shuffle` is reached only if the
        // `ShuffleReaderExec` sees a location with `executor_id !=
        // ctx.executor_id`. With our shared `ctx` (id = "local-executor")
        // and locations all tagged with the same id by `write_shuffle`,
        // this branch should not fire — but if it does, we read locally.
        self.ctx.shuffle_manager.read_partition(
            &location.job_uuid,
            location.stage_id,
            location.partition_id,
        )
    }
}

fn unique_shuffle_dir() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("/tmp/rquery-distributed-example-{nanos}")
}

/// Print every `(state, sum)` row in the result batches.
fn print_results(batches: &[RecordBatch]) {
    for batch in batches {
        let state_col = ArrowFieldVector::new(batch.column(0).clone());
        let sum_col = ArrowFieldVector::new(batch.column(1).clone());
        for row in 0..batch.num_rows() {
            let key = scalar_to_string(&state_col.get_value(row));
            let value = sum_col.get_value(row);
            println!("  {key}: {value:?}");
        }
    }
}

/// Stringify a `ScalarValue` for `(state, sum)` display. Same pattern as
/// `parallel_execution_example.rs`.
fn scalar_to_string(v: &ScalarValue) -> String {
    match v {
        ScalarValue::Utf8(s) => s.clone(),
        ScalarValue::Binary(b) => String::from_utf8_lossy(b).into_owned(),
        ScalarValue::Null => "null".to_string(),
        other => format!("{other:?}"),
    }
}
