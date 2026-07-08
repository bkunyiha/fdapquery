//! Distributed query demo using `SessionContext::standalone()` —
//! a `FlightExecutorClient` against an in-process `flight-server`
//! spawned automatically by the standalone constructor.
//!
//! ## What this shows
//!
//! `SessionContext::standalone().await` is fdapquery's mirror of
//! Ballista's zero-arg standalone constructor. It:
//!
//! 1. Spawns a `flight-server` on a random TCP port in a background
//!    thread with its own tokio runtime.
//! 2. Builds a `DistributedConfig` pointing at the bound address.
//! 3. Connects a `FlightExecutorClient` to it.
//! 4. Installs a `DistributedQueryPlanner` on the `SessionState` so
//!    every query routes through the in-process scheduler +
//!    real Flight gRPC.
//!
//! The caller sees a plain `SessionContext` — same type used for
//! single-node execution. Distribution is invisible below the
//! surface. That's the whole point of the extension-trait pattern
//! (see `PHASE_2_PLAN.md` → "The DataFusion / Ballista layering").
//!
//! Once `ctx.sql(...)` and `ctx.execute_data_frame(&df).await`
//! run, the pipeline goes:
//!
//! 1. `Scheduler::execute_stage` ships each stage-0 task via
//!    `FlightExecutorClient::execute_task` → `Client::do_action("execute_task")`
//!    → tonic gRPC → `FdapQueryFlightProducer::do_action` →
//!    `ShuffleWriterExec::write_shuffle(&ctx)` → Arrow IPC files on disk.
//! 2. `Scheduler::execute_final_stage` ships the stage-1 task via
//!    `FlightExecutorClient::execute_final_task` →
//!    `Client::do_get(protobuf::Action.task = Some(...))` → tonic gRPC →
//!    `FdapQueryFlightProducer::do_get` (distributed branch) →
//!    `task.plan.execute(&self.ctx)` → `AggregateExec(Final)` →
//!    `ShuffleReaderExec::execute(&ctx)` → batches streamed back through
//!    `FlightDataEncoder`.
//!
//! ## Single-executor cluster
//!
//! For demo simplicity the cluster has one executor — the same in-process
//! `flight-server` plays the role of every executor. With one executor, all
//! shuffle locations are local, so the stage-1 `ShuffleReaderExec` reads via
//! `ctx.shuffle_manager` and never exercises the cross-executor
//! `fetch_shuffle` path (which is currently unimplemented).
//!
//! ## How to run
//!
//! ```text
//! cd examples && cargo run --bin distributed_flight_example
//! ```

use std::time::Instant;

use fdapquery::SessionContext;
use fdapquery_datatypes::RecordBatch;
use fdapquery_datatypes::record_batch::to_csv;
use fdapquery_flight_client::SessionContextExt;
use futures::TryStreamExt;

const EMPLOYEE_CSV: &str = "../testdata/employee.csv";
const SQL: &str = "SELECT state, SUM(salary) FROM employee GROUP BY state";

#[tokio::main]
async fn main() {
    env_logger::init();

    println!("=== Distributed Query Execution Example (Flight) ===\n");
    println!("Query: {SQL}\n");

    // `SessionContext::standalone()` spawns the in-process
    // Flight-server executor and connects a `FlightExecutorClient`
    // to it. Zero args — mirror of Ballista's `standalone()`.
    let mut ctx = SessionContext::standalone()
        .await
        .expect("distributed_flight_example: SessionContext::standalone");
    ctx.register_csv("employee", EMPLOYEE_CSV);

    // Execute the query. `SessionContext::sql` is sync; the async
    // `execute_data_frame` drives the plan through the query
    // planner and returns a stream. Every Flight call is awaited
    // on the same tokio runtime; the result stream is decoded as
    // it arrives.
    println!(
        "Executing query (stage 0 → 3 shuffle-writer tasks via do_action, stage 1 → 1 final task via do_get):"
    );
    let start = Instant::now();
    let df = ctx.sql(SQL).expect("distributed_flight_example: sql");
    let stream = ctx
        .execute_data_frame(&df)
        .await
        .expect("distributed_flight_example: execute_data_frame");
    let results: Vec<RecordBatch> = stream
        .try_collect()
        .await
        .expect("distributed_flight_example: drain stream");
    let elapsed = start.elapsed().as_millis();
    println!("\nExecution completed in {elapsed}ms\n");

    println!("Results:");
    print_results(&results);

    println!("\n=== Example Complete ===");
}

fn print_results(batches: &[RecordBatch]) {
    if batches.is_empty() {
        println!("  (no results)");
        return;
    }
    for batch in batches {
        match to_csv(batch) {
            Ok(csv) => {
                for line in csv.lines() {
                    println!("  {line}");
                }
            }
            Err(e) => println!("  (error rendering batch: {e})"),
        }
    }
}
