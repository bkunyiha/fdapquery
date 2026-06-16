//! Distributed query demo using a real `FlightExecutorClient` against an
//! in-process `flight-server` — the rquery answer to "what does a real
//! distributed query look like?"
//!
//! ## What this shows
//!
//! Spawns a `flight-server` in a background thread bound to a random TCP port,
//! then constructs a `FlightExecutorClient` that talks to it over real Arrow
//! Flight gRPC. The `DistributedContext` drives the query through:
//!
//! 1. `Scheduler::execute_stage` ships each stage-0 task via
//!    `FlightExecutorClient::execute_task` → `Client::do_action("execute_task")`
//!    → tonic gRPC → `RQueryFlightProducer::do_action` →
//!    `ShuffleWriterExec::write_shuffle(&ctx)` → Arrow IPC files on disk.
//! 2. `Scheduler::execute_final_stage` ships the stage-1 task via
//!    `FlightExecutorClient::execute_final_task` →
//!    `Client::do_get(pb::Action.task = Some(...))` → tonic gRPC →
//!    `RQueryFlightProducer::do_get` (distributed branch) →
//!    `task.plan.execute(&self.ctx)` → `HashAggregateExec(Final)` →
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
//! Forcing 3 partitions via `DistributedConfig::with_default_partitions(3)`
//! means the shuffle is still real — stage 0 hash-partitions employee rows
//! into 3 partition files, stage 1's final aggregate reads all three.
//!
//! ## Threading model
//!
//! `Client::new` builds its own tokio runtime and uses `block_on` for the gRPC
//! connect. `block_on` panics inside an existing tokio runtime context, so we
//! can't use `#[tokio::main]` here. `main()` is plain sync. The server runs in
//! a `std::thread::spawn`ed background thread that owns its own tokio runtime;
//! an `mpsc` channel ships the bound address back to the main thread.
//!
//! ## How to run
//!
//! ```text
//! cd examples && cargo run --bin distributed_flight_example
//! ```
//!
//! The sibling binary `distributed_example` runs the same query through a
//! `LocalExecutorClient` (no flight-server) — useful for understanding the
//! scheduler shape without the gRPC layer.

use std::sync::mpsc;
use std::time::Instant;

use arrow_flight::flight_service_server::FlightServiceServer;
use client::FlightExecutorClient;
use datatypes::{ArrowFieldVector, ColumnVector, RecordBatch, ScalarValue};
use distributed::{DistributedConfig, DistributedContext, ExecutorConfig};
use flight_server::r_query_flight_producer::RQueryFlightProducer;
use physical_plan::{ExecutorContext, ShuffleManager};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;

const EMPLOYEE_CSV: &str = "../testdata/employee.csv";
const SQL: &str = "SELECT state, SUM(salary) FROM employee GROUP BY state";

fn main() {
    env_logger::init();

    println!("=== Distributed Query Execution Example (Flight) ===\n");
    println!("Query: {SQL}\n");

    // Spawn an in-process flight-server on a random TCP port. The bound
    // address is sent back through the mpsc channel.
    let (addr, shuffle_dir) = spawn_in_process_server("exec-1");
    println!("flight-server bound at {addr}");
    println!("shuffle directory: {shuffle_dir}\n");

    // Build the cluster config pointed at the in-process server. One
    // executor; force 3 partitions so the shuffle is real (otherwise
    // default_partitions = executor count = 1, no redistribution).
    let executors = vec![ExecutorConfig::new(
        "exec-1",
        "127.0.0.1",
        addr.port() as i32,
    )];
    let config = DistributedConfig::new(executors.clone()).with_default_partitions(3);

    println!(
        "Configured cluster with {} executor:",
        config.executors.len()
    );
    for e in &config.executors {
        println!("  - {} at {}:{}", e.id, e.host, e.port);
    }
    println!();

    // Build the real Flight client (it connects on construction; panics if
    // the in-process server isn't ready yet — but the mpsc handshake above
    // guarantees the server has bound its socket before we get here).
    let flight_client = FlightExecutorClient::new(&executors)
        .expect("FlightExecutorClient::new should connect to the in-process server");

    // Build the context and register the test data.
    let mut ctx = DistributedContext::new(config, flight_client);
    ctx.register_csv("employee", EMPLOYEE_CSV, true);

    // Execute the query. The scheduler drives every Flight call
    // synchronously; the result iterator buffers the decoded RecordBatches
    // from the server's streaming `do_get` response.
    println!(
        "Executing query (stage 0 → 3 shuffle-writer tasks via do_action, stage 1 → 1 final task via do_get):"
    );
    let start = Instant::now();
    let results: Vec<RecordBatch> = ctx.sql(SQL).collect();
    let elapsed = start.elapsed().as_millis();
    println!("\nExecution completed in {elapsed}ms\n");

    println!("Results:");
    print_results(&results);

    // Clean up shuffle files left by stage 0. (The server's tokio runtime
    // keeps running in the background — main() exits and the OS reaps it.)
    ShuffleManager::new(shuffle_dir).cleanup_all();

    println!("\n=== Example Complete ===");
}

/// Spawn an in-process flight-server in a background thread with its own
/// tokio runtime. Returns the bound `SocketAddr` (so the client can connect)
/// and the shuffle directory path (so `main` can clean up at the end).
///
/// Same pattern as
/// `client/tests/distributed_integration_test.rs::spawn_in_process_server`.
fn spawn_in_process_server(executor_id: &str) -> (std::net::SocketAddr, String) {
    let shuffle_dir = unique_shuffle_dir();
    let shuffle_dir_for_thread = shuffle_dir.clone();
    let executor_id_owned = executor_id.to_string();
    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("build server runtime");
        runtime.block_on(async move {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind random port");
            let addr = listener.local_addr().expect("local_addr");

            // The executor identity in the context must match the id/port
            // the scheduler dispatches against — otherwise shuffle reads
            // see locations with `executor_id != ctx.executor_id` and try
            // the cross-executor fetch path (currently unimplemented).
            let ctx = ExecutorContext::new(
                executor_id_owned,
                "127.0.0.1",
                addr.port() as i32,
                shuffle_dir_for_thread,
            );
            let producer = RQueryFlightProducer::new(ctx);

            tx.send(addr).expect("ship addr back to main thread");

            Server::builder()
                .add_service(FlightServiceServer::new(producer))
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await
                .expect("server serve");
        });
    });

    let addr = rx.recv().expect("server thread sent addr");
    (addr, shuffle_dir)
}

fn unique_shuffle_dir() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("/tmp/rquery-distributed-flight-example-{nanos}")
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
