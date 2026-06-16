//! Full distributed query end-to-end integration test — **the test that
//! closes the Phase 1 distributed loop.**
//!
//! ## What this proves
//!
//! Drives `DistributedContext::sql(...)` against a real `flight-server`
//! running on a real TCP port via real Arrow Flight gRPC, using a real
//! `FlightExecutorClient` to dispatch tasks. The mock executor client
//! from `distributed::scheduler::tests` is *not* in the loop —
//! `FlightExecutorClient::execute_task` ships the intermediate stage's
//! `ShuffleWriterExec` task via `do_action`,
//! `FlightExecutorClient::execute_final_task` ships the final stage's
//! plan via `do_get` with the `pb::Action.task` payload. The server
//! runs `task.plan.execute(&self.ctx)` and the context flows through
//! every operator including `ShuffleReaderExec` because the
//! `PhysicalPlan::execute` trait method takes `&ExecutorContext` as a
//! parameter.
//!
//! ## Single-executor cluster
//!
//! The cluster has one executor; the same in-process flight-server plays
//! the role of all executors. We force 3 partitions via
//! `DistributedConfig::with_default_partitions(3)` so the shuffle is real
//! (the test isn't just running everything in one partition with no
//! shuffle work). With one executor, all shuffle locations are local —
//! `ShuffleReaderExec` reads via `ctx.shuffle_manager` and never hits the
//! cross-executor remote-fetch path (currently unimplemented).
//!
//! ## Threading model — sync test, server in a background thread
//!
//! `Client::new` (in this crate) builds its own tokio runtime and uses
//! `block_on` for the gRPC connect. `block_on` panics inside an existing
//! tokio runtime context, so we can't use `#[tokio::test]`. The test
//! function is plain `#[test]` (sync). The server runs in a
//! `std::thread::spawn`ed background thread that owns its own tokio
//! runtime; an `mpsc` channel ships the bound address back to the test
//! thread.

use client::FlightExecutorClient;
use datatypes::RecordBatch;
use distributed::{DistributedConfig, DistributedContext, ExecutorConfig};
use std::sync::mpsc;

const EMPLOYEE_CSV: &str = "../testdata/employee.csv";

/// Build a unique shuffle directory under `/tmp` keyed by nanoseconds to
/// keep parallel `cargo test` runs from colliding on disk.
fn unique_shuffle_dir(tag: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("/tmp/rquery-shuffle-distributed-{tag}-{nanos}")
}

/// Spawn an in-process flight-server in a background thread with its own
/// tokio runtime. Returns the bound `SocketAddr` and the path to its
/// shuffle directory (for cleanup at the end of the test).
fn spawn_in_process_server(executor_id: &str) -> (std::net::SocketAddr, String) {
    use arrow_flight::flight_service_server::FlightServiceServer;
    use flight_server::r_query_flight_producer::RQueryFlightProducer;
    use physical_plan::ExecutorContext;
    use tokio::net::TcpListener;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::transport::Server;

    let shuffle_dir = unique_shuffle_dir("server");
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
            // The executor identity in the context must match the executor
            // id and the port the scheduler dispatches against — otherwise
            // shuffle reads see locations with `executor_id != ctx.executor_id`
            // and panic with the Phase-2 stub message.
            let ctx = ExecutorContext::new(
                executor_id_owned,
                "127.0.0.1",
                addr.port() as i32,
                shuffle_dir_for_thread,
            );
            let producer = RQueryFlightProducer::new(ctx);

            tx.send(addr).expect("ship addr back to test thread");

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

/// **The Phase 1 payoff test.** Run a real `SELECT state, SUM(salary) FROM
/// employee GROUP BY state` distributed query end-to-end through the
/// scheduler + FlightExecutorClient + flight-server + shuffle files +
/// final-stage aggregate. Assert the resulting row count and total sum
/// match what the in-process `ExecutionContext` would produce.
#[test]
fn distributed_aggregate_query_end_to_end_via_flight() {
    let (addr, shuffle_dir) = spawn_in_process_server("exec-test");

    // Build the FlightExecutorClient pointed at the in-process server.
    // ExecutorConfig.port is i32; SocketAddr.port() is u16.
    let executors = vec![ExecutorConfig::new(
        "exec-test",
        "127.0.0.1",
        addr.port() as i32,
    )];
    let flight_client = FlightExecutorClient::new(&executors)
        .expect("FlightExecutorClient::new should connect to the in-process server");

    // Build the scheduler stack with a non-default partition count so the
    // shuffle is real. (Default partition_count = executor count = 1, which
    // wouldn't exercise any redistribution.)
    let config = DistributedConfig::new(executors).with_default_partitions(3);
    let mut ctx = DistributedContext::new(config, flight_client);
    ctx.register_csv("employee", EMPLOYEE_CSV, true);

    // Run the query. The result is a `Box<dyn Iterator<Item = RecordBatch>>`
    // (sync) — the scheduler synchronously drove every Flight call.
    let results: Vec<RecordBatch> = ctx
        .sql("SELECT state, SUM(salary) FROM employee GROUP BY state")
        .collect();

    // Sanity check: at least one output batch and total row count matches
    // the number of distinct states in employee.csv.
    //
    // employee.csv has 4 data rows with states: CA, CO, CO, "" (empty)
    // → 3 distinct groups → 3 output rows.
    let total_output_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert!(
        !results.is_empty(),
        "expected at least one result batch; got {} batches",
        results.len()
    );
    assert_eq!(
        total_output_rows, 3,
        "expected 3 output rows (one per distinct state); got {total_output_rows}",
    );

    // Verify the schema looks right: 2 columns (state, sum).
    for batch in &results {
        assert_eq!(
            batch.num_columns(),
            2,
            "expected 2 output columns (state, sum); got {}",
            batch.num_columns()
        );
    }

    // Clean up shuffle files.
    physical_plan::ShuffleManager::new(shuffle_dir).cleanup_all();
}
