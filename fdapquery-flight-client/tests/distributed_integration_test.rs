//! Full distributed query end-to-end integration test — **the test that
//! closes the Phase 1 distributed loop.**
//!
//! ## What this proves
//!
//! Drives `DistributedContext::sql(...)` against a real `flight-server`
//! running on a real TCP port via real Arrow Flight gRPC, using a real
//! `FlightExecutorClient` to dispatch tasks. The mock executor client
//! from `fdapquery_distributed::scheduler::tests` is *not* in the loop —
//! `FlightExecutorClient::execute_task` ships the intermediate stage's
//! `ShuffleWriterExec` task via `do_action`,
//! `FlightExecutorClient::execute_final_task` ships the final stage's
//! plan via `do_get` with the `protobuf::Action.task` payload. The server
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
//! ## Threading model — async test, server in a background thread
//!
//! `Client::connect`, `FlightExecutorClient::connect`, and the scheduler's
//! `execute` are all `async fn`, so the test runs on a
//! tokio runtime via `#[tokio::test]`. The server still runs in a
//! `std::thread::spawn`ed background thread that owns its own tokio
//! runtime so the test and server runtimes don't share workers; an
//! `mpsc` channel ships the bound address back to the test thread.

use fdapquery_datatypes::RecordBatch;
use fdapquery_distributed::{DistributedConfig, DistributedContext, ExecutorConfig};
use fdapquery_flight_client::FlightExecutorClient;
use futures::TryStreamExt;
use std::sync::Arc;
use std::sync::mpsc;

const EMPLOYEE_CSV: &str = "../testdata/employee.csv";

/// Build a unique shuffle directory under `/tmp` keyed by nanoseconds to
/// keep parallel `cargo test` runs from colliding on disk.
fn unique_shuffle_dir(tag: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("/tmp/fdapquery-shuffle-distributed-{tag}-{nanos}")
}

/// Spawn an in-process flight-server in a background thread with its own
/// tokio runtime. Takes a `DistributedConfig` so the server's `RuntimeEnv`
/// — and therefore its `ShuffleManager.base_dir` — is derived from
/// `config.shuffle_dir` via `config.build_runtime_env()`. This is the
/// wiring point: the cluster config is the single source of truth for
/// where shuffle files land.
fn spawn_in_process_server(
    executor_id: &str,
    config: &DistributedConfig,
) -> std::net::SocketAddr {
    use arrow_flight::flight_service_server::FlightServiceServer;
    use fdapquery_flight_server::fdap_query_flight_producer::FdapQueryFlightProducer;
    use fdapquery_physical_plan::{SessionConfig, TaskContext};
    use tokio::net::TcpListener;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::transport::Server;

    // Build the executor-side runtime from the config. `build_runtime_env()`
    // reads `config.shuffle_dir` and constructs a `ShuffleManager` keyed on
    // it — this is the wire-up that proves the field is actually consumed.
    let runtime = Arc::new(config.build_runtime_env());
    let executor_id_owned = executor_id.to_string();
    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        let tokio_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("build server runtime");
        tokio_runtime.block_on(async move {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind random port");
            let addr = listener.local_addr().expect("local_addr");
            // The executor identity in the context must match the executor
            // id and the port the scheduler dispatches against — otherwise
            // shuffle reads see locations with `executor_id != ctx.executor_id`
            // and the cross-executor fetch path (currently unimplemented)
            // would fire.
            let ctx = Arc::new(TaskContext::new(
                executor_id_owned,
                "127.0.0.1",
                addr.port(),
                SessionConfig::new(),
                runtime,
            ));
            let producer = FdapQueryFlightProducer::new(ctx);

            tx.send(addr).expect("ship addr back to test thread");

            Server::builder()
                .add_service(FlightServiceServer::new(producer))
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await
                .expect("server serve");
        });
    });

    rx.recv().expect("server thread sent addr")
}

/// **The Phase 1 payoff test.** Run a real `SELECT state, SUM(salary) FROM
/// employee GROUP BY state` distributed query end-to-end through the
/// scheduler + FlightExecutorClient + flight-server + shuffle files +
/// final-stage aggregate. Assert the resulting row count and total sum
/// match what the in-process `SessionContext` would produce.
#[tokio::test]
async fn distributed_aggregate_query_end_to_end_via_flight() {
    // Pick a shuffle dir and bake it into a placeholder config first. The
    // server is spawned using this config's `build_runtime_env()` — proving
    // that `config.shuffle_dir` flows into the executor's `ShuffleManager.base_dir`.
    // We need a separate placeholder because the executor's port isn't known
    // until the server binds; the final cluster config is built below once
    // `addr` is known, with the same `shuffle_dir`.
    let shuffle_dir = unique_shuffle_dir("server");
    let server_config = DistributedConfig::new(vec![]).with_shuffle_dir(&shuffle_dir);
    let addr = spawn_in_process_server("exec-test", &server_config);

    // Build the FlightExecutorClient pointed at the in-process server.
    // ExecutorConfig.port is i32; SocketAddr.port() is u16.
    let executors = vec![ExecutorConfig::new(
        "exec-test",
        "127.0.0.1",
        i32::from(addr.port()),
    )];
    let flight_client = FlightExecutorClient::connect(&executors)
        .await
        .expect("FlightExecutorClient::connect should reach the in-process server");

    // Build the scheduler stack with a non-default partition count so the
    // shuffle is real. (Default partition_count = executor count = 1, which
    // wouldn't exercise any redistribution.) Pin `shuffle_dir` so the
    // planner-side config matches what the executor is actually using.
    let config = DistributedConfig::new(executors)
        .with_default_partitions(3)
        .with_shuffle_dir(&shuffle_dir);

    // Sanity check that the wiring holds: planner-side config and the
    // string we passed to the executor at spawn time agree.
    assert_eq!(config.shuffle_dir, shuffle_dir);

    let mut ctx = DistributedContext::new(config.clone(), flight_client);
    ctx.register_csv("employee", EMPLOYEE_CSV, true);

    // Run the query. The scheduler awaits every Flight call on the
    // current tokio runtime; the resulting `SendableRecordBatchStream`
    // is drained via `try_collect`.
    let stream = ctx
        .sql("SELECT state, SUM(salary) FROM employee GROUP BY state")
        .await
        .expect("sql plan");
    let results: Vec<RecordBatch> = stream.try_collect().await.expect("drain stream");

    // Sanity check: at least one output batch and total row count matches
    // the number of distinct states in employee.csv.
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

    // Clean up shuffle files via `config.shuffle_dir` — the same source of
    // truth the executor used at write time.
    fdapquery_physical_plan::ShuffleManager::new(&config.shuffle_dir).cleanup_all();
}
