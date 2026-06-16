//! Integration test for `flight-server` — exercises the full gRPC round-trip
//! against a real tonic server running on a random TCP port. Proves that an
//! actual Arrow Flight client can talk to `RQueryFlightProducer` over the
//! wire, and that `do_action("execute_task")` and `do_get` both behave
//! correctly end-to-end.
//!
//! ## Why this lives in `tests/`, not the `r_query_flight_producer.rs` unit
//! test module
//!
//! The unit tests in `r_query_flight_producer.rs` exercise the service
//! methods *directly* — they construct the producer in memory and call
//! `do_action(Request::new(...))` / `do_get(Request::new(...))` without ever
//! crossing a TCP socket. That's enough to prove the dispatch and stream
//! shapes are correct, but it skips the entire tonic / hyper / HTTP/2 /
//! protobuf framing stack.
//!
//! This file does the opposite: spins up a real `tonic::transport::Server`
//! on `127.0.0.1:0`, builds a real `FlightServiceClient` over a tonic
//! `Channel`, and verifies that the bytes flowing across the wire
//! deserialise into the expected `TaskResult` / `RecordBatch` shapes. Any
//! regression in framing, encoding, schema propagation, or stream lifecycle
//! surfaces here.

use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_flight::flight_service_server::FlightServiceServer;
use arrow_flight::{Action, Ticket};
use datasource::{CsvDataSource, DataSource};
use datatypes::RecordBatch;
use flight_server::r_query_flight_producer::RQueryFlightProducer;
use futures::StreamExt;
use logical_plan::{LogicalPlan, Scan};
use physical_plan::{
    ColumnExpression, ExecutorContext, PhysicalPlan, ScanExec, ShuffleWriterExec, Task,
};
use protobuf::{pb, serialize_logical_plan, serialize_task};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::Request;
use tonic::transport::{Channel, Server};

const EMPLOYEE_CSV: &str = "../testdata/employee.csv";

fn temp_dir(tag: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("/tmp/rquery-shuffle-test-{tag}-{nanos}")
}

/// Spawn an `RQueryFlightProducer`-backed tonic server bound to a random TCP
/// port on localhost. Returns the bound `addr` (so the client knows where to
/// connect) and the `JoinHandle` for the server task (so the test can drop
/// it at the end).
async fn spawn_flight_server(
    ctx: ExecutorContext,
) -> (
    std::net::SocketAddr,
    tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind random port");
    let addr = listener.local_addr().expect("local_addr");
    let listener_stream = TcpListenerStream::new(listener);

    let producer = RQueryFlightProducer::new(ctx);
    let server_handle = tokio::spawn(async move {
        Server::builder()
            .add_service(FlightServiceServer::new(producer))
            .serve_with_incoming(listener_stream)
            .await
    });

    (addr, server_handle)
}

/// Connect a Flight client to a local server.
async fn connect_client(addr: std::net::SocketAddr) -> FlightServiceClient<Channel> {
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("valid endpoint")
        .connect()
        .await
        .expect("connect to flight server");
    FlightServiceClient::new(channel)
}

fn build_employee_scan_plan() -> LogicalPlan {
    let ds: Arc<dyn DataSource> = Arc::new(CsvDataSource::new(EMPLOYEE_CSV, None, true, 1024));
    LogicalPlan::Scan(Scan::new(EMPLOYEE_CSV, ds, vec![]))
}

fn build_shuffle_writer_task() -> Task {
    let ds: Arc<dyn DataSource> = Arc::new(CsvDataSource::new(EMPLOYEE_CSV, None, true, 1024));
    let columns: Vec<String> = ds.schema().fields.iter().map(|f| f.name.clone()).collect();
    let scan: Arc<dyn PhysicalPlan> = Arc::new(ScanExec::new(Arc::clone(&ds), columns));
    let writer: Arc<dyn PhysicalPlan> = Arc::new(ShuffleWriterExec::new(
        scan,
        vec![Arc::new(ColumnExpression::new(0))],
        "test-job-integration",
        0,
        3,
    ));
    Task::new("test-job-integration", 0, 0, 0, writer)
}

/// End-to-end test of `do_action("execute_task")` over a real gRPC channel.
/// Builds a `ShuffleWriterExec` task in the test, serialises it, sends it via
/// the Flight client, receives the `TaskResult` bytes back, decodes them, and
/// asserts the identity fields + that the executor wrote at least one
/// partition file.
#[tokio::test]
async fn integration_do_action_execute_task() {
    let base = temp_dir("integration-action");
    let ctx = ExecutorContext::new("integration-exec", "127.0.0.1", 50099, &base);
    let shuffle_manager = Arc::clone(&ctx.shuffle_manager);
    let (addr, _server) = spawn_flight_server(ctx).await;
    let mut client = connect_client(addr).await;

    // Build a Task containing a ShuffleWriterExec. Serialise it; wrap in an
    // execute_task Action.
    let task = build_shuffle_writer_task();
    let task_info: pb::TaskInfo = serialize_task(&task);
    let body: Vec<u8> = prost::Message::encode_to_vec(&task_info);
    let action = Action {
        r#type: "execute_task".to_string(),
        body: body.into(),
    };

    // do_action → stream of arrow_flight::Result. Our handler emits exactly one.
    let response = client
        .do_action(Request::new(action))
        .await
        .expect("do_action request should succeed");
    let mut stream = response.into_inner();
    let result = stream
        .message()
        .await
        .expect("stream pull should succeed")
        .expect("at least one result expected");

    // Decode the result body as TaskResult.
    let task_result: pb::TaskResult =
        prost::Message::decode(result.body.as_ref()).expect("body should decode as TaskResult");

    assert_eq!(task_result.job_uuid, "test-job-integration");
    assert_eq!(task_result.stage_id, 0);
    assert_eq!(task_result.task_id, 0);
    assert_eq!(task_result.partition_id, 0);
    assert!(!task_result.shuffle_locations.is_empty());
    for loc in &task_result.shuffle_locations {
        assert_eq!(loc.executor_id, "integration-exec");
        assert_eq!(loc.executor_host, "127.0.0.1");
        assert_eq!(loc.executor_port, 50099);
    }

    shuffle_manager.cleanup_all();
}

/// End-to-end test of `do_get` over a real gRPC channel. Builds a
/// `LogicalPlan` that scans `employee.csv`, sends it as an Action via the
/// Flight client, decodes the response stream's `FlightData` items back into
/// `RecordBatch`es using `FlightRecordBatchStream`, and asserts the total row
/// count equals the input (4 rows in employee.csv).
#[tokio::test]
async fn integration_do_get_streams_record_batches() {
    use arrow_flight::decode::FlightRecordBatchStream;

    let base = temp_dir("integration-get");
    let ctx = ExecutorContext::new("integration-exec", "127.0.0.1", 50099, &base);
    let (addr, _server) = spawn_flight_server(ctx).await;
    let mut client = connect_client(addr).await;

    // Build a LogicalPlan and wrap in an Action protobuf, serialise to ticket bytes.
    let plan = build_employee_scan_plan();
    let plan_node = serialize_logical_plan(&plan);
    let action = pb::Action {
        query: Some(plan_node),
        task: None,
        settings: vec![],
    };
    let body: Vec<u8> = prost::Message::encode_to_vec(&action);
    let ticket = Ticket {
        ticket: body.into(),
    };

    let response = client
        .do_get(Request::new(ticket))
        .await
        .expect("do_get request should succeed");

    // Pipe the FlightData stream through FlightRecordBatchStream to decode
    // back into RecordBatches. The stream from `client.do_get` is
    // `Streaming<FlightData>` — convert to a Stream<Item = Result<FlightData, FlightError>>
    // first.
    let flight_data_stream = response
        .into_inner()
        .map(|r| r.map_err(|status| arrow_flight::error::FlightError::Tonic(Box::new(status))));
    let mut record_batch_stream = FlightRecordBatchStream::new_from_flight_data(flight_data_stream);

    let mut batches: Vec<RecordBatch> = Vec::new();
    while let Some(batch_result) = record_batch_stream.next().await {
        batches.push(batch_result.expect("batch decode should succeed"));
    }

    // employee.csv has 4 rows.
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 4, "round-trip row count must match input");

    // All batches share the same schema (the scan's output schema). At minimum,
    // every batch's column count matches the input.
    let ds: Arc<dyn DataSource> = Arc::new(CsvDataSource::new(EMPLOYEE_CSV, None, true, 1024));
    let expected_columns = ds.schema().fields.len();
    for b in &batches {
        assert_eq!(b.num_columns(), expected_columns);
    }
}

/// Smoke test that an unknown action type returns a tonic-canonical
/// `InvalidArgument` error all the way through the wire, not a generic
/// internal error or a silent success.
#[tokio::test]
async fn integration_unknown_action_returns_invalid_argument() {
    let base = temp_dir("integration-unknown");
    let ctx = ExecutorContext::new("integration-exec", "127.0.0.1", 50099, &base);
    let (addr, _server) = spawn_flight_server(ctx).await;
    let mut client = connect_client(addr).await;

    let action = Action {
        r#type: "nope_not_real".to_string(),
        body: Vec::new().into(),
    };

    let Err(status) = client.do_action(Request::new(action)).await else {
        panic!("unknown action must return Err");
    };
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert!(
        status.message().contains("nope_not_real"),
        "error message should name the unknown action: {}",
        status.message()
    );
}
