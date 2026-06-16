//! `RQueryFlightProducer` implements
//! [`arrow_flight::flight_service_server::FlightService`] and is constructed
//! by the binary in `src/bin/flight_server.rs`.
//!
//! ## Status
//!
//! | Method                       | State |
//! |------------------------------|-------|
//! | `do_action("execute_task")`  | **real** — drives intermediate-stage task execution; downcasts to `ShuffleWriterExec`, calls `write_shuffle(&ctx)`, returns `pb::TaskResult` with shuffle locations |
//! | `do_get`                     | **real** — streams `RecordBatch`es. Dispatches on the decoded `pb::Action`: `task` set → distributed final-stage path (runs `task.plan.execute(&ctx)`); `query` set → interactive path (runs the logical plan via `ExecutionContext`). Both branches share the same sync→async bridge (`spawn_blocking` → bounded mpsc → `FlightDataEncoder`) |
//! | `handshake`, `list_flights`, `get_flight_info`, `poll_flight_info`, `get_schema`, `do_put`, `do_exchange`, `list_actions` | stub — `Status::unimplemented` |
//!
//! ## Executor context
//! Per-executor state — `executor_id`, `executor_host`, `executor_port`, and
//! the `ShuffleManager` — is bundled into a single [`ExecutorContext`]
//! (`physical-plan/src/executor_context.rs`) held as one field on the
//! producer. The bin constructs one `ExecutorContext` at startup and hands
//! it to [`RQueryFlightProducer::new`].

use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_server::FlightService;
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
};
use datatypes::RecordBatch;
use execution::execution_context::ExecutionContext;
use futures::{Stream, TryStreamExt};
use physical_plan::{ExecutorContext, ShuffleWriterExec};
use protobuf::{deserialize_logical_plan, deserialize_task, pb};
use std::collections::HashMap;
use std::pin::Pin;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, info};

/// Arrow Flight producer for distributed query execution.
///
/// Held as the service implementation behind
/// `arrow_flight::flight_service_server::FlightServiceServer::new(producer)`.
/// The single field — [`ExecutorContext`] — carries the per-executor identity
/// and shuffle storage used by both `do_action("execute_task")` and `do_get`.
pub struct RQueryFlightProducer {
    ctx: ExecutorContext,
}

impl RQueryFlightProducer {
    /// Construct from a fully-built executor context. The bin in
    /// `src/bin/flight_server.rs` (or an integration test) builds
    /// the context from CLI / env / defaults at startup.
    pub fn new(ctx: ExecutorContext) -> Self {
        Self { ctx }
    }
}

/// Convenience alias matching the shape `arrow-flight` expects for the
/// streaming associated types — each method's response stream is a boxed,
/// pinned, `Send` stream of `Result<T, Status>`.
type FlightStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl FlightService for RQueryFlightProducer {
    type HandshakeStream = FlightStream<HandshakeResponse>;
    type ListFlightsStream = FlightStream<FlightInfo>;
    type DoGetStream = FlightStream<FlightData>;
    type DoPutStream = FlightStream<PutResult>;
    type DoActionStream = FlightStream<arrow_flight::Result>;
    type ListActionsStream = FlightStream<ActionType>;
    type DoExchangeStream = FlightStream<FlightData>;

    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        // No handshake protocol is implemented; respond with the
        // gRPC-canonical "unimplemented" status.
        Err(Status::unimplemented("handshake"))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Err(Status::unimplemented("list_flights"))
    }

    async fn get_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("get_flight_info"))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("get_schema"))
    }

    /// Stream the results of either a logical plan or a distributed final
    /// task as Arrow Flight data.
    ///
    /// ### Wire dispatch
    /// The `pb::Action` carries either `query` (an interactive logical plan)
    /// or `task` (a distributed final task). `do_get` checks `task` first
    /// and falls back to `query`:
    ///
    /// - **`action.task` set** — distributed final-stage path. Deserialise
    ///   to a `physical_plan::Task`, run `task.plan.execute(&self.ctx)`.
    ///   Works for any plan tree containing a `ShuffleReaderExec` because
    ///   the `PhysicalPlan::execute` trait method takes `&ExecutorContext`
    ///   and every operator threads it through. This is what
    ///   `FlightExecutorClient::execute_final_task` in module 14 calls.
    /// - **`action.query` set** — interactive path. Deserialise to a
    ///   `LogicalPlan`, run via a fresh `ExecutionContext`. The
    ///   `Context::sql` API in this crate uses this path.
    /// - **Neither set** — `Status::invalid_argument`.
    ///
    /// ### Sync→async bridge (both paths)
    /// `tokio::task::spawn_blocking` pumps the synchronous iterator into a
    /// bounded `tokio::sync::mpsc::channel(4)`. `ReceiverStream` wraps the
    /// receiver. `FlightDataEncoderBuilder` schema-encodes the first batch
    /// and data-encodes the rest as `FlightData` messages. `FlightError`
    /// is mapped to `tonic::Status::internal` for the response item type.
    ///
    /// Three properties: backpressure (bounded channel), tokio thread
    /// preservation (heavy work off the runtime), clean shutdown (receiver
    /// drop closes the channel; next `blocking_send` returns `Err`).
    /// ---------------------------------
    /// Retrieve a single stream associated with a particular descriptor
    /// associated with the referenced ticket. A Flight can be composed of one or
    /// more streams where each stream can be retrieved using a separate opaque
    /// ticket that the flight service uses for managing a collection of streams.
    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket: Ticket = request.into_inner();

        // 1 — decode Action from ticket bytes
        let action: pb::Action = prost::Message::decode(ticket.ticket.as_ref())
            .map_err(|e| Status::invalid_argument(format!("failed to decode Action: {e}")))?;

        // 2 — dispatch on which payload is set. `task` (distributed final
        // task) takes precedence over `query` (interactive logical plan).
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<RecordBatch, FlightError>>(4);

        if let Some(task_info) = action.task {
            // ── Distributed final-stage path ──
            // Deserialise to a `physical_plan::Task` (carries
            // `Arc<dyn PhysicalPlan>`), spawn_blocking, run
            // `task.plan.execute(&self.ctx)`. Works for any plan tree
            // containing a `ShuffleReaderExec` because the
            // `PhysicalPlan::execute` trait method takes `&ExecutorContext`
            // and every operator threads it through.
            let task = deserialize_task(&task_info);
            info!(
                "do_get executing final task: job={} stage={} task={} partition={}",
                task.job_uuid, task.stage_id, task.task_id, task.partition_id
            );
            let ctx = self.ctx.clone();
            tokio::task::spawn_blocking(move || {
                // Execute the task's plan and send the result batches to the client.
                // flight-server do_get
                //   -> task.plan.execute(&ctx)
                //      -> HashAggregateExec::execute(ctx)
                //         -> ShuffleReaderExec::execute(ctx)
                for batch in task.plan.execute(&ctx) {
                    if tx.blocking_send(Ok(batch)).is_err() {
                        debug!("do_get receiver dropped; halting executor");
                        break;
                    }
                }
            });
        } else if let Some(plan_node) = action.query {
            // Direct Flight logical-plan path, not the distributed scheduler path.
            // `client::Context::execute` sends `Action.query = Some(LogicalPlanNode)`
            // when one Flight server should execute the whole logical plan itself.
            // Distributed final stages use `Action.task = Some(TaskInfo)` above.
            let logical_plan = deserialize_logical_plan(&plan_node);
            info!("do_get executing logical plan: {}", logical_plan.pretty());
            tokio::task::spawn_blocking(move || {
                let exec_ctx = ExecutionContext::new(HashMap::new());
                for batch in exec_ctx.execute(&logical_plan) {
                    if tx.blocking_send(Ok(batch)).is_err() {
                        debug!("do_get receiver dropped; halting executor");
                        break;
                    }
                }
            });
        } else {
            return Err(Status::invalid_argument(
                "Action must have either `query` (logical plan) or `task` (final task)",
            ));
        }

        // 3 — wrap receiver as a Stream and pipe through FlightDataEncoder.
        // The encoder derives the Arrow schema from the first batch, emits a
        // schema FlightData message, then one or more data FlightData
        // messages per batch.
        let batches_stream = ReceiverStream::new(rx);
        let flight_stream = FlightDataEncoderBuilder::new().build(batches_stream);

        // 4 — map FlightError → tonic::Status for the response item type.
        let response_stream =
            flight_stream.map_err(|e| Status::internal(format!("Flight encode error: {e}")));

        Ok(Response::new(Box::pin(response_stream)))
    }

    ///
    /// Push a stream to the flight service associated with a particular
    /// flight stream. This allows a client of a flight service to upload a stream
    /// of data. Depending on the particular flight service, a client consumer
    /// could be allowed to upload a single stream per descriptor or an unlimited
    /// number. In the latter, the service might implement a 'seal' action that
    /// can be applied to a descriptor once all streams are uploaded.
    async fn do_put(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented("do_put"))
    }

    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("do_exchange"))
    }

    /// Dispatch on `action.type`. Only `"execute_task"` is implemented;
    /// returns a [`pb::TaskResult`] protobuf carrying the shuffle locations
    /// the task produced.
    ///
    /// ### Wire flow
    /// 1. `action.body` (bytes) is decoded as [`pb::TaskInfo`] via `prost::Message::decode`.
    /// 2. [`protobuf::deserialize_task`] converts it to a `physical_plan::Task`
    ///    (which carries `Arc<dyn PhysicalPlan>`).
    /// 3. Dispatch on the plan's concrete type via `as_any().downcast_ref::<ShuffleWriterExec>()`:
    ///    - `ShuffleWriterExec` → call [`ShuffleWriterExec::write_shuffle`],
    ///      get back `Vec<ShuffleLocation>`.
    ///    - Any other operator → drain `execute(&ctx)` for side effects and
    ///      return no locations.
    /// 4. Build a [`pb::TaskResult`] tagged with the task identity and the
    ///    location list, encode via `prost::Message::encode_to_vec`, wrap as
    ///    `arrow_flight::Result { body: bytes }`, return a one-element stream.
    ///
    /// ### Synchronous, not `spawn_blocking`
    /// The disk I/O inside `write_shuffle` runs on the tokio runtime thread.
    /// For small inputs that's fine; for production-shaped workloads the
    /// right move would be `tokio::task::spawn_blocking` to offload the
    /// CPU/disk work — which is what `do_get` uses for its streaming case.
    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        match action.r#type.as_str() {
            "execute_task" => {
                // 1 — decode TaskInfo from the action body
                let task_info: pb::TaskInfo = prost::Message::decode(action.body.as_ref())
                    .map_err(|e| {
                        Status::invalid_argument(format!("failed to decode TaskInfo: {e}"))
                    })?;
                info!(
                    "Executing task job={} stage={} task={} partition={}",
                    task_info.job_uuid,
                    task_info.stage_id,
                    task_info.task_id,
                    task_info.partition_id
                );

                // 2 — deserialise to a Task (carries Arc<dyn PhysicalPlan>)
                let task = deserialize_task(&task_info);
                debug!("Task plan: {}", task.plan);

                // 3 — dispatch on plan type
                let locations =
                    if let Some(writer) = task.plan.as_any().downcast_ref::<ShuffleWriterExec>() {
                        writer.write_shuffle(&self.ctx)
                    } else {
                        // Non-shuffle tasks only make sense here if the plan is a sink
                        // operator (write table/file, materialize cache, build stats, etc.).
                        // We drain to force those side effects; result rows belong on do_get.
                        // Examples:
                        // - INSERT INTO target SELECT ... The query computes rows, but the useful effect is writing them into target.
                        // - CREATE TABLE new_table AS SELECT ... The result rows are materialized into a new table/file, not returned to the client.
                        // - COPY (SELECT ...) TO 'file.parquet' The query output is written to external storage.
                        // - Shuffle/materialization stages in distributed execution A stage computes rows, partitions them, and writes them to disk/object storage so later stages can read them.
                        // - Cache population A query/subplan is executed to fill a cache; the caller may not need the rows immediately.
                        // - Index/statistics building The engine scans data and computes/writes index pages, zone maps, histograms, etc.
                        // - Validation/check operations A query may scan and verify constraints/data integrity, returning only success/failure or a count elsewhere.
                        task.plan.execute(&self.ctx).for_each(|_| {});
                        Vec::new()
                    };
                debug!("Task produced {} shuffle location(s)", locations.len());

                // 4 — build TaskResult, encode, wrap
                let task_result = pb::TaskResult {
                    job_uuid: task.job_uuid,
                    stage_id: task.stage_id,
                    task_id: task.task_id,
                    partition_id: task.partition_id,
                    shuffle_locations: locations.iter().map(pb::ShuffleLocation::from).collect(),
                };
                let body: Vec<u8> = prost::Message::encode_to_vec(&task_result);
                let result = arrow_flight::Result { body: body.into() };

                // 5 — single-item stream response. Flight's do_action returns
                // a stream of arrow_flight::Result; we emit exactly one.
                let stream = futures::stream::iter(vec![Ok(result)]);
                Ok(Response::new(Box::pin(stream)))
            }
            other => Err(Status::invalid_argument(format!(
                "Unknown action type: {other}"
            ))),
        }
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        Err(Status::unimplemented("list_actions"))
    }
}

#[cfg(test)]
mod tests {
    //! Direct method-level tests for `do_action`. We don't spin up a real
    //! tonic server here — that's `tests/integration_test.rs`.
    //! Instead we construct a `RQueryFlightProducer`, build an `Action` with a
    //! serialised `pb::TaskInfo` body, call `do_action(Request::new(action))`,
    //! collect the response stream, and assert on the decoded `pb::TaskResult`.

    use super::*;
    use arrow_flight::Action;
    use datasource::{CsvDataSource, DataSource};
    use futures::StreamExt;
    use physical_plan::{ColumnExpression, PhysicalPlan, ScanExec, ShuffleWriterExec, Task};
    use protobuf::serialize_task;
    use std::sync::Arc;

    const EMPLOYEE_CSV: &str = "../testdata/employee.csv";

    fn temp_dir(tag: &str) -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("/tmp/rquery-shuffle-test-{tag}-{nanos}")
    }

    fn build_task() -> Task {
        let ds: Arc<dyn DataSource> = Arc::new(CsvDataSource::new(EMPLOYEE_CSV, None, true, 1024));
        let columns: Vec<String> = ds.schema().fields.iter().map(|f| f.name.clone()).collect();
        let scan: Arc<dyn PhysicalPlan> = Arc::new(ScanExec::new(Arc::clone(&ds), columns));
        let writer: Arc<dyn PhysicalPlan> = Arc::new(ShuffleWriterExec::new(
            scan,
            vec![Arc::new(ColumnExpression::new(0))],
            "test-job-do-action",
            0,
            3,
        ));
        Task::new("test-job-do-action", 0, 0, 0, writer)
    }

    /// Encode a `Task` as `Action` body bytes via the protobuf round-trip the
    /// real client (module 14) will use.
    fn build_execute_task_action(task: &Task) -> Action {
        let task_info: pb::TaskInfo = serialize_task(task);
        let body: Vec<u8> = prost::Message::encode_to_vec(&task_info);
        Action {
            r#type: "execute_task".to_string(),
            body: body.into(),
        }
    }

    #[tokio::test]
    async fn execute_task_runs_shuffle_writer_and_returns_locations() {
        let base = temp_dir("do-action-writer");
        let ctx = ExecutorContext::new("exec-test", "127.0.0.1", 50099, &base);
        let producer = RQueryFlightProducer::new(ctx);

        let task = build_task();
        let action = build_execute_task_action(&task);

        let response = producer
            .do_action(Request::new(action))
            .await
            .expect("do_action should succeed");

        // Collect the response stream. do_action emits exactly one result.
        let results: Vec<arrow_flight::Result> = response
            .into_inner()
            .map(|r| r.expect("stream item should be Ok"))
            .collect()
            .await;
        assert_eq!(results.len(), 1, "exactly one result expected");

        // Decode the body as TaskResult and verify identity + locations.
        let task_result: pb::TaskResult = prost::Message::decode(results[0].body.as_ref())
            .expect("body should decode as TaskResult");

        assert_eq!(task_result.job_uuid, "test-job-do-action");
        assert_eq!(task_result.stage_id, 0);
        assert_eq!(task_result.task_id, 0);
        assert_eq!(task_result.partition_id, 0);
        assert!(!task_result.shuffle_locations.is_empty());
        assert!(task_result.shuffle_locations.len() <= 3);
        for loc in &task_result.shuffle_locations {
            assert_eq!(loc.executor_id, "exec-test");
            assert_eq!(loc.executor_host, "127.0.0.1");
            assert_eq!(loc.executor_port, 50099);
            assert_eq!(loc.job_uuid, "test-job-do-action");
            assert_eq!(loc.stage_id, 0);
        }

        // Clean up the tempdir.
        let base_clone = base.clone();
        physical_plan::ShuffleManager::new(base_clone).cleanup_all();
    }

    #[tokio::test]
    async fn unknown_action_type_returns_invalid_argument() {
        let base = temp_dir("do-action-unknown");
        let ctx = ExecutorContext::new("exec-test", "127.0.0.1", 50099, &base);
        let producer = RQueryFlightProducer::new(ctx);

        let action = Action {
            r#type: "totally_not_a_real_action".to_string(),
            body: Vec::new().into(),
        };

        // `expect_err` would require the Ok variant to be `Debug`; our
        // Response<Pin<Box<dyn Stream>>> isn't, so use `.err().expect(...)`
        // — `Option::expect` has no `Debug` bound on the inner value.
        let status = producer
            .do_action(Request::new(action))
            .await
            .err()
            .expect("unknown action must return Err");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status.message().contains("totally_not_a_real_action"),
            "error message should name the unknown action: {}",
            status.message()
        );
    }

    #[tokio::test]
    async fn do_get_streams_flight_data_for_a_logical_plan() {
        use futures::StreamExt;
        use logical_plan::{LogicalPlan, Scan};
        use protobuf::serialize_logical_plan;

        let base = temp_dir("do-get-happy");
        let ctx = ExecutorContext::new("exec-test", "127.0.0.1", 50099, &base);
        let producer = RQueryFlightProducer::new(ctx);

        // Build a LogicalPlan: scan employee.csv with all columns.
        let ds: Arc<dyn DataSource> = Arc::new(CsvDataSource::new(EMPLOYEE_CSV, None, true, 1024));
        let logical_plan = LogicalPlan::Scan(Scan::new(EMPLOYEE_CSV, ds, vec![]));

        // Serialise as Action protobuf and wrap in a Ticket.
        let plan_node = serialize_logical_plan(&logical_plan);
        let action = pb::Action {
            query: Some(plan_node),
            task: None,
            settings: vec![],
        };
        let ticket_bytes: Vec<u8> = prost::Message::encode_to_vec(&action);
        let ticket = arrow_flight::Ticket {
            ticket: ticket_bytes.into(),
        };

        let response = producer
            .do_get(Request::new(ticket))
            .await
            .expect("do_get should succeed");

        // Collect all FlightData items from the stream. The encoder emits at
        // least one schema message + one data message for a non-empty plan.
        let items: Vec<Result<arrow_flight::FlightData, tonic::Status>> =
            response.into_inner().collect().await;
        assert!(
            items.len() >= 2,
            "stream should emit at least a schema message + a data message, got {}",
            items.len()
        );
        for (i, item) in items.iter().enumerate() {
            assert!(item.is_ok(), "stream item {i} was Err: {:?}", item);
        }
    }

    #[tokio::test]
    async fn do_get_malformed_ticket_returns_invalid_argument() {
        let base = temp_dir("do-get-malformed");
        let ctx = ExecutorContext::new("exec-test", "127.0.0.1", 50099, &base);
        let producer = RQueryFlightProducer::new(ctx);

        let ticket = arrow_flight::Ticket {
            ticket: vec![0xff, 0xff, 0xff, 0xff].into(),
        };

        let status = producer
            .do_get(Request::new(ticket))
            .await
            .err()
            .expect("malformed ticket must return Err");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status.message().contains("failed to decode Action"),
            "error message should mention decode failure: {}",
            status.message()
        );
    }

    #[tokio::test]
    async fn do_get_missing_query_returns_invalid_argument() {
        let base = temp_dir("do-get-no-query");
        let ctx = ExecutorContext::new("exec-test", "127.0.0.1", 50099, &base);
        let producer = RQueryFlightProducer::new(ctx);

        // Valid Action protobuf bytes but with no query/task field.
        let action = pb::Action {
            query: None,
            task: None,
            settings: vec![],
        };
        let body: Vec<u8> = prost::Message::encode_to_vec(&action);
        let ticket = arrow_flight::Ticket {
            ticket: body.into(),
        };

        let status = producer
            .do_get(Request::new(ticket))
            .await
            .err()
            .expect("missing query must return Err");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status.message().contains("Action must have either"),
            "error message should mention missing query: {}",
            status.message()
        );
    }

    #[tokio::test]
    async fn malformed_task_body_returns_invalid_argument() {
        let base = temp_dir("do-action-malformed");
        let ctx = ExecutorContext::new("exec-test", "127.0.0.1", 50099, &base);
        let producer = RQueryFlightProducer::new(ctx);

        // execute_task action whose body is not a valid TaskInfo protobuf.
        let action = Action {
            r#type: "execute_task".to_string(),
            body: vec![0xff, 0xff, 0xff, 0xff].into(),
        };

        let status = producer
            .do_action(Request::new(action))
            .await
            .err()
            .expect("malformed body must return Err");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status.message().contains("failed to decode TaskInfo"),
            "error message should mention decode failure: {}",
            status.message()
        );
    }
}
