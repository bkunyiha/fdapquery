//! `FlightExecutorClient` — a concrete `fdapquery_distributed::ExecutorClient` that
//! drives the scheduler over real Arrow Flight gRPC.
//!
//! ## What it does
//!
//! Holds one [`Client`] per executor in the cluster (keyed by
//! `executor_id`). Each `async` [`fdapquery_distributed::ExecutorClient`]
//! method routes to the matching `Client` and `.await`s the gRPC call:
//!
//! | Trait method | Wire path | Server-side handler |
//! |--------------|-----------|---------------------|
//! | `execute_task(executor, task)` | `do_action("execute_task", TaskInfo)` → `TaskResult` | `fdap_query_flight_producer.rs::do_action` matches `ShuffleWriterExec`, calls `write_shuffle(Arc::clone(&ctx))`, returns shuffle locations |
//! | `execute_final_task(executor, task)` | `do_get(Action { task: Some(TaskInfo) })` → `FlightData` stream → `SendableRecordBatchStream` | `do_get` deserialises the Task, runs `task.plan.execute(0, Arc::clone(&self.ctx))`, streams batches |
//! | `fetch_shuffle(executor, location)` | not implemented | not implemented |
//!
//! `execute_final_task` *works* because `ExecutionPlan::execute` takes
//! `Arc<TaskContext>` as a trait-method parameter, so
//! `ShuffleReaderExec::execute(0, ctx)` honours the context. The final
//! stage's plan is `HashAggregateExec(Final)` wrapping
//! `ShuffleReaderExec`; when the server calls
//! `plan.execute(0, Arc::clone(&self.ctx))`, the context flows through
//! the aggregate to the reader, which reads its shuffle locations via
//! `ctx.runtime.shuffle_manager`. No special-case plan-tree rewriting
//! needed.
//!
//! ## What it doesn't do
//!
//! `fetch_shuffle` is not implemented. It would be called by a
//! `ShuffleReaderExec` that needs to read shuffle data from a different
//! executor. The current integration tests use a single executor for all
//! stages so all reads are local — `fetch_shuffle` returns an empty
//! stream. Wiring a Flight client into `TaskContext` (via `RuntimeEnv`)
//! so `ShuffleReaderExec` can call this for remote partitions is the
//! next step toward multi-executor distributed queries.
//!
//! ## How the trait shape supports this
//!
//! Because `ExecutionPlan::execute` takes `Arc<TaskContext>` as a
//! trait-method parameter, context-aware execution is the trait shape
//! itself. There is no need for a special-case plan-tree rewrite: the
//! context flows naturally through any operator that wraps a
//! `ShuffleReaderExec`. `FlightExecutorClient` is the real
//! `impl ExecutorClient` that drives a distributed query end-to-end.

use crate::client::Client;
use crate::endpoint::Endpoint;
use async_trait::async_trait;
use fdapquery_datatypes::{FdapQueryError, RecordBatch, Result, Schema};
use fdapquery_distributed::{ExecutorClient, ExecutorConfig};
use fdapquery_physical_plan::{
    RecordBatchStreamAdapter, SendableRecordBatchStream, ShuffleLocation, Task,
};
use fdapquery_proto::{pb, serialize_task};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

/// Concrete `fdapquery_distributed::ExecutorClient` that drives the Scheduler over
/// real Arrow Flight gRPC.
pub struct FlightExecutorClient {
    /// One Client per executor, keyed by `executor_id`. Each Client owns
    /// its own tokio runtime (Phase 1 simplification — see `Client`'s
    /// module doc).
    clients: HashMap<String, Client>,
}

impl FlightExecutorClient {
    /// Connect to every executor in the supplied configuration. Returns
    /// `Err` if any single connection fails — partial cluster initialisation
    /// is not supported (a missing executor would crash the scheduler the
    /// first time it tried to dispatch a task there).
    ///
    /// Async — call from within a tokio runtime and `.await`. Each
    /// `Client::connect` future is awaited sequentially; with N
    /// executors this is N round-trips. (A future optimisation would
    /// fan these out via `futures::future::try_join_all`, but for v0.1
    /// sequential is fine.)
    pub async fn connect(executors: &[ExecutorConfig]) -> anyhow::Result<Self> {
        let mut clients = HashMap::with_capacity(executors.len());
        for executor in executors {
            let endpoint = Endpoint::from(executor);
            info!(
                "FlightExecutorClient connecting to executor {} at {}",
                executor.id,
                endpoint.url()
            );
            let client = Client::connect(endpoint).await?;
            clients.insert(executor.id.clone(), client);
        }
        Ok(Self { clients })
    }

    /// Borrow the Client for a specific executor. Returns an error if
    /// `executor_id` isn't in the cluster — the scheduler shouldn't ever
    /// ask for an executor we don't have a connection to, but a hard
    /// failure surfaces the misconfiguration cleanly via the trait's
    /// `Result` return.
    fn client_for(&self, executor_id: &str) -> Result<&Client> {
        self.clients.get(executor_id).ok_or_else(|| {
            FdapQueryError::Internal(format!(
                "FlightExecutorClient: no connection to executor '{executor_id}' \
                 (cluster was configured with: {:?})",
                self.clients.keys().collect::<Vec<_>>()
            ))
        })
    }
}

/// Build an empty `SendableRecordBatchStream` over an empty schema —
/// used by `fetch_shuffle` as the not-implemented stub.
fn empty_stream() -> SendableRecordBatchStream {
    let schema = Arc::new(Schema::new(vec![]).to_arrow());
    Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::empty::<Result<RecordBatch>>(),
    ))
}

#[async_trait]
impl ExecutorClient for FlightExecutorClient {
    /// Ship a `ShuffleWriterExec` task to the executor via
    /// `do_action("execute_task", ...)`. Decodes the returned
    /// `pb::TaskResult` into a `Vec<ShuffleLocation>`. Async — the
    /// scheduler awaits this on its tokio runtime.
    async fn execute_task(
        &self,
        executor: &ExecutorConfig,
        task: Task,
    ) -> Result<Vec<ShuffleLocation>> {
        // Encode the physical task into the protobuf payload expected by Flight.
        let task_info: pb::TaskInfo = serialize_task(&task);
        let body: Vec<u8> = prost::Message::encode_to_vec(&task_info);

        debug!(
            "execute_task: job={} stage={} task={} partition={} → executor {}",
            task.job_uuid, task.stage_id, task.task_id, task.partition_id, executor.id,
        );

        let client = self.client_for(&executor.id)?;
        // Intermediate stages use do_action and return shuffle file locations.
        let response_bytes = client
            .do_action("execute_task", body)
            .await
            .map_err(|e| FdapQueryError::Internal(format!("execute_task do_action failed: {e}")))?;

        // The server replies with TaskResult, not data batches.
        let task_result: pb::TaskResult =
            prost::Message::decode(&response_bytes[..]).map_err(|e| {
                FdapQueryError::Internal(format!("execute_task: failed to decode TaskResult: {e}"))
            })?;

        Ok(task_result
            .shuffle_locations
            .into_iter()
            .map(|loc| {
                ShuffleLocation::new(
                    loc.job_uuid,
                    loc.stage_id,
                    loc.partition_id,
                    loc.executor_id,
                    loc.executor_host,
                    loc.executor_port,
                )
            })
            .collect())
    }

    /// Ship the final-stage task to the executor via `do_get` (with the
    /// `pb::Action.task` field). Returns the response as a
    /// `SendableRecordBatchStream` over the materialised batches.
    ///
    /// This path works for any plan tree containing `ShuffleReaderExec`
    /// because the `ExecutionPlan::execute` trait method takes
    /// `Arc<TaskContext>` and every operator threads it through. The
    /// server runs `task.plan.execute(0, Arc::clone(&ctx))`;
    /// `HashAggregateExec(Final).execute(0, ctx)` calls
    /// `ShuffleReaderExec.execute(0, ctx)` which reads shuffle files via
    /// `ctx.runtime.shuffle_manager`. No special-case plan-tree
    /// rewriting needed.
    ///
    /// v0.1 collects the wire stream into a `Vec<RecordBatch>` before
    /// wrapping it in a `RecordBatchStreamAdapter` — a Phase D
    /// optimisation will pipe `FlightRecordBatchStream` straight through
    /// without buffering.
    async fn execute_final_task(
        &self,
        executor: &ExecutorConfig,
        task: Task,
    ) -> Result<SendableRecordBatchStream> {
        let task_info: pb::TaskInfo = serialize_task(&task);
        // Final stages stream result batches through do_get using Action.task.
        let action = pb::Action {
            query: None,
            task: Some(task_info),
            settings: vec![],
        };
        let body: Vec<u8> = prost::Message::encode_to_vec(&action);

        debug!(
            "execute_final_task: job={} stage={} task={} partition={} → executor {}",
            task.job_uuid, task.stage_id, task.task_id, task.partition_id, executor.id,
        );

        // Get the client for the executor.
        let client = self.client_for(&executor.id)?;
        // Materialize the Flight stream, then expose it through the
        // trait's `SendableRecordBatchStream` API.
        let batches: Vec<RecordBatch> = client.do_get(body).await.map_err(|e| {
            FdapQueryError::Internal(format!("execute_final_task do_get failed: {e}"))
        })?;

        // Build a stream over the materialised batches, using the first
        // batch's arrow schema (or an empty schema if no batches came
        // back). The empty-schema fallback matches what
        // `do_get_streams_flight_data_for_a_logical_plan` exercises.
        let arrow_schema = match batches.first() {
            Some(batch) => batch.schema(),
            None => Arc::new(Schema::new(vec![]).to_arrow()),
        };
        let stream = futures::stream::iter(batches.into_iter().map(Ok::<_, FdapQueryError>));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema,
            stream,
        )))
    }

    /// Fetch one shuffle partition's data from a remote executor.
    ///
    /// Not implemented. Returns an empty `SendableRecordBatchStream`.
    /// The current integration test path uses a single in-process
    /// executor so all shuffle reads are local (handled inside
    /// `ShuffleReaderExec::execute` via
    /// `ctx.runtime.shuffle_manager`). Implementing this requires
    /// wiring a Flight client into `TaskContext` so the reader can
    /// fetch partitions from other executors over gRPC.
    async fn fetch_shuffle(
        &self,
        _executor: &ExecutorConfig,
        _location: &ShuffleLocation,
    ) -> Result<SendableRecordBatchStream> {
        // Cross-executor shuffle reads are not implemented; the
        // single-executor case is handled directly inside the server's
        // ShuffleReaderExec.
        debug!("fetch_shuffle: not implemented (cross-executor reads not supported)");
        Ok(empty_stream())
    }
}
