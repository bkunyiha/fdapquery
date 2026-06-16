//! `FlightExecutorClient` — a concrete `distributed::ExecutorClient` that
//! drives the scheduler over real Arrow Flight gRPC.
//!
//! ## What it does
//!
//! Holds one [`Client`] per executor in the cluster (keyed by
//! `executor_id`). Each [`distributed::ExecutorClient`] method routes to
//! the matching `Client`:
//!
//! | Trait method | Wire path | Server-side handler |
//! |--------------|-----------|---------------------|
//! | `execute_task(executor, task)` | `do_action("execute_task", TaskInfo)` → `TaskResult` | `r_query_flight_producer.rs::do_action` matches `ShuffleWriterExec`, calls `write_shuffle(&ctx)`, returns shuffle locations |
//! | `execute_final_task(executor, task)` | `do_get(Action { task: Some(TaskInfo) })` → `FlightData` stream | `do_get` deserialises the Task, runs `task.plan.execute(&self.ctx)`, streams batches |
//! | `fetch_shuffle(executor, location)` | not implemented | not implemented |
//!
//! `execute_final_task` *works* because `PhysicalPlan::execute` takes
//! `&ExecutorContext` as a trait-method parameter, so
//! `ShuffleReaderExec::execute(ctx)` honours the context. The final
//! stage's plan is `HashAggregateExec(Final)` wrapping `ShuffleReaderExec`;
//! when the server calls `plan.execute(&self.ctx)`, the context flows
//! through the aggregate to the reader, which reads its shuffle locations
//! via `ctx.shuffle_manager`. No special-case plan-tree rewriting needed.
//!
//! ## What it doesn't do
//!
//! `fetch_shuffle` is not implemented. It would be called by a
//! `ShuffleReaderExec` that needs to read shuffle data from a different
//! executor. The current integration tests use a single executor for all
//! stages so all reads are local — `fetch_shuffle` returns an empty
//! iterator. Wiring a Flight client into [`ExecutorContext`] so
//! `ShuffleReaderExec` can call this for remote partitions is the next
//! step toward multi-executor distributed queries.
//!
//! ## How the trait shape supports this
//!
//! Because `PhysicalPlan::execute` takes `&ExecutorContext` as a trait-method
//! parameter, context-aware execution is the trait shape itself. There is no
//! need for a special-case plan-tree rewrite: the context flows naturally
//! through any operator that wraps a `ShuffleReaderExec`. `FlightExecutorClient`
//! is the real `impl ExecutorClient` that drives a distributed query
//! end-to-end.

use crate::client::Client;
use crate::endpoint::Endpoint;
use anyhow::Result;
use datatypes::RecordBatch;
use distributed::{ExecutorClient, ExecutorConfig};
use physical_plan::{ShuffleLocation, Task};
use protobuf::{pb, serialize_task};
use std::collections::HashMap;
use tracing::{debug, info};

/// Concrete `distributed::ExecutorClient` that drives the Scheduler over
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
    /// **Not safe to call from inside an existing tokio runtime** — each
    /// `Client::new` block_ons against its own runtime. See `Client::new`
    /// for the workaround pattern (`std::thread::spawn(|| ...)`).
    pub fn new(executors: &[ExecutorConfig]) -> Result<Self> {
        let mut clients = HashMap::with_capacity(executors.len());
        for executor in executors {
            let endpoint = Endpoint::from(executor);
            info!(
                "FlightExecutorClient connecting to executor {} at {}",
                executor.id,
                endpoint.url()
            );
            let client = Client::new(endpoint)?;
            clients.insert(executor.id.clone(), client);
        }
        Ok(Self { clients })
    }

    /// Borrow the Client for a specific executor. Panics if `executor_id`
    /// isn't in the cluster — the scheduler shouldn't ever ask for an
    /// executor we don't have a connection to.
    fn client_for(&self, executor_id: &str) -> &Client {
        self.clients.get(executor_id).unwrap_or_else(|| {
            panic!(
                "FlightExecutorClient: no connection to executor '{executor_id}' \
                 (cluster was configured with: {:?})",
                self.clients.keys().collect::<Vec<_>>()
            )
        })
    }
}

impl ExecutorClient for FlightExecutorClient {
    /// Ship a `ShuffleWriterExec` task to the executor via
    /// `do_action("execute_task", ...)`. Decodes the returned
    /// `pb::TaskResult` into a `Vec<ShuffleLocation>`.
    fn execute_task(&self, executor: &ExecutorConfig, task: Task) -> Vec<ShuffleLocation> {
        // Encode the physical task into the protobuf payload expected by Flight.
        let task_info: pb::TaskInfo = serialize_task(&task);
        let body: Vec<u8> = prost::Message::encode_to_vec(&task_info);

        debug!(
            "execute_task: job={} stage={} task={} partition={} → executor {}",
            task.job_uuid, task.stage_id, task.task_id, task.partition_id, executor.id,
        );

        let client = self.client_for(&executor.id);
        // Intermediate stages use do_action and return shuffle file locations.
        let response_bytes = client
            .do_action("execute_task", body)
            .unwrap_or_else(|e| panic!("execute_task do_action failed: {e}"));

        // The server replies with TaskResult, not data batches.
        let task_result: pb::TaskResult = prost::Message::decode(&response_bytes[..])
            .unwrap_or_else(|e| panic!("execute_task: failed to decode TaskResult: {e}"));

        task_result
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
            .collect()
    }

    /// Ship the final-stage task to the executor via `do_get` (with the
    /// new `pb::Action.task` field). Pipes the response stream back as a
    /// `Vec<RecordBatch>` re-wrapped as an iterator.
    ///
    /// This path works for any plan tree containing `ShuffleReaderExec`
    /// because the `PhysicalPlan::execute` trait method takes
    /// `&ExecutorContext` and every operator threads it through. The server
    /// runs `task.plan.execute(&ctx)`; `HashAggregateExec(Final).execute(&ctx)`
    /// calls `ShuffleReaderExec.execute(&ctx)` which reads shuffle files
    /// via `ctx.shuffle_manager`. No special-case plan-tree rewriting needed.
    fn execute_final_task(
        &self,
        executor: &ExecutorConfig,
        task: Task,
    ) -> Box<dyn Iterator<Item = RecordBatch>> {
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
        let client = self.client_for(&executor.id);
        // Materialize the Flight stream, then expose it through the trait's iterator API.
        let batches: Vec<RecordBatch> = client
            .do_get(body)
            .unwrap_or_else(|e| panic!("execute_final_task do_get failed: {e}"));

        Box::new(batches.into_iter())
    }

    /// Fetch one shuffle partition's data from a remote executor.
    ///
    /// Not implemented. Returns an empty iterator. The current
    /// integration test path uses a single in-process executor so all
    /// shuffle reads are local (handled inside `ShuffleReaderExec::execute`
    /// via `ctx.shuffle_manager`). Implementing this requires wiring a
    /// Flight client into `ExecutorContext` so the reader can fetch
    /// partitions from other executors over gRPC.
    fn fetch_shuffle(
        &self,
        _executor: &ExecutorConfig,
        _location: &ShuffleLocation,
    ) -> Box<dyn Iterator<Item = RecordBatch>> {
        // Cross-executor shuffle reads are not implemented; the single-executor
        // case is handled directly inside the server's ShuffleReaderExec.
        debug!("fetch_shuffle: not implemented (cross-executor reads not supported)");
        Box::new(std::iter::empty())
    }
}
