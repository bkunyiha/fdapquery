//! # distributed
//!
//! Distributed query execution layer — scheduler, query-stage decomposition,
//! distributed planner, and the `QueryPlanner` / exec-node plumbing that
//! makes a `SessionContext` distribute. Mirror of Ballista's `ballista-core`.
//!
//! ## Modules
//! - [`distributed_config`] — `ExecutorConfig`, `DistributedConfig`
//! - [`query_stage`] — `QueryStage`
//! - [`distributed_planner`] — splits a single-node physical plan into stages
//!   at shuffle boundaries (currently only the two-stage aggregate pattern)
//! - [`scheduler`] — `Scheduler` plus the `ExecutorClient` abstraction boundary
//!   to the Flight world
//! - [`execution_plans`] — [`DistributedQueryExec`], the leaf `ExecutionPlan`
//!   whose `execute(0, ctx)` runs the physical planner + scheduler dispatch
//! - [`planner`] — [`DistributedQueryPlanner`], the `QueryPlanner`
//!   implementation that wraps every incoming logical plan in a
//!   `DistributedQueryExec`
//! - [`session_state_ext`] — [`SessionStateExt`], the trait that installs a
//!   `DistributedQueryPlanner` into a `SessionState` (mirror of Ballista's
//!   `SessionStateExt` at `ballista/core/src/extension.rs:101-345`)
//!
//! ## Where `SessionContextExt` lives
//!
//! `SessionContextExt` (the trait providing
//! `SessionContext::standalone().await` and `remote(url).await`) lives in
//! `fdapquery-flight-client` — the crate that mirrors `ballista-client`.
//! Ballista puts `SessionContextExt` in `ballista-client`, not
//! `ballista-core`, because `standalone()` needs to spawn the executor
//! process (would create a cycle if it lived in `core`). fdapquery follows
//! the same crate layout for the same reason. See
//! `fdapquery_flight_client::SessionContextExt`.
//!
//! ## Architectural notes
//! - **Async-native end-to-end.** Every call in the dispatch chain is
//!   `async fn` on a tokio runtime. `Scheduler::execute(plan).await` returns
//!   a `SendableRecordBatchStream`; every `ExecutorClient` method returns
//!   `impl Future`. The client-side entry point
//!   `SessionContextExt::standalone().await` is also async. Tokio handles
//!   both intra-process concurrency (per-batch stream polling) and the
//!   Flight gRPC boundary.
//! - **Per-stage dispatch is currently sequential.** `Scheduler::execute_stage`
//!   awaits each task's `execute_task` call in a `for` loop before dispatching
//!   the next partition, so wall time = sum of per-task times, not max.
//!   Parallel dispatch via `futures::future::try_join_all` is a future
//!   revision (see `SESSION-20c-PLAN.md` in the planning docs); it depends
//!   on the correctness fix that scopes each stage-0 task to a single input
//!   partition landing first.
//! - **`ExecutorClient` is the seam to Flight.** The trait has three methods
//!   (`execute_task` / `execute_final_task` / `fetch_shuffle`). The real
//!   implementation lives in the `fdapquery-flight-client` crate as
//!   `FlightExecutorClient`.
//! - **No `protobuf` dep.** Wire serialisation only happens at the Flight
//!   boundary; this crate deals in Rust types (`Arc<dyn ExecutionPlan>`,
//!   `Task`, `ShuffleLocation`) and hands the concrete client type
//!   whatever it needs via the `ExecutorClient` trait.

pub mod distributed_config;
pub mod distributed_planner;
pub mod execution_plans;
pub mod planner;
pub mod query_stage;
pub mod scheduler;
pub mod session_state_ext;

// Re-exports for ergonomic `use distributed::*;`.
pub use distributed_config::{DistributedConfig, ExecutorConfig};
pub use distributed_planner::DistributedPlanner;
pub use execution_plans::DistributedQueryExec;
pub use planner::DistributedQueryPlanner;
pub use query_stage::QueryStage;
pub use scheduler::{ExecutorClient, Scheduler};
pub use session_state_ext::SessionStateExt;
