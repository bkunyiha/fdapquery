//! # distributed
//!
//! Distributed query execution layer — scheduler, query-stage decomposition,
//! distributed planner, distributed context facade. The "minimal Ballista"
//! example from chapter 12 of *How Query Engines Work*.
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
//! - **Synchronous, sequential.** No async, no Tokio, no rayon. Each stage
//!   runs in dependency order; each task within a stage is dispatched one at
//!   a time, round-robin across executors. The module is a teaching artifact,
//!   not a production scheduler. Async lives one layer up at the Flight
//!   boundary (`flight-server` / `client`).
//! - **`ExecutorClient` is the seam to Flight.** The trait has three methods
//!   (`execute_task` / `execute_final_task` / `fetch_shuffle`). The real
//!   implementation lives in the `client` crate as `FlightExecutorClient`.
//! - **No `protobuf` dep.** Wire serialisation only happens at the Flight
//!   boundary.

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
