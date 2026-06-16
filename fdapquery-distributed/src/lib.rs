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
//! - [`distributed_context`] — facade matching `ExecutionContext`'s public API
//!   (`register_csv` / `register` / `sql` / `execute`)
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
pub mod distributed_context;
pub mod distributed_planner;
pub mod query_stage;
pub mod scheduler;

// Re-exports for ergonomic `use distributed::*;` are added per-batch as the
// types come into existence.
pub use distributed_config::{DistributedConfig, ExecutorConfig};
pub use distributed_context::DistributedContext;
pub use distributed_planner::DistributedPlanner;
pub use query_stage::QueryStage;
pub use scheduler::{ExecutorClient, Scheduler};
