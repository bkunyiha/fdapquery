//! # client
//!
//! Two roles in one crate:
//!
//! 1. **Interactive Flight client** — [`Client`] and [`Context`] wrap an Arrow
//!    Flight connection so a user can submit a `LogicalPlan` and receive
//!    `RecordBatch`es back.
//! 2. **Distributed-scheduler transport** — [`FlightExecutorClient`]
//!    implements `fdapquery_distributed::ExecutorClient` over a set of per-executor
//!    `Client` instances, so `Scheduler<FlightExecutorClient>::execute(...)`
//!    drives a real distributed query against real Flight executors.
//!
//! ## Modules
//! - [`client`] — `Client`: a synchronous Flight client (sync API over async tonic).
//! - [`endpoint`] — `Endpoint`: bundles a host+port into one address value.
//! - [`context`] — `Context`: high-level interactive API (`register_csv` / `sql` / `execute`).
//! - [`flight_executor_client`] — `FlightExecutorClient`:
//!   `impl fdapquery_distributed::ExecutorClient` over a per-executor `Client` map.
//! - [`extension`] — `SessionContextExt`: mirror of Ballista's `SessionContextExt`
//!   (in `ballista-client`). Provides
//!   `SessionContext::standalone()` / `standalone_with_state(state)` /
//!   `remote(url)` / `remote_with_state(url, state)`. The `standalone*` variants
//!   are gated behind `#[cfg(feature = "standalone")]`.

// ==============================================================
// Per-file modules.
// ==============================================================
pub mod client;
pub mod context;
pub mod endpoint;
pub mod extension;
pub mod flight_executor_client;

// ==============================================================
// Re-exports for convenient downstream `use client::FlightExecutorClient;`.
// ==============================================================
pub use client::Client;
pub use context::Context;
pub use endpoint::Endpoint;
pub use extension::SessionContextExt;
pub use flight_executor_client::FlightExecutorClient;
