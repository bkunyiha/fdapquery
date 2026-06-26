//! # execution
//!
//! Low-level runtime types: `TaskContext`, `RuntimeEnv`, `SessionConfig`,
//! and the per-process runtime resources they own (`ShuffleManager`,
//! `ShuffleLocation`).
//!
//! Session 15c moved these in from `fdapquery-physical-plan` and
//! moved `SessionContext` / `ParallelContext` out to the
//! `fdapquery` umbrella crate. The dep direction is now
//! `fdapquery-physical-plan` → `fdapquery-execution`, matching
//! DataFusion's `datafusion-physical-plan` → `datafusion-execution`.
//! The user-facing high-level API (`SessionContext` /
//! `ParallelContext`) lives in `fdapquery`.

// ==============================================================
// Per-file modules.
// ==============================================================
pub mod shuffle_location;
pub mod shuffle_manager;
// Session 15d-1 #92 moved `stream` here from `fdapquery-physical-plan`
// so the canonical `SendableRecordBatchStream` type lives at the
// DataFusion-equivalent location (`datafusion-execution::stream`).
// Catalog now depends on execution to use the same type for
// `TableProvider::scan`; physical-plan re-exports for backwards
// compatibility.
pub mod stream;
pub mod task_context;

// ==============================================================
// Re-exports for convenient downstream `use fdapquery_execution::*;` ergonomics.
// ==============================================================
pub use shuffle_location::ShuffleLocation;
pub use shuffle_manager::ShuffleManager;
pub use stream::{RecordBatchStream, RecordBatchStreamAdapter, SendableRecordBatchStream};
pub use task_context::{RuntimeEnv, SessionConfig, TaskContext};
