//! # execution
//!
//! The runtime — `ExecutionContext` and `ParallelContext`.
//!
//! `ExecutionContext::sql(...)` / `execute(...)` runs a query end-to-end
//! (parse → optimize → plan → execute → `Vec<RecordBatch>`). `ParallelContext`
//! runs aggregate queries across rayon workers.

// ==============================================================
// Per-file modules.
// ==============================================================
pub mod execution_context;
pub mod parallel_context;

// ==============================================================
// Re-exports for `use execution::*;` ergonomics.
// ==============================================================
pub use execution_context::ExecutionContext;
pub use parallel_context::ParallelContext;
