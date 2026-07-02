//! Distributed-side [`ExecutionPlan`](fdapquery_physical_plan::ExecutionPlan)
//! implementations. Mirror of Ballista's
//! `ballista/core/src/execution_plans/` module.
//!
//! Currently:
//! - [`DistributedQueryExec`] — wraps a logical plan and, on
//!   `execute(0, ctx)`, lowers it to a physical plan and dispatches
//!   the whole thing through the in-process [`crate::Scheduler`].
//!   Mirror of Ballista's `DistributedQueryExec` at
//!   `ballista/core/src/execution_plans/distributed_query.rs`.
//!
//! Additional exec nodes (`ShuffleWriterExec`, `ShuffleReaderExec`)
//! live in `fdapquery-physical-plan` today; the strict-mirror moves
//! them here in Phase 3.

pub mod distributed_query;
pub use distributed_query::DistributedQueryExec;
