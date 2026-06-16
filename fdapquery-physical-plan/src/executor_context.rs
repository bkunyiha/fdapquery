//! Per-executor runtime context for distributed task execution.
//!
//! Bundles the per-executor identity (`executor_id`, `executor_host`,
//! `executor_port`) with the shuffle storage handle
//! (`Arc<ShuffleManager>`) into a single value that operators receive
//! through `execute(&ctx)`.
//!
//! ## Why a struct (and not separate args)
//!
//! The shuffle operators ([`crate::ShuffleWriterExec`] and
//! [`crate::ShuffleReaderExec`]) need all four pieces of state to do
//! their work — identity for the locations they report, plus the
//! [`crate::ShuffleManager`] for the local file I/O. Passing them as a
//! single context value keeps the operator signatures stable and
//! matches DataFusion's `Arc<TaskContext>` idiom (single context value
//! flowing through every operator's `execute()`).
//!
//! ## Crate placement
//!
//! `ExecutorContext` lives in `physical-plan/` because that's where it
//! is consumed — `ShuffleWriterExec::write_shuffle(&ctx)` and
//! `ShuffleReaderExec::execute(&ctx)` both read its fields. Every other
//! candidate crate (`execution`, `distributed`, `flight-server`)
//! transitively depends on `physical-plan`, so placing the struct
//! anywhere else would create a dependency cycle.
//!
//! DataFusion solves the same layering question by introducing a
//! separate `datafusion-execution` crate below `datafusion-physical-plan`;
//! Ballista layers `ballista-executor` on top to hold per-process
//! identity.

use crate::shuffle_manager::ShuffleManager;
use std::sync::Arc;

/// Per-executor identity + shuffle storage handle. One instance per
/// `flight-server` process.
///
/// Constructed once by the `flight-server` binary and held inside
/// `RQueryFlightProducer`. `do_action("execute_task")` and `do_get` pass a
/// reference to the relevant shuffle operator's `execute(&ctx)` /
/// `write_shuffle(&ctx)` method.
///
/// The [`ShuffleManager`] is held inside an [`Arc`] so multiple in-flight
/// tasks on the same executor can share the same shuffle-storage handle
/// without copying — same pattern as `Task::plan` (see
/// `task.rs` translation note).
// `Debug` not derived: `ShuffleManager` has no `Debug` impl yet (it would
// be a one-line `#[derive(Debug)]` since its only field is a `String`, but
// adding it touches a separate operator file — deferred to whoever needs
// `{:?}` on an `ExecutorContext`). `Clone` is required for handing the
// context across the `spawn_blocking` boundary in `do_get`.
#[derive(Clone)]
pub struct ExecutorContext {
    /// Unique identifier for this executor in the cluster. Mirrors
    /// `DistributedConfig::ExecutorConfig::id`.
    pub executor_id: String,

    /// Hostname or IP this executor listens on.
    pub executor_host: String,

    /// Port this executor's Arrow Flight server listens on. Stored as `i32`
    /// to match [`crate::ShuffleLocation::executor_port`] (one less
    /// conversion at every `ShuffleLocation::new` call site).
    pub executor_port: i32,

    /// Local shuffle storage. Held in an `Arc` so per-task closures
    /// (`do_get`'s `spawn_blocking` body) can clone a cheap handle rather
    /// than borrowing across an async boundary.
    pub shuffle_manager: Arc<ShuffleManager>,
}

impl ExecutorContext {
    /// Construct from raw fields plus the shuffle directory. The
    /// [`ShuffleManager`] is built internally.
    pub fn new(
        executor_id: impl Into<String>,
        executor_host: impl Into<String>,
        executor_port: i32,
        shuffle_dir: impl Into<String>,
    ) -> Self {
        Self {
            executor_id: executor_id.into(),
            executor_host: executor_host.into(),
            executor_port,
            shuffle_manager: Arc::new(ShuffleManager::new(shuffle_dir)),
        }
    }

    /// Construct from an existing [`ShuffleManager`] handle. Useful in
    /// integration tests where a temporary directory is built outside
    /// the context and shared with the verifier.
    pub fn with_shuffle_manager(
        executor_id: impl Into<String>,
        executor_host: impl Into<String>,
        executor_port: i32,
        shuffle_manager: Arc<ShuffleManager>,
    ) -> Self {
        Self {
            executor_id: executor_id.into(),
            executor_host: executor_host.into(),
            executor_port,
            shuffle_manager,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_builds_owned_shuffle_manager_at_supplied_path() {
        // Doesn't write anything — `ShuffleManager::new` is pure, the
        // directory is only created on `write_partition`. We assert the
        // field carries the path we asked for and the identity fields
        // round-trip through `Into<String>`.
        let ctx = ExecutorContext::new("exec-7", "127.0.0.1", 50057, "/tmp/rquery-shuffle-test");
        assert_eq!(ctx.executor_id, "exec-7");
        assert_eq!(ctx.executor_host, "127.0.0.1");
        assert_eq!(ctx.executor_port, 50057);
        assert_eq!(ctx.shuffle_manager.base_dir, "/tmp/rquery-shuffle-test");
    }

    #[test]
    fn clone_is_cheap_and_shares_the_shuffle_manager() {
        // `Clone` on `ExecutorContext` is `String` clones (cheap, small
        // strings) + an `Arc::clone` (refcount bump). The two contexts
        // should observably share the same shuffle-manager allocation —
        // assertable via `Arc::strong_count` on `shuffle_manager`.
        let original = ExecutorContext::new("exec-0", "localhost", 50051, "/tmp/share");
        let strong_before = Arc::strong_count(&original.shuffle_manager);
        let twin = original.clone();
        let strong_after = Arc::strong_count(&original.shuffle_manager);
        assert_eq!(strong_after, strong_before + 1);
        // Both handles point at the same underlying ShuffleManager.
        assert!(Arc::ptr_eq(
            &original.shuffle_manager,
            &twin.shuffle_manager
        ));
    }

    #[test]
    fn with_shuffle_manager_reuses_existing_handle() {
        // The test-helper constructor takes an Arc<ShuffleManager> the
        // caller already owns and stores it verbatim — no second Arc
        // allocation, no rebuild.
        let sm = Arc::new(ShuffleManager::new("/tmp/byo"));
        let strong_before = Arc::strong_count(&sm);
        let ctx = ExecutorContext::with_shuffle_manager("exec-1", "h", 50052, Arc::clone(&sm));
        // Two Arcs now: the one we still hold (`sm`) and the one inside `ctx`.
        let strong_after = Arc::strong_count(&sm);
        assert_eq!(strong_after, strong_before + 1);
        assert!(Arc::ptr_eq(&ctx.shuffle_manager, &sm));
    }
}
