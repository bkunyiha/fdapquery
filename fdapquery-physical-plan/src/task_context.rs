//! Per-task runtime context — `TaskContext`, `SessionConfig`, `RuntimeEnv`.
//!
//! These types are the runtime currency the new `ExecutionPlan::execute`
//! signature passes around as `Arc<TaskContext>`. They replace the rquery
//! `ExecutorContext` struct that conflated three concerns DataFusion keeps
//! separate: per-task identity (now `TaskContext::executor_id`), per-session
//! configuration (now `SessionConfig`), and per-process runtime resources
//! (now `RuntimeEnv`). The `ShuffleManager` migrates from the old
//! `ExecutorContext` into `RuntimeEnv::shuffle_manager`.
//!
//! Same shape as DataFusion's `datafusion-execution::TaskContext`,
//! `SessionConfig`, and `RuntimeEnv`. Phase C migrates them to
//! `fdapquery-execution` (they only live here in Phase B because the
//! `ExecutionPlan` trait that references them is in this crate).

use crate::shuffle_manager::ShuffleManager;
use std::collections::HashMap;
use std::sync::Arc;

/// Per-session configuration. Settings keyed by string, parsed lazily by
/// typed accessors. Grows one setting at a time as the engine acquires
/// tunable surfaces.
///
/// Same shape as DataFusion's `SessionConfig`. v0.1 carries just the CSV
/// batch-size setting from rquery's `ExecutionContext::settings`.
#[derive(Debug, Clone, Default)]
pub struct SessionConfig {
    pub settings: HashMap<String, String>,
}

impl SessionConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// CSV batch size — the one setting carried over from rquery's
    /// `ExecutionContext`. Returns `1024` if unset.
    pub fn csv_batch_size(&self) -> usize {
        self.settings
            .get("rquery.csv.batchSize")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024)
    }

    /// Builder-style setter.
    pub fn with_setting(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.settings.insert(key.into(), value.into());
        self
    }
}

/// Per-executor runtime resources. In v0.1 this carries just the shuffle
/// manager; later sessions add a `MemoryManager` for spillable operators
/// and a function registry for UDFs.
///
/// Same shape as DataFusion's `RuntimeEnv`.
#[derive(Debug)]
pub struct RuntimeEnv {
    pub shuffle_manager: Arc<ShuffleManager>,
}

impl RuntimeEnv {
    pub fn new(shuffle_manager: Arc<ShuffleManager>) -> Self {
        Self { shuffle_manager }
    }

    /// Convenience constructor — builds a default `ShuffleManager` (uses
    /// `/tmp/rquery-shuffle` as the base dir) and wraps it. Useful for
    /// tests and single-node contexts that don't care about the shuffle
    /// directory location.
    pub fn default_local() -> Self {
        Self::new(Arc::new(ShuffleManager::default()))
    }
}

/// Per-task runtime context, threaded through every
/// `ExecutionPlan::execute(partition, ctx)` call.
///
/// Operators receive `Arc<TaskContext>` (not `&TaskContext`) so they can
/// move it across `await` points without lifetime tracking. The extra
/// ref-count cost (one atomic increment per `execute` call) is negligible
/// compared to per-batch work.
///
/// Same shape as DataFusion's `TaskContext`.
#[derive(Debug)]
pub struct TaskContext {
    /// Identifies which executor is running this task. Used by
    /// `ShuffleReaderExec` to decide whether a shuffle location is local
    /// (read from disk) or remote (fetch via Flight client).
    pub executor_id: String,
    /// Tunable settings for this query session.
    pub session_config: SessionConfig,
    /// Per-process runtime resources (shuffle manager today, more later).
    pub runtime: Arc<RuntimeEnv>,
}

impl TaskContext {
    pub fn new(
        executor_id: impl Into<String>,
        session_config: SessionConfig,
        runtime: Arc<RuntimeEnv>,
    ) -> Self {
        Self {
            executor_id: executor_id.into(),
            session_config,
            runtime,
        }
    }

    /// Convenience constructor for single-node tests: a `"test"`
    /// executor id, a default `SessionConfig`, and a default
    /// `RuntimeEnv` (the `/tmp/rquery-shuffle` directory). Useful for
    /// tests that don't exercise shuffle.
    pub fn default_test() -> Self {
        Self::new(
            "test",
            SessionConfig::new(),
            Arc::new(RuntimeEnv::default_local()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_config_default_csv_batch_size() {
        let config = SessionConfig::new();
        assert_eq!(config.csv_batch_size(), 1024);
    }

    #[test]
    fn session_config_overridden_csv_batch_size() {
        let config = SessionConfig::new().with_setting("rquery.csv.batchSize", "2048");
        assert_eq!(config.csv_batch_size(), 2048);
    }

    #[test]
    fn session_config_invalid_csv_batch_size_falls_back_to_default() {
        // A non-numeric value falls back to the default rather than panicking.
        let config = SessionConfig::new().with_setting("rquery.csv.batchSize", "not-a-number");
        assert_eq!(config.csv_batch_size(), 1024);
    }

    #[test]
    fn task_context_default_test_constructor() {
        let ctx = TaskContext::default_test();
        assert_eq!(ctx.executor_id, "test");
        assert_eq!(ctx.session_config.csv_batch_size(), 1024);
    }

    #[test]
    fn task_context_construction() {
        let runtime = Arc::new(RuntimeEnv::default_local());
        let config = SessionConfig::new();
        let ctx = TaskContext::new("exec-1", config, Arc::clone(&runtime));
        assert_eq!(ctx.executor_id, "exec-1");
        assert!(Arc::ptr_eq(&ctx.runtime, &runtime));
    }
}
