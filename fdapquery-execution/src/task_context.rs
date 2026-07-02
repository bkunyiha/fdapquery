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
//! `SessionConfig`, and `RuntimeEnv`. Future work migrates them to
//! `fdapquery-execution` (they only live here because the
//! `ExecutionPlan` trait that references them is in this crate).

use crate::shuffle_manager::ShuffleManager;
use fdapquery_common::config::{ConfigOptions, KEY_BATCH_SIZE, KEY_TARGET_PARTITIONS};
use std::collections::HashMap;
use std::sync::Arc;

/// Legacy string-map key for the CSV batch size — the rquery-DNA key that
/// existing consumers (`SessionContext::new`, `ParallelContext::with_parallelism`)
/// still read from the `settings` map. Kept as a fallback alias in
/// [`SessionConfig::csv_batch_size`] alongside DataFusion's canonical
/// `datafusion.execution.batch_size` key.
const LEGACY_KEY_CSV_BATCH_SIZE: &str = "rquery.csv.batchSize";

/// Per-session configuration.
///
/// Scaled-down mirror of DataFusion's
/// [`datafusion::execution::config::SessionConfig`]. Two fields:
///
/// - [`Self::options`] — typed [`ConfigOptions`] (mirror of DataFusion's
///   `datafusion_common::config::ConfigOptions`). Preferred read path.
///   Grows one field at a time as fdapquery acquires new tunable
///   surfaces. Task #149.
/// - [`Self::settings`] — legacy `HashMap<String, String>` from the
///   rquery-DNA shape. Preserved so existing consumers that read
///   `ctx.settings.get("key")` continue to work without change. Every
///   `with_*` setter mirrors the typed field into the map (using the
///   `datafusion.execution.*` keys, matching DataFusion's `set_str`
///   convention) and vice versa, so both views stay in sync.
///
/// New code should prefer the typed accessors (`target_partitions`,
/// `batch_size`, `options`). The `settings` field is a compatibility
/// shim and will migrate to strictly typed access over Phase 3 as
/// consumers move to `ConfigOptions`.
#[derive(Debug, Clone, Default)]
pub struct SessionConfig {
    /// Typed configuration. Mirror of DataFusion's `SessionConfig::options`
    /// (`Arc<ConfigOptions>` there; we hold a plain `ConfigOptions` because
    /// no fdapquery consumer clones the config yet).
    options: ConfigOptions,
    /// Legacy string-map settings. Kept public for backward compatibility
    /// with existing consumers (`SessionContext.settings`,
    /// `ParallelContext.settings`, `SessionStateBuilder` re-mirroring).
    /// The `with_setting` / `with_target_partitions` / `with_batch_size`
    /// setters keep this in sync with `options` for the fields that have
    /// both a typed and a string-key representation.
    pub settings: HashMap<String, String>,
}

impl SessionConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a reference to the typed [`ConfigOptions`]. Preferred read
    /// path for new consumers.
    pub fn options(&self) -> &ConfigOptions {
        &self.options
    }

    /// Return the target-partitions count. Mirrors DataFusion's
    /// [`SessionConfig::target_partitions`].
    ///
    /// Consumed by Phase 3 task #125 (`RepartitionExec` +
    /// `EnforceDistribution` optimizer rule).
    pub fn target_partitions(&self) -> usize {
        self.options.execution.target_partitions
    }

    /// Return the read/buffer batch size. Mirrors DataFusion's
    /// [`SessionConfig::batch_size`].
    pub fn batch_size(&self) -> usize {
        self.options.execution.batch_size
    }

    /// CSV batch size accessor kept as an alias for [`Self::batch_size`]
    /// so existing rquery-DNA consumers (`SessionContext`,
    /// `ParallelContext`, tests) compile without change. Reads the typed
    /// [`ExecutionOptions::batch_size`](fdapquery_common::config::ExecutionOptions::batch_size)
    /// field; the legacy `rquery.csv.batchSize` string key is honoured
    /// via [`Self::with_setting`], which mirrors it into the typed
    /// field at construction time.
    pub fn csv_batch_size(&self) -> usize {
        self.batch_size()
    }

    /// Typed builder for [`ExecutionOptions::target_partitions`](fdapquery_common::config::ExecutionOptions::target_partitions).
    /// Updates both the typed field and the `datafusion.execution.target_partitions`
    /// entry in [`Self::settings`] so the two views stay in sync.
    /// Mirrors DataFusion's [`SessionConfig::with_target_partitions`].
    pub fn with_target_partitions(mut self, n: usize) -> Self {
        self.options.execution.target_partitions = n;
        self.settings
            .insert(KEY_TARGET_PARTITIONS.to_string(), n.to_string());
        self
    }

    /// Typed builder for [`ExecutionOptions::batch_size`](fdapquery_common::config::ExecutionOptions::batch_size).
    /// Updates both the typed field and the `datafusion.execution.batch_size`
    /// entry in [`Self::settings`]. Mirrors DataFusion's
    /// [`SessionConfig::with_batch_size`].
    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.options.execution.batch_size = n;
        self.settings
            .insert(KEY_BATCH_SIZE.to_string(), n.to_string());
        self
    }

    /// String-map builder — the rquery-DNA setter kept for
    /// backward compatibility. When the incoming key matches a known
    /// typed field, the typed field is updated too so
    /// [`Self::target_partitions`] / [`Self::batch_size`] readers see the
    /// change. Unknown keys are stored in the map unchanged so
    /// user-supplied extensions keep working.
    ///
    /// Recognised keys:
    /// - `datafusion.execution.target_partitions` (DataFusion canonical)
    /// - `datafusion.execution.batch_size` (DataFusion canonical)
    /// - `rquery.csv.batchSize` (legacy fdapquery/rquery key, aliased
    ///   onto `execution.batch_size`)
    pub fn with_setting(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let key = key.into();
        let value = value.into();
        // Mirror the string value into the typed field if we recognise the
        // key. Non-numeric values for numeric fields are ignored on the
        // typed side (matching the pre-Task-#149 behaviour of
        // `csv_batch_size` on a malformed string, which fell back to the
        // default rather than panicking).
        match key.as_str() {
            KEY_TARGET_PARTITIONS => {
                if let Ok(n) = value.parse::<usize>() {
                    self.options.execution.target_partitions = n;
                }
            }
            KEY_BATCH_SIZE | LEGACY_KEY_CSV_BATCH_SIZE => {
                if let Ok(n) = value.parse::<usize>() {
                    self.options.execution.batch_size = n;
                }
            }
            _ => {}
        }
        self.settings.insert(key, value);
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
    /// `/tmp/fdapquery-shuffle` as the base dir) and wraps it. Useful for
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
/// ## Network identity here for now
/// `executor_host` and `executor_port` live on this struct
/// because every existing `ExecutorContext::new(id, host, port, dir)`
/// call site migrates one-to-one to `TaskContext::new(...)`.
/// Future work reshapes — these fields belong in `RuntimeEnv` for a clean
/// DataFusion-shape, and the move happens as part of the
/// broader `fdapquery-execution` crate reorg.
///
/// Same shape as DataFusion's `TaskContext` modulo the network identity
/// note above.
#[derive(Debug)]
pub struct TaskContext {
    /// Identifies which executor is running this task. Used by
    /// `ShuffleReaderExec` to decide whether a shuffle location is local
    /// (read from disk) or remote (fetch via Flight client).
    pub executor_id: String,
    /// Hostname or IP this executor listens on. Tagged onto the
    /// `ShuffleLocation`s a `ShuffleWriterExec` produces so downstream
    /// readers know which executor to fetch from. Future work migrates this
    /// to `RuntimeEnv`.
    pub executor_host: String,
    /// Port this executor listens on. Same story as
    /// `executor_host`.
    pub executor_port: u16,
    /// Tunable settings for this query session.
    pub session_config: SessionConfig,
    /// Per-process runtime resources (shuffle manager today, more later).
    pub runtime: Arc<RuntimeEnv>,
}

impl TaskContext {
    pub fn new(
        executor_id: impl Into<String>,
        executor_host: impl Into<String>,
        executor_port: u16,
        session_config: SessionConfig,
        runtime: Arc<RuntimeEnv>,
    ) -> Self {
        Self {
            executor_id: executor_id.into(),
            executor_host: executor_host.into(),
            executor_port,
            session_config,
            runtime,
        }
    }

    /// Convenience constructor for single-node tests: a `"test"`
    /// executor id, `localhost:0` as the network identity, a default
    /// `SessionConfig`, and a default `RuntimeEnv` (the
    /// `/tmp/fdapquery-shuffle` directory). Useful for tests that don't
    /// exercise shuffle.
    pub fn default_test() -> Self {
        Self::new(
            "test",
            "localhost",
            0,
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
    fn session_config_default_target_partitions_is_positive() {
        // Defaults to `available_parallelism().unwrap_or(1)`.
        let config = SessionConfig::new();
        assert!(config.target_partitions() >= 1);
    }

    #[test]
    fn session_config_with_target_partitions_updates_typed_and_map() {
        let config = SessionConfig::new().with_target_partitions(4);
        assert_eq!(config.target_partitions(), 4);
        // The bridge mirrors the typed field into the DataFusion-canonical
        // string key so string-map consumers observe the same value.
        assert_eq!(
            config.settings.get("datafusion.execution.target_partitions"),
            Some(&"4".to_string()),
        );
    }

    #[test]
    fn session_config_with_batch_size_updates_typed_and_map() {
        let config = SessionConfig::new().with_batch_size(2048);
        assert_eq!(config.batch_size(), 2048);
        assert_eq!(config.csv_batch_size(), 2048); // alias
        assert_eq!(
            config.settings.get("datafusion.execution.batch_size"),
            Some(&"2048".to_string()),
        );
    }

    #[test]
    fn session_config_with_setting_bridges_datafusion_key_to_typed_field() {
        let config =
            SessionConfig::new().with_setting("datafusion.execution.target_partitions", "8");
        assert_eq!(config.target_partitions(), 8);
    }

    #[test]
    fn session_config_with_setting_bridges_legacy_key_to_typed_batch_size() {
        // Legacy rquery.csv.batchSize continues to feed the typed field.
        let config = SessionConfig::new().with_setting("rquery.csv.batchSize", "4096");
        assert_eq!(config.batch_size(), 4096);
        assert_eq!(config.csv_batch_size(), 4096);
    }

    #[test]
    fn session_config_with_setting_preserves_unknown_keys() {
        let config = SessionConfig::new().with_setting("user.custom.key", "custom-value");
        assert_eq!(
            config.settings.get("user.custom.key"),
            Some(&"custom-value".to_string()),
        );
    }

    #[test]
    fn session_config_options_accessor_returns_typed_view() {
        let config = SessionConfig::new()
            .with_target_partitions(3)
            .with_batch_size(2048);
        assert_eq!(config.options().execution.target_partitions, 3);
        assert_eq!(config.options().execution.batch_size, 2048);
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
        let ctx = TaskContext::new("exec-1", "10.0.0.1", 50051, config, Arc::clone(&runtime));
        assert_eq!(ctx.executor_id, "exec-1");
        assert_eq!(ctx.executor_host, "10.0.0.1");
        assert_eq!(ctx.executor_port, 50051);
        assert!(Arc::ptr_eq(&ctx.runtime, &runtime));
    }
}
