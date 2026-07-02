//! `ExecutorConfig` + `DistributedConfig` data structs. Holds the cluster-
//! level configuration and a single `build_runtime_env()` helper that wires
//! `shuffle_dir` into a runtime an executor process can use.

use fdapquery_physical_plan::{RuntimeEnv, ShuffleManager};
use std::sync::Arc;

/// Configuration for a single executor in the distributed cluster.
///
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutorConfig {
    /// Unique identifier for this executor.
    pub id: String,
    /// Hostname or IP address.
    pub host: String,
    /// Port for the Arrow Flight service.
    pub port: i32,
}

impl ExecutorConfig {
    pub fn new(id: impl Into<String>, host: impl Into<String>, port: i32) -> Self {
        Self {
            id: id.into(),
            host: host.into(),
            port,
        }
    }
}

/// Configuration for the distributed query execution environment.
///
/// Builder methods set defaults for the shuffle directory and partition count.
///
/// ## Shuffle directory
/// The default shuffle directory is `/tmp/fdapquery-shuffle`.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// List of executors in the cluster.
    pub executors: Vec<ExecutorConfig>,
    /// Base directory for shuffle data on each executor.
    pub shuffle_dir: String,
    /// Default number of partitions for shuffle operations. `0` is a sentinel
    /// meaning "use the number of executors" — see [`Self::partition_count`].
    pub default_partitions: i32,
}

impl DistributedConfig {
    /// Default shuffle directory.
    pub const DEFAULT_SHUFFLE_DIR: &str = "/tmp/fdapquery-shuffle";

    /// Construct with sensible defaults — empty `shuffle_dir`,
    /// `default_partitions = 0` (i.e., use executor count).
    pub fn new(executors: Vec<ExecutorConfig>) -> Self {
        Self {
            executors,
            shuffle_dir: Self::DEFAULT_SHUFFLE_DIR.to_string(),
            default_partitions: 0,
        }
    }

    /// Builder: override the default shuffle directory.
    pub fn with_shuffle_dir(mut self, dir: impl Into<String>) -> Self {
        self.shuffle_dir = dir.into();
        self
    }

    /// Builder: override the default partition count. Pass `0` to fall back to
    /// the executor count.
    pub fn with_default_partitions(mut self, n: i32) -> Self {
        self.default_partitions = n;
        self
    }

    /// Effective partition count for shuffle operations.
    /// otherwise the executor count is used.
    pub fn partition_count(&self) -> i32 {
        if self.default_partitions > 0 {
            self.default_partitions
        } else {
            self.executors.len() as i32
        }
    }

    /// Build a `RuntimeEnv` whose `ShuffleManager` is keyed on this config's
    /// `shuffle_dir`. Use this whenever you spawn an executor (in-process for
    /// testing or as a separate process) so the cluster config is the single
    /// source of truth for where shuffle files land.
    ///
    /// Without this helper, the executor process and the cluster config can
    /// silently drift apart — the planner believes shuffle files live at
    /// `config.shuffle_dir` while the executor writes them somewhere else.
    /// Routing every executor construction through `build_runtime_env()`
    /// makes that mismatch impossible by construction.
    pub fn build_runtime_env(&self) -> RuntimeEnv {
        RuntimeEnv::new(Arc::new(ShuffleManager::new(self.shuffle_dir.clone())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_count_defaults_to_executor_count() {
        let cfg = DistributedConfig::new(vec![
            ExecutorConfig::new("exec-1", "localhost", 50051),
            ExecutorConfig::new("exec-2", "localhost", 50052),
            ExecutorConfig::new("exec-3", "localhost", 50053),
        ]);
        assert_eq!(cfg.partition_count(), 3);
    }

    #[test]
    fn explicit_default_partitions_overrides_executor_count() {
        let cfg = DistributedConfig::new(vec![ExecutorConfig::new("exec-1", "localhost", 50051)])
            .with_default_partitions(8);
        assert_eq!(cfg.partition_count(), 8);
    }

    #[test]
    fn default_shuffle_dir_is_fdapquery_path() {
        let cfg = DistributedConfig::new(vec![]);
        assert_eq!(cfg.shuffle_dir, "/tmp/fdapquery-shuffle");
    }

    #[test]
    fn build_runtime_env_uses_configured_shuffle_dir() {
        let cfg = DistributedConfig::new(vec![]).with_shuffle_dir("/tmp/fdap-test-build-env");
        let runtime = cfg.build_runtime_env();
        assert_eq!(runtime.shuffle_manager.base_dir, "/tmp/fdap-test-build-env");
    }

    #[test]
    fn build_runtime_env_picks_up_default_when_unset() {
        let cfg = DistributedConfig::new(vec![]);
        let runtime = cfg.build_runtime_env();
        assert_eq!(runtime.shuffle_manager.base_dir, "/tmp/fdapquery-shuffle");
    }
}
