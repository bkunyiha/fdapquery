//! Typed configuration options for fdapquery.
//!
//! Scaled-down mirror of DataFusion's [`datafusion_common::config::ConfigOptions`]
//! (`datafusion/common/src/config.rs`). DataFusion's `ConfigOptions` groups
//! 40+ fields across five sub-structs (`ExecutionOptions`,
//! `OptimizerOptions`, `SqlParserOptions`, `CatalogOptions`,
//! `ExplainOptions`). At v0.1 fdapquery ships only the fields it actually
//! consumes today or in imminent Phase 3 work — currently
//! [`ExecutionOptions::target_partitions`] (consumed by the future
//! `RepartitionExec` + `EnforceDistribution` optimizer rule) and
//! [`ExecutionOptions::batch_size`] (mirrors fdapquery's existing
//! `rquery.csv.batchSize` string setting).
//!
//! Additional sub-structs land per-consumer in Phase 3, not up-front. When a
//! new field is introduced, add it to the sub-struct that DataFusion places
//! it in and mirror the DataFusion field name exactly.
//!
//! Naming policy: every public method and field name mirrors DataFusion's
//! public surface (`target_partitions`, `with_target_partitions`,
//! `batch_size`, `with_batch_size`). A developer reading fdapquery source
//! after DataFusion source should not have to translate identifiers.

use std::num::NonZeroUsize;

/// String-map key for `ExecutionOptions::target_partitions`. Matches
/// DataFusion's `datafusion.execution.target_partitions` key so
/// `set_str`-style APIs are wire-compatible with DataFusion when they
/// arrive.
pub const KEY_TARGET_PARTITIONS: &str = "datafusion.execution.target_partitions";

/// String-map key for `ExecutionOptions::batch_size`. Matches DataFusion's
/// `datafusion.execution.batch_size` key for the same wire-compatibility
/// reason. The legacy fdapquery key `rquery.csv.batchSize` is kept as a
/// fallback alias on `SessionConfig` so rquery-DNA callers continue to
/// work.
pub const KEY_BATCH_SIZE: &str = "datafusion.execution.batch_size";

/// Top-level typed configuration for a session.
///
/// Scaled-down mirror of DataFusion's `ConfigOptions`. Fields are grouped
/// into sub-structs so the DataFusion namespace hierarchy carries through
/// (e.g. `options.execution.target_partitions`).
///
/// Sub-structs that DataFusion ships (`optimizer`, `sql_parser`, `catalog`,
/// `explain`) are intentionally omitted at v0.1 — they gain a fdapquery
/// consumer per-rule during Phase 3 and are added at that point, not
/// pre-declared empty.
#[derive(Debug, Clone, Default)]
pub struct ConfigOptions {
    /// Execution-time options: partitioning target, batch size.
    pub execution: ExecutionOptions,
}

/// Execution-time knobs — the group that DataFusion places under
/// `datafusion.execution.*`.
#[derive(Debug, Clone)]
pub struct ExecutionOptions {
    /// Number of partitions execution should be distributed across.
    ///
    /// Consumed by the future `RepartitionExec` + `EnforceDistribution`
    /// optimizer rule. Defaults to the number of available CPU cores
    /// when constructed via [`Default::default`], falling back to `1`
    /// if the system's parallelism cannot be determined (mirrors
    /// DataFusion's `datafusion_common::utils::get_available_parallelism`).
    pub target_partitions: usize,

    /// Default batch size for readers and buffers.
    ///
    /// Mirrors DataFusion's `datafusion.execution.batch_size`. fdapquery
    /// v0.1 uses this for CSV read chunk size — the same knob rquery
    /// exposed as `rquery.csv.batchSize`. Defaults to `1024`, matching
    /// fdapquery's rquery-era default; DataFusion defaults to `8192`, but
    /// fdapquery keeps `1024` for continuity with existing tests that
    /// depend on it.
    pub batch_size: usize,
}

impl Default for ExecutionOptions {
    fn default() -> Self {
        Self {
            target_partitions: default_target_partitions(),
            batch_size: 1024,
        }
    }
}

/// Available parallelism, or `1` if it cannot be determined. Mirrors
/// DataFusion's `datafusion_common::utils::get_available_parallelism`
/// behaviour without the `LazyLock` cache (fdapquery only calls this at
/// `SessionConfig::default()` time, so the cache is unnecessary).
fn default_target_partitions() -> usize {
    std::thread::available_parallelism().map_or(1, NonZeroUsize::get)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_options_default_batch_size_is_1024() {
        let opts = ExecutionOptions::default();
        assert_eq!(opts.batch_size, 1024);
    }

    #[test]
    fn execution_options_default_target_partitions_is_positive() {
        let opts = ExecutionOptions::default();
        assert!(opts.target_partitions >= 1);
    }

    #[test]
    fn config_options_default_composes_execution_defaults() {
        let opts = ConfigOptions::default();
        assert_eq!(opts.execution.batch_size, 1024);
        assert!(opts.execution.target_partitions >= 1);
    }
}
