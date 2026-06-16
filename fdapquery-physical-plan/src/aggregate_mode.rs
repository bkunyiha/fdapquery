//!
//! The mode of a `HashAggregateExec`, used to support distributed (two-stage)
//! aggregation.

/// How a `HashAggregateExec` aggregates.
///
/// In distributed execution, aggregation runs in two stages: each executor
/// computes `Partial` aggregates on its local data, then a coordinator merges
/// them in a `Final` stage. `Complete` is single-node execution where all data is
/// aggregated in one pass — the default and the only mode exercised until the
/// `distributed` module (14) is ported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AggregateMode {
    /// Single-node aggregation — all data is aggregated in one step.
    #[default]
    Complete,
    /// First stage of distributed aggregation — outputs intermediate state.
    Partial,
    /// Second stage of distributed aggregation — merges intermediate states.
    Final,
}
