//!
//! The mode of an `AggregateExec`, used to support distributed (two-stage)
//! aggregation. Strict mirror of DataFusion's
//! `datafusion::physical_plan::aggregates::AggregateMode`: the variant names
//! (`Partial`, `Final`, `FinalPartitioned`, `Single`, `SinglePartitioned`,
//! `PartialReduce`), their order, and the derive list match
//! `datafusion/physical-plan/src/aggregates/mod.rs` exactly.

/// Aggregation modes
///
/// See `Accumulator::state` for background information on multi-phase
/// aggregation and how these modes are used.
///
/// # Variants and their input/output modes
///
/// ```text
///                       | Input: Raw data           | Input: Partial state
/// Output: Final values  | Single, SinglePartitioned | Final, FinalPartitioned
/// Output: Partial state | Partial                   | PartialReduce
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum AggregateMode {
    /// One of multiple layers of aggregation, any input partitioning
    ///
    /// Partial aggregate that can be applied in parallel across input
    /// partitions.
    ///
    /// This is the first phase of a multi-phase aggregation.
    Partial,
    /// *Final* of multiple layers of aggregation, in exactly one partition
    ///
    /// Final aggregate that produces a single partition of output by combining
    /// the output of multiple partial aggregates.
    ///
    /// This is the second phase of a multi-phase aggregation.
    Final,
    /// *Final* of multiple layers of aggregation, input is *Partitioned*
    ///
    /// Final aggregate that works on pre-partitioned data.
    FinalPartitioned,
    /// *Single* layer of Aggregation, input is exactly one partition
    ///
    /// Applies the entire logical aggregation operation in a single operator,
    /// as opposed to Partial / Final modes which apply the logical aggregation
    /// using two operators.
    Single,
    /// *Single* layer of Aggregation, input is *Partitioned*
    ///
    /// Applies the entire logical aggregation operation in a single operator,
    /// as opposed to Partial / Final modes which apply the logical aggregation
    /// using two operators.
    SinglePartitioned,
    /// Combine multiple partial aggregations to produce a new partial
    /// aggregation.
    ///
    /// Input is intermediate accumulator state (like Final), but output is
    /// also intermediate accumulator state (like Partial). This enables
    /// tree-reduce aggregation strategies where partial results from
    /// multiple workers are combined in multiple stages before a final
    /// evaluation.
    PartialReduce,
}

impl Default for AggregateMode {
    /// Single-node aggregation — the only mode the single-node engine uses.
    /// DataFusion does not derive `Default` for `AggregateMode`; fdapquery
    /// keeps a default because the planner relies on it before the
    /// `distributed` module wires up multi-stage planning.
    fn default() -> Self {
        AggregateMode::Single
    }
}
