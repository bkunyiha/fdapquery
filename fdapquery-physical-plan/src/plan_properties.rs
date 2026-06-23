//! Static properties of an `ExecutionPlan`'s output.
//!
//! The optimiser reads these to decide where to insert `RepartitionExec`,
//! whether to reorder joins, etc. v0.1 ships only `output_partitioning`;
//! `output_ordering`, equivalence classes, emission cadence, and
//! boundedness are reserved for later sessions when the optimiser rules
//! that consume them ship (Phase C / Session 11+).

use crate::partitioning::Partitioning;

/// Static properties an operator's output has.
///
/// Same shape as DataFusion's `PlanProperties`. fdapquery v0.1 carries
/// the `output_partitioning` field only; DataFusion's
/// `eq_properties`/`emission_type`/`boundedness` fields gain populated
/// values in later sessions alongside the optimiser rules that consume
/// them.
#[derive(Debug, Clone)]
pub struct PlanProperties {
    /// How this operator's output is partitioned across downstream
    /// workers.
    pub output_partitioning: Partitioning,
    // Reserved: `output_ordering`, `eq_properties`, `emission_type`,
    // `boundedness`. Added one at a time alongside the optimiser rule
    // that needs them.
}

impl PlanProperties {
    /// Construct from an explicit partitioning.
    pub fn new(output_partitioning: Partitioning) -> Self {
        Self {
            output_partitioning,
        }
    }

    /// Convenience constructor — single-partition output with an
    /// unknown distribution shape. Most leaf operators (single-file
    /// scans) use this.
    pub fn single_partition_unknown() -> Self {
        Self::new(Partitioning::UnknownPartitioning(1))
    }

    /// The number of output partitions. Shortcut for
    /// `self.output_partitioning.partition_count()`.
    pub fn partition_count(&self) -> usize {
        self.output_partitioning.partition_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_partition_convenience_constructor() {
        let props = PlanProperties::single_partition_unknown();
        assert_eq!(props.partition_count(), 1);
    }

    #[test]
    fn partition_count_delegates_to_partitioning() {
        let props = PlanProperties::new(Partitioning::RoundRobinBatch(7));
        assert_eq!(props.partition_count(), 7);
    }
}
