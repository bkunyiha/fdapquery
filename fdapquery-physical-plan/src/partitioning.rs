//! How an operator's output is partitioned. Mirrors DataFusion's
//! `Partitioning` exactly — three variants because each carries
//! different "how do you assign rows to partitions" semantics.

use crate::Expression;
use std::sync::Arc;

/// Output partitioning of an `ExecutionPlan`.
///
/// Three variants:
/// - `RoundRobinBatch(n)` — each input batch goes to a partition picked
///   round-robin. Used by `RepartitionExec` to even out load across
///   downstream workers.
/// - `Hash(keys, n)` — each row is sent to partition
///   `hash(keys) mod n`. Used to co-locate matching keys for joins and
///   aggregates.
/// - `UnknownPartitioning(n)` — the engine knows there are `n` output
///   partitions but not how rows are distributed. Used by leaf
///   operators (scans) where the source dictates partitioning.
///
/// Same shape as `datafusion-physical-expr::Partitioning`.
///
/// No `Debug` derive: `dyn Expression` is not `Debug`-bound (operators
/// implement `Display` for human-readable rendering instead). The
/// `Display` impl below renders the variant in a tree-printer-friendly
/// form.
#[derive(Clone)]
pub enum Partitioning {
    /// Each output partition receives input batches round-robin.
    RoundRobinBatch(usize),
    /// Each row goes to `hash(keys) mod partition_count`.
    Hash(Vec<Arc<dyn Expression>>, usize),
    /// Partitioned somehow, the engine doesn't know the shape.
    UnknownPartitioning(usize),
}

impl Partitioning {
    /// The number of output partitions, regardless of variant.
    pub fn partition_count(&self) -> usize {
        match self {
            Self::RoundRobinBatch(n) | Self::UnknownPartitioning(n) => *n,
            Self::Hash(_, n) => *n,
        }
    }
}

impl std::fmt::Debug for Partitioning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RoundRobinBatch(n) => write!(f, "RoundRobinBatch({n})"),
            Self::Hash(keys, n) => {
                let key_str: Vec<String> = keys.iter().map(|k| k.to_string()).collect();
                write!(f, "Hash([{}], {n})", key_str.join(", "))
            }
            Self::UnknownPartitioning(n) => write!(f, "UnknownPartitioning({n})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_count_round_robin() {
        assert_eq!(Partitioning::RoundRobinBatch(4).partition_count(), 4);
    }

    #[test]
    fn partition_count_unknown() {
        assert_eq!(Partitioning::UnknownPartitioning(1).partition_count(), 1);
        assert_eq!(Partitioning::UnknownPartitioning(8).partition_count(), 8);
    }

    #[test]
    fn partition_count_hash() {
        let keys: Vec<Arc<dyn Expression>> = vec![];
        assert_eq!(Partitioning::Hash(keys, 3).partition_count(), 3);
    }
}
