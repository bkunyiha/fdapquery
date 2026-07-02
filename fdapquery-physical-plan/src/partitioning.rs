//! How an operator's output is partitioned. Mirrors DataFusion's
//! `Partitioning` exactly — three variants because each carries
//! different "how do you assign rows to partitions" semantics.

use crate::PhysicalExpr;
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
/// No `Debug` derive: the manual `Debug` impl below renders `Hash` keys
/// via each `PhysicalExpr`'s `Display`, which produces SQL-like operator
/// labels (`a + b`, `c < 10`) rather than the noisy struct dump a derived
/// `Debug` would emit. Mirrors DataFusion's `Partitioning::Debug`.
#[derive(Clone)]
pub enum Partitioning {
    /// Each output partition receives input batches round-robin.
    RoundRobinBatch(usize),
    /// Each row goes to `hash(keys) mod partition_count`.
    Hash(Vec<Arc<dyn PhysicalExpr>>, usize),
    /// Partitioned somehow, the engine doesn't know the shape.
    UnknownPartitioning(usize),
}

impl Partitioning {
    /// The number of output partitions, regardless of variant.
    pub fn partition_count(&self) -> usize {
        match self {
            Self::RoundRobinBatch(n) | Self::UnknownPartitioning(n) | Self::Hash(_, n) => *n,
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

/// Strict mirror of DataFusion's
/// `impl Display for Partitioning`
/// (`datafusion/physical-expr/src/partitioning.rs`). The format strings
/// match DataFusion's character-for-character so plan dumps that include
/// a `Partitioning` render identically in fdapquery and DataFusion.
///
/// DataFusion's `Partitioning` has a fourth variant `Range(RangePartitioning)`;
/// fdapquery doesn't yet support range partitioning, so it's not in the
/// match. When `Range` lands as a Phase 3 addition it appends to this
/// `Display` impl with `write!(f, "{range}")`.
impl std::fmt::Display for Partitioning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RoundRobinBatch(size) => write!(f, "RoundRobinBatch({size})"),
            Self::Hash(phy_exprs, size) => {
                let phy_exprs_str = phy_exprs
                    .iter()
                    .map(|e| format!("{e}"))
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(f, "Hash([{phy_exprs_str}], {size})")
            }
            Self::UnknownPartitioning(size) => write!(f, "UnknownPartitioning({size})"),
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
        let keys: Vec<Arc<dyn PhysicalExpr>> = vec![];
        assert_eq!(Partitioning::Hash(keys, 3).partition_count(), 3);
    }

    /// Byte-for-byte mirror of DataFusion's
    /// `impl Display for Partitioning`. Format strings verified against
    /// `/Users/bkunyiha/Rust/datafusion/datafusion/physical-expr/src/partitioning.rs`.
    #[test]
    fn display_round_robin_matches_datafusion() {
        assert_eq!(
            format!("{}", Partitioning::RoundRobinBatch(8)),
            "RoundRobinBatch(8)"
        );
    }

    #[test]
    fn display_unknown_partitioning_matches_datafusion() {
        assert_eq!(
            format!("{}", Partitioning::UnknownPartitioning(4)),
            "UnknownPartitioning(4)"
        );
    }

    #[test]
    fn display_hash_empty_keys_matches_datafusion() {
        let keys: Vec<Arc<dyn PhysicalExpr>> = vec![];
        assert_eq!(format!("{}", Partitioning::Hash(keys, 1)), "Hash([], 1)");
    }

    #[test]
    fn display_hash_with_single_column_key_matches_datafusion() {
        use crate::Column;
        let keys: Vec<Arc<dyn PhysicalExpr>> = vec![Arc::new(Column::new("a", 0))];
        assert_eq!(
            format!("{}", Partitioning::Hash(keys, 8)),
            "Hash([a@0], 8)"
        );
    }

    #[test]
    fn display_hash_with_two_column_keys_matches_datafusion() {
        use crate::Column;
        let keys: Vec<Arc<dyn PhysicalExpr>> = vec![
            Arc::new(Column::new("a", 0)),
            Arc::new(Column::new("b", 1)),
        ];
        assert_eq!(
            format!("{}", Partitioning::Hash(keys, 4)),
            "Hash([a@0, b@1], 4)"
        );
    }
}
