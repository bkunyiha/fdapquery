//! Sort functionalities.
//!
//! Strict mirror of `datafusion::physical_plan::sorts`
//! (`datafusion/physical-plan/src/sorts/mod.rs`). DataFusion's module
//! declares the following children:
//!
//! ```text
//! mod builder;
//! mod cursor;
//! mod merge;
//! mod multi_level_merge;
//! pub mod partial_sort;
//! pub mod partitioned_topk;
//! pub mod sort;
//! pub mod sort_preserving_merge;
//! mod stream;
//! pub mod streaming_merge;
//! ```
//!
//! fdapquery v0.1 ports only `sort` (the public `SortExec` operator,
//! in-memory path). The other submodules support DataFusion's external /
//! spilling sort (`ExternalSorter`, k-way merge, on-disk cursors) and the
//! sort-preserving-merge operator — all of which depend on
//! `RuntimeEnv::disk_manager`, the `MemoryReservation` / `MemoryPool`
//! types, and the `arrow-row` row format. None of those substrates is
//! ported yet; see the deferral note in `sort.rs` and the parent task's
//! follow-up list.

pub mod sort;
