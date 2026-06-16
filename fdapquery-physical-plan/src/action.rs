//!
//! Wire-protocol action types for distributed execution: a request is either a
//! whole query (`QueryAction`) or a request for a previously-computed shuffle
//! partition (`ShuffleIdAction`). These are scaffolding for the `distributed` /
//! `flight-server` modules; `ShuffleIdAction` is defined but not yet wired up.
//!
//! ## Shape
//! `trait Action` plus two structs. `QueryAction` can derive only `Clone`, because
//! `logical_plan::LogicalPlan` derives only `Clone` (no `Debug`/`PartialEq`).

use datatypes::ShuffleId;
use logical_plan::LogicalPlan;

/// Marker trait for a distributed-execution action.
pub trait Action {}

/// Execute a whole logical plan.
#[derive(Clone)]
pub struct QueryAction {
    pub plan: LogicalPlan,
}

impl QueryAction {
    pub fn new(plan: LogicalPlan) -> Self {
        Self { plan }
    }
}

impl Action for QueryAction {}

/// Fetch a previously-produced shuffle partition.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShuffleIdAction {
    pub shuffle_id: ShuffleId,
}

impl ShuffleIdAction {
    pub fn new(shuffle_id: ShuffleId) -> Self {
        Self { shuffle_id }
    }
}

impl Action for ShuffleIdAction {}
