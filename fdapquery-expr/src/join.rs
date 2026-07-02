//!
//! `JoinType` enum and the `Join` logical plan. The output schema concatenates
//! the two input schemas, dropping the right (or left, for a right join)
//! duplicate of any join key whose left and right names are identical.
//!
//! ## Strict mirror of DataFusion
//! `JoinType` (10 variants), `JoinSide` (3 variants), and `NullEquality`
//! (2 variants) match `datafusion_common::join_type` and
//! `datafusion_common::null_equality` byte-for-byte, including the
//! `Display` impls. fdapquery currently only *executes* `Inner`, `Left`,
//! and `Right` (see `Join::schema()` below and
//! `fdapquery_physical_plan::HashJoinExec::execute`) — the remaining
//! variants are API-shape parity so that consumers (planner, serializer,
//! distributed) can mirror DataFusion's `match` arms. Runtime divergences
//! fail at execution time with a clear error rather than at compile time
//! with a missing variant.

use crate::logical_plan::LogicalPlan;
use fdapquery_datatypes::{Field, Result, Schema};
use std::collections::HashSet;
use std::fmt;

/// Join type. Strict mirror of `datafusion_common::JoinType`.
///
/// 10 variants — `Inner`, `Left`, `Right`, `Full`, `LeftSemi`, `RightSemi`,
/// `LeftAnti`, `RightAnti`, `LeftMark`, `RightMark` — match DataFusion
/// exactly. The `Display` impl renders each variant to the same string
/// DataFusion does (`"Inner"`, `"LeftSemi"`, etc.) so the `HashJoinExec`
/// Default DisplayAs format string is byte-equivalent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinType {
    /// Inner Join — returns only rows where there is a matching value in both
    /// tables based on the join condition.
    Inner,
    /// Left Join — returns all rows from the left table and matching rows from
    /// the right table. If no match, NULLs are returned for right columns.
    Left,
    /// Right Join — returns all rows from the right table and matching rows
    /// from the left table. If no match, NULLs are returned for left columns.
    Right,
    /// Full Join (Full Outer Join) — returns all rows from both tables,
    /// matching where possible and padding with NULLs otherwise.
    Full,
    /// Left Semi Join — returns rows from the left table that have matching
    /// rows in the right table. Only left columns are returned.
    LeftSemi,
    /// Right Semi Join — returns rows from the right table that have matching
    /// rows in the left table. Only right columns are returned.
    RightSemi,
    /// Left Anti Join — returns rows from the left table that do NOT have a
    /// matching row in the right table.
    LeftAnti,
    /// Right Anti Join — returns rows from the right table that do NOT have a
    /// matching row in the left table.
    RightAnti,
    /// Left Mark Join — returns one record per left row with an additional
    /// `"mark"` column that is true if there is at least one match in the
    /// right input. Used to decorrelate `EXISTS` subqueries.
    LeftMark,
    /// Right Mark Join — symmetric to `LeftMark` for the right input.
    RightMark,
}

impl fmt::Display for JoinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            JoinType::Inner => "Inner",
            JoinType::Left => "Left",
            JoinType::Right => "Right",
            JoinType::Full => "Full",
            JoinType::LeftSemi => "LeftSemi",
            JoinType::RightSemi => "RightSemi",
            JoinType::LeftAnti => "LeftAnti",
            JoinType::RightAnti => "RightAnti",
            JoinType::LeftMark => "LeftMark",
            JoinType::RightMark => "RightMark",
        };
        write!(f, "{s}")
    }
}

/// Join side. Strict mirror of `datafusion_common::JoinSide`. Stores the
/// referred table side during column-index calculations on join output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinSide {
    /// Left side of the join
    Left,
    /// Right side of the join
    Right,
    /// Neither side — used for Mark joins, where the `"mark"` column does
    /// not belong to either input.
    None,
}

impl fmt::Display for JoinSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            JoinSide::Left => "left",
            JoinSide::Right => "right",
            JoinSide::None => "none",
        };
        write!(f, "{s}")
    }
}

/// Null-handling behaviour for join equality. Strict mirror of
/// `datafusion_common::NullEquality`. When `NullEqualsNothing`,
/// `null != null` (standard SQL semantics); when `NullEqualsNull`, two null
/// keys are considered equal (used by some `INTERSECT` / array-comparison
/// contexts).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NullEquality {
    /// Null is *not* equal to anything (`null != null`).
    NullEqualsNothing,
    /// Null is equal to null (`null == null`).
    NullEqualsNull,
}

#[derive(Clone)]
pub struct Join {
    pub left: Box<LogicalPlan>,
    pub right: Box<LogicalPlan>,
    pub join_type: JoinType,
    /// Join keys as `(left_name, right_name)` pairs.
    pub on: Vec<(String, String)>,
}

impl Join {
    pub fn new(
        left: LogicalPlan,
        right: LogicalPlan,
        join_type: JoinType,
        on: Vec<(String, String)>,
    ) -> Self {
        Self {
            left: Box::new(left),
            right: Box::new(right),
            join_type,
            on,
        }
    }

    pub fn schema(&self) -> Result<Schema> {
        // Keys whose left and right names are identical produce a single output
        // column rather than two ie if you join two tables using columns with the same name,
        // the output schema should include that join column only once.
        let duplicate_keys: HashSet<String> = self
            .on
            .iter()
            .filter(|(l, r)| l == r)
            .map(|(l, _)| l.clone())
            .collect();

        let left_schema = self.left.schema()?;
        let right_schema = self.right.schema()?;

        // arrow's `Fields` is `Arc<[Field]>` (immutable); collect into a
        // fresh `Vec<Field>` via `.iter().map(...).cloned()` before mutating.
        //
        // fdapquery only computes logical-plan schemas for the three classical
        // join shapes (`Inner` / `Left` / `Right`). The remaining variants
        // (`Full`, `LeftSemi`, `RightSemi`, `LeftAnti`, `RightAnti`,
        // `LeftMark`, `RightMark`) are API-shape parity with DataFusion's
        // 10-variant enum; their schema-derivation rules live in DataFusion's
        // `build_join_schema` and will land here when the logical planner
        // grows to emit them. Until then this match catches them with an
        // `unimplemented!` so the failure is loud and clear at logical-plan
        // build time, not at execute time.
        let fields: Vec<Field> = match self.join_type {
            JoinType::Inner | JoinType::Left => {
                let mut fs: Vec<Field> = left_schema
                    .fields()
                    .iter()
                    .map(|f| f.as_ref().clone())
                    .collect();
                fs.extend(
                    right_schema
                        .fields()
                        .iter()
                        .filter(|f| !duplicate_keys.contains(f.name()))
                        .map(|f| f.as_ref().clone()),
                );
                fs
            }
            JoinType::Right => {
                let mut fs: Vec<Field> = left_schema
                    .fields()
                    .iter()
                    .filter(|f| !duplicate_keys.contains(f.name()))
                    .map(|f| f.as_ref().clone())
                    .collect();
                fs.extend(right_schema.fields().iter().map(|f| f.as_ref().clone()));
                fs
            }
            JoinType::Full
            | JoinType::LeftSemi
            | JoinType::RightSemi
            | JoinType::LeftAnti
            | JoinType::RightAnti
            | JoinType::LeftMark
            | JoinType::RightMark => {
                unimplemented!(
                    "Join::schema for {:?} is not yet emitted by fdapquery's logical planner; \
                     mirror parity with DataFusion's 10-variant JoinType only.",
                    self.join_type
                )
            }
        };
        Ok(Schema::new(fields))
    }

    pub fn children(&self) -> Vec<&LogicalPlan> {
        vec![self.left.as_ref(), self.right.as_ref()]
    }
}

impl fmt::Display for Join {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let on: Vec<String> = self.on.iter().map(|(l, r)| format!("({l}, {r})")).collect();
        write!(f, "Join: type={}, on=[{}]", self.join_type, on.join(", "))
    }
}
