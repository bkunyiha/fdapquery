//!
//! `JoinType` enum and the `Join` logical plan. The output schema concatenates
//! the two input schemas, dropping the right (or left, for a right join)
//! duplicate of any join key whose left and right names are identical.

use crate::logical_plan::LogicalPlan;
use datatypes::{Field, Schema};
use std::collections::HashSet;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
}

impl fmt::Display for JoinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            JoinType::Inner => "Inner",
            JoinType::Left => "Left",
            JoinType::Right => "Right",
        };
        write!(f, "{s}")
    }
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

    pub fn schema(&self) -> Schema {
        // Keys whose left and right names are identical produce a single output
        // column rather than two ie if you join two tables using columns with the same name,
        // the output schema should include that join column only once.
        let duplicate_keys: HashSet<String> = self
            .on
            .iter()
            .filter(|(l, r)| l == r)
            .map(|(l, _)| l.clone())
            .collect();

        let fields: Vec<Field> = match self.join_type {
            JoinType::Inner | JoinType::Left => {
                let mut fs = self.left.schema().fields;
                fs.extend(
                    self.right
                        .schema()
                        .fields
                        .into_iter()
                        .filter(|f| !duplicate_keys.contains(&f.name)),
                );
                fs
            }
            JoinType::Right => {
                let mut fs: Vec<Field> = self
                    .left
                    .schema()
                    .fields
                    .into_iter()
                    .filter(|f| !duplicate_keys.contains(&f.name))
                    .collect();
                fs.extend(self.right.schema().fields);
                fs
            }
        };
        Schema::new(fields)
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
