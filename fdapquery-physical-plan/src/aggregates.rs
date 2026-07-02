//!
//! `PhysicalGroupBy` — strict mirror of DataFusion's
//! `datafusion::physical_plan::aggregates::PhysicalGroupBy`.
//!
//! The shape (private `expr`/`null_expr`/`groups`/`has_grouping_set` fields,
//! the `new` / `new_single` constructors, the accessor names `expr()`,
//! `null_expr()`, `groups()`, etc.) matches DataFusion's source
//! (`datafusion/physical-plan/src/aggregates/mod.rs` lines 264–410) verbatim.
//!
//! fdapquery's planner only emits the simple `GROUP BY a, b, …` shape today
//! (built via `PhysicalGroupBy::new_single`). The full grouping-set surface
//! (`null_expr`, `groups` matrix, `has_grouping_set`) is wired through so
//! `AggregateExec`'s display path and accessor surface are byte-for-byte
//! identical to DataFusion's, leaving the cube/rollup planner path as the
//! only remaining work to enable the multi-group case.

use crate::PhysicalExpr;
use std::sync::Arc;

/// Represents `GROUP BY` clause in the plan (including the more general GROUPING SET)
/// In the case of a simple `GROUP BY a, b` clause, this will contain the expression [a, b]
/// and a single group [false, false].
/// In the case of `GROUP BY GROUPING SETS/CUBE/ROLLUP` the planner will expand the expression
/// into multiple groups, using null expressions to align each group.
/// For example, with a group by clause `GROUP BY GROUPING SETS ((a,b),(a),(b))` the planner should
/// create a `PhysicalGroupBy` like
/// ```text
/// PhysicalGroupBy {
///     expr: [(col(a), a), (col(b), b)],
///     null_expr: [(NULL, a), (NULL, b)],
///     groups: [
///         [false, false], // (a,b)
///         [false, true],  // (a) <=> (a, NULL)
///         [true, false]   // (b) <=> (NULL, b)
///     ]
/// }
/// ```
#[derive(Clone, Debug, Default)]
pub struct PhysicalGroupBy {
    /// Distinct (Physical Expr, Alias) in the grouping set
    pub(crate) expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
    /// Corresponding NULL expressions for expr
    pub(crate) null_expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
    /// Null mask for each group in this grouping set. Each group is
    /// composed of either one of the group expressions in expr or a null
    /// expression in null_expr. If `groups[i][j]` is true, then the
    /// j-th expression in the i-th group is NULL, otherwise it is `expr[j]`.
    pub(crate) groups: Vec<Vec<bool>>,
    /// True when GROUPING SETS/CUBE/ROLLUP are used so `__grouping_id` should
    /// be included in the output schema.
    pub(crate) has_grouping_set: bool,
}

impl PhysicalGroupBy {
    /// Create a new `PhysicalGroupBy`
    pub fn new(
        expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
        null_expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
        groups: Vec<Vec<bool>>,
        has_grouping_set: bool,
    ) -> Self {
        Self {
            expr,
            null_expr,
            groups,
            has_grouping_set,
        }
    }

    /// Create a GROUPING SET with only a single group. This is the "standard"
    /// case when building a plan from an expression such as `GROUP BY a,b,c`
    pub fn new_single(expr: Vec<(Arc<dyn PhysicalExpr>, String)>) -> Self {
        let num_exprs = expr.len();
        Self {
            expr,
            null_expr: vec![],
            groups: vec![vec![false; num_exprs]],
            has_grouping_set: false,
        }
    }

    /// Returns the group expressions
    pub fn expr(&self) -> &[(Arc<dyn PhysicalExpr>, String)] {
        &self.expr
    }

    /// Returns the null expressions
    pub fn null_expr(&self) -> &[(Arc<dyn PhysicalExpr>, String)] {
        &self.null_expr
    }

    /// Returns the group null masks
    pub fn groups(&self) -> &[Vec<bool>] {
        &self.groups
    }

    /// Returns true if this grouping uses GROUPING SETS, CUBE or ROLLUP.
    pub fn has_grouping_set(&self) -> bool {
        self.has_grouping_set
    }

    /// Returns true if this `PhysicalGroupBy` has no group expressions
    pub fn is_empty(&self) -> bool {
        self.expr.is_empty()
    }

    /// Returns true if this is a "simple" GROUP BY (not using GROUPING SETS/CUBE/ROLLUP).
    /// This determines whether the `__grouping_id` column is included in the output schema.
    pub fn is_single(&self) -> bool {
        !self.has_grouping_set
    }

    /// Returns true if this has no grouping at all (including no GROUPING SETS)
    pub fn is_true_no_grouping(&self) -> bool {
        self.is_empty() && !self.has_grouping_set
    }

    /// Calculate GROUP BY expressions according to input schema.
    pub fn input_exprs(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        self.expr
            .iter()
            .map(|(expr, _alias)| Arc::clone(expr))
            .collect()
    }
}
