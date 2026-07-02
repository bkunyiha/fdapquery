//! `PhysicalSortExpr` and `LexOrdering` — sort-expression types consumed
//! by `SortExec` and equivalence-property reasoning.
//!
//! Strict mirror of DataFusion's `datafusion::physical_expr_common::sort_expr`:
//! - `PhysicalSortExpr { expr: Arc<dyn PhysicalExpr>, options: SortOptions }`
//!   matches `datafusion_physical_expr_common::sort_expr::PhysicalSortExpr`.
//! - `Display` produces `"{expr} {ASC|DESC} [NULLS FIRST|NULLS LAST]"` —
//!   `"a@0 ASC NULLS LAST"` etc. — byte-for-byte equivalent to DataFusion's
//!   `impl Display for datafusion_physical_expr_common::sort_expr::PhysicalSortExpr`,
//!   which delegates to the `to_str(SortOptions)` mapping defined in
//!   the same `datafusion_physical_expr_common::sort_expr` module.
//!
//! ## `LexOrdering`
//! DataFusion's `datafusion_physical_expr_common::sort_expr::LexOrdering` is a non-degenerate,
//! duplicate-free wrapper around `Vec<PhysicalSortExpr>` with an additional
//! `IndexSet` for deduplication. fdapquery's v0.1 mirror uses a plain
//! `Vec<PhysicalSortExpr>` type alias because the equivalence-property
//! machinery that benefits from the deduplicating set (`reorder`,
//! `extract_common_sort_prefix`) is not yet ported — see the deferral note
//! in `fdapquery-physical-plan/src/sorts/sort.rs`. The alias keeps the
//! public type name identical so consumer code reads the same; promoting
//! the alias to a real struct is a follow-up that lands alongside the
//! equivalence-property port.

use crate::expressions::PhysicalExpr;
use arrow::compute::SortOptions;
use std::fmt;
use std::sync::Arc;

/// Represents Sort operation for a column in a `RecordBatch`.
///
/// Strict mirror of DataFusion's `PhysicalSortExpr`
/// (`datafusion_physical_expr_common::sort_expr::PhysicalSortExpr`):
///
/// ```text
/// pub struct PhysicalSortExpr {
///     pub expr: Arc<dyn PhysicalExpr>,
///     pub options: SortOptions,
/// }
/// ```
#[derive(Clone, Debug)]
pub struct PhysicalSortExpr {
    /// Physical expression representing the column to sort.
    pub expr: Arc<dyn PhysicalExpr>,
    /// How the given column should be sorted (ASC/DESC, NULLS FIRST/LAST).
    pub options: SortOptions,
}

impl PhysicalSortExpr {
    /// Create a new `PhysicalSortExpr`. Mirrors DataFusion's
    /// `datafusion_physical_expr_common::sort_expr::PhysicalSortExpr::new`.
    pub fn new(expr: Arc<dyn PhysicalExpr>, options: SortOptions) -> Self {
        Self { expr, options }
    }

    /// Create a new `PhysicalSortExpr` with default `SortOptions`
    /// (ASC, NULLS LAST per arrow's `SortOptions::default()`).
    pub fn new_default(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self::new(expr, SortOptions::default())
    }
}

impl PartialEq for PhysicalSortExpr {
    fn eq(&self, other: &Self) -> bool {
        // Compare options first (cheap) before the trait-object expr eq.
        self.options == other.options && self.expr.to_string() == other.expr.to_string()
    }
}

impl fmt::Display for PhysicalSortExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.expr, to_str(self.options))
    }
}

/// Returns the SQL string representation of the given `SortOptions`.
///
/// Byte-for-byte mirror of DataFusion's `to_str` helper at
/// the private `to_str` helper in
/// `datafusion_physical_expr_common::sort_expr`:
///
/// ```text
/// fn to_str(options: SortOptions) -> &str {
///     match (options.descending, options.nulls_first) {
///         (true, true)   => "DESC",
///         (true, false)  => "DESC NULLS LAST",
///         (false, true)  => "ASC",
///         (false, false) => "ASC NULLS LAST",
///     }
/// }
/// ```
#[inline]
fn to_str(options: SortOptions) -> &'static str {
    match (options.descending, options.nulls_first) {
        (true, true) => "DESC",
        (true, false) => "DESC NULLS LAST",
        (false, true) => "ASC",
        (false, false) => "ASC NULLS LAST",
    }
}

/// A lexicographical ordering: a vector of [`PhysicalSortExpr`] applied in
/// order (first key dominates, second key breaks ties, etc.).
///
/// DataFusion's `LexOrdering` is a non-degenerate, duplicate-free wrapper
/// struct (`Vec<PhysicalSortExpr>` plus an `IndexSet<Arc<dyn PhysicalExpr>>`
/// for deduplication) — see `datafusion_physical_expr_common::sort_expr::LexOrdering`.
/// fdapquery v0.1 uses a plain type alias because the equivalence-property
/// machinery that consumes the deduplicating set (`LexOrdering::push`,
/// `reorder`, `extract_common_sort_prefix`) is not yet ported. The alias
/// keeps the public type name identical so consumer code (`SortExec::new`,
/// `SortExec::expr()`) reads the same as DataFusion. Promoting the alias
/// to a real struct is a follow-up that lands alongside the
/// equivalence-property port.
pub type LexOrdering = Vec<PhysicalSortExpr>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column_expression::Column;

    /// `Display` round-trips through the SQL convention DataFusion uses:
    /// `"{expr} ASC|DESC [NULLS FIRST|NULLS LAST]"`. The `to_str` mapping
    /// is byte-for-byte mirrored from
    /// the private `to_str` helper in
    /// `datafusion_physical_expr_common::sort_expr`.
    #[test]
    fn display_asc_nulls_last() {
        let col: Arc<dyn PhysicalExpr> = Arc::new(Column::new("a", 0));
        let sort = PhysicalSortExpr::new(
            col,
            SortOptions {
                descending: false,
                nulls_first: false,
            },
        );
        assert_eq!(format!("{sort}"), "a@0 ASC NULLS LAST");
    }

    #[test]
    fn display_asc_nulls_first() {
        let col: Arc<dyn PhysicalExpr> = Arc::new(Column::new("a", 0));
        let sort = PhysicalSortExpr::new(
            col,
            SortOptions {
                descending: false,
                nulls_first: true,
            },
        );
        assert_eq!(format!("{sort}"), "a@0 ASC");
    }

    #[test]
    fn display_desc_nulls_first() {
        let col: Arc<dyn PhysicalExpr> = Arc::new(Column::new("b", 1));
        let sort = PhysicalSortExpr::new(
            col,
            SortOptions {
                descending: true,
                nulls_first: true,
            },
        );
        assert_eq!(format!("{sort}"), "b@1 DESC");
    }

    #[test]
    fn display_desc_nulls_last() {
        let col: Arc<dyn PhysicalExpr> = Arc::new(Column::new("b", 1));
        let sort = PhysicalSortExpr::new(
            col,
            SortOptions {
                descending: true,
                nulls_first: false,
            },
        );
        assert_eq!(format!("{sort}"), "b@1 DESC NULLS LAST");
    }

    #[test]
    fn new_default_is_asc_nulls_first() {
        // arrow's `SortOptions::default()` is `descending=false, nulls_first=true`,
        // and DataFusion's `to_str` maps that to bare `"ASC"` (the NULLS FIRST
        // is implicit). Verified against
        // the private `to_str` helper in
        // `datafusion_physical_expr_common::sort_expr`.
        let col: Arc<dyn PhysicalExpr> = Arc::new(Column::new("c", 2));
        let sort = PhysicalSortExpr::new_default(col);
        assert_eq!(format!("{sort}"), "c@2 ASC");
    }
}
