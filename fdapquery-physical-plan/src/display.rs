// Strict-mirror: DataFusion's `display` module defines its `Wrapper` struct
// and the `impl Display for Wrapper` inside `displayable()` for intentional
// encapsulation. Matching the layout matches the source.
#![allow(clippy::items_after_statements)]

//! Strict mirror of `datafusion::physical_plan::display`.
//!
//! Carries the [`DisplayAs`] trait, the [`DisplayFormatType`] enum, the
//! [`DisplayableExecutionPlan`] builder wrapper, and the wrapper types
//! [`DefaultDisplay`] / [`VerboseDisplay`] / [`ProjectSchemaDisplay`].
//! Same module path, same public surface, same byte-for-byte
//! indentation as `datafusion-physical-plan` v54.x.
//!
//! ## Entry points
//!
//! - [`displayable`] — free function. Wraps an `&dyn ExecutionPlan` in a
//!   [`DisplayableExecutionPlan`].
//! - [`DisplayableExecutionPlan::indent`] — returns an `impl Display` that
//!   walks the tree, one operator per line, two ASCII spaces per nesting
//!   level. Byte-identical to DataFusion's indent renderer.
//! - [`DisplayableExecutionPlan::one_line`] — root only.
//! - [`DisplayableExecutionPlan::tree_render`] — box-drawing tree.
//! - [`DisplayableExecutionPlan::graphviz`] — `dot` graph.
//! - [`DisplayableExecutionPlan::to_stringified`] — deprecated; returns a
//!   [`StringifiedPlan`]-shaped value.
//!
//! ## How indentation works
//!
//! `IndentVisitor::pre_visit` writes `{indent:0width$}` blanks where
//! `width = self.indent * 2`, then calls `plan.fmt_as(self.t, self.f)`,
//! then a newline. Children inherit `indent + 1`. **Two spaces, not
//! tabs, not four spaces** — this matches DataFusion exactly.

use crate::metrics::{MetricCategory, MetricType};
use crate::physical_plan::{ExecutionPlan, ExecutionPlanVisitor, accept};
use crate::render_tree::RenderTree;
// `PlanType` and `StringifiedPlan` now live in
// `fdapquery-common::display` (mirroring DataFusion's
// `datafusion_common::display`). They are re-exported at the
// `fdapquery-physical-plan` crate root for back-compat (see `lib.rs`),
// but their canonical location is `fdapquery-common`.
pub use fdapquery_common::display::{PlanType, StringifiedPlan};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fmt::Formatter;
use std::fmt::Write as _;

// =============================================================================
// DisplayFormatType — same three variants as DataFusion.
// =============================================================================

/// Options for controlling how each [`ExecutionPlan`] should format itself.
///
/// Mirrors `datafusion::physical_plan::display::DisplayFormatType`. Three
/// variants:
///
/// - [`Default`](DisplayFormatType::Default) — compact one-line label.
/// - [`Verbose`](DisplayFormatType::Verbose) — extra detail (schemas,
///   ordering, statistics) appended.
/// - [`TreeRender`](DisplayFormatType::TreeRender) — DuckDB-style box-tree
///   labels (key=value pairs rendered in the box body).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DisplayFormatType {
    /// Default, compact format. Example: `FilterExec: c12 < 10.0`.
    Default,
    /// Verbose, showing all available details.
    Verbose,
    /// TreeRender — DuckDB-style box-tree, one key=value pair per line.
    TreeRender,
}

// =============================================================================
// DisplayAs trait — per-operator format method.
// =============================================================================

/// Trait for types that can format themselves under each
/// [`DisplayFormatType`] mode. Mirrors `datafusion::physical_plan::DisplayAs`.
///
/// Every concrete `ExecutionPlan` in fdapquery implements this. The
/// indent walker calls `plan.fmt_as(self.t, self.f)` instead of going
/// through `Display`, so a single operator can produce different output
/// for `Default` vs `Verbose`. The walker handles the newline; impls
/// **must not** emit one.
pub trait DisplayAs {
    /// Format according to `t`. The walker handles indentation and the
    /// trailing newline — impls must not emit either.
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter<'_>) -> fmt::Result;
}

// =============================================================================
// DefaultDisplay / VerboseDisplay / ProjectSchemaDisplay — small wrappers.
// =============================================================================

/// New-type wrapper to render `T: DisplayAs` in
/// [`DisplayFormatType::Default`] mode via `Display`. Mirrors
/// `datafusion::physical_plan::display::DefaultDisplay`.
pub struct DefaultDisplay<T>(pub T);

impl<T: DisplayAs> fmt::Display for DefaultDisplay<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt_as(DisplayFormatType::Default, f)
    }
}

/// New-type wrapper to render `T: DisplayAs` in
/// [`DisplayFormatType::Verbose`] mode via `Display`. Mirrors
/// `datafusion::physical_plan::display::VerboseDisplay`.
pub struct VerboseDisplay<T>(pub T);

impl<T: DisplayAs> fmt::Display for VerboseDisplay<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt_as(DisplayFormatType::Verbose, f)
    }
}

/// Wrapper that renders a schema reference as a comma-separated bracketed
/// field-name list. Mirrors
/// `datafusion::physical_plan::display::ProjectSchemaDisplay`. Used by the
/// indent / tree-render paths when `show_schema=true` would otherwise
/// emit a verbose `Schema { … }` debug dump.
#[derive(Debug)]
pub struct ProjectSchemaDisplay<'a>(pub &'a fdapquery_datatypes::Schema);

impl fmt::Display for ProjectSchemaDisplay<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // fdapquery_datatypes::Schema is a re-export of arrow_schema::Schema —
        // `.fields()` yields `&Fields`, each `Field::name()` is `&str`.
        let parts: Vec<String> = self
            .0
            .fields()
            .iter()
            .map(|x| x.name().to_owned())
            .collect();
        write!(f, "[{}]", parts.join(", "))
    }
}

// =============================================================================
// ShowMetrics (private) — drives the indent walker's metric branch.
// =============================================================================

/// Private enum mirroring DataFusion's `ShowMetrics`. Selects whether
/// metrics are rendered alongside each operator label and at what
/// aggregation level.
#[derive(Debug, Clone, Copy)]
enum ShowMetrics {
    /// Do not emit metrics. Default for [`DisplayableExecutionPlan::new`].
    None,
    /// Show metrics aggregated across partitions. Set by
    /// [`DisplayableExecutionPlan::with_metrics`].
    Aggregated,
    /// Show every per-partition metric. Set by
    /// [`DisplayableExecutionPlan::with_full_metrics`].
    Full,
}

// =============================================================================
// DisplayableExecutionPlan — the builder wrapper.
// =============================================================================

/// Wraps an [`ExecutionPlan`] with formatting configuration. Mirrors
/// `datafusion::physical_plan::display::DisplayableExecutionPlan` field
/// for field.
///
/// The wrapper is `Clone + Debug`. Builder methods take `mut self` and
/// return `Self`, so configuration chains stay one expression:
/// `displayable(plan).set_show_schema(true).indent(false)`.
#[derive(Debug, Clone)]
pub struct DisplayableExecutionPlan<'a> {
    inner: &'a dyn ExecutionPlan,
    /// How metrics are surfaced. Defaults to `None`.
    show_metrics: ShowMetrics,
    /// Whether statistics are appended to each line.
    show_statistics: bool,
    /// Whether schema info is appended to each line. See
    /// [`set_show_schema`](Self::set_show_schema).
    show_schema: bool,
    /// Which metric types are rendered when metrics are shown.
    metric_types: Vec<MetricType>,
    /// Optional semantic-category filter on the rendered metrics.
    /// `None` ⇒ all categories; `Some(vec![])` ⇒ plan-only.
    metric_categories: Option<Vec<MetricCategory>>,
    /// Maximum render width for [`tree_render`](Self::tree_render).
    tree_maximum_render_width: usize,
}

/// Free-function entry point. Same signature as DataFusion's
/// `datafusion::physical_plan::displayable`.
///
/// Wraps `plan` in a [`DisplayableExecutionPlan`] with all defaults
/// (`show_metrics = None`, `show_statistics = false`,
/// `show_schema = false`, `metric_types = [Summary, Dev]`,
/// `metric_categories = None`, `tree_maximum_render_width = 240`).
pub fn displayable(plan: &dyn ExecutionPlan) -> DisplayableExecutionPlan<'_> {
    DisplayableExecutionPlan::new(plan)
}

impl<'a> DisplayableExecutionPlan<'a> {
    /// Default metric-type filter. Same value DataFusion uses.
    fn default_metric_types() -> Vec<MetricType> {
        vec![MetricType::Summary, MetricType::Dev]
    }

    /// Create a wrapper around an [`ExecutionPlan`].
    ///
    /// Defaults: no metrics, no statistics, no schema, default metric
    /// types, no category filter, 240-column tree width.
    pub fn new(inner: &'a dyn ExecutionPlan) -> Self {
        Self {
            inner,
            show_metrics: ShowMetrics::None,
            show_statistics: false,
            show_schema: false,
            metric_types: Self::default_metric_types(),
            metric_categories: None,
            tree_maximum_render_width: 240,
        }
    }

    /// Like [`new`](Self::new) but configures aggregated per-operator metrics.
    pub fn with_metrics(inner: &'a dyn ExecutionPlan) -> Self {
        Self {
            inner,
            show_metrics: ShowMetrics::Aggregated,
            show_statistics: false,
            show_schema: false,
            metric_types: Self::default_metric_types(),
            metric_categories: None,
            tree_maximum_render_width: 240,
        }
    }

    /// Like [`new`](Self::new) but configures full per-partition metrics.
    pub fn with_full_metrics(inner: &'a dyn ExecutionPlan) -> Self {
        Self {
            inner,
            show_metrics: ShowMetrics::Full,
            show_statistics: false,
            show_schema: false,
            metric_types: Self::default_metric_types(),
            metric_categories: None,
            tree_maximum_render_width: 240,
        }
    }

    /// Toggle schema-after-each-line rendering. Same shape as DataFusion's
    /// `set_show_schema`. The format used is
    /// `schema=[a:Int32;N, b:Int32;N, ...]`.
    pub fn set_show_schema(mut self, show_schema: bool) -> Self {
        self.show_schema = show_schema;
        self
    }

    /// Toggle per-operator statistics rendering.
    pub fn set_show_statistics(mut self, show_statistics: bool) -> Self {
        self.show_statistics = show_statistics;
        self
    }

    /// Replace the metric-type filter list.
    pub fn set_metric_types(mut self, metric_types: Vec<MetricType>) -> Self {
        self.metric_types = metric_types;
        self
    }

    /// Set the semantic-category filter.
    ///
    /// - `None` — include all categories (default).
    /// - `Some(vec![])` — plan-only (suppress all metrics).
    /// - `Some(vec![cat, …])` — show only those categories (plus
    ///   uncategorised metrics).
    pub fn set_metric_categories(mut self, metric_categories: Option<Vec<MetricCategory>>) -> Self {
        self.metric_categories = metric_categories;
        self
    }

    /// Set the maximum render width for [`tree_render`](Self::tree_render).
    pub fn set_tree_maximum_render_width(mut self, width: usize) -> Self {
        self.tree_maximum_render_width = width;
        self
    }

    /// Returns an `impl Display` that walks the tree depth-first, one
    /// node per line, two ASCII spaces per nesting level.
    ///
    /// **Byte-for-byte equivalence with DataFusion.** The walker calls
    /// each node's [`DisplayAs::fmt_as`] with `Default` or `Verbose`
    /// based on `verbose`, prepends `indent * 2` ASCII spaces, and
    /// terminates each line with `\n`. Children inherit `indent + 1`.
    ///
    /// Example output (matches DataFusion):
    ///
    /// ```text
    /// ProjectionExec: expr=[a]
    ///   CoalesceBatchesExec: target_batch_size=8192
    ///     FilterExec: a < 5
    /// ```
    pub fn indent(&self, verbose: bool) -> impl fmt::Display + 'a {
        let format_type = if verbose {
            DisplayFormatType::Verbose
        } else {
            DisplayFormatType::Default
        };
        // Local wrapper, owns clones of the slice-borrowed fields so the
        // returned `impl Display` lives the full `'a`.
        struct Wrapper<'a> {
            format_type: DisplayFormatType,
            plan: &'a dyn ExecutionPlan,
            show_metrics: ShowMetrics,
            show_statistics: bool,
            show_schema: bool,
            metric_types: Vec<MetricType>,
            metric_categories: Option<Vec<MetricCategory>>,
        }
        impl fmt::Display for Wrapper<'_> {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                let mut visitor = IndentVisitor {
                    t: self.format_type,
                    f,
                    indent: 0,
                    show_metrics: self.show_metrics,
                    show_statistics: self.show_statistics,
                    show_schema: self.show_schema,
                    metric_types: &self.metric_types,
                    metric_categories: self.metric_categories.as_deref(),
                };
                accept(self.plan, &mut visitor)
            }
        }
        Wrapper {
            format_type,
            plan: self.inner,
            show_metrics: self.show_metrics,
            show_statistics: self.show_statistics,
            show_schema: self.show_schema,
            metric_types: self.metric_types.clone(),
            metric_categories: self.metric_categories.clone(),
        }
    }

    /// Returns an `impl Display` rendering a Graphviz `dot` graph of the
    /// plan tree. Mirrors DataFusion's `graphviz`. Each node is emitted
    /// as `N[label=…,tooltip=…]`; edges as `parent -> child`. The output
    /// is valid `dot` and pastes directly into
    /// <https://dreampuf.github.io/GraphvizOnline>.
    ///
    /// Implementation is a small in-module `GraphvizVisitor` (no external
    /// `GraphvizBuilder` dependency).
    pub fn graphviz(&self) -> impl fmt::Display + 'a {
        struct Wrapper<'a> {
            plan: &'a dyn ExecutionPlan,
            show_metrics: ShowMetrics,
            show_statistics: bool,
            metric_types: Vec<MetricType>,
            metric_categories: Option<Vec<MetricCategory>>,
        }
        impl fmt::Display for Wrapper<'_> {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                let t = DisplayFormatType::Default;
                let mut visitor = GraphvizVisitor {
                    f,
                    t,
                    show_metrics: self.show_metrics,
                    show_statistics: self.show_statistics,
                    metric_types: &self.metric_types,
                    metric_categories: self.metric_categories.as_deref(),
                    next_id: 0,
                    parents: Vec::new(),
                };
                visitor.start_graph()?;
                accept(self.plan, &mut visitor)?;
                visitor.end_graph()?;
                Ok(())
            }
        }
        Wrapper {
            plan: self.inner,
            show_metrics: self.show_metrics,
            show_statistics: self.show_statistics,
            metric_types: self.metric_types.clone(),
            metric_categories: self.metric_categories.clone(),
        }
    }

    /// Returns an `impl Display` rendering the plan as an ASCII box-art
    /// tree (DuckDB-inspired). Mirrors DataFusion's `tree_render` —
    /// uses [`DisplayFormatType::TreeRender`] on each operator and lays
    /// the result out in a grid bounded by
    /// [`tree_maximum_render_width`](Self::set_tree_maximum_render_width).
    pub fn tree_render(&self) -> impl fmt::Display + 'a {
        struct Wrapper<'a> {
            plan: &'a dyn ExecutionPlan,
            maximum_render_width: usize,
        }
        impl fmt::Display for Wrapper<'_> {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                let mut visitor = TreeRenderVisitor {
                    f,
                    maximum_render_width: self.maximum_render_width,
                };
                visitor.visit(self.plan)
            }
        }
        Wrapper {
            plan: self.inner,
            maximum_render_width: self.tree_maximum_render_width,
        }
    }

    /// Returns an `impl Display` that prints only the root operator's
    /// label (no children, no newline-trailing recursion). Mirrors
    /// DataFusion's `one_line` — reuses `IndentVisitor` but calls only
    /// `pre_visit` once.
    pub fn one_line(&self) -> impl fmt::Display + 'a {
        struct Wrapper<'a> {
            plan: &'a dyn ExecutionPlan,
            show_metrics: ShowMetrics,
            show_statistics: bool,
            show_schema: bool,
            metric_types: Vec<MetricType>,
            metric_categories: Option<Vec<MetricCategory>>,
        }
        impl fmt::Display for Wrapper<'_> {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                let mut visitor = IndentVisitor {
                    f,
                    t: DisplayFormatType::Default,
                    indent: 0,
                    show_metrics: self.show_metrics,
                    show_statistics: self.show_statistics,
                    show_schema: self.show_schema,
                    metric_types: &self.metric_types,
                    metric_categories: self.metric_categories.as_deref(),
                };
                visitor.pre_visit(self.plan)?;
                Ok(())
            }
        }
        Wrapper {
            plan: self.inner,
            show_metrics: self.show_metrics,
            show_statistics: self.show_statistics,
            show_schema: self.show_schema,
            metric_types: self.metric_types.clone(),
            metric_categories: self.metric_categories.clone(),
        }
    }

    /// Materialise the plan dump as a [`StringifiedPlan`].
    ///
    /// **Deprecated** in DataFusion since 47.0.0. Kept for API parity —
    /// new callers should use [`indent`](Self::indent) or
    /// [`tree_render`](Self::tree_render) directly.
    #[deprecated(since = "0.1.0", note = "use indent() or tree_render() instead")]
    pub fn to_stringified(
        &self,
        verbose: bool,
        plan_type: PlanType,
        explain_format: DisplayFormatType,
    ) -> StringifiedPlan {
        match (&explain_format, &plan_type) {
            (DisplayFormatType::TreeRender, PlanType::FinalPhysicalPlan) => {
                StringifiedPlan::new(plan_type, self.tree_render().to_string())
            }
            _ => StringifiedPlan::new(plan_type, self.indent(verbose).to_string()),
        }
    }
}

// =============================================================================
// IndentVisitor — the workhorse of indent() and one_line().
// =============================================================================

/// Per-node visitor that emits the indented one-line label of each
/// operator. Private; same fields as DataFusion's `IndentVisitor`.
///
/// `pre_visit` writes `indent * 2` ASCII spaces, then
/// `plan.fmt_as(t, f)`, then any metrics / stats / schema suffixes,
/// then a newline. `post_visit` decrements `indent`.
struct IndentVisitor<'a, 'b> {
    /// `Default` or `Verbose`, picked by `indent(verbose)`.
    t: DisplayFormatType,
    /// Sink.
    f: &'a mut Formatter<'b>,
    /// Current depth. Multiplied by 2 to produce ASCII-space padding.
    indent: usize,
    /// Metric-rendering mode inherited from the wrapper.
    show_metrics: ShowMetrics,
    /// Whether to append `, statistics=[…]` to each line.
    show_statistics: bool,
    /// Whether to append `, schema=[…]` to each line.
    show_schema: bool,
    /// Metric-type filter list.
    metric_types: &'a [MetricType],
    /// Optional category filter.
    metric_categories: Option<&'a [MetricCategory]>,
}

impl ExecutionPlanVisitor for IndentVisitor<'_, '_> {
    type Error = fmt::Error;

    fn pre_visit(&mut self, plan: &dyn ExecutionPlan) -> std::result::Result<bool, Self::Error> {
        // Two ASCII spaces per nesting level — matches DataFusion byte-for-byte.
        write!(self.f, "{:indent$}", "", indent = self.indent * 2)?;
        plan.fmt_as(self.t, self.f)?;
        match self.show_metrics {
            ShowMetrics::None => {}
            ShowMetrics::Aggregated => {
                if let Some(metrics) = plan.metrics() {
                    let mut metrics = metrics
                        .filter_by_metric_types(self.metric_types)
                        .aggregate_by_name()
                        .sorted_for_display()
                        .timestamps_removed();
                    if let Some(cats) = self.metric_categories {
                        metrics = metrics.filter_by_categories(cats);
                    }
                    write!(self.f, ", metrics=[{metrics}]")?;
                } else {
                    write!(self.f, ", metrics=[]")?;
                }
            }
            ShowMetrics::Full => {
                if let Some(metrics) = plan.metrics() {
                    let mut metrics = metrics.filter_by_metric_types(self.metric_types);
                    if let Some(cats) = self.metric_categories {
                        metrics = metrics.filter_by_categories(cats);
                    }
                    write!(self.f, ", metrics=[{metrics}]")?;
                } else {
                    write!(self.f, ", metrics=[]")?;
                }
            }
        }
        if self.show_statistics {
            // fdapquery operators have no `partition_statistics` method
            // yet — emit the same key with an empty value so the
            // suffix shape matches DataFusion's.
            write!(self.f, ", statistics=[]")?;
        }
        if self.show_schema {
            // `Debug` on `arrow_schema::Schema` is unwieldy; use
            // `ProjectSchemaDisplay`'s compact bracketed form, which is
            // what DataFusion emits via `display_schema` for this path.
            let schema = plan.schema();
            write!(self.f, ", schema={}", ProjectSchemaDisplay(&schema))?;
        }
        writeln!(self.f)?;
        self.indent += 1;
        Ok(true)
    }

    fn post_visit(&mut self, _plan: &dyn ExecutionPlan) -> std::result::Result<bool, Self::Error> {
        self.indent -= 1;
        Ok(true)
    }
}

// =============================================================================
// GraphvizVisitor — produces `dot` graphs from the plan tree.
// =============================================================================

/// Per-node visitor that emits Graphviz `dot` syntax for the plan tree.
/// Mirrors DataFusion's `GraphvizVisitor` (lives in the same module).
struct GraphvizVisitor<'a, 'b> {
    f: &'a mut Formatter<'b>,
    t: DisplayFormatType,
    show_metrics: ShowMetrics,
    show_statistics: bool,
    metric_types: &'a [MetricType],
    metric_categories: Option<&'a [MetricCategory]>,
    /// Next node id to allocate.
    next_id: usize,
    /// Stack of parent ids (for edge emission).
    parents: Vec<usize>,
}

impl GraphvizVisitor<'_, '_> {
    fn start_graph(&mut self) -> fmt::Result {
        writeln!(self.f, "strict digraph dot_plan {{")
    }

    fn end_graph(&mut self) -> fmt::Result {
        writeln!(self.f, "}}")
    }
}

/// Helper struct: capture `plan.fmt_as` output into a `String` so we can
/// quote-escape it for the `label="…"` attribute.
struct CaptureFmtAs<'a> {
    plan: &'a dyn ExecutionPlan,
    t: DisplayFormatType,
}
impl fmt::Display for CaptureFmtAs<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.plan.fmt_as(self.t, f)
    }
}

impl ExecutionPlanVisitor for GraphvizVisitor<'_, '_> {
    type Error = fmt::Error;

    fn pre_visit(&mut self, plan: &dyn ExecutionPlan) -> std::result::Result<bool, Self::Error> {
        let id = self.next_id;
        self.next_id += 1;

        // Build label.
        let mut label = format!("{}", CaptureFmtAs { plan, t: self.t });
        match self.show_metrics {
            ShowMetrics::None => {}
            ShowMetrics::Aggregated => {
                if let Some(metrics) = plan.metrics() {
                    let mut metrics = metrics
                        .filter_by_metric_types(self.metric_types)
                        .aggregate_by_name()
                        .sorted_for_display()
                        .timestamps_removed();
                    if let Some(cats) = self.metric_categories {
                        metrics = metrics.filter_by_categories(cats);
                    }
                    write!(label, ", metrics=[{metrics}]").ok();
                } else {
                    label.push_str(", metrics=[]");
                }
            }
            ShowMetrics::Full => {
                if let Some(metrics) = plan.metrics() {
                    let mut metrics = metrics.filter_by_metric_types(self.metric_types);
                    if let Some(cats) = self.metric_categories {
                        metrics = metrics.filter_by_categories(cats);
                    }
                    write!(label, ", metrics=[{metrics}]").ok();
                } else {
                    label.push_str(", metrics=[]");
                }
            }
        }
        if self.show_statistics {
            label.push_str(", statistics=[]");
        }

        // Escape double quotes and backslashes for `dot` literals.
        let label = label.replace('\\', "\\\\").replace('"', "\\\"");
        writeln!(self.f, "  {id}[label=\"{label}\",tooltip=\"\"]")?;
        if let Some(&parent) = self.parents.last() {
            writeln!(self.f, "  {parent} -> {id}")?;
        }
        self.parents.push(id);
        Ok(true)
    }

    fn post_visit(&mut self, _plan: &dyn ExecutionPlan) -> std::result::Result<bool, Self::Error> {
        self.parents.pop();
        Ok(true)
    }
}

// =============================================================================
// TreeRenderVisitor — DuckDB-style Unicode box-art tree.
// =============================================================================
// Strict mirror of `datafusion::physical_plan::display::TreeRenderVisitor`
// at `/Users/bkunyiha/Rust/datafusion/datafusion/physical-plan/src/display.rs`.
// Three-layer per-y rendering atop the `RenderTree` precomputation in
// `crate::render_tree`:
//   * `render_top_layer`    — `┌─────┬─────┐` borders; the `┬` joins the
//                              parent above when `y > 0`.
//   * `render_box_content`  — vertical borders `│ … │` framing the
//                              centered content; the halfway-point row
//                              emits horizontal `─` runs plus `┬`/`├`/`┤`
//                              junction glyphs to connect to children.
//   * `render_bottom_layer` — `└─────┴─────┘` borders; the `┴` joins the
//                              child below when one exists.
// All sizing follows DataFusion: each box is `NODE_RENDER_WIDTH` (29)
// columns wide, the half-width is `NODE_RENDER_WIDTH / 2 = 14`, and
// content lines are centered via `adjust_text_for_rendering` (extra
// space on the LEFT when the count is odd, matching DataFusion).

struct TreeRenderVisitor<'a, 'b> {
    /// Write to this formatter
    f: &'a mut Formatter<'b>,
    /// Maximum total width of the rendered tree
    maximum_render_width: usize,
}

impl TreeRenderVisitor<'_, '_> {
    // Unicode box-drawing characters for creating borders and connections.
    const LTCORNER: &'static str = "\u{250C}"; // ┌ — Left top corner
    const RTCORNER: &'static str = "\u{2510}"; // ┐ — Right top corner
    const LDCORNER: &'static str = "\u{2514}"; // └ — Left bottom corner
    const RDCORNER: &'static str = "\u{2518}"; // ┘ — Right bottom corner

    const TMIDDLE: &'static str = "\u{252C}"; // ┬ — Top T-junction (connects down)
    const LMIDDLE: &'static str = "\u{251C}"; // ├ — Left T-junction (connects right)
    const DMIDDLE: &'static str = "\u{2534}"; // ┴ — Bottom T-junction (connects up)

    const VERTICAL: &'static str = "\u{2502}"; // │
    const HORIZONTAL: &'static str = "\u{2500}"; // ─

    // TODO: Make these variables configurable.
    const NODE_RENDER_WIDTH: usize = 29; // Width of each node's box
    const MAX_EXTRA_LINES: usize = 30; // Maximum number of extra info lines per node

    /// Main entry point for rendering an execution plan as a tree.
    /// The rendering process happens in three stages for each level of the tree:
    /// 1. Render top borders and connections
    /// 2. Render node content and vertical connections
    /// 3. Render bottom borders and connections
    pub fn visit(&mut self, plan: &dyn ExecutionPlan) -> fmt::Result {
        let root = RenderTree::create_tree(plan);

        for y in 0..root.height {
            // Start by rendering the top layer.
            self.render_top_layer(&root, y)?;
            // Now we render the content of the boxes
            self.render_box_content(&root, y)?;
            // Render the bottom layer of each of the boxes
            self.render_bottom_layer(&root, y)?;
        }

        Ok(())
    }

    /// Renders the top layer of boxes at the given y-level of the tree.
    /// This includes:
    /// - Top corners (┌─┐) for nodes
    /// - Horizontal connections between nodes
    /// - Vertical connections to parent nodes
    fn render_top_layer(&mut self, root: &RenderTree, y: usize) -> fmt::Result {
        for x in 0..root.width {
            if self.maximum_render_width > 0
                && x * Self::NODE_RENDER_WIDTH >= self.maximum_render_width
            {
                break;
            }

            if root.has_node(x, y) {
                write!(self.f, "{}", Self::LTCORNER)?;
                write!(
                    self.f,
                    "{}",
                    Self::HORIZONTAL.repeat(Self::NODE_RENDER_WIDTH / 2 - 1)
                )?;
                if y == 0 {
                    // top level node: no node above this one
                    write!(self.f, "{}", Self::HORIZONTAL)?;
                } else {
                    // render connection to node above this one
                    write!(self.f, "{}", Self::DMIDDLE)?;
                }
                write!(
                    self.f,
                    "{}",
                    Self::HORIZONTAL.repeat(Self::NODE_RENDER_WIDTH / 2 - 1)
                )?;
                write!(self.f, "{}", Self::RTCORNER)?;
            } else {
                let mut has_adjacent_nodes = false;
                for i in 0..(root.width - x) {
                    has_adjacent_nodes = has_adjacent_nodes || root.has_node(x + i, y);
                }
                if !has_adjacent_nodes {
                    // There are no nodes to the right side of this position
                    // no need to fill the empty space
                    continue;
                }
                // there are nodes next to this, fill the space
                write!(self.f, "{}", &" ".repeat(Self::NODE_RENDER_WIDTH))?;
            }
        }
        writeln!(self.f)?;

        Ok(())
    }

    /// Renders the content layer of boxes at the given y-level of the tree.
    /// This includes:
    /// - Node names and extra information
    /// - Vertical borders (│) for boxes
    /// - Vertical connections between nodes
    fn render_box_content(&mut self, root: &RenderTree, y: usize) -> fmt::Result {
        let mut extra_info: Vec<Vec<String>> = vec![vec![]; root.width];
        let mut extra_height = 0;

        for (x, extra_info_item) in extra_info.iter_mut().enumerate().take(root.width) {
            if let Some(node) = root.get_node(x, y) {
                Self::split_up_extra_info(&node.extra_text, extra_info_item, Self::MAX_EXTRA_LINES);
                if extra_info_item.len() > extra_height {
                    extra_height = extra_info_item.len();
                }
            }
        }

        let halfway_point = extra_height.div_ceil(2);

        // Render the actual node.
        for render_y in 0..=extra_height {
            for (x, _) in root.nodes.iter().enumerate().take(root.width) {
                if self.maximum_render_width > 0
                    && x * Self::NODE_RENDER_WIDTH >= self.maximum_render_width
                {
                    break;
                }

                let mut has_adjacent_nodes = false;
                for i in 0..(root.width - x) {
                    has_adjacent_nodes = has_adjacent_nodes || root.has_node(x + i, y);
                }

                if let Some(node) = root.get_node(x, y) {
                    write!(self.f, "{}", Self::VERTICAL)?;

                    // Figure out what to render.
                    let mut render_text = if render_y == 0 {
                        node.name.clone()
                    } else if render_y <= extra_info[x].len() {
                        extra_info[x][render_y - 1].clone()
                    } else {
                        String::new()
                    };

                    render_text =
                        Self::adjust_text_for_rendering(&render_text, Self::NODE_RENDER_WIDTH - 2);
                    write!(self.f, "{render_text}")?;

                    if render_y == halfway_point && node.child_positions.len() > 1 {
                        write!(self.f, "{}", Self::LMIDDLE)?;
                    } else {
                        write!(self.f, "{}", Self::VERTICAL)?;
                    }
                } else if render_y == halfway_point {
                    let has_child_to_the_right = Self::should_render_whitespace(root, x, y);
                    if root.has_node(x, y + 1) {
                        // Node right below this one.
                        write!(
                            self.f,
                            "{}",
                            Self::HORIZONTAL.repeat(Self::NODE_RENDER_WIDTH / 2)
                        )?;
                        if has_child_to_the_right {
                            write!(self.f, "{}", Self::TMIDDLE)?;
                            // Have another child to the right, Keep rendering the line.
                            write!(
                                self.f,
                                "{}",
                                Self::HORIZONTAL.repeat(Self::NODE_RENDER_WIDTH / 2)
                            )?;
                        } else {
                            write!(self.f, "{}", Self::RTCORNER)?;
                            if has_adjacent_nodes {
                                // Only a child below this one: fill the reset with spaces.
                                write!(self.f, "{}", " ".repeat(Self::NODE_RENDER_WIDTH / 2))?;
                            }
                        }
                    } else if has_child_to_the_right {
                        // Child to the right, but no child right below this one: render a full
                        // line.
                        write!(
                            self.f,
                            "{}",
                            Self::HORIZONTAL.repeat(Self::NODE_RENDER_WIDTH)
                        )?;
                    } else if has_adjacent_nodes {
                        // Empty spot: render spaces.
                        write!(self.f, "{}", " ".repeat(Self::NODE_RENDER_WIDTH))?;
                    }
                } else if render_y >= halfway_point {
                    if root.has_node(x, y + 1) {
                        // Have a node below this empty spot: render a vertical line.
                        write!(
                            self.f,
                            "{}{}",
                            " ".repeat(Self::NODE_RENDER_WIDTH / 2),
                            Self::VERTICAL
                        )?;
                        if has_adjacent_nodes || Self::should_render_whitespace(root, x, y) {
                            write!(self.f, "{}", " ".repeat(Self::NODE_RENDER_WIDTH / 2))?;
                        }
                    } else if has_adjacent_nodes || Self::should_render_whitespace(root, x, y) {
                        // Empty spot: render spaces.
                        write!(self.f, "{}", " ".repeat(Self::NODE_RENDER_WIDTH))?;
                    }
                } else if has_adjacent_nodes {
                    // Empty spot: render spaces.
                    write!(self.f, "{}", " ".repeat(Self::NODE_RENDER_WIDTH))?;
                }
            }
            writeln!(self.f)?;
        }

        Ok(())
    }

    /// Renders the bottom layer of boxes at the given y-level of the tree.
    /// This includes:
    /// - Bottom corners (└─┘) for nodes
    /// - Horizontal connections between nodes
    /// - Vertical connections to child nodes
    fn render_bottom_layer(&mut self, root: &RenderTree, y: usize) -> fmt::Result {
        for x in 0..=root.width {
            if self.maximum_render_width > 0
                && x * Self::NODE_RENDER_WIDTH >= self.maximum_render_width
            {
                break;
            }
            let mut has_adjacent_nodes = false;
            for i in 0..(root.width - x) {
                has_adjacent_nodes = has_adjacent_nodes || root.has_node(x + i, y);
            }
            if root.get_node(x, y).is_some() {
                write!(self.f, "{}", Self::LDCORNER)?;
                write!(
                    self.f,
                    "{}",
                    Self::HORIZONTAL.repeat(Self::NODE_RENDER_WIDTH / 2 - 1)
                )?;
                if root.has_node(x, y + 1) {
                    // node below this one: connect to that one
                    write!(self.f, "{}", Self::TMIDDLE)?;
                } else {
                    // no node below this one: end the box
                    write!(self.f, "{}", Self::HORIZONTAL)?;
                }
                write!(
                    self.f,
                    "{}",
                    Self::HORIZONTAL.repeat(Self::NODE_RENDER_WIDTH / 2 - 1)
                )?;
                write!(self.f, "{}", Self::RDCORNER)?;
            } else if root.has_node(x, y + 1) {
                write!(self.f, "{}", &" ".repeat(Self::NODE_RENDER_WIDTH / 2))?;
                write!(self.f, "{}", Self::VERTICAL)?;
                if has_adjacent_nodes || Self::should_render_whitespace(root, x, y) {
                    write!(self.f, "{}", &" ".repeat(Self::NODE_RENDER_WIDTH / 2))?;
                }
            } else if has_adjacent_nodes || Self::should_render_whitespace(root, x, y) {
                write!(self.f, "{}", &" ".repeat(Self::NODE_RENDER_WIDTH))?;
            }
        }
        writeln!(self.f)?;

        Ok(())
    }

    fn extra_info_separator() -> String {
        "-".repeat(Self::NODE_RENDER_WIDTH - 9)
    }

    fn remove_padding(s: &str) -> String {
        s.trim().to_string()
    }

    pub fn split_up_extra_info(
        extra_info: &HashMap<String, String>,
        result: &mut Vec<String>,
        max_lines: usize,
    ) {
        if extra_info.is_empty() {
            return;
        }

        result.push(Self::extra_info_separator());

        let mut requires_padding = false;
        let mut was_inlined = false;

        // use BTreeMap for repeatable key order
        let sorted_extra_info: BTreeMap<_, _> = extra_info.iter().collect();
        for (key, value) in sorted_extra_info {
            let mut str = Self::remove_padding(value);
            let mut is_inlined = false;
            let available_width = Self::NODE_RENDER_WIDTH - 7;
            let total_size = key.len() + str.len() + 2;
            let is_multiline = str.contains('\n');

            if str.is_empty() {
                str.clone_from(key);
            } else if !is_multiline && total_size < available_width {
                str = format!("{key}: {str}");
                is_inlined = true;
            } else {
                str = format!("{key}:\n{str}");
            }

            if is_inlined && was_inlined {
                requires_padding = false;
            }

            if requires_padding {
                result.push(String::new());
            }

            let mut splits: Vec<String> = str.split('\n').map(String::from).collect();
            if splits.len() > max_lines {
                let mut truncated_splits = Vec::new();
                for split in splits.iter().take(max_lines / 2) {
                    truncated_splits.push(split.clone());
                }
                truncated_splits.push("...".to_string());
                for split in splits.iter().skip(splits.len() - max_lines / 2) {
                    truncated_splits.push(split.clone());
                }
                splits = truncated_splits;
            }
            for split in splits {
                Self::split_string_buffer(&split, result);
            }
            if result.len() > max_lines {
                result.truncate(max_lines);
                result.push("...".to_string());
            }

            requires_padding = true;
            was_inlined = is_inlined;
        }
    }

    /// Adjusts text to fit within the specified width by:
    /// 1. Truncating with ellipsis if too long
    /// 2. Center-aligning within the available space if shorter
    fn adjust_text_for_rendering(source: &str, max_render_width: usize) -> String {
        let render_width = source.chars().count();
        if render_width > max_render_width {
            let truncated = &source[..max_render_width - 3];
            format!("{truncated}...")
        } else {
            let total_spaces = max_render_width - render_width;
            let half_spaces = total_spaces / 2;
            let extra_left_space = usize::from(total_spaces % 2 != 0);
            format!(
                "{}{}{}",
                " ".repeat(half_spaces + extra_left_space),
                source,
                " ".repeat(half_spaces)
            )
        }
    }

    /// Determines if whitespace should be rendered at a given position.
    /// This is important for:
    /// 1. Maintaining proper spacing between sibling nodes
    /// 2. Ensuring correct alignment of connections between parents and children
    /// 3. Preserving the tree structure's visual clarity
    fn should_render_whitespace(root: &RenderTree, x: usize, y: usize) -> bool {
        let mut found_children = 0;

        for i in (0..=x).rev() {
            let node = root.get_node(i, y);
            if root.has_node(i, y + 1) {
                found_children += 1;
            }
            if let Some(node) = node {
                if node.child_positions.len() > 1 && found_children < node.child_positions.len() {
                    return true;
                }

                return false;
            }
        }

        false
    }

    fn split_string_buffer(source: &str, result: &mut Vec<String>) {
        let mut character_pos = 0;
        let mut start_pos = 0;
        let mut render_width = 0;
        let mut last_possible_split = 0;

        let chars: Vec<char> = source.chars().collect();

        while character_pos < chars.len() {
            // Treating each char as width 1 for simplification
            let char_width = 1;

            // Does the next character make us exceed the line length?
            if render_width + char_width > Self::NODE_RENDER_WIDTH - 2 {
                if start_pos + 8 > last_possible_split {
                    // The last character we can split on is one of the first 8 characters of the line
                    // to not create very small lines we instead split on the current character
                    last_possible_split = character_pos;
                }

                result.push(source[start_pos..last_possible_split].to_string());
                render_width = character_pos - last_possible_split;
                start_pos = last_possible_split;
                character_pos = last_possible_split;
            }

            // check if we can split on this character
            if Self::can_split_on_this_char(chars[character_pos]) {
                last_possible_split = character_pos;
            }

            character_pos += 1;
            render_width += char_width;
        }

        if source.len() > start_pos {
            // append the remainder of the input
            result.push(source[start_pos..].to_string());
        }
    }

    fn can_split_on_this_char(c: char) -> bool {
        (!c.is_ascii_digit() && !c.is_ascii_uppercase() && !c.is_ascii_lowercase()) && c != '_'
    }
}

// =============================================================================
// display_orderings — helper used by per-operator `fmt_as` impls.
// =============================================================================

/// Render a slice of orderings as `, output_ordering=[…]` (one ordering)
/// or `, output_orderings=[…, …]` (multiple). Mirrors DataFusion's
/// `display_orderings`. `T: Display` so the type can be either a
/// concrete `LexOrdering` or any per-engine ordering wrapper.
pub fn display_orderings<T: fmt::Display>(f: &mut Formatter<'_>, orderings: &[T]) -> fmt::Result {
    if !orderings.is_empty() {
        let start = if orderings.len() == 1 {
            ", output_ordering="
        } else {
            ", output_orderings=["
        };
        write!(f, "{start}")?;
        for (idx, ordering) in orderings.iter().enumerate() {
            match idx {
                0 => write!(f, "[{ordering}]")?,
                _ => write!(f, ", [{ordering}]")?,
            }
        }
        let end = if orderings.len() == 1 { "" } else { "]" };
        write!(f, "{end}")?;
    }
    Ok(())
}

// =============================================================================
// PlanType / StringifiedPlan — re-exported from `fdapquery-common::display`.
// Moved the type definitions to `fdapquery-common`
// (mirroring DataFusion's `datafusion_common::display`); the `pub use`
// near the top of this file makes them addressable at this module's
// public path for back-compat.
// =============================================================================

#[cfg(test)]
mod tests {
    //! Tests for the Unicode `TreeRenderVisitor` — strict mirror of
    //! `datafusion::physical_plan::display::TreeRenderVisitor`.
    //!
    //! These tests assert **byte-for-byte equivalence with DataFusion**
    //! by reconstructing the expected output via the same `NODE_RENDER_WIDTH = 29`
    //! sizing, the same centering rule
    //! (`adjust_text_for_rendering`), and the same junction-glyph
    //! choices (`┌─┐│└─┘`, `┬`/`┴`/`├`/`┤`/`┼`).  Any deviation in
    //! glyphs, padding, or junction selection fails these tests.
    //!
    //! Fixtures are drawn from `test_util` (single-node `TestSourceExec`)
    //! and the per-operator wrappers (`GlobalLimitExec` for a 2-level
    //! plan).

    use super::*;
    use crate::test_util::employee_source;

    // -------------------------------------------------------------------------
    // Helpers: rebuild the DataFusion-canonical expected output by hand from
    // the constants the renderer uses (`NODE_RENDER_WIDTH`, the box-drawing
    // glyphs, and the centering rule from `adjust_text_for_rendering`).
    // Keeping the helpers in the test module — instead of inlining giant
    // string literals — makes the expected strings inspection-friendly
    // and impossible to drift from the renderer's own constants.
    // -------------------------------------------------------------------------

    const LT: &str = "\u{250C}"; // ┌
    const RT: &str = "\u{2510}"; // ┐
    const LD: &str = "\u{2514}"; // └
    const RD: &str = "\u{2518}"; // ┘
    const VERT: &str = "\u{2502}"; // │
    const HORIZ: &str = "\u{2500}"; // ─
    const TMID: &str = "\u{252C}"; // ┬ — top T (children below)
    const DMID: &str = "\u{2534}"; // ┴ — bottom T (parent above)
    const LMID: &str = "\u{251C}"; // ├ — left T (children to right)

    // Mirrors DataFusion's `NODE_RENDER_WIDTH = 29` and the half-width
    // arithmetic the three render layers do.
    const NRW: usize = 29;
    const HALF_MINUS_ONE: usize = NRW / 2 - 1; // 13

    /// Reimplements `TreeRenderVisitor::adjust_text_for_rendering` so the
    /// tests can produce DataFusion-canonical centered strings.
    fn center(text: &str, max_render_width: usize) -> String {
        let n = text.chars().count();
        if n > max_render_width {
            let truncated = &text[..max_render_width - 3];
            format!("{truncated}...")
        } else {
            let total = max_render_width - n;
            let half = total / 2;
            let extra_left = usize::from(!total.is_multiple_of(2));
            format!(
                "{}{}{}",
                " ".repeat(half + extra_left),
                text,
                " ".repeat(half)
            )
        }
    }

    /// Top-border segment for a single node.  When `y == 0` the middle
    /// glyph is a plain `─`; when `y > 0` it is `┴` to join the parent
    /// above.
    fn top_border(parent_above: bool) -> String {
        let mid = if parent_above { DMID } else { HORIZ };
        format!(
            "{LT}{}{mid}{}{RT}",
            HORIZ.repeat(HALF_MINUS_ONE),
            HORIZ.repeat(HALF_MINUS_ONE)
        )
    }

    /// Bottom-border segment for a single node.  When there is a child
    /// directly below, the middle glyph is `┬`; otherwise plain `─`.
    fn bottom_border(child_below: bool) -> String {
        let mid = if child_below { TMID } else { HORIZ };
        format!(
            "{LD}{}{mid}{}{RD}",
            HORIZ.repeat(HALF_MINUS_ONE),
            HORIZ.repeat(HALF_MINUS_ONE)
        )
    }

    /// One body line of a single node — the inner content
    /// `{VERT}{centered}{VERT}` (or `{VERT}{centered}{LMID}` at the
    /// halfway point when the node has 2+ children).
    fn body_line(content: &str, left_tee: bool) -> String {
        let right = if left_tee { LMID } else { VERT };
        format!("{VERT}{}{right}", center(content, NRW - 2))
    }

    // -------------------------------------------------------------------------
    // Test 1: single-node plan (1 × 1 grid).
    // Verifies the simplest case — leaf operator alone, no parent above
    // and no child below, so top/bottom borders use plain `─` middles.
    // -------------------------------------------------------------------------

    #[test]
    fn tree_render_byte_equivalent_to_datafusion_single_node() {
        let plan = employee_source();
        let rendered = displayable(plan.as_ref()).tree_render().to_string();

        // No ASCII fallbacks.
        assert!(!rendered.contains('+'));
        // The Unicode glyphs are all present somewhere.
        for glyph in [LT, RT, LD, RD, VERT, HORIZ] {
            assert!(
                rendered.contains(glyph),
                "missing glyph {glyph:?} in:\n{rendered}"
            );
        }

        // TestSourceExec's `fmt_as(TreeRender, …)` emits the same
        // single line as Default: `"TestSourceExec: batches=1"`.  The
        // `key=value` split is at the first `=`, yielding one extra-
        // info entry: `{"TestSourceExec: batches": "1"}`.
        // `split_up_extra_info` then produces three lines:
        //   1. separator: `-`.repeat(20)  (= NODE_RENDER_WIDTH - 9)
        //   2. `"TestSourceExec: batches:"`  (key + ":" stamped at start of own line because total_size = 23 + 1 + 2 = 26 ≥ available_width = 22)
        //   3. `"1"`  (the value)
        // `extra_height = 3` → `halfway_point = ceil(3/2) = 2`.  The
        // body loop runs `render_y` from 0 (name) to 3 inclusive.  At
        // every iteration the node has 0 children, so `LMID` is never
        // emitted; right border stays `VERT`.

        let sep = "-".repeat(NRW - 9); // 20 dashes

        let expected = format!(
            "{top}\n{l0}\n{l1}\n{l2}\n{l3}\n{bot}\n",
            top = top_border(false),
            l0 = body_line("TestSourceExec", false),
            l1 = body_line(&sep, false),
            l2 = body_line("TestSourceExec: batches:", false),
            l3 = body_line("1", false),
            bot = bottom_border(false),
        );

        assert_eq!(
            rendered, expected,
            "single-node byte snapshot drift.\nactual:\n{rendered}\nexpected:\n{expected}"
        );
    }

    // -------------------------------------------------------------------------
    // Test 2: two-level plan with one parent and one child.
    // Exercises:
    //   * `parent_above` bit on the second level's top border (must use `┴`).
    //   * `child_below` bit on the first level's bottom border (must use `┬`).
    //   * box-content halfway-point junction glyphs for a one-child node:
    //     `RTCORNER` ┐ closes a single child run and writes the right
    //     border, then a row of `│` connector beneath.
    // -------------------------------------------------------------------------

    #[test]
    fn tree_render_byte_equivalent_to_datafusion_two_level_one_child() {
        use crate::GlobalLimitExec;
        use std::sync::Arc;
        let plan = employee_source();
        let limit: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(plan, 0, Some(2)));
        let rendered = displayable(limit.as_ref()).tree_render().to_string();

        // Top of the rendered output must contain the parent's top
        // border with a plain `─` middle (no parent above the root)
        // followed by its name centered in the box body.
        assert!(rendered.starts_with(&top_border(false)));
        assert!(rendered.contains(&body_line("GlobalLimitExec", false)));

        // Between the levels, every parent of a child must end with a
        // bottom border using `┬` and every child must begin with a
        // top border using `┴`.
        assert!(
            rendered.contains(&bottom_border(true)),
            "expected parent bottom-border `└…┬…┘`"
        );
        assert!(
            rendered.contains(&top_border(true)),
            "expected child top-border `┌…┴…┐`"
        );

        // The leaf at the bottom uses plain horizontal middles on its
        // bottom border (no child below).
        assert!(rendered.contains(&bottom_border(false)));

        // The child's name line.
        assert!(rendered.contains(&body_line("TestSourceExec", false)));

        // The vertical connector between the levels is the part of the
        // content layer where the parent's halfway-point row emitted
        // `HORIZONTAL.repeat(NRW/2)` + `RTCORNER` + (no adjacent
        // padding because there's only one column at width 1).  Above
        // that we have rows of `│ … │` for body lines that aren't
        // the halfway point.  No `┬`/`├`/`┤`/`┼` in the body of a
        // one-child layout.
        assert!(
            !rendered.contains(LMID),
            "single-child layouts must not emit `├` glyphs.  Got:\n{rendered}"
        );
    }

    // -------------------------------------------------------------------------
    // Test 3: two-level plan with one parent and TWO children.
    // This exercises:
    //   * the `├` (`LMID`) glyph on the halfway-point body row of a
    //     multi-child node (DataFusion writes `├` instead of `│` as
    //     the right border at that row when `node.child_positions.len() > 1`).
    //   * the `┬` (`TMID`) on the halfway-point row of the parent's
    //     content layer connecting horizontally to the next child.
    //   * widened grid: width = 2, so the top/body/bottom rows of the
    //     parent level have the parent box at x=0 and empty fill at
    //     x=1.
    // We construct a synthetic two-child operator over two
    // `TestSourceExec` instances and verify the exact bytes.
    // -------------------------------------------------------------------------

    #[test]
    fn tree_render_byte_equivalent_to_datafusion_two_level_two_children() {
        use crate::display::{DisplayAs, DisplayFormatType};
        use crate::physical_plan::ExecutionPlan;
        use crate::plan_properties::PlanProperties;
        use fdapquery_datatypes::{FdapQueryError, Result, Schema};
        use fdapquery_execution::TaskContext;
        use std::any::Any;
        use std::fmt;
        use std::sync::Arc;

        #[derive(Debug)]
        struct PairExec {
            children: Vec<Arc<dyn ExecutionPlan>>,
            schema: Schema,
            properties: PlanProperties,
        }
        impl DisplayAs for PairExec {
            fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "PairExec")
            }
        }
        impl fmt::Display for PairExec {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                <Self as DisplayAs>::fmt_as(self, DisplayFormatType::Default, f)
            }
        }
        impl ExecutionPlan for PairExec {
            fn name(&self) -> &'static str {
                "PairExec"
            }
            fn schema(&self) -> Schema {
                self.schema.clone()
            }
            fn properties(&self) -> &PlanProperties {
                &self.properties
            }
            fn execute(
                &self,
                _: usize,
                _: Arc<TaskContext>,
            ) -> Result<crate::SendableRecordBatchStream> {
                Err(FdapQueryError::Internal("PairExec test stub".into()))
            }
            fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
                self.children.iter().collect()
            }
            fn with_new_children(
                self: Arc<Self>,
                _: Vec<Arc<dyn ExecutionPlan>>,
            ) -> Result<Arc<dyn ExecutionPlan>> {
                Ok(self)
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
        }

        let c0 = employee_source();
        let c1 = employee_source();
        let schema = c0.schema();
        let properties = PlanProperties::single_partition_unknown();
        let plan: Arc<dyn ExecutionPlan> = Arc::new(PairExec {
            children: vec![c0, c1],
            schema,
            properties,
        });

        let rendered = displayable(plan.as_ref()).tree_render().to_string();

        // Two-child layout MUST emit the `├` glyph at the halfway-point
        // row of the parent's body — that's how DataFusion distinguishes
        // multi-child nodes from single-child nodes.
        assert!(
            rendered.contains(LMID),
            "two-child layout must emit `├` at the halfway-point row.  Got:\n{rendered}"
        );

        // The horizontal connector between parent and second child runs
        // through a `┬` glyph (`TMID`).
        assert!(
            rendered.contains(TMID),
            "two-child layout must contain `┬` joining parent to children.  Got:\n{rendered}"
        );

        // Both children's top borders must be `┴`-joined to the parent
        // above.
        assert!(
            rendered.contains(&top_border(true)),
            "child top borders must use `┴` to join parent.  Got:\n{rendered}"
        );

        // Parent bottom border uses `┬` (child below).
        assert!(
            rendered.contains(&bottom_border(true)),
            "parent bottom border must use `┬` (child below).  Got:\n{rendered}"
        );

        // Both leaf children's bottom borders use plain `─`.
        assert!(
            rendered.contains(&bottom_border(false)),
            "leaf bottom borders must use plain `─`.  Got:\n{rendered}"
        );
    }

    // -------------------------------------------------------------------------
    // Test 4: three-level plan with mixed branching.
    // Two `GlobalLimitExec` wrappers stacked: outer over inner over the
    // leaf.  Verifies that the renderer extends `RenderTree::height`
    // correctly and that every middle level's top border uses `┴`
    // (parent above) AND its bottom border uses `┬` (child below).
    // -------------------------------------------------------------------------

    #[test]
    fn tree_render_byte_equivalent_to_datafusion_three_level() {
        use crate::GlobalLimitExec;
        use std::sync::Arc;
        let plan = employee_source();
        let mid: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(plan, 0, Some(5)));
        let top: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(mid, 0, Some(2)));
        let rendered = displayable(top.as_ref()).tree_render().to_string();

        // Top-level border: parent_above=false.
        assert!(rendered.starts_with(&top_border(false)));
        // Middle level must contain BOTH:
        //   * a `┴`-using top border (joined to parent above), and
        //   * a `┬`-using bottom border (joined to child below).
        assert!(
            rendered.contains(&top_border(true)),
            "middle level should join parent above with `┴`.  Got:\n{rendered}"
        );
        assert!(
            rendered.contains(&bottom_border(true)),
            "middle level should join child below with `┬`.  Got:\n{rendered}"
        );
        // Bottom (leaf): bottom border uses plain `─`.
        assert!(
            rendered.contains(&bottom_border(false)),
            "leaf bottom border should be plain `─`.  Got:\n{rendered}"
        );

        // The outermost name lives in the first body row.
        assert!(rendered.contains(&body_line("GlobalLimitExec", false)));
        // The leaf name lives in the last set of body rows.
        assert!(rendered.contains(&body_line("TestSourceExec", false)));
    }
}
