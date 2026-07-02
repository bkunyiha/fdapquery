//! Strict mirror of `datafusion::physical_plan::render_tree`.
//!
//! Two-pass grid model used by `crate::display::TreeRenderVisitor` (the
//! Unicode box-drawing renderer that backs `displayable(plan).tree_render()`).
//!
//! Pass 1 — `create_tree` — walks the plan and produces a `RenderTree`: a
//! flattened 2D `Vec<Option<Arc<RenderTreeNode>>>` of `width × height`
//! cells. Each `RenderTreeNode` stores the operator's name, its
//! `TreeRender`-mode extra-info key/value pairs, and the `(x, y)` grid
//! coordinates of its children in the next-deeper row.
//!
//! Pass 2 — the `TreeRenderVisitor::visit` loop in `display.rs` — emits
//! three layers per `y` (top borders, body lines, bottom borders) using
//! the grid's `has_node` / `get_node` queries.
//!
//! Byte-for-byte equivalence with DataFusion: same field set, same
//! `get_position` formula (`y * width + x`), same recursive
//! `create_tree_recursive` that returns the subtree width and records
//! child positions in BFS order. Mirrors
//! `/Users/bkunyiha/Rust/datafusion/datafusion/physical-plan/src/render_tree.rs`.

use std::collections::HashMap;
use std::fmt::Formatter;
use std::sync::Arc;
use std::{cmp, fmt};

use crate::display::DisplayFormatType;
use crate::physical_plan::ExecutionPlan;

/// Represents a 2D coordinate in the rendered tree.
///
/// Used to track positions of nodes and their connections.  Mirrors
/// DataFusion's `Coordinate` (also currently unused — see the `dead_code`
/// allow — but kept for API parity).
pub struct Coordinate {
    /// Horizontal position in the tree
    #[allow(dead_code)]
    pub x: usize,
    /// Vertical position in the tree
    #[allow(dead_code)]
    pub y: usize,
}

impl Coordinate {
    pub fn new(x: usize, y: usize) -> Self {
        Coordinate { x, y }
    }
}

/// Represents a node in the render tree, containing information about an
/// execution plan operator and its relationships to other operators.
///
/// `name` is `plan.name()`.  `extra_text` carries every `key=value` (or
/// bare-line) pair the operator's `DisplayAs::fmt_as(TreeRender, …)`
/// implementation produced.  `child_positions` is the grid coordinates of
/// each child in the next-deeper `y` row.
pub struct RenderTreeNode {
    /// The name of the physical `ExecutionPlan`.
    pub name: String,
    /// Execution info collected from `ExecutionPlan`.
    pub extra_text: HashMap<String, String>,
    /// Positions of child nodes in the rendered tree.
    pub child_positions: Vec<Coordinate>,
}

impl RenderTreeNode {
    pub fn new(name: String, extra_text: HashMap<String, String>) -> Self {
        RenderTreeNode {
            name,
            extra_text,
            child_positions: vec![],
        }
    }

    fn add_child_position(&mut self, x: usize, y: usize) {
        self.child_positions.push(Coordinate::new(x, y));
    }
}

/// Main structure for rendering an execution plan as a tree.
/// Manages a 2D grid of nodes and their layout information.
pub struct RenderTree {
    /// Storage for tree nodes in a flattened 2D grid.
    pub nodes: Vec<Option<Arc<RenderTreeNode>>>,
    /// Total width of the rendered tree.
    pub width: usize,
    /// Total height of the rendered tree.
    pub height: usize,
}

impl RenderTree {
    /// Creates a new render tree from an execution plan.  Mirrors
    /// DataFusion's `RenderTree::create_tree`.
    pub fn create_tree(plan: &dyn ExecutionPlan) -> Self {
        let (width, height) = get_tree_width_height(plan);

        let mut result = Self::new(width, height);

        create_tree_recursive(&mut result, plan, 0, 0);

        result
    }

    fn new(width: usize, height: usize) -> Self {
        RenderTree {
            nodes: vec![None; (width + 1) * (height + 1)],
            width,
            height,
        }
    }

    pub fn get_node(&self, x: usize, y: usize) -> Option<Arc<RenderTreeNode>> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let pos = self.get_position(x, y);
        self.nodes.get(pos).and_then(|node| node.clone())
    }

    pub fn set_node(&mut self, x: usize, y: usize, node: Arc<RenderTreeNode>) {
        let pos = self.get_position(x, y);
        if let Some(slot) = self.nodes.get_mut(pos) {
            *slot = Some(node);
        }
    }

    pub fn has_node(&self, x: usize, y: usize) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }

        let pos = self.get_position(x, y);
        self.nodes.get(pos).is_some_and(|node| node.is_some())
    }

    fn get_position(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }
}

/// Calculates the required dimensions of the tree.
/// This ensures we allocate enough space for the entire tree structure.
///
/// # Arguments
/// * `plan` - The execution plan to measure
///
/// # Returns
/// * A tuple of (width, height) representing the dimensions needed for the tree
fn get_tree_width_height(plan: &dyn ExecutionPlan) -> (usize, usize) {
    let children = plan.children();

    // Leaf nodes take up 1x1 space
    if children.is_empty() {
        return (1, 1);
    }

    let mut width = 0;
    let mut height = 0;

    for child in children {
        let (child_width, child_height) = get_tree_width_height(child.as_ref());
        width += child_width;
        height = cmp::max(height, child_height);
    }

    height += 1;

    (width, height)
}

fn fmt_display(plan: &dyn ExecutionPlan) -> impl fmt::Display + '_ {
    struct Wrapper<'a> {
        plan: &'a dyn ExecutionPlan,
    }

    impl fmt::Display for Wrapper<'_> {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            self.plan.fmt_as(DisplayFormatType::TreeRender, f)?;
            Ok(())
        }
    }

    Wrapper { plan }
}

/// Recursively builds the render tree structure.
/// Traverses the execution plan and creates corresponding render nodes while
/// maintaining proper positioning and parent-child relationships.
///
/// # Arguments
/// * `result` - The render tree being constructed
/// * `plan` - Current execution plan node being processed
/// * `x` - Horizontal position in the tree
/// * `y` - Vertical position in the tree
///
/// # Returns
/// * The width of the subtree rooted at the current node
fn create_tree_recursive(
    result: &mut RenderTree,
    plan: &dyn ExecutionPlan,
    x: usize,
    y: usize,
) -> usize {
    let display_info = fmt_display(plan).to_string();
    let mut extra_info = HashMap::new();

    // Parse the key-value pairs from the formatted string.
    // See DisplayFormatType::TreeRender for details
    for line in display_info.lines() {
        if let Some((key, value)) = line.split_once('=') {
            extra_info.insert(key.to_string(), value.to_string());
        } else {
            extra_info.insert(line.to_string(), String::new());
        }
    }

    let mut node = RenderTreeNode::new(plan.name().to_string(), extra_info);

    let children = plan.children();

    if children.is_empty() {
        result.set_node(x, y, Arc::new(node));
        return 1;
    }

    let mut width = 0;
    for child in children {
        let child_x = x + width;
        let child_y = y + 1;
        node.add_child_position(child_x, child_y);
        width += create_tree_recursive(result, child.as_ref(), child_x, child_y);
    }

    result.set_node(x, y, Arc::new(node));

    width
}
