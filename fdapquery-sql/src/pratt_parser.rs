//! The Pratt (Top-Down Operator Precedence) parsing loop. See
//! <https://tdop.github.io/> for Pratt's original paper.
//!
//! The parser is exposed as a trait with a provided `parse` method and three
//! required hooks: `next_precedence`, `parse_prefix`, and `parse_infix`.
//! Callers invoke `parse(&mut self, 0)` to parse a full expression.

use crate::expressions::SqlExpr;

/// A Pratt parser.
pub trait PrattParser {
    /// Parse an expression, consuming infix operators that bind tighter than
    /// `precedence`.
    fn parse(&mut self, precedence: i32) -> Option<SqlExpr> {
        let mut expr = self.parse_prefix()?;
        while precedence < self.next_precedence() {
            // Compute the next precedence into a local first: `parse_infix`
            // borrows `self` mutably, so it can't also take `self.next_precedence()`
            // as an argument in the same call.
            let next = self.next_precedence();
            expr = self.parse_infix(expr, next);
        }
        Some(expr)
    }

    /// Precedence of the next token (0 if none / not an operator).
    fn next_precedence(&self) -> i32;

    /// Parse the next prefix expression.
    fn parse_prefix(&mut self) -> Option<SqlExpr>;

    /// Parse the next infix expression, given the already-parsed `left`.
    fn parse_infix(&mut self, left: SqlExpr, precedence: i32) -> SqlExpr;
}
