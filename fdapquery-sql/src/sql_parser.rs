//! The concrete [`PrattParser`] for SQL: the precedence table plus the prefix
//! and infix parse logic that build the [`SqlExpr`] AST.
//!
//! Parse errors surface as `FdapQueryError::SqlParse(_)` from the public
//! `parse` entry point.

use crate::expressions::{SqlExpr, SqlSelect};
use crate::pratt_parser::PrattParser;
use crate::token_stream::TokenStream;
use crate::tokens::{Keyword, Literal, Symbol, TokenType};
use fdapquery_datatypes::{FdapQueryError, Result};

/// SQL parser over a token stream.
pub struct SqlParser {
    pub tokens: TokenStream,
}

impl SqlParser {
    pub fn new(tokens: TokenStream) -> Self {
        Self { tokens }
    }

    fn parse_order(&mut self) -> Result<Vec<SqlExpr>> {
        let mut sort_list = Vec::new();
        let mut sort = self.parse_expr()?;
        while let Some(s) = sort {
            let normalized = match s {
                SqlExpr::Identifier(name) => SqlExpr::Sort {
                    expr: Box::new(SqlExpr::Identifier(name)),
                    asc: true,
                },
                s @ SqlExpr::Sort { .. } => s,
                other => {
                    return Err(FdapQueryError::SqlParse(format!(
                        "unexpected expression {other:?} after ORDER BY"
                    )));
                }
            };
            sort_list.push(normalized);

            if matches!(
                self.tokens.peek().map(|t| t.token_type),
                Some(TokenType::Symbol(Symbol::Comma))
            ) {
                self.tokens.next();
            } else {
                break;
            }
            sort = self.parse_expr()?;
        }
        Ok(sort_list)
    }

    fn parse_cast(&mut self) -> Result<SqlExpr> {
        if !self
            .tokens
            .consume_token_type(&TokenType::Symbol(Symbol::LeftParen))
        {
            return Err(FdapQueryError::SqlParse(
                "expected '(' after CAST".into(),
            ));
        }
        let expr = self.parse_expr()?.ok_or_else(|| {
            FdapQueryError::SqlParse("expected expression in CAST, found EOF".into())
        })?;
        let (inner, alias) = match expr {
            SqlExpr::Alias { expr, alias } => (expr, alias),
            _ => {
                return Err(FdapQueryError::SqlParse(
                    "expected 'AS type' in CAST expression".into(),
                ));
            }
        };
        if !self
            .tokens
            .consume_token_type(&TokenType::Symbol(Symbol::RightParen))
        {
            return Err(FdapQueryError::SqlParse(
                "expected ')' after CAST expression".into(),
            ));
        }
        Ok(SqlExpr::Cast {
            expr: inner,
            data_type: alias,
        })
    }

    fn parse_date(&mut self) -> Result<SqlExpr> {
        let token = self.tokens.next().ok_or_else(|| {
            FdapQueryError::SqlParse("expected date string after DATE keyword, found EOF".into())
        })?;
        if !matches!(token.token_type, TokenType::Literal(Literal::String)) {
            return Err(FdapQueryError::SqlParse(format!(
                "expected date string after DATE keyword, found {token:?}"
            )));
        }
        Ok(SqlExpr::Date(token.text))
    }

    fn parse_interval(&mut self) -> Result<SqlExpr> {
        let token = self.tokens.next().ok_or_else(|| {
            FdapQueryError::SqlParse(
                "expected interval string after INTERVAL keyword, found EOF".into(),
            )
        })?;
        if !matches!(token.token_type, TokenType::Literal(Literal::String)) {
            return Err(FdapQueryError::SqlParse(format!(
                "expected interval string after INTERVAL keyword, found {token:?}"
            )));
        }
        Ok(SqlExpr::Interval(token.text))
    }

    fn parse_select(&mut self) -> Result<SqlSelect> {
        let projection = self.parse_expr_list()?;

        if !self.tokens.consume_keyword("FROM") {
            return Err(FdapQueryError::SqlParse(format!(
                "expected FROM keyword, found {:?}",
                self.tokens.peek()
            )));
        }

        let table_expr = self.parse_expr()?.ok_or_else(|| {
            FdapQueryError::SqlParse("expected table name after FROM, found EOF".into())
        })?;
        let table_name = match table_expr {
            SqlExpr::Identifier(id) => id,
            other => {
                return Err(FdapQueryError::SqlParse(format!(
                    "expected table name after FROM, found {other:?}"
                )));
            }
        };

        // optional WHERE
        let mut selection = None;
        if self.tokens.consume_keyword("WHERE") {
            selection = self.parse_expr()?;
        }

        // optional GROUP BY
        let mut group_by = Vec::new();
        if self.tokens.consume_keywords(&["GROUP", "BY"]) {
            group_by = self.parse_expr_list()?;
        }

        // optional HAVING
        let mut having = None;
        if self.tokens.consume_keyword("HAVING") {
            having = self.parse_expr()?;
        }

        // optional ORDER BY
        let mut order_by = Vec::new();
        if self.tokens.consume_keywords(&["ORDER", "BY"]) {
            order_by = self.parse_order()?;
        }

        // optional LIMIT
        let mut limit = None;
        if self.tokens.consume_keyword("LIMIT") {
            let limit_expr = self.parse_expr()?.ok_or_else(|| {
                FdapQueryError::SqlParse("expected limit value after LIMIT, found EOF".into())
            })?;
            limit = match limit_expr {
                SqlExpr::Long(v) => Some(v as i32),
                _ => {
                    return Err(FdapQueryError::SqlParse(
                        "LIMIT must be a numeric value".into(),
                    ));
                }
            };
        }

        Ok(SqlSelect {
            projection,
            selection,
            group_by,
            order_by,
            having,
            limit,
            table_name,
        })
    }

    fn parse_expr_list(&mut self) -> Result<Vec<SqlExpr>> {
        let mut list = Vec::new();
        let mut expr = self.parse_expr()?;
        while let Some(e) = expr {
            list.push(e);
            if matches!(
                self.tokens.peek().map(|t| t.token_type),
                Some(TokenType::Symbol(Symbol::Comma))
            ) {
                self.tokens.next();
            } else {
                break;
            }
            expr = self.parse_expr()?;
        }
        Ok(list)
    }

    fn parse_expr(&mut self) -> Result<Option<SqlExpr>> {
        self.parse(0)
    }

    /// Parse the next token, requiring it to be an identifier. Returns the
    /// identifier's text.
    fn parse_identifier(&mut self) -> Result<String> {
        let expr = self.parse_expr()?.ok_or_else(|| {
            FdapQueryError::SqlParse("expected identifier, found EOF".into())
        })?;
        match expr {
            SqlExpr::Identifier(id) => Ok(id),
            other => Err(FdapQueryError::SqlParse(format!(
                "expected identifier, found {other:?}"
            ))),
        }
    }
}

impl PrattParser for SqlParser {
    fn next_precedence(&self) -> i32 {
        let token = match self.tokens.peek() {
            Some(t) => t,
            None => return 0,
        };
        match token.token_type {
            // Keywords
            TokenType::Keyword(Keyword::As)
            | TokenType::Keyword(Keyword::Asc)
            | TokenType::Keyword(Keyword::Desc) => 10,
            TokenType::Keyword(Keyword::Or) => 20,
            TokenType::Keyword(Keyword::And) => 30,
            // Symbols
            TokenType::Symbol(Symbol::Lt)
            | TokenType::Symbol(Symbol::LtEq)
            | TokenType::Symbol(Symbol::Eq)
            | TokenType::Symbol(Symbol::BangEq)
            | TokenType::Symbol(Symbol::GtEq)
            | TokenType::Symbol(Symbol::Gt) => 40,
            TokenType::Symbol(Symbol::Plus) | TokenType::Symbol(Symbol::Sub) => 50,
            TokenType::Symbol(Symbol::Star) | TokenType::Symbol(Symbol::Slash) => 60,
            TokenType::Symbol(Symbol::LeftParen) => 70,
            _ => 0,
        }
    }

    fn parse_prefix(&mut self) -> Result<Option<SqlExpr>> {
        let token = match self.tokens.next() {
            Some(t) => t,
            None => return Ok(None),
        };
        let expr = match &token.token_type {
            // Keywords
            TokenType::Keyword(Keyword::Select) => SqlExpr::Select(Box::new(self.parse_select()?)),
            TokenType::Keyword(Keyword::Cast) => self.parse_cast()?,
            TokenType::Keyword(Keyword::Date) => self.parse_date()?,
            TokenType::Keyword(Keyword::Interval) => self.parse_interval()?,
            TokenType::Keyword(Keyword::Min)
            | TokenType::Keyword(Keyword::Max)
            | TokenType::Keyword(Keyword::Sum)
            | TokenType::Keyword(Keyword::Avg)
            | TokenType::Keyword(Keyword::Count) => SqlExpr::Identifier(token.text.clone()),

            // type keywords used as identifiers
            TokenType::Keyword(Keyword::Int) => SqlExpr::Identifier(token.text.clone()),
            TokenType::Keyword(Keyword::Double) => SqlExpr::Identifier(token.text.clone()),

            // Literals
            TokenType::Literal(Literal::Identifier) => SqlExpr::Identifier(token.text.clone()),
            TokenType::Literal(Literal::String) => SqlExpr::String(token.text.clone()),
            TokenType::Literal(Literal::Long) => SqlExpr::Long(token.text.parse::<i64>().map_err(
                |e| {
                    FdapQueryError::SqlParse(format!(
                        "invalid long literal '{}': {e}",
                        token.text
                    ))
                },
            )?),
            TokenType::Literal(Literal::Double) => {
                SqlExpr::Double(token.text.parse::<f64>().map_err(|e| {
                    FdapQueryError::SqlParse(format!(
                        "invalid double literal '{}': {e}",
                        token.text
                    ))
                })?)
            }

            // Parenthesized expression
            TokenType::Symbol(Symbol::LeftParen) => {
                let expr = self.parse_expr()?.ok_or_else(|| {
                    FdapQueryError::SqlParse("expected expression after '(', found EOF".into())
                })?;
                if !self
                    .tokens
                    .consume_token_type(&TokenType::Symbol(Symbol::RightParen))
                {
                    return Err(FdapQueryError::SqlParse(
                        "expected ')' after expression".into(),
                    ));
                }
                expr
            }

            // Star, for count(*)
            TokenType::Symbol(Symbol::Star) => SqlExpr::Identifier("*".to_string()),

            _ => {
                return Err(FdapQueryError::SqlParse(format!(
                    "unexpected token {token:?}"
                )));
            }
        };
        Ok(Some(expr))
    }

    fn parse_infix(&mut self, left: SqlExpr, precedence: i32) -> Result<SqlExpr> {
        let token = self
            .tokens
            .peek()
            .ok_or_else(|| FdapQueryError::SqlParse("expected infix token, found EOF".into()))?;
        let expr = match &token.token_type {
            // Arithmetic and comparison operators
            TokenType::Symbol(Symbol::Plus)
            | TokenType::Symbol(Symbol::Sub)
            | TokenType::Symbol(Symbol::Star)
            | TokenType::Symbol(Symbol::Slash)
            | TokenType::Symbol(Symbol::Eq)
            | TokenType::Symbol(Symbol::Gt)
            | TokenType::Symbol(Symbol::Lt)
            | TokenType::Symbol(Symbol::GtEq)
            | TokenType::Symbol(Symbol::LtEq)
            | TokenType::Symbol(Symbol::BangEq)
            | TokenType::Symbol(Symbol::LtGt) => {
                let op = token.text.clone();
                self.tokens.next(); // consume the operator
                let r = self.parse(precedence)?.ok_or_else(|| {
                    FdapQueryError::SqlParse(format!(
                        "expected right operand for '{op}', found EOF"
                    ))
                })?;
                SqlExpr::BinaryExpr {
                    l: Box::new(left),
                    op,
                    r: Box::new(r),
                }
            }

            // AS — aliasing
            TokenType::Keyword(Keyword::As) => {
                self.tokens.next(); // consume AS
                SqlExpr::Alias {
                    expr: Box::new(left),
                    alias: self.parse_identifier()?,
                }
            }

            // boolean operators
            TokenType::Keyword(Keyword::And) | TokenType::Keyword(Keyword::Or) => {
                let op = token.text.clone();
                self.tokens.next(); // consume the keyword
                let r = self.parse(precedence)?.ok_or_else(|| {
                    FdapQueryError::SqlParse(format!(
                        "expected right operand for '{op}', found EOF"
                    ))
                })?;
                SqlExpr::BinaryExpr {
                    l: Box::new(left),
                    op,
                    r: Box::new(r),
                }
            }

            // sort direction
            TokenType::Keyword(Keyword::Asc) | TokenType::Keyword(Keyword::Desc) => {
                let asc = matches!(token.token_type, TokenType::Keyword(Keyword::Asc));
                self.tokens.next();
                SqlExpr::Sort {
                    expr: Box::new(left),
                    asc,
                }
            }

            // function call
            TokenType::Symbol(Symbol::LeftParen) => {
                if let SqlExpr::Identifier(id) = &left {
                    let id = id.clone();
                    self.tokens.next(); // consume the left paren
                    let args = if matches!(
                        self.tokens.peek().map(|t| t.token_type),
                        Some(TokenType::Symbol(Symbol::RightParen))
                    ) {
                        Vec::new()
                    } else {
                        self.parse_expr_list()?
                    };
                    if !matches!(
                        self.tokens.next().map(|t| t.token_type),
                        Some(TokenType::Symbol(Symbol::RightParen))
                    ) {
                        return Err(FdapQueryError::SqlParse(
                            "expected ')' after function arguments".into(),
                        ));
                    }
                    SqlExpr::Function { id, args }
                } else {
                    return Err(FdapQueryError::SqlParse(
                        "unexpected '(' after non-identifier expression".into(),
                    ));
                }
            }

            _ => {
                return Err(FdapQueryError::SqlParse(format!(
                    "unexpected infix token {token:?}"
                )));
            }
        };
        Ok(expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql_tokenizer::SqlTokenizer;

    fn parse(sql: &str) -> Option<SqlExpr> {
        let tokens = SqlTokenizer::new(sql).tokenize().unwrap();
        SqlParser::new(tokens).parse(0).unwrap()
    }

    /// Like `parse` but returns the Result so error-path tests can inspect it.
    fn try_parse(sql: &str) -> Result<Option<SqlExpr>> {
        let tokens = SqlTokenizer::new(sql).tokenize()?;
        SqlParser::new(tokens).parse(0)
    }

    fn parse_select(sql: &str) -> SqlSelect {
        match parse(sql) {
            Some(SqlExpr::Select(s)) => *s,
            other => panic!("expected SELECT, found {other:?}"),
        }
    }

    fn id(s: &str) -> SqlExpr {
        SqlExpr::Identifier(s.to_string())
    }
    fn bin(l: SqlExpr, op: &str, r: SqlExpr) -> SqlExpr {
        SqlExpr::BinaryExpr {
            l: Box::new(l),
            op: op.to_string(),
            r: Box::new(r),
        }
    }

    #[test]
    fn precedence_add_then_mult() {
        let expected = bin(
            SqlExpr::Long(1),
            "+",
            bin(SqlExpr::Long(2), "*", SqlExpr::Long(3)),
        );
        assert_eq!(parse("1 + 2 * 3"), Some(expected));
    }

    #[test]
    fn precedence_mult_then_add() {
        let expected = bin(
            bin(SqlExpr::Long(1), "*", SqlExpr::Long(2)),
            "+",
            SqlExpr::Long(3),
        );
        assert_eq!(parse("1 * 2 + 3"), Some(expected));
    }

    #[test]
    fn simple_select() {
        let select = parse_select("SELECT id, first_name, last_name FROM employee");
        assert_eq!(select.table_name, "employee");
        assert_eq!(
            select.projection,
            vec![id("id"), id("first_name"), id("last_name")]
        );
    }

    #[test]
    fn projection_with_binary_expression() {
        let select = parse_select("SELECT salary * 0.1 FROM employee");
        assert_eq!(select.table_name, "employee");
        assert_eq!(
            select.projection,
            vec![bin(id("salary"), "*", SqlExpr::Double(0.1))]
        );
    }

    #[test]
    fn projection_with_aliased_binary_expression() {
        let select = parse_select("SELECT salary * 0.1 AS bonus FROM employee");
        assert_eq!(select.table_name, "employee");
        let expected = SqlExpr::Alias {
            expr: Box::new(bin(id("salary"), "*", SqlExpr::Double(0.1))),
            alias: "bonus".to_string(),
        };
        assert_eq!(select.projection, vec![expected]);
    }

    #[test]
    fn parse_select_with_where() {
        let select =
            parse_select("SELECT id, first_name, last_name FROM employee WHERE state = 'CO'");
        assert_eq!(
            select.projection,
            vec![id("id"), id("first_name"), id("last_name")]
        );
        assert_eq!(
            select.selection,
            Some(bin(id("state"), "=", SqlExpr::String("CO".to_string())))
        );
        assert_eq!(select.table_name, "employee");
    }

    #[test]
    fn parse_select_with_order() {
        let select = parse_select("SELECT state, salary FROM employee ORDER BY salary desc, state");
        assert_eq!(select.projection, vec![id("state"), id("salary")]);
        assert_eq!(
            select.order_by,
            vec![
                SqlExpr::Sort {
                    expr: Box::new(id("salary")),
                    asc: false
                },
                SqlExpr::Sort {
                    expr: Box::new(id("state")),
                    asc: true
                },
            ]
        );
    }

    #[test]
    fn parse_select_with_aggregates() {
        let select = parse_select("SELECT state, MAX(salary) FROM employee GROUP BY state");
        assert_eq!(
            select.projection,
            vec![
                id("state"),
                SqlExpr::Function {
                    id: "MAX".to_string(),
                    args: vec![id("salary")]
                },
            ]
        );
        assert_eq!(select.group_by, vec![id("state")]);
        assert_eq!(select.table_name, "employee");
    }

    #[test]
    fn parse_select_with_aliased_aggregates() {
        let select =
            parse_select("SELECT state, MAX(salary) AS top_wage FROM employee GROUP BY state");
        let max = SqlExpr::Function {
            id: "MAX".to_string(),
            args: vec![id("salary")],
        };
        let alias = SqlExpr::Alias {
            expr: Box::new(max),
            alias: "top_wage".to_string(),
        };
        assert_eq!(select.projection, vec![id("state"), alias]);
        assert_eq!(select.group_by, vec![id("state")]);
        assert_eq!(select.table_name, "employee");
    }

    #[test]
    fn parse_select_with_aggregates_and_having() {
        let select = parse_select(
            "SELECT state, MAX(salary) AS top_wage FROM employee \
             GROUP BY state HAVING MAX(salary) > 10 AND MAX(salary) < 100",
        );
        let max = SqlExpr::Function {
            id: "MAX".to_string(),
            args: vec![id("salary")],
        };
        let alias = SqlExpr::Alias {
            expr: Box::new(max),
            alias: "top_wage".to_string(),
        };
        assert_eq!(select.projection, vec![id("state"), alias]);
        assert_eq!(select.group_by, vec![id("state")]);
        assert_eq!(select.table_name, "employee");
    }

    #[test]
    fn parse_select_with_aggregates_and_cast() {
        let select =
            parse_select("SELECT state, MAX(CAST(salary AS double)) FROM employee GROUP BY state");
        let cast = SqlExpr::Cast {
            expr: Box::new(id("salary")),
            data_type: "double".to_string(),
        };
        let max = SqlExpr::Function {
            id: "MAX".to_string(),
            args: vec![cast],
        };
        assert_eq!(select.projection, vec![id("state"), max]);
        assert_eq!(select.group_by, vec![id("state")]);
        assert_eq!(select.table_name, "employee");
    }

    #[test]
    fn parse_select_with_limit() {
        let select = parse_select("SELECT id, first_name FROM employee LIMIT 10");
        assert_eq!(select.projection, vec![id("id"), id("first_name")]);
        assert_eq!(select.limit, Some(10));
        assert_eq!(select.table_name, "employee");
    }

    #[test]
    fn parse_select_with_where_and_limit() {
        let select = parse_select("SELECT id, first_name FROM employee WHERE state = 'CO' LIMIT 5");
        assert_eq!(select.projection, vec![id("id"), id("first_name")]);
        assert_eq!(
            select.selection,
            Some(bin(id("state"), "=", SqlExpr::String("CO".to_string())))
        );
        assert_eq!(select.limit, Some(5));
        assert_eq!(select.table_name, "employee");
    }

    #[test]
    fn parse_select_with_order_by_and_limit() {
        let select = parse_select("SELECT id, salary FROM employee ORDER BY salary DESC LIMIT 3");
        assert_eq!(select.projection, vec![id("id"), id("salary")]);
        assert_eq!(
            select.order_by,
            vec![SqlExpr::Sort {
                expr: Box::new(id("salary")),
                asc: false
            }]
        );
        assert_eq!(select.limit, Some(3));
        assert_eq!(select.table_name, "employee");
    }

    #[test]
    fn parse_date_literal() {
        assert_eq!(
            parse("date '1998-12-01'"),
            Some(SqlExpr::Date("1998-12-01".to_string()))
        );
    }

    #[test]
    fn parse_select_with_date_literal_in_where() {
        let select = parse_select("SELECT id FROM orders WHERE order_date < date '1998-12-01'");
        assert_eq!(select.projection, vec![id("id")]);
        assert_eq!(
            select.selection,
            Some(bin(
                id("order_date"),
                "<",
                SqlExpr::Date("1998-12-01".to_string())
            ))
        );
        assert_eq!(select.table_name, "orders");
    }

    #[test]
    fn parse_interval_literal() {
        assert_eq!(
            parse("interval '68 days'"),
            Some(SqlExpr::Interval("68 days".to_string()))
        );
    }

    #[test]
    fn parse_select_with_date_and_interval_arithmetic() {
        let select = parse_select(
            "SELECT id FROM orders WHERE order_date < date '1998-12-01' - interval '68 days'",
        );
        assert_eq!(select.projection, vec![id("id")]);
        let rhs = bin(
            SqlExpr::Date("1998-12-01".to_string()),
            "-",
            SqlExpr::Interval("68 days".to_string()),
        );
        assert_eq!(select.selection, Some(bin(id("order_date"), "<", rhs)));
        assert_eq!(select.table_name, "orders");
    }

    fn assert_sql_parse_err(sql: &str, expected_substring: &str) {
        let err = try_parse(sql).expect_err("parse should fail");
        assert!(
            matches!(err, FdapQueryError::SqlParse(_)),
            "expected SqlParse(_), got {err:?}"
        );
        assert!(
            err.to_string().contains(expected_substring),
            "expected message containing '{expected_substring}', got '{}'",
            err
        );
    }

    #[test]
    fn function_call_missing_closing_paren_returns_sql_parse_error() {
        assert_sql_parse_err(
            "SELECT MAX(salary FROM employee",
            "expected ')' after function arguments",
        );
    }

    #[test]
    fn cast_missing_opening_paren_returns_sql_parse_error() {
        assert_sql_parse_err(
            "SELECT CAST salary AS double) FROM employee",
            "expected '(' after CAST",
        );
    }

    #[test]
    fn cast_missing_as_returns_sql_parse_error() {
        assert_sql_parse_err(
            "SELECT CAST(salary) FROM employee",
            "expected 'AS type' in CAST expression",
        );
    }

    #[test]
    fn cast_missing_closing_paren_returns_sql_parse_error() {
        assert_sql_parse_err(
            "SELECT CAST(salary AS double FROM employee",
            "expected ')' after CAST expression",
        );
    }

    #[test]
    fn from_with_non_identifier_returns_sql_parse_error() {
        assert_sql_parse_err("SELECT a FROM 123", "expected table name after FROM");
    }

    #[test]
    fn from_with_string_literal_returns_sql_parse_error() {
        assert_sql_parse_err("SELECT a FROM 'table'", "expected table name after FROM");
    }
}
