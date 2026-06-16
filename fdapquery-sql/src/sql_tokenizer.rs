//! Hand-written lexer that turns a SQL string into a [`TokenStream`].
//!
//! ## Notes
//! - Rust `String` is UTF-8 and indexes by byte, so the tokenizer holds the
//!   input as a `Vec<char>` and works in char offsets. This makes
//!   `end_offset` a stable character index and avoids byte/char-boundary
//!   hazards.
//! - Tokenization errors are reported via `panic!` (§3.6).

use crate::token_stream::TokenStream;
use crate::tokens::{Keyword, Literal, Symbol, Token, TokenType};

/// Lexer over a SQL string.
pub struct SqlTokenizer {
    chars: Vec<char>,
    pub offset: usize,
}

impl SqlTokenizer {
    pub fn new(sql: &str) -> Self {
        Self {
            chars: sql.chars().collect(),
            offset: 0,
        }
    }

    /// Length in characters.
    fn len(&self) -> usize {
        self.chars.len()
    }

    /// `chars[start..end]` as an owned `String`.
    fn substring(&self, start: usize, end: usize) -> String {
        self.chars[start..end].iter().collect()
    }

    /// Tokenize the whole input.
    pub fn tokenize(&mut self) -> TokenStream {
        let mut list = Vec::new();
        while let Some(token) = self.next_token() {
            list.push(token);
        }
        TokenStream::new(list)
    }

    fn next_token(&mut self) -> Option<Token> {
        self.offset = self.skip_whitespace(self.offset);
        if self.offset >= self.len() {
            return None;
        }
        let ch = self.chars[self.offset];
        if Literal::is_identifier_start(ch) {
            let token = self.scan_identifier(self.offset);
            self.offset = token.end_offset;
            Some(token)
        } else if Literal::is_number_start(ch) {
            let token = self.scan_number(self.offset);
            self.offset = token.end_offset;
            Some(token)
        } else if Symbol::is_symbol_start(ch) {
            let token = self.scan_symbol(self.offset);
            self.offset = token.end_offset;
            Some(token)
        } else if Literal::is_chars_start(ch) {
            let token = self.scan_chars(self.offset, ch);
            self.offset = token.end_offset;
            Some(token)
        } else {
            panic!("Unexpected character '{}' at position {}", ch, self.offset);
        }
    }

    fn skip_whitespace(&self, start: usize) -> usize {
        self.index_of_first(start, |ch| !ch.is_whitespace())
    }

    fn scan_number(&self, start: usize) -> Token {
        // The `-` branch is dead in the main path (a leading `-` is a `Symbol`,
        // not a number start) but is preserved for completeness.
        let mut end = if self.chars[start] == '-' {
            self.index_of_first(start + 1, |ch| !ch.is_ascii_digit())
        } else {
            self.index_of_first(start, |ch| !ch.is_ascii_digit())
        };
        if end == self.len() {
            return Token::new(
                self.substring(start, end),
                TokenType::Literal(Literal::Long),
                end,
            );
        }
        let is_float = self.chars[end] == '.';
        if is_float {
            end = self.index_of_first(end + 1, |ch| !ch.is_ascii_digit());
        }
        let lit = if is_float {
            Literal::Double
        } else {
            Literal::Long
        };
        Token::new(self.substring(start, end), TokenType::Literal(lit), end)
    }

    fn scan_identifier(&self, start: usize) -> Token {
        // Back-quoted identifier: `like this`.
        if self.chars[start] == '`' {
            let end = self.get_offset_until_terminated_char('`', start + 1);
            return Token::new(
                self.substring(start + 1, end),
                TokenType::Literal(Literal::Identifier),
                end + 1,
            );
        }
        let end = self.index_of_first(start, |ch| !Literal::is_identifier_part(ch));
        let text = self.substring(start, end);
        if self.is_ambiguous_identifier(&text) {
            let token_type = self.process_ambiguous_identifier(end, &text);
            Token::new(text, token_type, end)
        } else {
            let token_type = match Keyword::text_of(&text) {
                Some(keyword) => TokenType::Keyword(keyword),
                None => TokenType::Literal(Literal::Identifier),
            };
            Token::new(text, token_type, end)
        }
    }

    /// `ORDER` / `GROUP` are keywords only when followed by `BY`; otherwise they
    /// are ordinary identifiers (e.g. a table named `order`).
    fn is_ambiguous_identifier(&self, text: &str) -> bool {
        text.eq_ignore_ascii_case(Keyword::Order.name())
            || text.eq_ignore_ascii_case(Keyword::Group.name())
    }

    fn process_ambiguous_identifier(&self, start: usize, text: &str) -> TokenType {
        let skip = self.skip_whitespace(start);
        if skip + 2 <= self.len()
            && self
                .substring(skip, skip + 2)
                .eq_ignore_ascii_case(Keyword::By.name())
        {
            TokenType::Keyword(
                Keyword::text_of(text).expect("ambiguous identifier must be a keyword"),
            )
        } else {
            TokenType::Literal(Literal::Identifier)
        }
    }

    fn get_offset_until_terminated_char(&self, terminated: char, start: usize) -> usize {
        match self.chars[start..].iter().position(|&c| c == terminated) {
            Some(pos) => start + pos,
            None => panic!("Must contain {terminated} in remain sql[{start} .. end]"),
        }
    }

    /// text starting at `start`, shrinking until a valid symbol is found.
    fn scan_symbol(&self, start: usize) -> Token {
        let mut end = self.index_of_first(start, |ch| !Symbol::is_symbol(ch));
        let mut text = self.substring(start, end);
        let mut symbol = Symbol::text_of(&text);
        while symbol.is_none() {
            end -= 1;
            text = self.substring(start, end);
            symbol = Symbol::text_of(&text);
        }
        let symbol = symbol.expect("text must be a Symbol");
        Token::new(text, TokenType::Symbol(symbol), end)
    }

    /// quote escape (`''` or `""`).
    fn scan_chars(&self, start: usize, terminated: char) -> Token {
        let mut builder = String::new();
        let mut i = start + 1;
        while i < self.len() {
            let ch = self.chars[i];
            if ch == terminated {
                if i + 1 < self.len() && self.chars[i + 1] == terminated {
                    builder.push(terminated);
                    i += 2;
                } else {
                    return Token::new(builder, TokenType::Literal(Literal::String), i + 1);
                }
            } else {
                builder.push(ch);
                i += 1;
            }
        }
        panic!("Unterminated string starting at position {start}");
    }

    /// First index `>= start` whose char satisfies `predicate`, or the input
    /// length if none does.
    fn index_of_first(&self, start: usize, predicate: impl Fn(char) -> bool) -> usize {
        let mut idx = start;
        while idx < self.len() {
            if predicate(self.chars[idx]) {
                return idx;
            }
            idx += 1;
        }
        self.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kw(text: &str, k: Keyword, end: usize) -> Token {
        Token::new(text, TokenType::Keyword(k), end)
    }
    fn sym(text: &str, s: Symbol, end: usize) -> Token {
        Token::new(text, TokenType::Symbol(s), end)
    }
    fn lit(text: &str, l: Literal, end: usize) -> Token {
        Token::new(text, TokenType::Literal(l), end)
    }
    fn tokenize(sql: &str) -> Vec<Token> {
        SqlTokenizer::new(sql).tokenize().tokens
    }

    #[test]
    fn tokenize_simple_select() {
        let expected = vec![
            kw("SELECT", Keyword::Select, 6),
            lit("id", Literal::Identifier, 9),
            sym(",", Symbol::Comma, 10),
            lit("first_name", Literal::Identifier, 21),
            sym(",", Symbol::Comma, 22),
            lit("last_name", Literal::Identifier, 32),
            kw("FROM", Keyword::From, 37),
            lit("employee", Literal::Identifier, 46),
        ];
        assert_eq!(
            tokenize("SELECT id, first_name, last_name FROM employee"),
            expected
        );
    }

    #[test]
    fn projection_with_binary_expression() {
        let expected = vec![
            kw("SELECT", Keyword::Select, 6),
            lit("salary", Literal::Identifier, 13),
            sym("*", Symbol::Star, 15),
            lit("0.1", Literal::Double, 19),
            kw("FROM", Keyword::From, 24),
            lit("employee", Literal::Identifier, 33),
        ];
        assert_eq!(tokenize("SELECT salary * 0.1 FROM employee"), expected);
    }

    #[test]
    fn projection_with_aliased_binary_expression() {
        let expected = vec![
            kw("SELECT", Keyword::Select, 6),
            lit("salary", Literal::Identifier, 13),
            sym("*", Symbol::Star, 15),
            lit("0.1", Literal::Double, 19),
            kw("AS", Keyword::As, 22),
            lit("bonus", Literal::Identifier, 28),
            kw("FROM", Keyword::From, 33),
            lit("employee", Literal::Identifier, 42),
        ];
        assert_eq!(
            tokenize("SELECT salary * 0.1 AS bonus FROM employee"),
            expected
        );
    }

    #[test]
    fn tokenize_select_with_where() {
        let expected = vec![
            kw("SELECT", Keyword::Select, 6),
            lit("a", Literal::Identifier, 8),
            sym(",", Symbol::Comma, 9),
            lit("b", Literal::Identifier, 11),
            kw("FROM", Keyword::From, 16),
            lit("employee", Literal::Identifier, 25),
            kw("WHERE", Keyword::Where, 31),
            lit("state", Literal::Identifier, 37),
            sym("=", Symbol::Eq, 39),
            lit("CO", Literal::String, 44),
        ];
        assert_eq!(
            tokenize("SELECT a, b FROM employee WHERE state = 'CO'"),
            expected
        );
    }

    #[test]
    fn tokenize_select_with_aggregates() {
        let expected = vec![
            kw("SELECT", Keyword::Select, 6),
            lit("state", Literal::Identifier, 12),
            sym(",", Symbol::Comma, 13),
            kw("MAX", Keyword::Max, 17),
            sym("(", Symbol::LeftParen, 18),
            lit("salary", Literal::Identifier, 24),
            sym(")", Symbol::RightParen, 25),
            kw("FROM", Keyword::From, 30),
            lit("employee", Literal::Identifier, 39),
            kw("GROUP", Keyword::Group, 45),
            kw("BY", Keyword::By, 48),
            lit("state", Literal::Identifier, 54),
        ];
        assert_eq!(
            tokenize("SELECT state, MAX(salary) FROM employee GROUP BY state"),
            expected
        );
    }

    #[test]
    fn tokenize_select_with_aggregates_and_having() {
        let expected = vec![
            kw("SELECT", Keyword::Select, 6),
            lit("state", Literal::Identifier, 12),
            sym(",", Symbol::Comma, 13),
            kw("MAX", Keyword::Max, 17),
            sym("(", Symbol::LeftParen, 18),
            lit("salary", Literal::Identifier, 24),
            sym(")", Symbol::RightParen, 25),
            kw("FROM", Keyword::From, 30),
            lit("employee", Literal::Identifier, 39),
            kw("GROUP", Keyword::Group, 45),
            kw("BY", Keyword::By, 48),
            lit("state", Literal::Identifier, 54),
            kw("HAVING", Keyword::Having, 61),
            kw("MAX", Keyword::Max, 65),
            sym("(", Symbol::LeftParen, 66),
            lit("salary", Literal::Identifier, 72),
            sym(")", Symbol::RightParen, 73),
            sym(">", Symbol::Gt, 75),
            lit("10", Literal::Long, 78),
        ];
        assert_eq!(
            tokenize(
                "SELECT state, MAX(salary) FROM employee GROUP BY state HAVING MAX(salary) > 10"
            ),
            expected
        );
    }

    #[test]
    fn tokenize_compound_operators() {
        let expected = vec![
            lit("a", Literal::Identifier, 1),
            sym(">=", Symbol::GtEq, 4),
            lit("b", Literal::Identifier, 6),
            kw("OR", Keyword::Or, 9),
            lit("a", Literal::Identifier, 11),
            sym("<=", Symbol::LtEq, 14),
            lit("b", Literal::Identifier, 16),
            kw("OR", Keyword::Or, 19),
            lit("a", Literal::Identifier, 21),
            sym("<>", Symbol::LtGt, 24),
            lit("b", Literal::Identifier, 26),
            kw("OR", Keyword::Or, 29),
            lit("a", Literal::Identifier, 31),
            sym("!=", Symbol::BangEq, 34),
            lit("b", Literal::Identifier, 36),
        ];
        assert_eq!(tokenize("a >= b OR a <= b OR a <> b OR a != b"), expected);
    }

    #[test]
    fn tokenize_long_values() {
        let expected = vec![
            lit("123456789", Literal::Long, 9),
            sym("+", Symbol::Plus, 11),
            lit("987654321", Literal::Long, 21),
        ];
        assert_eq!(tokenize("123456789 + 987654321"), expected);
    }

    #[test]
    fn tokenize_float_double_values() {
        let expected = vec![
            lit("123456789.00", Literal::Double, 12),
            sym("+", Symbol::Plus, 14),
            lit("987654321.001", Literal::Double, 28),
        ];
        assert_eq!(tokenize("123456789.00 + 987654321.001"), expected);
    }

    #[test]
    fn tokenize_table_group() {
        let expected = vec![
            kw("select", Keyword::Select, 6),
            sym("*", Symbol::Star, 8),
            kw("from", Keyword::From, 13),
            lit("group", Literal::Identifier, 19),
        ];
        assert_eq!(tokenize("select * from group"), expected);
    }

    #[test]
    fn tokenize_symbol_after_identifier() {
        let expected = vec![
            lit("a", Literal::Identifier, 1),
            sym("+", Symbol::Plus, 2),
            lit("b", Literal::Identifier, 3),
        ];
        assert_eq!(tokenize("a+b"), expected);
    }

    #[test]
    fn tokenize_multiple_symbols_without_spaces() {
        let expected = vec![
            lit("x", Literal::Identifier, 1),
            sym("*", Symbol::Star, 2),
            lit("y", Literal::Identifier, 3),
            sym("+", Symbol::Plus, 4),
            lit("z", Literal::Identifier, 5),
        ];
        assert_eq!(tokenize("x*y+z"), expected);
    }

    #[test]
    fn tokenize_order_as_table_name_at_end_of_query() {
        let expected = vec![
            kw("SELECT", Keyword::Select, 6),
            sym("*", Symbol::Star, 8),
            kw("FROM", Keyword::From, 13),
            lit("order", Literal::Identifier, 19),
        ];
        assert_eq!(tokenize("SELECT * FROM order"), expected);
    }

    #[test]
    fn tokenize_order_as_table_name_with_trailing_space() {
        let expected = vec![
            kw("SELECT", Keyword::Select, 6),
            sym("*", Symbol::Star, 8),
            kw("FROM", Keyword::From, 13),
            lit("order", Literal::Identifier, 19),
        ];
        assert_eq!(tokenize("SELECT * FROM order "), expected);
    }

    #[test]
    fn tokenize_backtick_identifier() {
        let expected = vec![
            kw("SELECT", Keyword::Select, 6),
            lit("my column", Literal::Identifier, 18),
            kw("FROM", Keyword::From, 23),
            lit("t", Literal::Identifier, 25),
        ];
        assert_eq!(tokenize("SELECT `my column` FROM t"), expected);
    }

    #[test]
    fn tokenize_escaped_single_quotes_in_string() {
        let expected = vec![lit("it's", Literal::String, 7)];
        assert_eq!(tokenize("'it''s'"), expected);
    }

    #[test]
    fn tokenize_number_starting_with_dot() {
        let expected = vec![lit(".5", Literal::Double, 2)];
        assert_eq!(tokenize(".5"), expected);
    }

    #[test]
    #[should_panic(expected = "Unexpected character")]
    fn tokenize_unrecognized_character_should_fail() {
        SqlTokenizer::new("SELECT $ FROM t").tokenize();
    }

    #[test]
    fn tokenize_date_literal() {
        let expected = vec![
            kw("date", Keyword::Date, 4),
            lit("1998-12-01", Literal::String, 17),
        ];
        assert_eq!(tokenize("date '1998-12-01'"), expected);
    }

    #[test]
    fn tokenize_interval_literal() {
        let expected = vec![
            kw("interval", Keyword::Interval, 8),
            lit("68 days", Literal::String, 18),
        ];
        assert_eq!(tokenize("interval '68 days'"), expected);
    }
}
