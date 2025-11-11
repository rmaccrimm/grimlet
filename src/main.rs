use anyhow::{Result, anyhow};
use std::io::stdin;

#[derive(PartialEq, Debug)]
enum Token {
    EndOfFile,
    Def,
    Extern,
    Identifier { name: String },
    Number { value: f64 },
    Other { ascii_value: u32 },
}

struct Lexer {
    input: Vec<char>,
    pos: usize,
}

impl Lexer {
    fn from_stdin() -> Result<Self> {
        let mut buf = String::new();
        stdin().read_line(&mut buf)?;
        Ok(Self {
            input: buf.chars().collect(),
            pos: 0,
        })
    }

    fn from_str(s: &str) -> Self {
        Self {
            input: s.chars().collect(),
            pos: 0,
        }
    }

    fn get_char(&mut self) -> Option<char> {
        if self.pos == self.input.len() {
            return None;
        }
        let c = self.input[self.pos];
        self.pos += 1;
        Some(c)
    }

    fn get_token(&mut self) -> Result<Token> {
        let mut last_char = ' ';
        while last_char.is_whitespace() {
            let next = self.get_char();
            last_char = match next {
                Some(c) => c,
                None => {
                    return Ok(Token::EndOfFile);
                }
            }
        }

        if last_char.is_alphabetic() {
            let mut ident = String::new();
            while last_char.is_alphabetic() {
                ident += &last_char.to_string();
                last_char = match self.get_char() {
                    Some(c) => c,
                    None => {
                        break;
                    }
                }
            }
            return Ok(Token::Identifier { name: ident });
        }
        Err(anyhow!("Failed to get token"))
    }
}

fn main() -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_token_identifiers() {
        assert_eq!(
            Lexer::from_str("a").get_token().unwrap(),
            Token::Identifier { name: "a".into() }
        );
        assert_eq!(
            Lexer::from_str("    anIdentifier then others 1.234")
                .get_token()
                .unwrap(),
            Token::Identifier {
                name: "anIdentifier".into()
            }
        );
        assert_eq!(Lexer::from_str(" ").get_token().unwrap(), Token::EndOfFile);
    }
}
