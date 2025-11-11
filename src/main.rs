use anyhow::{Result, anyhow};
use std::io::stdin;

#[derive(PartialEq, Debug)]
enum Token {
    EndOfFile,
    Def,
    Extern,
    Identifier { name: String },
    Number { value: f64 },
    Other(char),
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
        self.pos += 1;
        if self.pos == self.input.len() {
            return None;
        }
        Some(self.input[self.pos])
    }

    fn get_token(&mut self) -> Result<Token> {
        if self.pos == self.input.len() {
            return Ok(Token::EndOfFile);
        }

        let mut current = self.input[self.pos];
        while current.is_whitespace() {
            let next = self.get_char();
            current = match next {
                Some(c) => c,
                None => {
                    return Ok(Token::EndOfFile);
                }
            }
        }

        if current.is_alphabetic() {
            let mut ident = String::new();
            while current.is_alphabetic() {
                ident += &current.to_string();
                current = match self.get_char() {
                    Some(c) => c,
                    None => {
                        break;
                    }
                }
            }
            if &ident == "extern" {
                return Ok(Token::Extern);
            }
            if &ident == "def" {
                return Ok(Token::Def);
            }
            return Ok(Token::Identifier { name: ident });
        }

        if current.is_ascii_digit() {
            let mut x = String::new();
            while current.is_ascii_digit() || current == '.' {
                x += &current.to_string();
                current = match self.get_char() {
                    Some(c) => c,
                    None => {
                        break;
                    }
                }
            }
            let value: f64 = x.parse()?;
            return Ok(Token::Number { value });
        }

        self.pos += 1;
        Ok(Token::Other(current))
    }
}

fn main() -> Result<()> {
    let mut lexer = Lexer::from_stdin()?;
    loop {
        let t = lexer.get_token()?;
        println!("{:#?}", t);
        if t == Token::EndOfFile {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_token() {
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
        assert_eq!(
            Lexer::from_str("1").get_token().unwrap(),
            Token::Number { value: 1.0 }
        );
        assert_eq!(
            Lexer::from_str("1.234").get_token().unwrap(),
            Token::Number { value: 1.234 }
        );

        assert!(Lexer::from_str("1.234.34").get_token().is_err());
    }
}
