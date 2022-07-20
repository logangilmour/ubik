use std::{
    iter::Peekable,
    str::{from_utf8_unchecked, CharIndices},
};

fn main() {
    println!("Hello, world!");
}

pub struct Tokenizer<'a> {
    src_chars: Peekable<CharIndices<'a>>,
    src_bytes: &'a [u8],
}

impl<'a> Tokenizer<'a> {
    pub fn new(src: &'a str) -> Self {
        Tokenizer {
            src_chars: src.char_indices().peekable(),
            src_bytes: src.as_bytes(),
        }
    }
}

pub fn is_brace(c: char) -> bool {
    c == '(' || c == ')'
}

pub fn parse(src: &str) -> Expr {
    let mut tokens = Tokenizer::new(src);
    tokens.next();
    parse_exp(&mut tokens)
}

#[derive(Debug)]
pub enum Expr {
    E(Vec<Expr>),
    V(i32),
    S(String),
}

pub fn eval(e: &Expr) -> i32 {
    match e {
        Expr::E(vals) => match &vals[0] {
            Expr::S(op) => match op.as_str() {
                "+" => eval(&vals[1]) + eval(&vals[2]),
                "*" => eval(&vals[1]) * eval(&vals[2]),
                "-" => eval(&vals[1]) - eval(&vals[2]),
                "☹️" => eval(&vals[1]) * 100,
                _ => panic!("Unknown op"),
            },
            _ => panic!("First thing must be op"),
        },
        Expr::V(val) => *val,
        Expr::S(val) => val.parse::<i32>().unwrap(),
    }
}

pub fn parse_exp(tokens: &mut Tokenizer) -> Expr {
    let mut e: Vec<Expr> = Default::default();
    loop {
        let token = tokens.next().unwrap().to_owned();

        match token.as_str() {
            "(" => e.push(parse_exp(tokens)),
            ")" => return Expr::E(e),
            _ => e.push(Expr::S(token)),
        }
    }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let mut token_width;
        let token_start;
        loop {
            if let Some((offset, first_char)) = self.src_chars.next() {
                let width = first_char.len_utf8();
                if first_char.is_ascii_whitespace() {
                    continue;
                }
                if is_brace(first_char) {
                    unsafe {
                        return Some(from_utf8_unchecked(&self.src_bytes[offset..offset + width]));
                    }
                }
                token_start = offset;
                token_width = width;
                break;
            } else {
                return None;
            }
        }
        while let Some((_, next_char)) = self.src_chars.peek() {
            let width = next_char.len_utf8();

            if next_char.is_ascii_whitespace() || is_brace(*next_char) {
                break;
            }

            token_width += width;
            self.src_chars.next();
        }
        unsafe {
            Some(from_utf8_unchecked(
                &self.src_bytes[token_start..token_start + token_width],
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{eval, parse, Tokenizer};

    #[test]
    fn test_tokenize() {
        let test = "(☹️️)";
        let parse_test = vec!["(", "☹️️", ")"];
        assert!(Tokenizer::new(test)
            .zip(parse_test.iter())
            .all(|(a, b)| (&a).eq(b)));

        println!("{:?}", eval(&parse("(☹️(- (+ (* 3 2) (* 2 2)) 1))")));
    }
}
