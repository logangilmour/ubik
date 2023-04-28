use std::{
    iter::Peekable,
    str::{from_utf8_unchecked, CharIndices},
};

use num_derive::FromPrimitive;
fn main() {
    let tokens = Tokenizer::new("1 2 + 1 +")
        .flat_map(|token| match token._type {
            TokenType::Number => vec![Op::PUSHI as u32, token.src.parse::<u32>().unwrap()],
            TokenType::Name => match token.src {
                "+" => vec![Op::ADDI as u32],
                _ => panic!(),
            },
        })
        .collect::<Vec<_>>();
    let mut prog_counter = 0;
    let mut stack = vec![];
    while prog_counter < tokens.len() {
        let op: Op = num::FromPrimitive::from_u32(tokens[prog_counter]).unwrap();
        match op {
            Op::ADDI => {
                let v1 = stack.pop().unwrap();
                let v2 = stack.pop().unwrap();

                stack.push(v1 + v2);
            }
            Op::PUSHI => {
                prog_counter += 1;
                stack.push(tokens[prog_counter]);
            }
        }
        prog_counter += 1;
    }
    println!("STACK: {:?}", stack);
}

#[repr(u32)]
#[derive(FromPrimitive)]
pub enum Op {
    ADDI,
    PUSHI,
}

#[derive(Debug)]
pub struct Token<'a> {
    _type: TokenType,
    src: &'a str,
}

#[derive(Debug)]
pub enum TokenType {
    Number,
    Name,
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

impl<'a> Iterator for Tokenizer<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut token_width;
        let token_start;
        let token_type;
        loop {
            if let Some((offset, first_char)) = self.src_chars.next() {
                let width = first_char.len_utf8();
                if first_char.is_ascii_whitespace() {
                    continue;
                }
                if first_char.is_numeric() {
                    token_type = TokenType::Number;
                } else {
                    token_type = TokenType::Name;
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

            if next_char.is_ascii_whitespace() {
                break;
            }

            token_width += width;
            self.src_chars.next();
        }
        unsafe {
            Some(Token {
                _type: token_type,
                src: from_utf8_unchecked(&self.src_bytes[token_start..token_start + token_width]),
            })
        }
    }
}
