use std::{
    collections::HashMap,
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

#[derive(Debug)]
pub struct Type {
    size: usize,
    alignment: usize,
    elements: usize,
}

pub fn parse(src: &str) -> (Vec<u8>, Vec<Type>) {
    let mut tokens = Tokenizer::new(src);
    let mut types = vec![];
    let mut data = vec![];
    parse_exp(&mut tokens, &mut data, &mut types);
    (data, types)
}

#[derive(Debug)]
pub enum Expr {
    E(Vec<Expr>),
    V(i32),
    S(String),
}

pub fn eval(data: &[u8], types: &[Type]) -> i32 {
    let mut intermediates = vec![];
    let mut offset = 0;
    let mut evaluating: Vec<(usize, usize)> = vec![];

    let symbols: HashMap<String, i32> = [("+".to_owned(), 0), ("-".to_owned(), 1)]
        .into_iter()
        .collect();

    for (i, t) in types.iter().enumerate() {
        if t.elements == 0 {
            let s = unsafe { from_utf8_unchecked(&data[offset..offset + t.size]) };
            offset += t.size;
            if let Ok(num) = s.parse::<i32>() {
                intermediates.push(num);
            } else {
                intermediates.push(symbols[s]);
            }
            evaluating.last_mut().unwrap().1 += 1;
        } else {
            evaluating.push((i, 0));
        }
        while let Some(ev) = evaluating.last() {
            if types[ev.0].elements == ev.1 {
                assert!(ev.1 == 3, "Wrong number of args for fn");
                let args = (
                    intermediates.pop().unwrap(),
                    intermediates.pop().unwrap(),
                    intermediates.pop().unwrap(),
                );
                intermediates.push(match args.2 {
                    0 => args.1 + args.0,
                    1 => args.1 - args.0,
                    _ => panic!("Unknown op"),
                });
                evaluating.pop();
                if let Some(parent) = evaluating.last_mut() {
                    parent.1 += 1;
                }
            } else {
                break;
            }
        }
    }
    intermediates.pop().unwrap()
}

pub fn parse_exp(tokens: &mut Tokenizer, data: &mut Vec<u8>, types: &mut Vec<Type>) {
    let mut building_types: Vec<usize> = vec![];
    for token in tokens.by_ref() {
        println!("Parsing: {}", token);
        match token {
            "(" => {
                building_types.push(types.len());
                types.push(Type {
                    size: 0,
                    alignment: 1,
                    elements: 0,
                });
            }
            ")" => {
                let finished_type_idx = building_types.pop().unwrap();

                if let Some(bt) = building_types.last() {
                    println!("FINALIZING");
                    types[*bt].elements += 1;
                    types[*bt].size += types[finished_type_idx].size;
                    types[*bt].alignment =
                        types[*bt].alignment.max(types[finished_type_idx].alignment);
                }
            }
            s => {
                let bytes = s.as_bytes();
                data.extend(bytes);
                let finished_type = Type {
                    elements: 0,
                    size: bytes.len(),
                    alignment: 1,
                };
                if let Some(bt) = building_types.last() {
                    types[*bt].elements += 1;
                    types[*bt].size += finished_type.size;
                    types[*bt].alignment = types[*bt].alignment.max(finished_type.alignment);
                }
                types.push(finished_type);
            }
        }
    }
    assert!(building_types.is_empty());
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

        let (data, types) = parse("(- (- 2 3) 1)");
        println!("{:?}", eval(&data, &types));
    }
}
