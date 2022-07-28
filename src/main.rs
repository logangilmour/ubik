use std::{
    collections::{HashMap, VecDeque},
    iter::Peekable,
    str::{from_utf8_unchecked, CharIndices},
};

extern crate llvm_sys;

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
    c == '(' || c == ')' || c == '{' || c == '}'
}

#[derive(Debug, Clone)]
pub enum Expr {
    Parse(Vec<Expr>),
    List(Vec<Expr>),
    Symbol(String),
    Number(i32),
}

#[derive(Default)]
pub struct Env<'a> {
    parent: Option<&'a Env<'a>>,
    symbols: HashMap<String, Expr>,
}

impl<'a> Env<'a> {
    fn store(&mut self, symbol: &'a Expr, value: &'a Expr) {
        if let Expr::Symbol(name) = symbol {
            self.symbols.insert(name.clone(), value.clone());
        } else {
            panic!("Name must be symbol.")
        }
    }

    fn load(&self, symbol: &'a Expr) -> Expr {
        if let Expr::Symbol(name) = symbol {
            if let Some(stored) = self.symbols.get(name.as_str()) {
                stored.clone()
            } else if let Some(parent) = self.parent {
                parent.load(symbol)
            } else {
                panic!("No such symbol")
            }
        } else {
            panic!("Name must be symbol")
        }
    }
}

pub fn eval(expr: Expr) {
    if let Expr::Parse(exprs) = expr {
        let mut symbols = Default::default();

        for expr in &exprs {
            println!("Output: {}", eval_recursive(expr, &mut symbols));
        }
    }
}
pub fn eval_recursive<'a>(expr: &'a Expr, env: &Env) -> i32 {
    match expr {
        Expr::List(exprs) => {
            if let Expr::Symbol(name) = &exprs[0] {
                match name.as_str() {
                    /*
                    "fn" => {
                        if let Expr::Symbol(name) = &exprs[1] {
                            symbols.entry(name).or_default().push(expr);
                            return 0;
                        } else {
                            panic!("Must have name!")
                        }
                    }
                    */
                    "+" => return eval_recursive(&exprs[1], env) + eval_recursive(&exprs[2], env),
                    "-" => return eval_recursive(&exprs[1], env) - eval_recursive(&exprs[2], env),
                    "*" => return eval_recursive(&exprs[1], env) * eval_recursive(&exprs[2], env),
                    other => {
                        /*
                        let v = env.load(&exprs[0]);

                        if let Some(stored) = symbols.get(other) {
                            if let Some(Expr::Number(val)) = v {
                                return *val;
                            } else if let Some(Expr::List(stored_exprs)) = stored.last() {
                                if let Expr::Symbol(name) = &stored_exprs[0] {
                                    match name.as_str() {
                                        "fn" => {
                                            if let Expr::List(params) = &stored_exprs[2] {
                                                assert!(
                                                    params.len() == exprs.len() - 1,
                                                    "Fn must have correct number of args"
                                                );
                                                for param in params {
                                                    if let Expr::Symbol(name) = param {
                                                        symbols
                                                            .entry(name)
                                                            .or_default()
                                                            .push(&Number(eval()))
                                                    } else {
                                                        panic!("parameter def must be symbols")
                                                    }
                                                }
                                            } else {
                                                panic!("Must provide args")
                                            }
                                        }
                                        _ => panic!("Only valid stored expr is a fn!"),
                                    }
                                }
                            } else {
                                panic!("WTF")
                            }
                        } else {
                            panic!("unkown symbol!")
                        }
                        */
                    }
                }
            }
            panic!("Not sure whats going on")
        }
        Expr::Number(val) => *val,
        Expr::Symbol(name) => {
            /*
            env.load(expr) {
                if let Some(Expr::Number(val)) = defs.last() {
                    *val
                } else {
                    panic!("Not a number!")
                }
            } else {
                panic!("No such symbol!")
            }
            */
            0
        }
        Expr::Parse(_) => panic!("Shouldn't get here"),
    }
}

/*
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
*/

pub fn parse_exp(tokens: &mut Tokenizer) -> Expr {
    let mut items: Vec<Expr> = vec![];
    while let Some(token) = tokens.next() {
        match token {
            "(" | "{" => {
                let sub_expr = parse_exp(tokens);
                assert!(!matches!(sub_expr, Expr::Parse(_)), "Uh oh");
                items.push(sub_expr);
            }
            ")" | "}" => {
                return Expr::List(items);
            }
            s => {
                if s.chars().next().unwrap().is_numeric() {
                    items.push(Expr::Number(s.parse().expect("Bad number")));
                } else {
                    items.push(Expr::Symbol(s.to_owned()));
                }
            }
        }
    }
    Expr::Parse(items)
}

/*
pub fn parse(src: &str) -> Expr {
    let mut tokens = Tokenizer::new(src);
    let mut types = vec![];
    let mut data = vec![];
    let mut building_types: Vec<usize> = vec![];
    for token in tokens.by_ref() {
        match token {
            "(" | "{" => {
                building_types.push(types.len());
                types.push(Type {
                    kind: if token == "(" {
                        Kind::Expr
                    } else {
                        Kind::Macro
                    },
                    size: 0,
                    alignment: 1,
                    elements: 0,
                    total_elements: 0,
                });
            }
            ")" | "}" => {
                let finished_type_idx = building_types.pop().unwrap();

                if let Some(bt) = building_types.last() {
                    println!("FINALIZING");
                    types[*bt].elements += 1;
                    types[*bt].size += types[finished_type_idx].size;
                    types[*bt].total_elements += types[finished_type_idx].total_elements + 1;
                    types[*bt].alignment =
                        types[*bt].alignment.max(types[finished_type_idx].alignment);
                }
                assert!(
                    token
                        == if types[finished_type_idx].kind == Kind::Expr {
                            ")"
                        } else {
                            "}"
                        },
                    "Mismatched braces!"
                );
            }
            s => {
                if s.chars().next().unwrap().is_numeric() {}
                let bytes = s.as_bytes();
                data.extend(bytes);
                let finished_type = Type {
                    kind: Kind::Atom,
                    elements: 0,
                    total_elements: 0,
                    size: bytes.len(),
                    alignment: 1,
                };
                if let Some(bt) = building_types.last() {
                    types[*bt].elements += 1;
                    types[*bt].total_elements += 1;
                    types[*bt].size += finished_type.size;
                    types[*bt].alignment = types[*bt].alignment.max(finished_type.alignment);
                }
                types.push(finished_type);
            }
        }
    }
    assert!(building_types.is_empty());

    let mut deferred: VecDeque<usize> = Default::default();

    let mut offsets = vec![];
    let mut max_visited = 0;
    while max_visited < types.len() {
        deferred.push_back(max_visited);

        while let Some(cur_idx) = deferred.pop_front() {
            {
                let mut count = 0;
                let mut offset = cur_idx + 1;
                while count < types[cur_idx].elements {
                    let child = &types[offset];
                    offsets.push(offset);

                    if child.kind != Kind::Atom {
                        deferred.push_back(offset);
                        offset += child.total_elements + 1;
                    } else {
                        offset += 1;
                    }
                    max_visited = max_visited.max(offset);
                    count += 1;
                }
            }
        }
    }

    /*
    let mut child_counts: VecDeque<(usize, usize)> = Default::default();
    let mut child_offsets = vec![];
    let mut level = 0;
    let mut cur_type = 0;
    let type_it = types.iter().enumerate().skip(1);
    child_counts.push_back((0, 0));
    while !child_counts.is_empty() {
        let (t_idx, t) = type_it.next().unwrap();
        if t.kind != Kind::Atom {
            child_counts.push_back((t_idx, 0));
            level += 1;
        } else {
            let cur = child_counts.back_mut().unwrap();
            cur.1 += 1;
            if types[cur.0].elements == cur.1 {
                level -= 1;
            }
        }

        let (idx, count) = child_counts.front().unwrap();
        if *count == types[*idx].elements {
            child_counts.pop_front();
        }
        if child_count.last().unwrap() == types[*intermediate_offsets.front().unwrap()].elements {
            intermediate_offsets.pop_front();
            child_count = 0;
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
    */
    (data, types, offsets)
}
*/

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
    use std::ptr;
    use std::str::from_utf8;

    use crate::eval;
    use crate::parse_exp;
    use crate::Tokenizer;

    #[test]
    fn test_tokenize() {
        let test = "(☹️️)";
        let parse_test = vec!["(", "☹️️", ")"];
        assert!(Tokenizer::new(test)
            .zip(parse_test.iter())
            .all(|(a, b)| (&a).eq(b)));

        println!(
            "{:?}",
            eval(parse_exp(&mut Tokenizer::new("(+ 1 2) (+ 2 3)")))
        );

        //println!("{:?}", eval(&data, &types));
    }

    macro_rules! c_str {
        ($s:expr) => {
            concat!($s, "\0").as_ptr() as *const i8
        };
    }

    use llvm_sys::bit_writer::*;
    use llvm_sys::core::*;
    use llvm_sys::prelude::LLVMBool;

    #[test]
    fn test_llvm() {
        unsafe {
            let context = LLVMContextCreate();
            let module = LLVMModuleCreateWithNameInContext(c_str!("hello"), context);
            let builder = LLVMCreateBuilderInContext(context);

            // types
            let int_8_type = LLVMInt8TypeInContext(context);
            let int_8_type_ptr = LLVMPointerType(int_8_type, 0);
            let int_32_type = LLVMInt32TypeInContext(context);

            // puts function
            let puts_function_args_type = [int_8_type_ptr].as_ptr() as *mut _;

            let puts_function_type = LLVMFunctionType(int_32_type, puts_function_args_type, 1, 0);
            let puts_function = LLVMAddFunction(module, c_str!("puts"), puts_function_type);
            // end

            // main function
            let main_function_type = LLVMFunctionType(int_32_type, ptr::null_mut(), 0, 0);
            let main_function = LLVMAddFunction(module, c_str!("main"), main_function_type);

            let entry = LLVMAppendBasicBlockInContext(context, main_function, c_str!("entry"));
            LLVMPositionBuilderAtEnd(builder, entry);

            let puts_function_args = [LLVMBuildPointerCast(
                builder, // cast [14 x i8] type to int8 pointer
                LLVMBuildGlobalString(builder, c_str!("Hello, World!"), c_str!("hello")), // build hello string constant
                int_8_type_ptr,
                c_str!("0"),
            )]
            .as_ptr() as *mut _;

            LLVMBuildCall(builder, puts_function, puts_function_args, 1, c_str!("i"));
            LLVMBuildRet(builder, LLVMConstInt(int_32_type, 0, 0));
            // end

            //LLVMDumpModule(module); // dump module to STDOUT
            LLVMPrintModuleToFile(module, c_str!("hello.ll"), ptr::null_mut());

            // clean memory
            LLVMDisposeBuilder(builder);
            LLVMDisposeModule(module);
            LLVMContextDispose(context);
        }
    }
}
