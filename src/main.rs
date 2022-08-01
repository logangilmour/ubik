use std::{
    collections::{HashMap, VecDeque},
    fmt::{self, Display},
    hash::BuildHasherDefault,
    iter::Peekable,
    ptr,
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
    c == '(' || c == ')' || c == '{' || c == '}' || c == '[' || c == ']'
}

#[derive(Debug, Clone)]
pub enum Expr {
    Parse(Vec<Expr>),
    List(Vec<Expr>),
    Symbol(String),
    Number(i32),
}

impl Expr {
    pub fn num(&self) -> i32 {
        if let Expr::Number(n) = self {
            return *n;
        }
        panic!("Must be a number!")
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Parse(a) => {
                for (idx, child) in a.iter().enumerate() {
                    child.fmt(f).unwrap();
                    if idx != a.len() - 1 {
                        write!(f, " ").unwrap();
                    }
                }
            }
            Expr::List(a) => {
                write!(f, "(").unwrap();
                for (idx, child) in a.iter().enumerate() {
                    child.fmt(f).unwrap();
                    if idx != a.len() - 1 {
                        write!(f, " ").unwrap();
                    }
                }
                write!(f, ")").unwrap();
            }
            Expr::Number(n) => {
                write!(f, "{}", n).unwrap();
            }
            Expr::Symbol(s) => {
                write!(f, "{}", s).unwrap();
            }
        }
        Ok(())
    }
}

#[derive(Default, Debug)]
pub struct Env<'a> {
    parent: Option<&'a Env<'a>>,
    symbols: HashMap<String, Expr>,
}

impl<'a> Env<'a> {
    fn store(&mut self, symbol: Expr, value: Expr) {
        if let Expr::Symbol(name) = symbol {
            self.symbols.insert(name.clone(), value);
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
        let mut symbols: Env = Default::default();

        for expr in &exprs {
            if let Expr::List(exprs) = expr {
                if let Expr::Symbol(name) = &exprs[0] {
                    if name.as_str() == "fn" {
                        symbols.store(exprs[1].clone(), Expr::List(exprs[2..].to_vec()));
                        continue;
                    }
                }
            }
            println!("Output: {}", eval_recursive(expr, &mut symbols));
        }
    }
}

pub fn eval_recursive(expr: &Expr, env: &mut Env) -> Expr {
    match expr {
        Expr::List(exprs) => {
            if let Expr::Symbol(name) = &exprs[0] {
                match name.as_str() {
                    "set" => {
                        let val = eval_recursive(&exprs[2], env);
                        env.store(exprs[1].clone(), val);
                        return Expr::List(vec![]);
                    }
                    "=" => {
                        return Expr::Number(
                            (eval_recursive(&exprs[1], env).num()
                                == eval_recursive(&exprs[2], env).num())
                                as i32,
                        )
                    }
                    "eval" => {
                        let mut subenv = Env {
                            parent: Some(env),
                            ..Default::default()
                        };
                        let mut ret = Expr::List(vec![]);
                        for expr in exprs[1..].iter() {
                            ret = eval_recursive(&eval_recursive(expr, &mut subenv), &mut subenv);
                        }
                        return ret;
                    }
                    "quote" => return Expr::List(exprs[1..].to_vec()),
                    "print" => {
                        print!("PRINT: ");
                        for expr in &exprs[1..] {
                            print!("{} ", eval_recursive(expr, env));
                        }
                        println!();
                        return Expr::List(vec![]);
                    }
                    "+" => {
                        return Expr::Number(
                            eval_recursive(&exprs[1], env).num()
                                + eval_recursive(&exprs[2], env).num(),
                        )
                    }
                    "-" => {
                        return Expr::Number(
                            eval_recursive(&exprs[1], env).num()
                                - eval_recursive(&exprs[2], env).num(),
                        )
                    }
                    "*" => {
                        return Expr::Number(
                            eval_recursive(&exprs[1], env).num()
                                * eval_recursive(&exprs[2], env).num(),
                        )
                    }
                    "if" => {
                        let mut clause_index = None;
                        let mut last_evaluated = Expr::Number(0);
                        for (idx, name) in exprs.iter().enumerate().filter_map(|(idx, expr)| {
                            if let Expr::Symbol(name) = expr {
                                if name == "elseif" || name == "else" || name == "if" {
                                    Some((idx, name))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }) {
                            match name.as_str() {
                                "if" | "elseif" => {
                                    assert!(idx + 1 < exprs.len());
                                    if eval_recursive(&exprs[idx + 1], env).num() != 0 {
                                        clause_index = Some(idx + 2);
                                        break;
                                    }
                                }
                                "else" => {
                                    clause_index = Some(idx + 1);
                                    break;
                                }
                                _ => unreachable!("Should never get here"),
                            }
                        }
                        if let Some(idx) = clause_index {
                            for expr in exprs[idx..].iter().take_while(|expr| {
                                !if let Expr::Symbol(name) = expr {
                                    name == "elseif" || name == "else"
                                } else {
                                    false
                                }
                            }) {
                                last_evaluated = eval_recursive(expr, env);
                            }
                        }
                        return last_evaluated;
                    }
                    _ => {
                        let expr = env.load(&exprs[0]);
                        match &expr {
                            Expr::List(fun) => {
                                if let Expr::List(params) = &fun[0] {
                                    let mut fnenv = Env {
                                        parent: None,
                                        ..Default::default()
                                    };
                                    assert!(
                                        params.len() == exprs.len() - 1,
                                        "Must have right number of args"
                                    );
                                    for (idx, param) in params.iter().enumerate() {
                                        assert!(matches!(param, Expr::Symbol(_)));
                                        let val = eval_recursive(&exprs[idx + 1], env);
                                        fnenv.store(param.clone(), val);
                                    }
                                    let mut ret = Expr::List(vec![]);
                                    fnenv.parent = Some(env);
                                    for fnexpr in &fun[1..] {
                                        ret = eval_recursive(fnexpr, &mut fnenv);
                                    }
                                    return ret;
                                }
                            }
                            Expr::Number(val) => return expr,
                            _ => panic!("How'd that get in the symbol table?"),
                        }
                    }
                }
            }
            panic!("Not sure whats going on")
        }
        Expr::Number(_) => expr.clone(),
        Expr::Symbol(_) => env.load(expr),
        Expr::Parse(_) => panic!("Shouldn't get here"),
    }
}

pub fn parse_exp(tokens: &mut Tokenizer, brace: &str) -> Expr {
    let mut items: Vec<Expr> = vec![];
    while let Some(token) = tokens.next() {
        match token {
            "(" | "{" | "[" => {
                let sub_expr = parse_exp(tokens, token);
                assert!(
                    !matches!(sub_expr, Expr::Parse(_)),
                    "Got to the end of the file without closing all braces!"
                );
                items.push(sub_expr);
            }
            ")" | "}" | "]" => {
                if brace == "(" {
                    assert!(token == ")", "Mismatched brace!");
                }
                if brace == "{" {
                    assert!(token == "}", "Mismatched brace!");
                }
                if brace == "[" {
                    assert!(token == "]", "Mismatched brace!");
                }
                if brace == "" {
                    panic!("Extra closing brace!")
                }
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

macro_rules! c_str {
    ($s:expr) => {
        concat!($s, "\0").as_ptr() as *const i8
    };
}

use llvm_sys::{
    analysis::LLVMVerifyFunction, bit_writer::*, prelude::LLVMModuleRef, LLVMContext, LLVMModule,
    LLVMType, LLVMValue,
};
use llvm_sys::{core::*, error_handling::LLVMEnablePrettyStackTrace};
use llvm_sys::{prelude::LLVMBool, LLVMBuilder};

pub struct LL {
    builder: *mut LLVMBuilder,
    module: *mut LLVMModule,
    context: *mut LLVMContext,
}

pub fn compile(expr: &Expr) {
    unsafe {
        let context = LLVMContextCreate();
        let module = LLVMModuleCreateWithNameInContext(c_str!("hello"), context);
        let builder = LLVMCreateBuilderInContext(context);

        let llvm = LL {
            context,
            module,
            builder,
        };

        LLVMEnablePrettyStackTrace();

        // types
        let int_8_type = LLVMInt8TypeInContext(llvm.context);
        let int_8_type_ptr = LLVMPointerType(int_8_type, 0);
        let int_8_array_type = LLVMArrayType(int_8_type, 2);
        let int_32_type = LLVMInt32TypeInContext(llvm.context);

        let int_64_type = LLVMInt64TypeInContext(llvm.context);

        // puts function
        //let puts_function_args_type = [int_8_type_ptr].as_ptr() as *mut _;

        //let puts_function_type = LLVMFunctionType(int_32_type, puts_function_args_type, 1, 0);
        //let puts_function = //LLVMAddFunction(llvm.module, c_str!("puts"), puts_function_type);
        // end

        let mut symbols: CEnv = Default::default();
        symbols.initialize_builtins();

        if let Expr::Parse(exprs) = expr {
            for expr in exprs {
                if let Expr::List(fdef) = expr {
                    match &fdef[0] {
                        Expr::Symbol(name) => match name.as_str() {
                            "fn_extern" => {
                                compile_fn_proto(expr, &mut symbols, &llvm);
                            }
                            _ => (),
                        },
                        _ => (),
                    }
                }
            }
        }

        // main function
        let main_function_type = LLVMFunctionType(int_32_type, ptr::null_mut(), 0, 0);
        let main_function = LLVMAddFunction(module, c_str!("main"), main_function_type);

        let entry = LLVMAppendBasicBlockInContext(context, main_function, c_str!("entry"));

        LLVMPositionBuilderAtEnd(builder, entry);

        let alo = LLVMBuildAlloca(builder, int_8_array_type, c_str!("summed"));

        let mut ret = None;

        if let Expr::Parse(exprs) = expr {
            for expr in exprs {
                if let Expr::List(fdef) = expr {
                    match &fdef[0] {
                        Expr::Symbol(name) => match name.as_str() {
                            "fn_extern" => (),
                            _ => {
                                if let Some(cval) = compile_recursive(expr, &mut symbols, &llvm) {
                                    ret = Some(cval);
                                }
                            }
                        },
                        _ => (),
                    }
                }
            }
        }
        let summed = ret;

        let small = LLVMBuildIntCast(builder, summed.unwrap().val, int_8_type, c_str!("lil"));

        let zoffi64 = LLVMConstInt(int_64_type, 0, 0);
        let zoffi32 = LLVMConstInt(int_32_type, 0, 0);
        let ptrargs = [zoffi64, zoffi32];

        let numaddr = LLVMBuildGEP2(
            builder,
            int_8_array_type,
            alo,
            ptrargs.as_ptr() as *mut _,
            2,
            c_str!("null"),
        );

        LLVMBuildStore(builder, small, numaddr);

        let ooffi32 = LLVMConstInt(int_32_type, 1, 0);
        let ptrargs1 = [zoffi64, ooffi32];

        let endaddr = LLVMBuildGEP2(
            builder,
            int_8_array_type,
            alo,
            ptrargs1.as_ptr() as *mut _,
            2,
            c_str!("null"),
        );

        LLVMBuildStore(builder, LLVMConstInt(int_8_type, 0, 0), endaddr);

        let puts_function_args = [
            //LLVMBuildPointerCast(
            //builder, // cast [14 x i8] type to int8 pointer
            numaddr, // build hello string constant
                    //int_8_type_ptr,
                    //c_str!("0"),
                    //)
        ]
        .as_ptr() as *mut _;

        let puts_function = if let FunImpl::User(f) = symbols.functions.get("puts").unwrap().val {
            f
        } else {
            panic!("Needs a real one")
        };

        let puts_function_type = eval_type(
            &symbols.functions.get("puts").unwrap()._type,
            &symbols,
            &llvm,
        );

        LLVMBuildCall2(
            builder,
            puts_function_type,
            puts_function,
            puts_function_args,
            1,
            c_str!("i"),
        );
        let fun = LLVMBuildRet(builder, LLVMConstInt(int_32_type, 0, 0));
        // end
        LLVMVerifyFunction(
            main_function,
            llvm_sys::analysis::LLVMVerifierFailureAction::LLVMAbortProcessAction,
        );
        //LLVMDumpModule(module); // dump module to STDOUT
        LLVMPrintModuleToFile(module, c_str!("hello.ll"), ptr::null_mut());

        // clean memory
        LLVMDisposeBuilder(builder);
        LLVMDisposeModule(module);
        LLVMContextDispose(context);
    }
}

impl std::fmt::Debug for FunImpl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User(arg0) => f.debug_tuple("User").finish(),
            Self::Builtin(arg0) => f.debug_tuple("Builtin").finish(),
        }
    }
}

pub fn compile_fn_proto(expr: &Expr, env: &mut CEnv, llvm: &LL) -> *mut LLVMValue {
    if let Expr::List(exprs) = expr {
        if let [Expr::Symbol(_fntype), Expr::Symbol(name), Expr::List(params), ret] = &exprs[..] {
            let fntype = Expr::List(vec![
                Expr::Symbol(":fn".to_string()),
                Expr::List(
                    params
                        .iter()
                        .skip(1)
                        .step_by(2)
                        .cloned()
                        .collect::<Vec<_>>(),
                ),
                ret.clone(),
            ]);

            let ft = eval_type(&fntype, env, llvm);
            let mut cname = name.clone().into_bytes();
            cname.push(0);
            let fun = unsafe { LLVMAddFunction(llvm.module, cname.as_ptr() as *const i8, ft) };
            env.functions.insert(
                name.clone(),
                CFun {
                    val: FunImpl::User(fun),
                    _type: fntype,
                },
            );

            return fun;
        }
    }
    panic!("Not a function!")
}

pub fn eval_type(expr: &Expr, env: &CEnv, llvm: &LL) -> *mut LLVMType {
    match expr {
        Expr::List(exprs) => {
            if exprs.is_empty() {
                return unsafe { LLVMVoidType() };
            }
            if let Expr::Symbol(name) = &exprs[0] {
                match name.as_str() {
                    "*" => return unsafe { LLVMPointerType(eval_type(&exprs[1], env, llvm), 0) },
                    ":fn" => {
                        if let Expr::List(params) = &exprs[1] {
                            let pt = params
                                .iter()
                                .map(|p| eval_type(p, env, llvm))
                                .collect::<Vec<_>>();
                            let rt = eval_type(&exprs[2], env, llvm);

                            return unsafe {
                                LLVMFunctionType(rt, pt.as_ptr() as *mut _, pt.len() as u32, 0)
                            };
                        }
                    }

                    other => panic!("Unknown type fn {}", other),
                }
            }
        }
        Expr::Symbol(name) => match name.as_str() {
            ":i32" => return unsafe { LLVMInt32TypeInContext(llvm.context) },
            ":u8" => return unsafe { LLVMInt8TypeInContext(llvm.context) },
            _ => panic!("Unknown type"),
        },
        _ => panic!("No other stuff in here yet."),
    }
    panic!("Nothing Matched type!")
}

pub fn compile_recursive(expr: &Expr, env: &mut CEnv, llvm: &LL) -> Option<CVal> {
    match expr {
        Expr::List(exprs) => {
            if let Expr::Symbol(name) = &exprs[0] {
                match name.as_str() {
                    "var" => {
                        let sym = if let Expr::Symbol(name) = &exprs[1] {
                            name
                        } else {
                            panic!("var expects a symbol as first arg");
                        };

                        let assigned = compile_recursive(&exprs[2], env, llvm).unwrap();
                        let _type = eval_type(&assigned._type, env, llvm);
                        let loc = unsafe { LLVMBuildAlloca(llvm.builder, _type, c_str!("alloca")) };
                        unsafe { LLVMBuildStore(llvm.builder, assigned.val, loc) };
                        env.store_var(
                            sym,
                            CVal {
                                val: loc,
                                _type: assigned._type.clone(),
                            },
                        );
                        return None;
                    }
                    _ => (),
                }

                let args = &exprs[1..]
                    .iter()
                    .map(|expr| compile_recursive(expr, env, llvm))
                    .collect::<Vec<_>>();
                let params = Expr::List(
                    args.iter()
                        .map(|cval| cval.as_ref().unwrap()._type.clone())
                        .collect::<Vec<_>>(),
                );
                let values = args
                    .iter()
                    .map(|cval| cval.as_ref().unwrap().val)
                    .collect::<Vec<_>>();
                let fun = env.lookup_fn(name, &params);
                let funtype = if let Expr::List(exprs) = &fun._type {
                    exprs[2].clone()
                } else {
                    panic!("Bad shaped function")
                };

                if let FunImpl::Builtin(fun) = fun.val {
                    return Some(CVal {
                        val: fun(&values, llvm, env),
                        _type: funtype,
                    });
                } else {
                    panic!("oh no")
                };
            }
            panic!("Not sure whats going on")
        }
        Expr::Number(val) => unsafe {
            return Some(CVal {
                val: LLVMConstInt(LLVMInt32TypeInContext(llvm.context), *val as u64, 0),
                _type: Expr::Symbol(":i32".to_string()),
            });
        },
        Expr::Symbol(name) => {
            let var = env.load_var(name);
            return Some(CVal {
                val: unsafe {
                    LLVMBuildLoad2(
                        llvm.builder,
                        eval_type(&var._type, env, llvm),
                        var.val,
                        c_str!("load"),
                    )
                },
                _type: var._type.clone(),
            });
        }
        Expr::Parse(_) => panic!("Shouldn't get here"),
    }
}

#[derive(Default, Debug)]
pub struct CEnv {
    //parent: Option<&'a Env<'a>>,
    functions: HashMap<String, CFun>,
    vars: HashMap<String, CVal>,
}

#[derive(Debug, Clone)]
pub struct CVal {
    val: *mut LLVMValue,
    _type: Expr,
}

#[derive(Debug)]
pub struct CFun {
    val: FunImpl,
    _type: Expr,
}

type Op = fn(&[*mut LLVMValue], &LL, &mut CEnv) -> *mut LLVMValue;

pub enum FunImpl {
    User(*mut LLVMValue),
    Builtin(Op),
}

fn fullname(name: &str, params: &Expr) -> String {
    let mut full_name = name.to_string();
    full_name.push_str(&params.to_string());

    full_name
}

impl CEnv {
    pub fn initialize_builtins(&mut self) {
        self.store_builtin("+", "(:fn  [:i32 :i32] :i32)", |exprs, llvm, env| unsafe {
            LLVMBuildAdd(llvm.builder, exprs[0], exprs[1], c_str!("add"))
        });

        self.store_builtin("-", "(:fn  [:i32 :i32] :i32)", |exprs, llvm, env| unsafe {
            LLVMBuildSub(llvm.builder, exprs[0], exprs[1], c_str!("add"))
        });

        self.store_builtin("*", "(:fn  [:i32 :i32] :i32)", |exprs, llvm, env| unsafe {
            LLVMBuildMul(llvm.builder, exprs[0], exprs[1], c_str!("add"))
        });

        self.store_builtin("/", "(:fn  [:i32 :i32] :i32)", |exprs, llvm, env| unsafe {
            LLVMBuildSDiv(llvm.builder, exprs[0], exprs[1], c_str!("add"))
        });
    }

    pub fn store_var(&mut self, name: &str, var: CVal) {
        self.vars.insert(name.to_string(), var);
    }

    pub fn load_var(&self, name: &str) -> &CVal {
        println!("Loading var {} from {:?}", name, self.vars);
        self.vars.get(name).unwrap()
    }

    pub fn lookup_fn(&self, name: &str, params: &Expr) -> &CFun {
        let full_name = fullname(name, params);
        println!("Looking up {} in {:?}", full_name, self.functions);
        self.functions.get(&full_name).unwrap()
    }

    pub fn store_builtin(&mut self, name: &str, _type: &str, fun: Op) {
        let _type = parse(_type);
        if let Expr::List(exprs) = &_type {
            let full_name = fullname(name, &exprs[1]);
            self.functions.insert(
                full_name,
                CFun {
                    val: FunImpl::Builtin(fun),
                    _type,
                },
            );
        } else {
            panic!("Bad def: {} {}", name, _type);
        }
    }
}

pub fn parse(src: &str) -> Expr {
    if let Expr::Parse(mut exprs) = parse_exp(&mut Tokenizer::new(src), "") {
        return exprs.pop().unwrap();
    }
    panic!("No expressions!");
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_llvm() {
        println!("Test: {}", parse("[fn [i32 i32] i8]"));
        compile(&parse_exp(
            &mut Tokenizer::new(
                "
        {fn_extern puts (s [* :u8]) :i32}
        (var A 2)
        (+ A (+ 48 (* 2 3)))",
            ),
            "",
        ));
    }

    use std::ptr;
    use std::str::from_utf8;

    use crate::compile;
    use crate::eval;
    use crate::parse;
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
            eval(parse_exp(
                &mut Tokenizer::new(
                    "{fn add (a b)
                     (+ a b)
                }
                {fn add_twice (a b) 
                    {set test 44}
                    (add 
                        test
                        (+ a b))
                    }

                {fn pow (a b)
                    (print a b)
                    {if (= b 1)
                        a
                    else
                        (* a (pow a (- b 1)))
                    }
                }

                (pow 2 5)

                (eval (quote + 1 100))

                {set threshold 48}
                {if (= (add_twice 2 3) threshold)
                    11 
                elseif (= (add_twice 2 3) 5)
                    
                    12 
                else 
                    2
                }
                
                threshold
                "
                ),
                ""
            ))
        );

        //println!("{:?}", eval(&data, &types));
    }
}
