use clap::Parser;
use std::fmt;
use std::fs::File;
use std::io::{prelude::*, BufReader};
use std::path::{Path, PathBuf};

macro_rules! die {
    ($code:expr, $($arg:tt)*) => {{
        ::std::eprintln!($($arg)*);
        ::std::process::exit($code)
    }}
}

fn main() {
    let args = Args::parse();
    let Some(file_name) = args.file_names.get(0) else {
        die!(1, "must provide filename");
    };
    let csv = match CSV::from_file(file_name, !args.no_headers) {
        Ok(csv) => csv,
        Err(e) => die!(1, "{e}"),
    };
    if args.mean.as_ref().map(|s| s != "*").unwrap_or(true) {
        println!("{}", csv.headers.join(","));
        println!("{}", csv.types.iter().map(|t| t.to_type_str()).collect::<Vec<_>>().join(","));
        csv.values
            .iter()
            .for_each(|row| {
                println!("{}", row.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
            });
        //csv.values.iter().for_each(|r| println!("{:?}", r));
        return;
    }
}

#[allow(dead_code)]
struct CSVParser {
    //
}

struct CSV {
    headers: Vec<String>,
    types: Vec<Value>,
    values: Vec<Vec<Value>>,
}

impl CSV {
    fn new() -> Self {
        Self {
            headers: Vec::new(),
            types: Vec::new(),
            values: Vec::new(),
        }
    }

    fn from_file<P: AsRef<Path>>(path: P, headers: bool) -> anyhow::Result<Self> {
        let mut csv = CSV::new();
        csv.parse_file(path, headers)?;
        Ok(csv)
    }

    fn parse_file<P: AsRef<Path>>(&mut self, path: P, headers: bool) -> anyhow::Result<()> {
        let mut lines = BufReader::new(File::open(path)?)
            .lines()
            .enumerate()
            .map(|(i, l)| (i + 1, l));

        if headers {
            let Some((_, line)) = lines.next() else {
                return Ok(());
            };
            let line = line.map_err(|e| anyhow::anyhow!("error reading file: {e}"))?;
            self.headers = line.split(',').map(String::from).collect();
        }

        let Some((lineno, line)) = lines.next() else {
            return Ok(());
        };
        let line = line.map_err(|e| anyhow::anyhow!("error reading file: {e}"))?;
        let row = line.split(',').collect::<Vec<_>>();
        let ncols = if headers {
            let ncols = self.headers.len();
            if row.len() != ncols {
                anyhow::bail!("expected {} cols, found {} on line {}", ncols, row.len(), lineno);
            }
            ncols
        } else {
            row.len()
        };
        self.types = Vec::with_capacity(ncols);
        let mut values = Vec::with_capacity(ncols);
        for (col, val) in row.into_iter().enumerate().map(incr_en) {
            if val == "" {
                self.types.push(Value::Null);
                values.push(Value::Null);
            } else if is_float(&val) {
                self.types.push(Value::F64(0.0));
                match val.parse() {
                    Ok(f) => values.push(Value::F64(f)),
                    Err(e) => anyhow::bail!("error parsing value in row {lineno} col {col}: {e}"),
                }
            } else {
                self.types.push(Value::Str(String::new()));
                values.push(Value::Str(val.to_string()));
            }
        }
        self.values.push(values);

        for (lineno, line) in lines {
            let line = line.map_err(|e| anyhow::anyhow!("error reading file: {e}"))?;
            let row = line.split(',').collect::<Vec<_>>();
            if row.len() != ncols {
                anyhow::bail!("expected {} cols, found {} on line {}", ncols, row.len(), lineno);
            }
            let mut values = Vec::with_capacity(ncols);
            for (i, val) in row.into_iter().enumerate() {
                if val == "" {
                    values.push(Value::Null);
                } else {
                    match self.types[i].parse_as_self(val) {
                        Ok(value) => {
                            if self.types[i].is_null() {
                                self.types[i] = value.to_type();
                            }
                            values.push(value);
                        }
                        Err(e) => anyhow::bail!("error parsing value in row {lineno} col {0}: {e}", i + 1),
                    }
                }
            }
            self.values.push(values);
        }
        Ok(())
    }

    fn split_line_into<'a>(v: &mut Vec<&'a str>, line: &'a str) {
        let mut prev_char = '\0';
        let mut quote_count = 0usize;
        let mut last_char = 0;
        for (i, c) in line.char_indices() {
            if c == '"" {
                quote_count += 1;
            } else if c == ',' && quote_count % 2 == 0 {
                v.push(&line[last_char..i]);
                last_char = i + 1;
            }
        }
        v.push(&line[last_char..]);
    }
}

struct CSVStr(Box<str>);

impl std::str::FromStr for CSVStr {
    type Err;

    fn from_str(s: &str) -> Self {
        // Number of quotes before non-quote character
        let bqc = s.chars().take_while(|c| c == &'"').count();
        //
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
enum Value {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),

    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),

    F32(f32),
    F64(f64),

    Bool(bool),

    Str(String),
    Null,
}

#[allow(dead_code)]
impl Value {
    fn is_int(&self) -> bool {
        use Value::*;
        matches!(
            self,
            I8(_) | I16(_) | I32(_) | I64(_) | I128(_) |
            U8(_) | U16(_) | U32(_) | U64(_) | U128(_),
        )
    }

    fn is_signed(&self) -> bool {
        use Value::*;
        matches!(self, I8(_) | I16(_) | I32(_) | I64(_) | I128(_))
    }

    fn is_unsigned(&self) -> bool {
        use Value::*;
        matches!(self, U8(_) | U16(_) | U32(_) | U64(_) | U128(_))
    }

    fn is_float(&self) -> bool {
        use Value::*;
        matches!(self, F32(_) | F64(_))
    }

    fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    // Attempts to parse the given value as the type of self. If self is Null, it will attempt to
    // parse it as a float, then as a string, or Null if it's empty.
    fn parse_as_self<S: AsRef<str>>(&self, s: S) -> anyhow::Result<Self> {
        use Value::*;
        let s = s.as_ref();
        if s == "" {
            return Ok(Value::Null);
        }
        match self {
            I8(_) => Ok(I8(s.parse()?)),
            I16(_) => Ok(I16(s.parse()?)),
            I32(_) => Ok(I32(s.parse()?)),
            I64(_) => Ok(I64(s.parse()?)),
            I128(_) => Ok(I128(s.parse()?)),
            U8(_) => Ok(U8(s.parse()?)),
            U16(_) => Ok(U16(s.parse()?)),
            U32(_) => Ok(U32(s.parse()?)),
            U64(_) => Ok(U64(s.parse()?)),
            U128(_) => Ok(U128(s.parse()?)),
            F32(_) => Ok(F32(s.parse()?)),
            F64(_) => Ok(F64(s.parse()?)),
            Bool(_) => Ok(Bool(s.parse()?)),
            Str(_) => Ok(Str(s.to_string())),
            Null => {
                if is_float(s) {
                    s.parse().map(|f| F64(f)).map_err(|e| e.into())
                } else {
                    Ok(Str(s.to_string()))
                }
            },
        }
    }

    // Returns self but with the default/"empty" value for the interior type
    fn to_type(&self) -> Self {
        use Value::*;
        match self {
            I8(_) => I8(0),
            I16(_) => I16(0),
            I32(_) => I32(0),
            I64(_) => I64(0),
            I128(_) => I128(0),
            U8(_) => U8(0),
            U16(_) => U16(0),
            U32(_) => U32(0),
            U64(_) => U64(0),
            U128(_) => U128(0),
            F32(_) => F32(0.0),
            F64(_) => F64(0.0),
            Bool(_) => Bool(false),
            Str(_) => Str(String::new()),
            Null => Null,
        }
    }

    fn to_type_str(&self) -> &'static str {
        use Value::*;
        match self {
            I8(_) => "i8",
            I16(_) => "i16",
            I32(_) => "i32",
            I64(_) => "i64",
            I128(_) => "i128",
            U8(_) => "u8",
            U16(_) => "u16",
            U32(_) => "u32",
            U64(_) => "u64",
            U128(_) => "u128",
            F32(_) => "f32",
            F64(_) => "f64",
            Bool(_) => "bool",
            Str(_) => "str",
            Null => "null",
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Value::*;
        match self {
            I8(v) => write!(f, "{v}"),
            I16(v) => write!(f, "{v}"),
            I32(v) => write!(f, "{v}"),
            I64(v) => write!(f, "{v}"),
            I128(v) => write!(f, "{v}"),
            U8(v) => write!(f, "{v}"),
            U16(v) => write!(f, "{v}"),
            U32(v) => write!(f, "{v}"),
            U64(v) => write!(f, "{v}"),
            U128(v) => write!(f, "{v}"),
            F32(v) => write!(f, "{v}"),
            F64(v) => write!(f, "{v}"),
            Bool(v) => write!(f, "{v}"),
            // TODO: Quotes?
            Str(v) => write!(f, "\"{v}\""),
            Null => write!(f, "null"),
        }
    }
}

trait CSVOp {
    type Output;
    fn add(&mut self);
    fn finalize(self) -> Self::Output;
}

#[allow(dead_code)]
fn is_int(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii_digit())
}

fn is_float(s: &str) -> bool {
    let mut dec_found = false;
    s.chars()
        .all(|c| {
            if c == '.' {
                dec_found = !dec_found;
                dec_found
            } else {
                c.is_ascii_digit()
            }
        })
}

// Used as func to pass to .map() after calling .enumerate() to increment the num.
fn incr_en<T>((i, v): (usize, T)) -> (usize, T) {
    (i + 1, v)
}

#[derive(Parser, Debug)]
struct Args {
    file_names: Vec<PathBuf>,
    #[arg(long)]
    no_headers: bool,
    #[arg(long)]
    mean: Option<String>,
    #[arg(long)]
    mode: Option<String>,
    #[arg(long)]
    sd: Option<String>,
    #[arg(long)]
    dt: Option<String>,
}
