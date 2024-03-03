use std::borrow::Cow;
use std::fmt;
use std::fs::File;
use std::io::{prelude::*, BufReader, Error as IOError};
use std::mem;
use std::ops::{
    Bound, Index, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
};
use std::path::Path;
use std::slice::SliceIndex;

/// A row in a CSV.
#[derive(Clone, Copy, PartialEq)]
pub struct CSVRow<'a>(&'a CSV, &'a [Value]);

impl<'a> CSVRow<'a> {
    /// Returns a reference to the value in the row associated with the column
    pub fn get_col(&'a self, name: &str) -> Option<&'a Value> {
        let headers = self.0.headers.as_ref()?;
        headers
            .into_iter()
            .position(|s| s == name)
            .map(|i| &self.as_slice()[i])
    }

    /// Returns the length of the row.
    pub fn len(&'a self) -> usize {
        self.1.len()
    }

    /// Returns a slice of the row's values.
    pub fn as_slice(&'a self) -> &'a [Value] {
        self.1
    }

    /// Returns the CSV the row belongs to.
    pub fn csv(&'a self) -> &'a CSV {
        self.0
    }
}

impl<'a> Index<usize> for CSVRow<'a> {
    type Output = Value;

    /// Gets the value in the row with the index (column-index).
    ///
    /// # Panics
    ///
    /// Panics if index >= self.len().
    fn index(&self, index: usize) -> &Self::Output {
        &self.1[index]
    }
}

macro_rules! csv_row_index {
    ($($index_type:ty),+ $(,)?) => {
        $(
            impl<'a> Index<$index_type> for CSVRow<'a> {
                type Output = [Value];

                /// Gets the values in the row with the specified indexes (column-indexes).
                ///
                /// # Panics
                ///
                /// Panics if self.as_slice(index) panics.
                fn index(&self, index: $index_type) -> &Self::Output {
                    &self.1[index]
                }
            }
        )+
    }
}

csv_row_index!(
    Range<usize>,
    RangeFrom<usize>,
    RangeFull,
    RangeInclusive<usize>,
    RangeTo<usize>,
    RangeToInclusive<usize>
);

impl<'a> Index<&str> for CSVRow<'a> {
    type Output = Value;

    /// Gets the value in the row assocated with the column.
    ///
    /// # Panics
    ///
    /// Panics if the column doesn't exist.
    fn index(&self, name: &str) -> &Self::Output {
        self.get_col(name).expect("no col found")
    }
}

impl<'a> fmt::Debug for CSVRow<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.1)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum CSVData {
    Rows(Vec<Vec<Value>>),
    Cols(Vec<Vec<Value>>),
}

/// A CSV with information on headers, types, and houses the data, stored either row-wise (default)
/// or column-wise.
#[derive(Clone, PartialEq)]
pub struct CSV {
    headers: Option<Vec<String>>,
    types: Option<Vec<Type>>,
    data: CSVData,
}

impl CSV {
    /// Creates a new empty CSV.
    pub const fn new() -> Self {
        Self {
            headers: None,
            types: None,
            data: CSVData::Rows(Vec::new()),
        }
    }

    /// Parses the CSV pointed to by `path`.
    pub fn open<P: AsRef<Path>>(path: P) -> CSVRes<Self> {
        CSVReaderBuilder::default().open(path)
    }

    /// Returns a CSVReaderBuilder.
    pub fn reader_builder() -> CSVReaderBuilder {
        CSVReaderBuilder::new()
    }

    /// Returns the number of rows.
    pub fn num_rows(&self) -> usize {
        match &self.data {
            CSVData::Rows(rows) => rows.len(),
            CSVData::Cols(cols) => cols.get(0).map(|c| c.len()).unwrap_or(0),
        }
    }

    /// Returns the number of columns.
    pub fn num_cols(&self) -> usize {
        if let Some(n) = self.types.as_ref().map(|t| t.len()) {
            return n;
        }
        match &self.data {
            CSVData::Rows(rows) => rows.get(0).map(|r| r.len()).unwrap_or(0),
            CSVData::Cols(cols) => cols.len(),
        }
    }

    /// Returns the headers, if there are any.
    pub fn headers(&self) -> Option<&[String]> {
        self.headers.as_ref().map(Vec::as_slice)
    }

    /// Returns the types, if there are any.
    pub fn types(&self) -> Option<&[Type]> {
        self.types.as_ref().map(Vec::as_slice)
    }

    /// Gets the row(s) or column(s) at the specified index (depending on whether the CSV is stored
    /// row-wise or column-wise).
    pub fn get<I>(&self, i: I) -> Option<&<I as SliceIndex<[Vec<Value>]>>::Output>
    where
        I: SliceIndex<[Vec<Value>]>,
    {
        match &self.data {
            CSVData::Rows(rows) => rows.get(i),
            CSVData::Cols(cols) => cols.get(i),
        }
    }

    /// Gets the row(s) at the specified index. Returns None if the CSV isn't row-major.
    pub fn get_row<I>(&self, index: I) -> Option<&<I as SliceIndex<[Vec<Value>]>>::Output>
    where
        I: SliceIndex<[Vec<Value>]>,
    {
        match &self.data {
            CSVData::Cols(_) => None,
            CSVData::Rows(rows) => rows.get(index),
        }
    }

    /// Gets the row at the specified index. Constructs a new row and returns Cow::Owned if the
    /// CSV is column-major.
    pub fn get_row_cow<'a>(&'a self, index: usize) -> Option<Cow<'a, Vec<Value>>> {
        let cols = match &self.data {
            CSVData::Cols(cols) => cols,
            CSVData::Rows(rows) => return rows.get(index).map(Cow::Borrowed),
        };
        let mut row = Vec::with_capacity(cols.len());
        for col in cols {
            row.push(col.get(index)?.clone());
        }
        Some(Cow::Owned(row))
    }

    /// Gets the row at the specified index, returning a CSVRow. Returns None if the CSV isn't
    /// row-major.
    pub fn get_csv_row<'a>(&'a self, index: usize) -> Option<CSVRow<'a>> {
        self.get_row(index).map(|r| CSVRow(self, r))
    }

    /// Gets the rows at the specified index, returning a Vec<CSVRow>. Returns None if the CSV
    /// isn't row-major.
    pub fn get_csv_rows<'a, R: RangeBounds<usize>>(&'a self, range: R) -> Option<Vec<CSVRow<'a>>> {
        match &self.data {
            CSVData::Cols(_) => None,
            CSVData::Rows(rows) => {
                let s = match range.start_bound() {
                    Bound::Included(s) => *s,
                    Bound::Excluded(s) => s.saturating_sub(1),
                    Bound::Unbounded => 0,
                };
                let e = match range.end_bound() {
                    Bound::Included(e) => *e + 1,
                    Bound::Excluded(e) => *e,
                    Bound::Unbounded => rows.len(),
                };
                if s >= rows.len() || e > rows.len() {
                    None
                } else {
                    Some((&rows[s..e]).into_iter().map(|r| CSVRow(self, r)).collect())
                }
            }
        }
    }

    /// Gets the column with the speified name. Returns None if the CSV isn't column-major.
    pub fn get_col(&self, name: &str) -> Option<&[Value]> {
        let cols = match &self.data {
            CSVData::Rows(_) => return None,
            CSVData::Cols(cols) => cols,
        };
        let headers = self.headers.as_ref()?;
        let i = headers.into_iter().position(|s| s == name)?;
        Some(&cols[i])
    }

    /// Gets the column at the specified name. Constructs a new column and returns Cow::Owned if
    /// the CSV is row-major.
    pub fn get_col_cow<'a>(&'a self, name: &str) -> Option<Cow<'a, [Value]>> {
        let headers = self.headers.as_ref()?;
        let i = headers.into_iter().position(|s| s == name)?;
        let rows = match &self.data {
            CSVData::Rows(rows) => rows,
            CSVData::Cols(cols) => return Some(Cow::Borrowed(&cols[i])),
        };
        let mut col = Vec::with_capacity(rows.len());
        for row in rows {
            col.push(row[i].clone());
        }
        Some(Cow::Owned(col))
    }

    /// Gets the columns(s) at the specified index. Returns None if the CSV isn't column-major.
    pub fn get_col_index<I>(&self, ci: I) -> Option<&<I as SliceIndex<[Vec<Value>]>>::Output>
    where
        I: SliceIndex<[Vec<Value>]>,
    {
        match &self.data {
            CSVData::Rows(_) => return None,
            CSVData::Cols(cols) => cols.get(ci),
        }
    }

    /// Gets the column at the specified index. Constructs a new column and returns Cow::Owned
    /// if the CSV is row-major.
    pub fn get_col_index_cow<'a>(&'a self, ci: usize) -> Option<Cow<'a, Vec<Value>>> {
        let rows = match &self.data {
            CSVData::Rows(rows) => rows,
            CSVData::Cols(cols) => return cols.get(ci).map(Cow::Borrowed),
        };
        let mut col = Vec::with_capacity(rows.len());
        for row in rows {
            col.push(row.get(ci)?.clone());
        }
        Some(Cow::Owned(col))
    }

    /// Converts the CSV's internal storage to store data row-wise. Noop if already row-wise.
    pub fn make_row_wise(&mut self) {
        let cols = match &mut self.data {
            CSVData::Rows(_) => return,
            CSVData::Cols(cols) => mem::replace(cols, Vec::new()),
        };
        if cols.len() == 0 {
            self.data = CSVData::Rows(Vec::new());
            return;
        }
        let mut rows = vec![vec![Value::I8(0); cols.len()]; cols[0].len()];
        for (c, col) in cols.into_iter().enumerate() {
            for (r, v) in col.into_iter().enumerate() {
                rows[r][c] = v;
            }
        }
        self.data = CSVData::Rows(rows);
    }

    /// Converts the CSV's internal storage to store data column-wise. Noop if already column-wise.
    pub fn make_col_wise(&mut self) {
        let rows = match &mut self.data {
            CSVData::Cols(_) => return,
            CSVData::Rows(rows) => mem::replace(rows, Vec::new()),
        };
        if rows.len() == 0 {
            self.data = CSVData::Cols(Vec::new());
            return;
        }
        let mut cols = vec![vec![Value::I8(0); rows.len()]; rows[0].len()];
        for (r, row) in rows.into_iter().enumerate() {
            for (c, v) in row.into_iter().enumerate() {
                cols[c][r] = v;
            }
        }
        self.data = CSVData::Cols(cols);
    }

    /// Returns whether the CSV is stored row-wise.
    pub fn is_row_wise(&self) -> bool {
        match self.data {
            CSVData::Rows(_) => true,
            _ => false,
        }
    }

    /// Returns whether the CSV is stored column-wise.
    pub fn is_col_wise(&self) -> bool {
        match self.data {
            CSVData::Cols(_) => true,
            _ => false,
        }
    }
}

impl Default for CSV {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for CSV {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn print_row<T: ToString>(f: &mut fmt::Formatter, s: &[T]) -> fmt::Result {
            let mut iter = s.into_iter();
            if let Some(next) = iter.next() {
                write!(f, "{}", next.to_string())?;
            }
            for next in iter {
                write!(f, ",{}", next.to_string())?;
            }
            Ok(())
        }
        match &self.data {
            CSVData::Rows(rows) => {
                if let Some(headers) = self.headers.as_ref() {
                    print_row(f, &headers)?;
                    f.write_str("\n")?;
                }
                for row in rows {
                    print_row(f, &row)?;
                    f.write_str("\n")?;
                }
            }
            CSVData::Cols(cols) => {
                let headers = self.headers.as_ref().map(Vec::as_slice).unwrap_or(&[]);
                for (i, col) in cols.into_iter().enumerate() {
                    if let Some(h) = headers.get(i) {
                        if col.len() != 0 {
                            write!(f, "{},", h.to_string())?;
                        } else {
                            write!(f, "{}", h.to_string())?;
                        }
                    }
                    print_row(f, &col)?;
                    f.write_str("\n")?;
                }
            }
        }
        Ok(())
    }
}

/*
impl Index<usize> for CSV {
    type Output = Vec<Type>;

    fn index(&self, index: usize) -> &Self::Output {
        /*
        match &self.data {
            CSVData::Rows(rows) => {
            }
        }
        */
        todo!()
    }
}
*/

/// A CSV reader with contains information on column types and headers
pub struct CSVReader<B: BufRead> {
    reader: B,
    types: Option<Vec<Type>>,
    headers: Option<Vec<String>>,
    no_headers: bool,
    line_buf: String,
}

impl<R: Read> CSVReader<BufReader<R>> {
    /// Creates a new CSVReader with the internal buf reader set to std::io::BufReader<R> (will
    /// attempt to parse headers).
    pub fn with_reader(r: R) -> CSVRes<Self> {
        Self::builder().build_with_reader(r)
    }
}

impl<B: BufRead> CSVReader<B> {
    /// Creates a new CSVReader (will attempt to parse headers).
    pub fn new(br: B) -> CSVRes<Self> {
        Self::builder().build(br)
    }

    /// Returns a CSVReaderBuilder.
    pub fn builder() -> CSVReaderBuilder {
        CSVReaderBuilder::default()
    }

    /// Returns the types, if there are any. None if types were not initially provided and nothing
    /// has been parsed yet. This will be automatically set, if not already set, when parsing a
    /// line, whether it be headers or the first line in the CSV.
    pub fn types(&self) -> Option<&[Type]> {
        self.types.as_ref().map(Vec::as_slice)
    }

    /// Returns the headers, if there are any. None if no_headers was set to true.
    pub fn headers(&self) -> Option<&[String]> {
        self.headers.as_ref().map(Vec::as_slice)
    }

    //fn parse_headers(&mut self) -> CSVRes<Vec<String>> {
    fn set_headers(&mut self) -> CSVRes<()> {
        self.line_buf.clear();
        if self.reader.read_line(&mut self.line_buf)? == 0 {
            return Ok(());
        }
        let line = &self.line_buf[..self.line_buf.len() - 1];
        if let Some(types) = self.types.as_ref() {
            let strs = vec![Type::Str; types.len()];
            let vals = parse_line_with_types(line, &strs)?;
            self.headers = Some(vals.into_iter().map(|v| v.into()).collect());
        } else {
            let vals = parse_line(line)?;
            self.types = Some(vec![Type::Raw; vals.len()]);
            self.headers = Some(vals.into_iter().map(|v| v.into()).collect());
        }
        Ok(())
    }
}

impl<B: BufRead> Iterator for CSVReader<B> {
    type Item = CSVRes<Vec<Value>>;

    /// Parses and returns the next row in the CSV (headers not included, if applicable).
    /// If the types were not set and no_headers was set to true, the first call to this sets the
    /// types to Type::Raw.
    fn next(&mut self) -> Option<Self::Item> {
        self.line_buf.clear();
        match self.reader.read_line(&mut self.line_buf) {
            Ok(0) => return None,
            Ok(_) => (),
            Err(e) => return Some(Err(e.into())),
        }
        let line = &self.line_buf[..self.line_buf.len() - 1];
        if let Some(types) = self.types.as_ref() {
            Some(parse_line_with_types(line, types))
        } else {
            match parse_line(line) {
                Ok(vals) => {
                    self.types = Some(vec![Type::Raw; vals.len()]);
                    Some(Ok(vals))
                }
                Err(e) => Some(Err(e)),
            }
        }
    }
}

/*
pub struct CSVLineReader<I: Iterator<Item=String>> {
    iter: I,
    types: Option<Vec<Type>>,
    headers: Option<Vec<String>>,
    no_headers: bool,
    line_buf: String,
}

impl<I: Iterator<Item=String>> CSVLineReader<I> {
    pub fn new(iter: I) -> CSVRes<Self> {
        Self::builder().build_from_iter(iter)
    }

    pub fn builder() -> CSVReaderBuilder {
        CSVReaderBuilder::default()
    }

    pub fn types(&self) -> Option<&[Type]> {
        self.types.as_ref().map(Vec::as_slice)
    }

    pub fn headers(&self) -> Option<&[Type]> {
        self.headers.as_ref().map(Vec::as_slice)
    }

    //fn parse_headers(&mut self) -> CSVRes<Vec<String>> {
    fn set_headers(&mut self) -> CSVRes<()> {
        if self.reader.read_line(&mut self.line_buf)? == 0 {
            return Ok(());
        }
        if let Some(types) = self.types.as_ref() {
            let vals = parse_line_with_types(&self.line_buf, types)?;
            self.headers = Some(vals.into_iter().map(|v| v.into()).collect());
        } else {
            let vals = parse_line(&self.line_buf)?;
            self.types = Some(vec![Type::Raw; vals.len()]);
            self.headers = Some(vals.into_iter().map(|v| v.into()).collect());
        }
        Ok(())
    }
}

impl<I: Iterator<Item=String>> Iterator for CSVLineReader<I> {
    type Item = CSVRes<Vec<Value>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.read_line(&mut self.line_buf) {
            Ok(0) => return None,
            Ok(_) => (),
            Err(e) => return Some(Err(e.into())),
        }
        if let Some(types) = self.types.as_ref() {
            Some(parse_line_with_types(&self.line_buf, types))
        } else {
            match parse_line(&self.line_buf) {
                Ok(vals) => {
                    self.types = Some(vec![Type::Raw; vals.len()]);
                    Some(Ok(vals))
                }
                Err(e) => Some(Err(e)),
            }
        }
    }
}
*/

/// A builder to set options for reading a CSV.
#[derive(Default)]
pub struct CSVReaderBuilder {
    types: Option<Vec<Type>>,
    no_headers: bool,
}

impl CSVReaderBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            types: None,
            no_headers: false,
        }
    }

    /// Set the types of the CSV.
    pub fn types(self, types: Option<Vec<Type>>) -> Self {
        Self { types, ..self }
    }

    /// Set whether the CSV has headers.
    pub fn no_headers(self, no_headers: bool) -> Self {
        Self { no_headers, ..self }
    }

    /// Return a new CSVReader with the internal buf reader set as B.
    /// Attempts to parse the headers if appropriate.
    pub fn build<B: BufRead>(self, reader: B) -> CSVRes<CSVReader<B>> {
        let mut reader = CSVReader {
            reader,
            types: self.types,
            no_headers: self.no_headers,
            headers: None,
            line_buf: String::new(),
        };
        if !reader.no_headers {
            reader.set_headers()?;
        }
        Ok(reader)
    }

    /// Return a new CSVReader with the internal buf reader set as std::io::BufReader<B>.
    /// Attempts to parse the headers if appropriate.
    pub fn build_with_reader<R: Read>(self, reader: R) -> CSVRes<CSVReader<BufReader<R>>> {
        self.build(BufReader::new(reader))
    }

    /// Parse the CSV pointed to by path with the options set by this builder.
    pub fn open<P: AsRef<Path>>(self, path: P) -> CSVRes<CSV> {
        let mut reader = self.build_with_reader(File::open(path)?)?;
        let mut rows = Vec::new();
        //reader.try_for_each(|row| Ok::<(), CSVError>(rows.push(row?)))?;
        for (i, row) in reader.by_ref().enumerate() {
            match row {
                Ok(row) => rows.push(row),
                Err(e) => {
                    let r = if reader.headers.is_none() {
                        i + 1
                    } else {
                        i + 2
                    };
                    return Err(CSVError::ParseRow(Box::new(e), r));
                }
            }
        }
        Ok(CSV {
            headers: reader.headers,
            types: reader.types,
            data: CSVData::Rows(rows),
        })
    }
}

pub type CSVRes<T> = Result<T, CSVError>;

/// An error return by CSV parsing operations. Columns in errors are 1-based indexes.
#[derive(Debug)]
pub enum CSVError {
    /// IO error when reading from reader.
    IO(IOError),
    /// Invalid parse of Type at column.
    // Type, column
    BadParse(Type, usize),
    /// Invalid column.
    BadCol(usize),
    /// Expected column count and parsed column count mismatch.
    BadColCount { want: usize, got: usize },
    /// Error when parsing a row. Contains the row number (1-based).
    // Error, row num (1-based)
    ParseRow(Box<Self>, usize),
}

impl CSVError {
    fn bad_col_count(want: usize, got: usize) -> Self {
        Self::BadColCount { want, got }
    }
}

impl From<IOError> for CSVError {
    fn from(e: IOError) -> Self {
        CSVError::IO(e)
    }
}

impl fmt::Display for CSVError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CSVError::IO(e) => write!(f, "io error: {e}"),
            CSVError::BadParse(t, col) => write!(f, "bad {t} parse at col {col}"),
            CSVError::BadCol(col) => write!(f, "malformed col {col}"),
            CSVError::BadColCount { want, got } if want > got => {
                write!(f, "expected {want} cols, got {got}")
            }
            CSVError::BadColCount { want, got } => {
                write!(f, "expected {want} cols, got at least {got}")
            }
            CSVError::ParseRow(e, r) => write!(f, "row {r}: {e}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
    Str,
    /// A raw CSV representation (e.g., strings remain unescaped).
    Raw,
}

impl Type {
    /// Parses a raw CSV string and returns a value associated with the type, or None.
    pub fn value_from_raw_str(self, s: &str) -> Option<Value> {
        match self {
            Type::I8 => s.parse().map(Value::I8).ok(),
            Type::I16 => s.parse().map(Value::I16).ok(),
            Type::I32 => s.parse().map(Value::I32).ok(),
            Type::I64 => s.parse().map(Value::I64).ok(),
            Type::U8 => s.parse().map(Value::U8).ok(),
            Type::U16 => s.parse().map(Value::U16).ok(),
            Type::U32 => s.parse().map(Value::U32).ok(),
            Type::U64 => s.parse().map(Value::U64).ok(),
            Type::F32 => s.parse().map(Value::F32).ok(),
            Type::F64 => s.parse().map(Value::F64).ok(),
            Type::Bool => s.parse().map(Value::Bool).ok(),
            Type::Str => Some(Value::Str(s.replace("\"\"", "\""))),
            Type::Raw => Some(Value::Raw(s.to_string())),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::I8 => write!(f, "i8"),
            Type::I16 => write!(f, "i16"),
            Type::I32 => write!(f, "i32"),
            Type::I64 => write!(f, "i64"),
            Type::U8 => write!(f, "u8"),
            Type::U16 => write!(f, "u16"),
            Type::U32 => write!(f, "u32"),
            Type::U64 => write!(f, "u64"),
            Type::F32 => write!(f, "f32"),
            Type::F64 => write!(f, "f64"),
            Type::Bool => write!(f, "bool"),
            Type::Str => write!(f, "str"),
            Type::Raw => write!(f, "any"),
        }
    }
}

// Holds a (potentially) parsed value from a CSV.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    /// A raw CSV representation (e.g., strings remain unescaped).
    Raw(String),
}

impl Value {
    /// Returns the Type this is.
    pub fn typ(&self) -> Type {
        match self {
            Value::I8(_) => Type::I8,
            Value::I16(_) => Type::I16,
            Value::I32(_) => Type::I32,
            Value::I64(_) => Type::I64,
            Value::U8(_) => Type::U8,
            Value::U16(_) => Type::U16,
            Value::U32(_) => Type::U32,
            Value::U64(_) => Type::U64,
            Value::F32(_) => Type::F32,
            Value::F64(_) => Type::F64,
            Value::Bool(_) => Type::Bool,
            Value::Str(_) => Type::Str,
            Value::Raw(_) => Type::Raw,
        }
    }

    /// Returns a i8 if the type is Type::I8.
    pub fn as_i8(&self) -> Option<i8> {
        match self {
            Value::I8(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns a i16 if the type is Type::I16.
    pub fn as_i16(&self) -> Option<i16> {
        match self {
            Value::I16(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns a i32 if the type is Type::I32.
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Value::I32(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns a i64 if the type is Type::I64.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::I64(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns a u8 if the type is Type::U8.
    pub fn as_u8(&self) -> Option<u8> {
        match self {
            Value::U8(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns a u16 if the type is Type::U16.
    pub fn as_u16(&self) -> Option<u16> {
        match self {
            Value::U16(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns a u32 if the type is Type::U32.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Value::U32(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns a u64 if the type is Type::U64.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U64(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns a f32 if the type is Type::F32.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::F32(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns a f64 if the type is Type::F64.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns a bool if the type is Type::Bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns a string if the type is Type::Str.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(v) => Some(v.as_str()),
            _ => None,
        }
    }

    /// Returns a raw string if the type is Type::Raw.
    pub fn as_raw(&self) -> Option<&str> {
        match self {
            Value::Raw(v) => Some(v.as_str()),
            _ => None,
        }
    }

    /// Returns a i8 if the type is Type::I8, otherwise, returns the value.
    pub fn try_into_i8(self) -> Result<i8, Self> {
        match self {
            Value::I8(v) => Ok(v),
            v => Err(v),
        }
    }

    /// Returns a i16 if the type is Type::I16, otherwise, returns the value.
    pub fn try_into_i16(self) -> Result<i16, Self> {
        match self {
            Value::I16(v) => Ok(v),
            v => Err(v),
        }
    }

    /// Returns a i32 if the type is Type::I32, otherwise, returns the value.
    pub fn try_into_i32(self) -> Result<i32, Self> {
        match self {
            Value::I32(v) => Ok(v),
            v => Err(v),
        }
    }

    /// Returns a i64 if the type is Type::I64, otherwise, returns the value.
    pub fn try_into_i64(self) -> Result<i64, Self> {
        match self {
            Value::I64(v) => Ok(v),
            v => Err(v),
        }
    }

    /// Returns a u8 if the type is Type::U8, otherwise, returns the value.
    pub fn try_into_u8(self) -> Result<u8, Self> {
        match self {
            Value::U8(v) => Ok(v),
            v => Err(v),
        }
    }

    /// Returns a u16 if the type is Type::U16, otherwise, returns the value.
    pub fn try_into_u16(self) -> Result<u16, Self> {
        match self {
            Value::U16(v) => Ok(v),
            v => Err(v),
        }
    }

    /// Returns a u32 if the type is Type::U32, otherwise, returns the value.
    pub fn try_into_u32(self) -> Result<u32, Self> {
        match self {
            Value::U32(v) => Ok(v),
            v => Err(v),
        }
    }

    /// Returns a u64 if the type is Type::U64, otherwise, returns the value.
    pub fn try_into_u64(self) -> Result<u64, Self> {
        match self {
            Value::U64(v) => Ok(v),
            v => Err(v),
        }
    }

    /// Returns a f32 if the type is Type::F32, otherwise, returns the value.
    pub fn try_into_f32(self) -> Result<f32, Self> {
        match self {
            Value::F32(v) => Ok(v),
            v => Err(v),
        }
    }

    /// Returns a f64 if the type is Type::F64, otherwise, returns the value.
    pub fn try_into_f64(self) -> Result<f64, Self> {
        match self {
            Value::F64(v) => Ok(v),
            v => Err(v),
        }
    }

    /// Returns a bool if the type is Type::Bool, otherwise, returns the value.
    pub fn try_into_bool(self) -> Result<bool, Self> {
        match self {
            Value::Bool(v) => Ok(v),
            v => Err(v),
        }
    }

    /// Returns a string if the type is Type::Str, otherwise, returns the value.
    pub fn try_into_string(self) -> Result<String, Self> {
        match self {
            Value::Str(v) => Ok(v),
            v => Err(v),
        }
    }

    /// Returns a raw string if the type is Type::Raw, otherwise, returns the value.
    pub fn try_into_raw(self) -> Result<String, Self> {
        match self {
            Value::Raw(v) => Ok(v),
            v => Err(v),
        }
    }

    /*
    pub fn try_into_type(self, t: Type) -> Result<Self, Self> {
        use std::convert::TryInto;
        macro_rules! try_into_match {
            ($v:expr, $vt:expr, $t:expr) => {
                match $t {
                    Type::I8 => Value::I8($v.try_into().map_err(|_| $vt($v))?),
                    Type::I16 => Value::I16($v.try_into().map_err(|_| $vt($v))?),
                    Type::I32 => Value::I32($v.try_into().map_err(|_| $vt($v))?),
                    Type::I64 => Value::I64($v.try_into().map_err(|_| $vt($v))?),
                    Type::U8 => Value::U8($v.try_into().map_err(|_| $vt($v))?),
                    Type::U16 => Value::U16($v.try_into().map_err(|_| $vt($v))?),
                    Type::U32 => Value::U32($v.try_into().map_err(|_| $vt($v))?),
                    Type::U64 => Value::U64($v.try_into().map_err(|_| $vt($v))?),
                    Type::F32 => Value::F32($v.try_into().map_err(|_| $vt($v))?),
                    Type::F64 => Value::F64($v.try_into().map_err(|_| $vt($v))?),
                    Type::Bool => Value::Bool($v.try_into().map_err(|_| $vt($v))?),
                    _ => unreachable!(),
                }
            }
        }
        let st = self.typ();
        if t == st {
            return Ok(self);
        }
        if t == Type::Str {
            if st != Type::Raw {
                return Ok(self.to_string());
            }
            // TODO
            todo!()
        } else if t == Type::Raw {
            todo!()
        }
        match self {
            Value::I8(v) => try_into_match!(v, Value::I8, t),
            Value::I16(v) => try_into_match!(v, Value::I16, t),
            Value::I32(v) => try_into_match!(v, Value::I32, t),
            Value::I64(v) => try_into_match!(v, Value::I64, t),
            Value::U8(v) => try_into_match!(v, Value::U8, t),
            Value::U16(v) => try_into_match!(v, Value::U16, t),
            Value::U32(v) => try_into_match!(v, Value::U32, t),
            Value::U64(v) => try_into_match!(v, Value::U64, t),
            Value::F32(v) => try_into_match!(v, Value::F32, t),
            Value::F64(v) => try_into_match!(v, Value::F64, t),
            Value::Bool(v) => try_into_match!(v, Value::Bool, t),
            Value::Str(_) | Value::Raw(_) => unreachable!(),
        }
    }
    */

    /// Converts the value into a raw CSV string.
    pub fn to_csv_string(&self) -> String {
        match self {
            Value::I8(v) => v.to_string(),
            Value::I16(v) => v.to_string(),
            Value::I32(v) => v.to_string(),
            Value::I64(v) => v.to_string(),
            Value::U8(v) => v.to_string(),
            Value::U16(v) => v.to_string(),
            Value::U32(v) => v.to_string(),
            Value::U64(v) => v.to_string(),
            Value::F32(v) => v.to_string(),
            Value::F64(v) => v.to_string(),
            Value::Bool(v) => v.to_string(),
            Value::Str(s) => s.replace("\"", "\"\""),
            Value::Raw(s) => s.clone(),
        }
    }

    /// Converts the value into a raw CSV string.
    pub fn into_csv_string(self) -> String {
        self.to_csv_string()
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::I8(v) => write!(f, "{}", v),
            Value::I16(v) => write!(f, "{}", v),
            Value::I32(v) => write!(f, "{}", v),
            Value::I64(v) => write!(f, "{}", v),
            Value::U8(v) => write!(f, "{}", v),
            Value::U16(v) => write!(f, "{}", v),
            Value::U32(v) => write!(f, "{}", v),
            Value::U64(v) => write!(f, "{}", v),
            Value::F32(v) => write!(f, "{}", v),
            Value::F64(v) => write!(f, "{}", v),
            Value::Bool(v) => write!(f, "{}", v),
            Value::Str(v) => f.write_str(&v),
            Value::Raw(v) => f.write_str(&v),
        }
    }
}

macro_rules! impl_value_from {
    ($($from:ty, $vt:expr),+ $(,)?) => {
        $(
            impl From<$from> for Value {
                fn from(v: $from) -> Self {
                    $vt(v)
                }
            }
        )+
    }
}

impl_value_from!(
    i8,
    Value::I8,
    i16,
    Value::I16,
    i32,
    Value::I32,
    i64,
    Value::I64,
    u8,
    Value::U8,
    u16,
    Value::U16,
    u32,
    Value::U32,
    u64,
    Value::U64,
    f32,
    Value::F32,
    f64,
    Value::F64,
    bool,
    Value::Bool,
    String,
    Value::Str,
);

impl From<Value> for String {
    fn from(v: Value) -> Self {
        v.to_string()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum QuotesState {
    NA,
    InQuotes,
    ExpectComma,
}

fn parse_line(line: &str) -> CSVRes<Vec<Value>> {
    let (mut start, mut prev) = (0, '\0');
    let mut quotes_state = QuotesState::NA;
    let mut values = Vec::new();
    let mut col = 1;
    for (i, c) in line.char_indices() {
        match c {
            _ if c != ',' && quotes_state == QuotesState::ExpectComma => {
                if c != '"' {
                    return Err(CSVError::BadCol(col));
                }
                // Escaped quote
                quotes_state = QuotesState::InQuotes;
                continue; // prev still == '"'
            }
            ',' if quotes_state != QuotesState::InQuotes => {
                let end = if prev != '"' {
                    i - 1
                } else {
                    quotes_state = QuotesState::NA;
                    i - 2
                };
                values.push(Value::Raw(line[start..end].to_string()));
                start = i + 1;
                col += 1;
            }
            // TODO: Return bad parse?
            '"' if prev != ',' && quotes_state != QuotesState::InQuotes => {
                return Err(CSVError::BadCol(col));
            }
            '"' if prev == '"' => (), // Escaped quote
            '"' if prev == ',' && quotes_state != QuotesState::InQuotes => {
                quotes_state = QuotesState::InQuotes;
                start = i + 1;
            }
            '"' if quotes_state == QuotesState::InQuotes => {
                quotes_state = QuotesState::ExpectComma;
            }
            _ => (),
        }
        prev = c;
    }
    if start != line.len() {
        if quotes_state == QuotesState::InQuotes {
            return Err(CSVError::BadCol(col));
        }
        let end = if prev != '"' {
            line.len()
        } else {
            line.len() - 1
        };
        values.push(Value::Raw(line[start..end].to_string()));
    }
    Ok(values)
}

fn parse_line_with_types(line: &str, types: &[Type]) -> CSVRes<Vec<Value>> {
    let (mut start, mut prev) = (0, '\0');
    let mut quotes_state = QuotesState::NA;
    let mut values = Vec::with_capacity(types.len());
    let mut col = 1;
    for (i, c) in line.char_indices() {
        // Still more cols
        if prev == ',' && quotes_state != QuotesState::InQuotes && col > types.len() {
            return Err(CSVError::bad_col_count(types.len(), col));
        }
        match c {
            _ if c != ',' && quotes_state == QuotesState::ExpectComma => {
                if c != '"' {
                    return Err(CSVError::BadCol(col));
                }
                // Escaped quote
                quotes_state = QuotesState::InQuotes;
                continue; // prev still == '"'
            }
            ',' if quotes_state != QuotesState::InQuotes => {
                let end = if prev != '"' {
                    i
                } else {
                    quotes_state = QuotesState::NA;
                    i - 1
                };
                let t = *types
                    .get(col - 1)
                    .ok_or(CSVError::bad_col_count(types.len(), col))?;
                let val = t
                    .value_from_raw_str(&line[start..end])
                    .ok_or(CSVError::BadParse(t, col))?;
                values.push(val);
                start = i + 1;
                col += 1;
            }
            // TODO: Return bad parse?
            '"' if prev != ',' && quotes_state != QuotesState::InQuotes => {
                return Err(CSVError::BadCol(col));
            }
            '"' if prev == ',' && quotes_state != QuotesState::InQuotes => {
                let t = types[col - 1];
                if t != Type::Str || t != Type::Raw {
                    return Err(CSVError::BadParse(t, col));
                }
                quotes_state = QuotesState::InQuotes;
                start = i + 1;
            }
            '"' if quotes_state == QuotesState::InQuotes => {
                quotes_state = QuotesState::ExpectComma;
            }
            _ => (),
        }
        prev = c;
    }
    if start != line.len() {
        if quotes_state == QuotesState::InQuotes {
            return Err(CSVError::BadCol(col));
        }
        let end = if prev != '"' {
            line.len()
        } else {
            line.len() - 1
        };
        let t = *types
            .get(col - 1)
            .ok_or(CSVError::bad_col_count(types.len(), col))?;
        let val = t
            .value_from_raw_str(&line[start..end])
            .ok_or(CSVError::BadParse(t, col))?;
        values.push(val);
    }
    if values.len() == types.len() {
        return Ok(values);
    }
    Err(CSVError::bad_col_count(types.len(), values.len()))
}
