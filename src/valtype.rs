#[derive(Debug,Clone,Copy)]
pub enum ValType {
    F(f32),
    D(f64),
    I(i32),
    L(i64),
}

use std::fmt;

impl fmt::Display for ValType {
    fn fmt(&self, f: &mut fmt::Formatter ) -> fmt::Result {
        write!( f, "{:?}", self )
    }
}

impl From<ValType> for f32 {
    fn from(s: ValType) -> Self {
        match s {
            ValType::F(x) => x as f32,
            ValType::D(x) => x as f32,
            ValType::I(x) => x as f32,
            ValType::L(x) => x as f32,
        }
    }
}
