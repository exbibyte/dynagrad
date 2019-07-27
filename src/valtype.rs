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
