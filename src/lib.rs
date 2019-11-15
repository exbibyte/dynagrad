// #[macro_use(s)]
// pub use ndarray;
#![feature(weak_into_raw)]

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

mod core;
mod ricci;
mod valtype;

mod interface {
    pub use crate::core::{Add, Cos, Div, Exp, Leaf, Ln, Mul, Pow, Sin, Tan};
    pub use crate::ricci::*;
    pub use crate::valtype::ValType;
}

pub use interface::*;
