// #[macro_use(s)]
// pub use ndarray;

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

mod core;
mod valtype;

mod interface {
    pub use crate::core::{Add, Cos, Div, Exp, Leaf, Ln, Mul, Pow, Sin, Tan};
    pub use crate::valtype::ValType;
}

pub use interface::*;
