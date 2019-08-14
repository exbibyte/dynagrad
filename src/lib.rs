// #[macro_use(s)]
// pub use ndarray;

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

mod core;
mod valtype;

mod interface {
    pub use crate::core::{Leaf,Mul,Add};
    pub use crate::valtype::ValType as ValType;
}

pub use interface::*;
