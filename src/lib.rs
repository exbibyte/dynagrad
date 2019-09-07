// #[macro_use(s)]
// pub use ndarray;
#![feature(weak_into_raw)]
#![feature(weak_ptr_eq)]

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

mod core;
mod valtype;
mod ricci;

mod interface {
    pub use crate::core::{Add, Leaf, Mul};
    pub use crate::valtype::ValType;
    pub use crate::ricci::*;
}

pub use interface::*;
