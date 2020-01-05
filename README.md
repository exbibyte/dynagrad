# Dynamic Automatic Differentiation in Rust

A pedagogical attempt at auto-differentiation. This is based on the autograd package and other variations of it as well as literature references (eg: The Art of Differentiating Computer Programs, An Introduction to Algorithmic Differentiation – Uwe Naumann).

Work in progress..

# Note:
- currently depends on Rust nightly

# Support:
- forward mode
- reverse mode
- a composition thereof for higher-order derivatives.

# Todo:
- Multidimension support, possibly with help of ndarray crate
- Add support for Ricci calculus notation for symbolic manipulation (reference: Computing Higher Order Derivatives of Matrix and Tensor Expressions by Laue et al.)
- More ops and tests (see src/core.rs)

# Plots:
<p align="center">
   <img src="images/eg_simple_plot_tan.png" alt="drawing" width="400"/>
   <img src="images/eg_simple_plot_sin.png" alt="drawing" width="400"/>
</p>
