//! An implementation of dynamic automatic differentiation
//! with forward and reverse modes
#![allow(non_snake_case)]

use std::borrow::{Borrow, BorrowMut};
use std::cell::RefCell;
use std::cmp::{Eq, PartialEq};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::Deref;
use std::rc::{Rc, Weak};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Clone, Debug)]
pub struct PtrVWrap(pub Rc<RefCell<VWrap>>);

impl Hash for PtrVWrap {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let p = Rc::downgrade(&self.0);
        (Weak::as_raw(&p) as usize).hash(state);
    }
}

impl PartialEq for PtrVWrap {
    fn eq(&self, other: &Self) -> bool {
        //decay both to Weak and compare
        Weak::ptr_eq(&Rc::downgrade(&self.0), &Rc::downgrade(&other.0))
    }
}

impl Eq for PtrVWrap {}

use crate::valtype::ValType;

#[cfg(test)]
lazy_static! {
    static ref ID: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
}

#[cfg(test)]
fn get_id() -> i32 {
    ID.fetch_add(1, Ordering::SeqCst) as _
}

/// wrapper for variable with recording of dependencies
// #[derive(Debug)]
pub struct VWrap {
    /// input dependencies
    pub inp: Vec<PtrVWrap>,

    /// source function
    raw: Box<dyn FWrap>,

    /// evaluted value
    pub val: Option<ValType>,

    #[cfg(test)]
    pub id: i32,

    pub eval_g: bool,

    /// adjoint accumulation expression
    pub adj_accum: Option<PtrVWrap>,
}
use std::fmt;

impl fmt::Debug for VWrap {
    #[cfg(test)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "VWrap {{ inp: {:#?}, raw:: {:?}, val: {:?}, id: {:?}, eval_g: {:?} }}",
            self.inp, self.raw, self.val, self.id, self.eval_g
        )
    }
    #[cfg(not(test))]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "VWrap {{ inp: {:#?}, raw:: {:?}, val: {:?}, eval_g: {:?} }}",
            self.inp, self.raw, self.val, self.eval_g
        )
    }
}

/// initializer functions
#[allow(dead_code)]
impl VWrap {
    fn new(v: Box<dyn FWrap>) -> PtrVWrap {
        PtrVWrap(Rc::new(RefCell::new(VWrap {
            inp: vec![],
            raw: v,
            val: None,
            #[cfg(test)]
            id: get_id(),
            eval_g: false,
            adj_accum: None,
        })))
    }

    fn new_with_input(f: Box<dyn FWrap>, v: Vec<PtrVWrap>) -> PtrVWrap {
        PtrVWrap(Rc::new(RefCell::new(VWrap {
            inp: v,
            raw: f,
            val: None,
            #[cfg(test)]
            id: get_id(),
            eval_g: false,
            adj_accum: None,
        })))
    }

    fn new_with_val(v: Box<dyn FWrap>, val: ValType) -> PtrVWrap {
        PtrVWrap(Rc::new(RefCell::new(VWrap {
            inp: vec![],
            raw: v,
            val: Some(val),
            #[cfg(test)]
            id: get_id(),
            eval_g: false,
            adj_accum: None,
        })))
    }
}

impl PtrVWrap {
    fn set_inp(&mut self, v: Vec<PtrVWrap>) {
        self.0.deref().borrow_mut().inp = v;
    }

    pub fn set_val(&mut self, v: ValType) {
        self.0.deref().borrow_mut().val = Some(v);
    }

    /// forward mode (tanget-linear)
    pub fn apply_fwd(&mut self) -> ValType {
        let mut args: Vec<(ValType, bool)> = vec![];

        //recursive apply
        for i in self.0.deref().borrow_mut().inp.iter_mut() {
            let val = i.apply_fwd();
            args.push((val, i.0.deref().borrow().eval_g));
        }

        let v = self.0.deref().borrow().raw.f()(args, self.0.deref().borrow().val);

        self.0.deref().borrow_mut().val = Some(v);

        v
    }

    /// reverse mode (adjoint)
    fn apply_rev_recurse(&mut self) -> ValType {
        let mut args: Vec<(ValType, bool)> = vec![];

        //recursive apply
        for i in self.0.deref().borrow_mut().inp.iter_mut() {
            let val = i.apply_rev_recurse();
            let temp = i.0.deref().borrow().eval_g;
            args.push((val, temp));
        }

        let v = self.0.deref().borrow().raw.f()(args, self.0.deref().borrow().val);

        self.0.deref().borrow_mut().val = Some(v);

        v
    }

    /// reverse mode (adjoint)
    pub fn apply_rev(&mut self) -> ValType {
        let v = self.apply_rev_recurse();

        v
    }

    /// create adjoint graph starting from current variable and go through input dependencies
    ///
    /// resulting sensitivity graphs are propagated to leaf nodes' adjoint accumulation
    /// where it can be collected
    pub fn rev(&self) -> HashMap<PtrVWrap, PtrVWrap> {
        use std::collections::VecDeque;

        let mut q = VecDeque::new();

        let mut adjoints_collected = HashMap::new();

        //initialization of sensitity=1 for starting node
        self.0.deref().borrow_mut().adj_accum = Some(VWrap::new(OpOne::new()));

        q.push_back(self.clone());

        let mut visited: HashSet<PtrVWrap> = HashSet::new();

        //breadth-first
        while !q.is_empty() {
            let mut n = q.pop_front().unwrap();

            if visited.contains(&n) {
                //skip already traversed nodes
                continue;
            }

            if n.0.deref().borrow_mut().adj_accum.is_none() {
                n.0.deref().borrow_mut().adj_accum = Some(VWrap::new(OpZero::new()));
            }

            //delegate adjoint calc to operation
            let mut adjoints = {
                let mut f = n.0.deref().borrow().raw.adjoint();
                f(
                    n.0.deref().borrow().inp.clone(),
                    n.0.deref()
                        .borrow()
                        .adj_accum
                        .as_ref()
                        .expect("adj_accum empty")
                        .clone(),
                    &n,
                )
            };

            assert_eq!(adjoints.len(), n.0.deref().borrow().inp.len());

            //propagate adjoints to inputs
            let l = adjoints.len();
            for idx in 0..l {
                if n.0.deref().borrow_mut().inp[idx]
                    .0
                    .deref()
                    .borrow_mut()
                    .adj_accum
                    .is_none()
                {
                    n.0.deref().borrow_mut().inp[idx]
                        .0
                        .deref()
                        .borrow_mut()
                        .adj_accum = Some(VWrap::new(OpZero::new()));
                }

                let temp = n.0.deref().borrow().inp[idx]
                    .0
                    .deref()
                    .borrow()
                    .adj_accum
                    .as_ref()
                    .unwrap()
                    .clone();

                n.0.deref().borrow_mut().inp[idx]
                    .0
                    .deref()
                    .borrow_mut()
                    .adj_accum = Some(Add(temp, adjoints[idx].clone()));
            }

            //reset adjoint accumulation for current node to zero
            if !n.0.deref().borrow().inp.is_empty() {
                //reset adjoints for internal nodes
                n.0.deref().borrow_mut().adj_accum = None;
            } else {
                //collect adjoints for leaf nodes
                let adj = n.0.deref().borrow_mut().adj_accum.take();
                adjoints_collected.insert(n.clone(), adj.expect("leaf adjoint missing"));
            }

            //do adjoints for inputs
            for i in n.0.deref().borrow().inp.iter() {
                q.push_back(i.clone());
            }

            visited.insert(n.clone());
        }

        adjoints_collected
    }

    /// create tangent-linear starting from current variable
    pub fn fwd(&self) -> PtrVWrap {
        let mut g = self.0.deref().borrow().raw.tangent();
        let ret = g(self.0.deref().borrow().inp.clone(), self);
        ret
    }

    /// indicator in fwd propagation
    pub fn active(&mut self) -> Self {
        self.0.deref().borrow_mut().eval_g = true;
        self.clone()
    }

    pub fn inactive(&mut self) -> Self {
        self.0.deref().borrow_mut().eval_g = false;
        self.clone()
    }

    pub fn adjoint(&self) -> Option<PtrVWrap> {
        self.0.deref().borrow().adj_accum.clone()
    }

    pub fn reset_adjoint(&mut self) {
        self.0.deref().borrow_mut().adj_accum = None;
    }
}

/// wrapper for function
trait FWrap: std::fmt::Debug {
    fn new() -> Box<dyn FWrap>
    where
        Self: Sized;

    /// creates a function to evaluate given values
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType, bool)>, Option<ValType>) -> ValType>;

    /// creates a function to evaluate given values for reverse pass
    fn f_rev(&self) -> Box<dyn FnMut(Vec<(ValType, bool)>, Option<ValType>) -> ValType> {
        self.f()
    }

    /// creates linear tangent function with given input dependencies and returns wrapped variable
    /// used in forward mode
    fn tangent(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, &PtrVWrap) -> PtrVWrap>;

    /// creates function to compute the adjoint for the input dependencies
    /// used in reverse mode
    fn adjoint(
        &self,
    ) -> Box<
        dyn FnMut(
            Vec<PtrVWrap>, /*inputs*/
            PtrVWrap,      /*accumulated adjoint*/
            &PtrVWrap,     /*self*/
        ) -> Vec<PtrVWrap>,
    >;
}

#[derive(Debug, Clone, Copy)]
struct OpMul {}
#[derive(Debug, Clone, Copy)]
struct OpAdd {}
#[derive(Debug, Clone, Copy)]
struct OpLeaf {}
#[derive(Debug, Clone, Copy)]
struct OpOne {}
/// special link to variable of interest for gradient calc
#[derive(Debug, Clone, Copy)]
struct OpLink {}
#[derive(Debug, Clone, Copy)]
struct OpZero {}
#[derive(Debug, Clone, Copy)]
struct OpConst {}

impl FWrap for OpMul {
    fn new() -> Box<dyn FWrap>
    where
        Self: Sized,
    {
        Box::new(OpMul {})
    }
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType, bool)>, Option<ValType>) -> ValType> {
        Box::new(move |x: Vec<(ValType, bool)>, _: Option<ValType>| {
            assert!(x.len() == 2);
            match (x[0].0, x[1].0) {
                (ValType::F(v0), ValType::F(v1)) => ValType::F(v0 * v1),
                (ValType::I(v0), ValType::I(v1)) => ValType::I(v0 * v1),
                (ValType::F(v0), ValType::I(v1)) => ValType::F(v0 * v1 as f32),
                (ValType::I(v0), ValType::F(v1)) => ValType::F(v0 as f32 * v1),
                _ => {
                    panic!("type not supported");
                }
            }
        })
    }
    fn tangent(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, &PtrVWrap) -> PtrVWrap> {
        Box::new(move |args: Vec<PtrVWrap>, _: &PtrVWrap| {
            assert!(args.len() == 2);

            //apply chain rule: (xy)' = x'y + xy'

            let a_prime = args[0].fwd();
            let m1 = VWrap::new_with_input(OpMul::new(), vec![a_prime, args[1].clone()]);

            let b_prime = args[1].fwd();
            let m2 = VWrap::new_with_input(OpMul::new(), vec![args[0].clone(), b_prime]);

            VWrap::new_with_input(OpAdd::new(), vec![m1, m2])
        })
    }
    fn adjoint(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, PtrVWrap, &PtrVWrap) -> Vec<PtrVWrap>> {
        Box::new(
            move |inputs: Vec<PtrVWrap>, out_adj: PtrVWrap, cur: &PtrVWrap| {
                assert_eq!(inputs.len(), 2);
                vec![
                    Mul(inputs[1].clone(), out_adj.clone()),
                    Mul(inputs[0].clone(), out_adj),
                ]
            },
        )
    }
}

impl FWrap for OpAdd {
    fn new() -> Box<dyn FWrap>
    where
        Self: Sized,
    {
        Box::new(OpAdd {})
    }
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType, bool)>, Option<ValType>) -> ValType> {
        Box::new(move |x: Vec<(ValType, bool)>, _: Option<ValType>| {
            assert_eq!(x.len(), 2);
            match (x[0].0, x[1].0) {
                (ValType::F(v0), ValType::F(v1)) => ValType::F(v0 + v1),
                (ValType::I(v0), ValType::I(v1)) => ValType::I(v0 + v1),
                _ => {
                    panic!("type not supported");
                }
            }
        })
    }
    fn tangent(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, &PtrVWrap) -> PtrVWrap> {
        Box::new(move |args: Vec<PtrVWrap>, _: &PtrVWrap| {
            //apply rule: (a+b+c+...)' = a'+b'+c'+...

            let mut inp_grad = vec![];

            for i in args.iter() {
                let d = i.fwd();
                inp_grad.push(d);
            }

            assert!(inp_grad.len() > 0);

            let count = inp_grad.len();

            if count > 1 {
                for i in 1..count {
                    let temp = VWrap::new_with_input(
                        OpAdd::new(),
                        vec![inp_grad[i - 1].clone(), inp_grad[i].clone()],
                    );

                    inp_grad[i] = temp;
                }
            }
            inp_grad[count - 1].clone()
        })
    }
    fn adjoint(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, PtrVWrap, &PtrVWrap) -> Vec<PtrVWrap>> {
        Box::new(
            move |inputs: Vec<PtrVWrap>, out_adj: PtrVWrap, cur: &PtrVWrap| {
                assert_eq!(inputs.len(), 2);
                vec![out_adj.clone(), out_adj]
            },
        )
    }
}

impl FWrap for OpLeaf {
    fn new() -> Box<dyn FWrap>
    where
        Self: Sized,
    {
        Box::new(OpLeaf {})
    }
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType, bool)>, Option<ValType>) -> ValType> {
        Box::new(move |_x: Vec<(ValType, bool)>, v: Option<ValType>| v.expect("leaf value missing"))
    }
    fn tangent(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, &PtrVWrap) -> PtrVWrap> {
        Box::new(move |_args: Vec<PtrVWrap>, self_ptr: &PtrVWrap| {
            VWrap::new_with_input(OpLink::new(), vec![self_ptr.clone()])
        })
    }
    fn adjoint(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, PtrVWrap, &PtrVWrap) -> Vec<PtrVWrap>> {
        Box::new(
            move |inputs: Vec<PtrVWrap>, out_adj: PtrVWrap, cur: &PtrVWrap| {
                assert_eq!(inputs.len(), 0);
                vec![]
            },
        )
    }
}

/// special construct for representing derivative of a variable created in tangent-linear pass
impl FWrap for OpLink {
    fn new() -> Box<dyn FWrap>
    where
        Self: Sized,
    {
        Box::new(OpLink {})
    }
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType, bool)>, Option<ValType>) -> ValType> {
        Box::new(move |x: Vec<(ValType, bool)>, _v: Option<ValType>| {
            assert!(x.len() == 1);
            if x[0].1 {
                //indicator for calculating gradient of the linked variable
                ValType::F(1.)
            } else {
                ValType::F(0.)
            }
        })
    }
    fn tangent(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, &PtrVWrap) -> PtrVWrap> {
        Box::new(move |_args: Vec<PtrVWrap>, self_ptr: &PtrVWrap| {
            VWrap::new_with_val(OpZero::new(), ValType::F(0.))
        })
    }
    fn adjoint(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, PtrVWrap, &PtrVWrap) -> Vec<PtrVWrap>> {
        Box::new(
            move |inputs: Vec<PtrVWrap>, out_adj: PtrVWrap, cur: &PtrVWrap| {
                vec![VWrap::new_with_val(OpZero::new(), ValType::F(0.)); inputs.len()]
            },
        )
    }
}

impl FWrap for OpConst {
    fn new() -> Box<dyn FWrap>
    where
        Self: Sized,
    {
        Box::new(OpConst {})
    }
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType, bool)>, Option<ValType>) -> ValType> {
        Box::new(move |_x: Vec<(ValType, bool)>, v: Option<ValType>| v.expect("leaf value missing"))
    }
    fn tangent(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, &PtrVWrap) -> PtrVWrap> {
        Box::new(move |_args: Vec<PtrVWrap>, self_ptr: &PtrVWrap| {
            VWrap::new_with_val(OpZero::new(), ValType::F(0.))
        })
    }
    fn adjoint(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, PtrVWrap, &PtrVWrap) -> Vec<PtrVWrap>> {
        Box::new(
            move |inputs: Vec<PtrVWrap>, out_adj: PtrVWrap, cur: &PtrVWrap| {
                assert_eq!(inputs.len(), 0);
                vec![]
            },
        )
    }
}

impl FWrap for OpOne {
    fn new() -> Box<dyn FWrap>
    where
        Self: Sized,
    {
        Box::new(OpOne {})
    }
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType, bool)>, Option<ValType>) -> ValType> {
        Box::new(move |_x: Vec<(ValType, bool)>, _v: Option<ValType>| ValType::F(1.))
    }
    fn tangent(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, &PtrVWrap) -> PtrVWrap> {
        Box::new(move |_args: Vec<PtrVWrap>, self_ptr: &PtrVWrap| {
            VWrap::new_with_val(OpZero::new(), ValType::F(0.))
        })
    }
    fn adjoint(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, PtrVWrap, &PtrVWrap) -> Vec<PtrVWrap>> {
        Box::new(
            move |inputs: Vec<PtrVWrap>, out_adj: PtrVWrap, cur: &PtrVWrap| {
                assert_eq!(inputs.len(), 0);
                vec![]
            },
        )
    }
}

impl FWrap for OpZero {
    fn new() -> Box<dyn FWrap>
    where
        Self: Sized,
    {
        Box::new(OpZero {})
    }
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType, bool)>, Option<ValType>) -> ValType> {
        Box::new(move |_x: Vec<(ValType, bool)>, _v: Option<ValType>| {
            //todo
            ValType::F(0.)
        })
    }
    fn tangent(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, &PtrVWrap) -> PtrVWrap> {
        Box::new(move |_args: Vec<PtrVWrap>, self_ptr: &PtrVWrap| {
            VWrap::new_with_val(OpZero::new(), ValType::F(0.))
        })
    }
    fn adjoint(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, PtrVWrap, &PtrVWrap) -> Vec<PtrVWrap>> {
        Box::new(
            move |inputs: Vec<PtrVWrap>, out_adj: PtrVWrap, cur: &PtrVWrap| {
                assert_eq!(inputs.len(), 0);
                vec![]
            },
        )
    }
}

#[allow(dead_code)]
pub fn Mul(arg0: PtrVWrap, arg1: PtrVWrap) -> PtrVWrap {
    let mut a = VWrap::new(OpMul::new());
    a.set_inp(vec![arg0, arg1]);
    a
}

#[allow(dead_code)]
pub fn Add(arg0: PtrVWrap, arg1: PtrVWrap) -> PtrVWrap {
    let mut a = VWrap::new(OpAdd::new());
    a.set_inp(vec![arg0, arg1]);
    a
}

#[allow(dead_code)]
pub fn Leaf(arg0: ValType) -> PtrVWrap {
    let a = VWrap::new_with_val(OpLeaf::new(), arg0);
    a
}

#[cfg(test)]
fn eq_f32(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.01
}

#[test]
fn test_loop_fwd() {
    let mut l0 = Leaf(ValType::F(2.)).active();

    let mut l = l0.clone();
    for _ in 0..10 {
        l = Mul(l, Leaf(ValType::F(2.)));
    }

    let vl = l.apply_fwd();

    dbg!(vl);

    assert!(eq_f32(vl.into(), 2048.));

    let mut g = l.fwd();
    let h = g.apply_fwd();

    dbg!(h);

    assert!(eq_f32(h.into(), 1024.));
}

#[test]
fn test_simple_fwd() {
    //(3x)' = 3

    let mut l0 = Leaf(ValType::F(4.)).active();
    let mut l1 = Leaf(ValType::F(3.));
    let mut a = Mul(l0.clone(), l1.clone());

    let mut b = a.fwd();

    let c = b.apply_fwd();

    dbg!(c);

    assert!(eq_f32(c.into(), 3.));
}

#[test]
fn test_square_fwd() {
    //(3x^2)' = 6x{x=4} = 24
    let mut l0 = Leaf(ValType::F(4.)).active();
    let mut l1 = Leaf(ValType::F(3.));
    let mut a = Mul(Mul(l0.clone(), l0.clone()), l1);

    let mut b = a.fwd();

    let c = b.apply_fwd();

    dbg!(&c);

    // dbg!( &l0 );

    assert!(eq_f32(c.into(), 24.));
}

#[test]
fn test_square_fwd_2() {
    //(x(4)^2)' = 16
    let mut l0 = Leaf(ValType::F(4.));
    let mut l1 = Leaf(ValType::F(3.)).active();
    let mut a = Mul(Mul(l0.clone(), l0.clone()), l1);

    let mut b = a.fwd();

    let c = b.apply_fwd();

    dbg!(&c);

    assert!(eq_f32(c.into(), 16.));
}

#[test]
fn test_simple_rev() {
    //(3x)' = 3
    let mut l0 = Leaf(ValType::F(4.));
    let mut l1 = Leaf(ValType::F(3.));
    let mut a = Mul(l0.clone(), l1.clone());

    let ret = a
        .rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing")
        .apply_rev();

    dbg!(ret);

    assert!(eq_f32(ret.into(), 3.));
}

#[test]
fn test_simple_rev_2() {
    //(3x^2)' = 6x{x=4} = 24
    let mut l0 = Leaf(ValType::F(4.));
    let mut l1 = Leaf(ValType::F(3.));
    let mut a = Mul(Mul(l0.clone(), l0.clone()), l1.clone());

    let ret = a
        .rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing")
        .apply_rev();

    dbg!(&ret);

    assert!(eq_f32(ret.into(), 24.));
}

#[test]
fn test_composite_fwd_over_fwd() {
    //y=3*x^2 where x=4
    //compute y'' = (6x)' = 6

    let mut l0 = Leaf(ValType::F(4.)).active();
    let mut l1 = Leaf(ValType::F(3.));
    let mut a = Mul(Mul(l0.clone(), l0.clone()), l1.clone());

    let mut gg = a.fwd().fwd();

    let ret = gg.apply_fwd();

    dbg!(&ret);

    assert!(eq_f32(ret.into(), 6.));
}

#[test]
fn test_composite_fwd_over_rev() {
    //y=x*3 where x=4
    //compute y'' = (3)' = 0

    let mut l0 = Leaf(ValType::F(4.)).active();
    let mut l1 = Leaf(ValType::F(3.));
    let mut a = Mul(l0.clone(), l1.clone());

    let mut adjoints = a.rev();

    let mut adj = adjoints.get_mut(&l0).expect("l0 adjoint missing");

    let mut g = adj.fwd();

    let ret = g.apply_fwd();

    dbg!(&ret);

    assert!(eq_f32(ret.into(), 0.));
}

#[test]
fn test_composite_fwd_over_rev_2() {
    //y=3*x^2 where x=4
    //compute y'' = (6x)' = 6

    let mut l0 = Leaf(ValType::F(4.)).active();
    let mut l1 = Leaf(ValType::F(3.));
    let mut a = Mul(Mul(l0.clone(), l0.clone()), l1.clone());

    let ret = a
        .rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing")
        .fwd()
        .apply_fwd();

    dbg!(&ret);

    assert!(eq_f32(ret.into(), 6.));
}

#[test]
fn test_composite_rev_over_rev() {
    //(3x^2)'' = 6
    let mut l0 = Leaf(ValType::F(4.));
    let mut l1 = Leaf(ValType::F(3.));
    let mut a = Mul(Mul(l0.clone(), l0.clone()), l1.clone());

    let ret = a
        .rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing")
        .rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing")
        .apply_rev();

    dbg!(&ret);

    assert!(eq_f32(ret.into(), 6.));
}

#[test]
fn test_composite_rev_over_fwd() {
    //(3x^2)'' = 6

    let mut l0 = Leaf(ValType::F(4.)).active();
    let mut l1 = Leaf(ValType::F(3.));
    let mut a = Mul(Mul(l0.clone(), l0.clone()), l1.clone());

    let ret = a
        .fwd()
        .rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing")
        .apply_rev();

    dbg!(&ret);

    assert!(eq_f32(ret.into(), 6.));
}

#[test]
fn test_composite_rev_over_fwd_change_input() {
    //(3x^2)'' = 6

    let mut l0 = Leaf(ValType::F(4.)).active();
    let mut l1 = Leaf(ValType::F(3.));
    let mut a = Mul(Mul(l0.clone(), l0.clone()), l1.clone());

    let mut gg = a
        .fwd()
        .rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing")
        .clone();

    let ret = gg.apply_rev();

    dbg!(&ret);

    assert!(eq_f32(ret.into(), 6.));

    //change to (7x^2)''=(14x)'=14
    l1.set_val(ValType::F(7.));

    let ret2 = gg.apply_rev();

    dbg!(&ret2);

    assert!(eq_f32(ret2.into(), 14.));
}

#[test]
fn test_composite_rev_over_rev_change_input() {
    //(3x^2)'' = 6

    let mut l0 = Leaf(ValType::F(4.));
    let mut l1 = Leaf(ValType::F(3.));
    let mut a = Mul(Mul(l0.clone(), l0.clone()), l1.clone());

    let mut gg = a
        .rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing")
        .rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing")
        .clone();

    let ret = gg.apply_rev();

    dbg!(&ret);

    assert!(eq_f32(ret.into(), 6.));

    //change to (7x^2)''=(14x)'=14
    l1.set_val(ValType::F(7.));

    let ret2 = gg.apply_rev();

    dbg!(&ret2);

    assert!(eq_f32(ret2.into(), 14.));
}

#[test]
fn test_composite_fwd_over_rev_change_input() {
    //(3x^2)'' = 6

    let mut l0 = Leaf(ValType::F(4.)).active();
    let mut l1 = Leaf(ValType::F(3.));
    let mut a = Mul(Mul(l0.clone(), l0.clone()), l1.clone());

    let mut gg = a.rev().get_mut(&l0).expect("l0 adjoint missing").fwd();

    let ret = gg.apply_fwd();

    dbg!(&ret);

    assert!(eq_f32(ret.into(), 6.));

    //change to (7x^2)''=(14x)'=14
    l1.set_val(ValType::F(7.));

    let ret2 = gg.apply_fwd();

    dbg!(&ret2);

    assert!(eq_f32(ret2.into(), 14.));
}

#[test]
fn test_composite_fwd_over_fwd_change_input() {
    //y=3*x^2 where x=4
    //compute y'' = (6x)' = 6

    let mut l0 = Leaf(ValType::F(4.)).active();
    let mut l1 = Leaf(ValType::F(3.));
    let mut a = Mul(Mul(l0.clone(), l0.clone()), l1.clone());

    let mut gg = a.fwd().fwd();

    let ret = gg.apply_fwd();

    dbg!(&ret);

    assert!(eq_f32(ret.into(), 6.));

    //change to (7x^2)''=(14x)'=14
    l1.set_val(ValType::F(7.));

    let ret2 = gg.apply_fwd();

    dbg!(&ret2);

    assert!(eq_f32(ret2.into(), 14.));
}

#[test]
fn test_2nd_partial_derivative() {
    //x=4
    //y=3
    //f=x^2 * y^2
    //d^2(f)/(dx dy) = d(d(x^2 * y^2)/dx)/dy = d(2x*y^2)/dy = 2x*2y = 4*4*3 = 48

    let mut l0 = Leaf(ValType::F(4.));
    let mut l1 = Leaf(ValType::F(3.));
    let mut a = Mul(Mul(l0.clone(), l0.clone()), Mul(l1.clone(), l1.clone()));

    let mut gg = a
        .rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing")
        .rev()
        .get_mut(&l1)
        .expect("l1 adjoint missing")
        .clone();

    let ret = gg.apply_rev();

    dbg!(&ret);

    assert!(eq_f32(ret.into(), 48.));
}
