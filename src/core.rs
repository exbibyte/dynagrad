//! An implementation of dynamic automatic differentiation
//! with forward and reverse modes

#![allow(non_snake_case)]

use std::rc::{Rc};
use std::sync::{Arc,atomic::{AtomicUsize,Ordering}};
use std::cell::RefCell;

#[derive(Clone,Debug)]
pub struct PtrVWrap(pub Rc<RefCell<VWrap>>);

use crate::valtype::ValType;

lazy_static! {
    static ref ID: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
}

fn get_id() -> i32 {
    ID.fetch_add(1,Ordering::SeqCst) as _
}

/// wrapper for variable with recording of dependencies
#[derive(Debug)]
pub struct VWrap {

    /// input dependencies
    pub inp: Vec<PtrVWrap>,

    /// source function
    raw: Box<dyn FWrap>,

    /// evaluted value
    pub val: Option<ValType>,

    /// for debugging
    pub id: i32,

    pub eval_g: bool,

    /// adjoint accumulation expression
    pub adj_accum: Option<PtrVWrap>,

    // /// adjoint accumulation value at fixpoint
    // pub adj_accum_val: f32,
}

/// initializer functions
#[allow(dead_code)]
impl VWrap {

    fn new( v: Box<dyn FWrap> ) -> PtrVWrap {
        PtrVWrap( Rc::new( RefCell::new( VWrap {
            inp: vec![],
            raw: v,
            val: None,
            id: get_id(),
            eval_g: false,
            adj_accum: None,
        } ) ) )
    }

    fn new_with_input( f: Box<dyn FWrap>, v: Vec<PtrVWrap> ) -> PtrVWrap {
        PtrVWrap( Rc::new( RefCell::new( VWrap {
            inp: v,
            raw: f,
            val: None,
            id: get_id(),
            eval_g: false,
            adj_accum: None,
        } ) ) )
    }

    fn new_with_val( v: Box<dyn FWrap>, val: ValType ) -> PtrVWrap {
        PtrVWrap( Rc::new( RefCell::new( VWrap {
            inp: vec![],
            raw: v,
            val: Some(val),
            id: get_id(),
            eval_g: false,
            adj_accum: None,
        } ) ) )
    }
}

impl PtrVWrap {

    fn set_inp( & mut self, v: Vec<PtrVWrap> ) {
        self.0.borrow_mut().inp = v;
    }

    fn set_val( & mut self, v: ValType ) {
        self.0.borrow_mut().val = Some(v);
    }

    /// forward mode (tanget-linear)
    fn apply_fwd(& mut self) -> ValType {
        
        let mut args : Vec<(ValType,bool)> = vec![];

        //recursive apply
        for i in self.0.borrow_mut().inp.iter_mut() {
            let val = i.apply_fwd();
            args.push((val, i.0.borrow().eval_g));
        }
        
        let v = self.0.borrow().raw.f()( args, self.0.borrow().val );
        
        self.0.borrow_mut().val = Some(v);

        v
    }

    /// reverse mode (adjoint)
    fn apply_rev(& mut self) -> ValType {
        unimplemented!();
    }

    /// create adjoint graph starting from current variable and go through input dependencies
    ///
    /// resulting sensitivity graphs are propagated to leaf nodes' adjoint accumulation
    /// where it can be collected
    fn rev(&self) {
        
        use std::collections::VecDeque;
        
        let mut q = VecDeque::new();
        
        //initialization of sensitity=1 for starting node
        self.0.borrow_mut().adj_accum = Some(VWrap::new( OpOne::new() ));
        
        q.push_back(self.clone());

        //breadth-first
        while !q.is_empty(){
            
            let mut n = q.pop_front().unwrap();

            if n.0.borrow_mut().adj_accum.is_none(){
                n.0.borrow_mut().adj_accum = Some(VWrap::new( OpZero::new() ));
            }

            //delegate adjoint calc to operation
            let mut f = n.0.borrow().raw.adjoint();
            let mut adjoints = f( n.0.borrow().inp.clone(),
                                  n.0.borrow().adj_accum.as_ref().expect("adj_accum empty").clone(),
                                  &n );

            assert_eq!( adjoints.len(), n.0.borrow().inp.len() );
            
            //propagate adjoints to inputs
            let l = adjoints.len();
            for idx in 0..l {
                if n.0.borrow_mut().inp[idx].0.borrow_mut().adj_accum.is_none(){
                    n.0.borrow_mut().inp[idx].0.borrow_mut().adj_accum = Some(VWrap::new( OpZero::new() ));
                }
                n.0.borrow_mut().inp[idx].0.borrow_mut().adj_accum = Some( Add(
                    n.0.borrow_mut().inp[idx].0.borrow().adj_accum.as_ref().unwrap().clone(),
                    adjoints[idx].clone() ) );
            }
            
            //reset adjoint accumulation for current node to zero
            n.0.borrow_mut().adj_accum = None;

            //do adjoints for inputs
            for i in n.0.borrow().inp.iter() {
                q.push_back(i.clone());
            }
        }
    }
    
    //create tangent-linear starting from current variable
    pub fn fwd(&self) -> PtrVWrap {
        let mut g = self.0.borrow().raw.tangent();
        let ret = g( self.0.borrow().inp.clone(), self );
        ret
    }

    fn active(&mut self) -> Self {
        self.0.borrow_mut().eval_g = true;
        self.clone()
    }
}

/// wrapper for function
trait FWrap : std::fmt::Debug {
    
    fn new() -> Box<dyn FWrap > where Self: Sized;

    /// creates a function to evaluate given values
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType,bool)>, Option<ValType>) -> ValType >;

    /// creates linear tangent function with given input dependencies and returns wrapped variable
    /// used in forward mode
    fn tangent(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, &PtrVWrap) -> PtrVWrap >;

    /// creates function to compute the adjoint for the input dependencies
    /// used in reverse mode
    fn adjoint(&self) -> Box<dyn FnMut(Vec<PtrVWrap>/*inputs*/, PtrVWrap/*accumulated adjoint*/, &PtrVWrap/*self*/) -> Vec<PtrVWrap> >;

}

#[derive(Debug,Clone,Copy)]
struct OpMul {}
#[derive(Debug,Clone,Copy)]
struct OpAdd {}
#[derive(Debug,Clone,Copy)]
struct OpLeaf {}
#[derive(Debug,Clone,Copy)]
struct OpOne {}
/// special link to variable of interest for gradient calc
#[derive(Debug,Clone,Copy)]
struct OpLink {}
#[derive(Debug,Clone,Copy)]
struct OpZero {}
#[derive(Debug,Clone,Copy)]
struct OpConst {}

impl FWrap for OpMul {
    fn new() -> Box<dyn FWrap> where Self: Sized {
        Box::new( OpMul{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType,bool)>, Option<ValType>) -> ValType > {
        
        Box::new( move |x:Vec<(ValType,bool)>,_:Option<ValType>| {
            assert!( x.len() == 2 );
            match (x[0].0,x[1].0) {
                (ValType::F(v0), ValType::F(v1)) => ValType::F(v0*v1),
                (ValType::I(v0), ValType::I(v1)) => ValType::I(v0*v1),
                (ValType::F(v0), ValType::I(v1)) => ValType::F(v0 * v1 as f32),
                (ValType::I(v0), ValType::F(v1)) => ValType::F(v0 as f32 * v1),
                _ => { panic!("type not supported"); },
            }
        } )
    }
    fn tangent(&self) -> Box<dyn FnMut(Vec<PtrVWrap>,&PtrVWrap) -> PtrVWrap > {
        Box::new( move |args:Vec<PtrVWrap>,_:&PtrVWrap| {
            
            assert!(args.len()==2);
            
            //apply chain rule: (xy)' = x'y + xy'

            let a_prime = args[0].fwd();
            let m1 = VWrap::new_with_input( OpMul::new(),
                                            vec![ a_prime,
                                                  args[1].clone() ] );

            let b_prime = args[1].fwd();
            let m2 = VWrap::new_with_input( OpMul::new(),
                                            vec![ args[0].clone(),
                                                  b_prime ] );

            VWrap::new_with_input( OpAdd::new(),
                                   vec![ m1, m2 ] )
        })
    }
    fn adjoint(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, PtrVWrap, &PtrVWrap) -> Vec<PtrVWrap> > {
        unimplemented!();
    }    
}

impl FWrap for OpAdd {
    fn new() -> Box<dyn FWrap> where Self: Sized {
        Box::new( OpAdd{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType,bool)>, Option<ValType>) -> ValType > {
        Box::new( move |x:Vec<(ValType,bool)>,_:Option<ValType>| {
            assert!( x.len() == 2 );
            match (x[0].0,x[1].0){
                (ValType::F(v0),ValType::F(v1)) => { ValType::F(v0+v1) },
                (ValType::I(v0),ValType::I(v1)) => { ValType::I(v0+v1) },
                _ => {panic!("type not supported");},
            }
        } )
    }
    fn tangent(&self) -> Box<dyn FnMut(Vec<PtrVWrap>,&PtrVWrap) -> PtrVWrap > {

        Box::new( move |args: Vec<PtrVWrap>,_:&PtrVWrap| {

            //apply rule: (a+b+c+...)' = a'+b'+c'+...
            
            let mut inp_grad = vec![];
            
            for i in args.iter() {
                let d = i.fwd();
                inp_grad.push(d);
            }

            assert!( inp_grad.len() > 0 );

            let count = inp_grad.len();
            
            if count > 1 {
                for i in 1..count {
                    
                    let temp = VWrap::new_with_input(
                        OpAdd::new(),
                        vec![ inp_grad[i-1].clone(), inp_grad[i].clone() ],
                    );

                    inp_grad[i] = temp;
                }
            }
            inp_grad[count-1].clone()
        })
    }
    fn adjoint(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, PtrVWrap, &PtrVWrap) -> Vec<PtrVWrap> > {
        unimplemented!();
    }
}

impl FWrap for OpLeaf {
    fn new() -> Box<dyn FWrap> where Self: Sized {
        Box::new( OpLeaf{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType,bool)>,Option<ValType>) -> ValType > {
        Box::new( move |_x:Vec<(ValType,bool)>,v:Option<ValType>| {
            v.expect("leaf value missing")
        } )
    }
    fn tangent(&self) -> Box<dyn FnMut(Vec<PtrVWrap>,&PtrVWrap) -> PtrVWrap > {
        Box::new( move |_args: Vec<PtrVWrap>,self_ptr:&PtrVWrap| {
            VWrap::new_with_input( OpLink::new(), vec![self_ptr.clone()] )
        })
    }
    fn adjoint(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, PtrVWrap, &PtrVWrap) -> Vec<PtrVWrap> > {
        unimplemented!();
    }
}

impl FWrap for OpLink {
    fn new() -> Box<dyn FWrap> where Self: Sized {
        Box::new( OpLink{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType,bool)>,Option<ValType>) -> ValType > {
        Box::new( move |x:Vec<(ValType,bool)>,_v:Option<ValType>| {
            assert!(x.len()==1);
            if x[0].1 { //indicator for calculating gradient of the linked variable
                ValType::F(1.)
            } else {
                ValType::F(0.)
            }
        } )
    }
    fn tangent(&self) -> Box<dyn FnMut(Vec<PtrVWrap>,&PtrVWrap) -> PtrVWrap > {
        Box::new( move |_args: Vec<PtrVWrap>,self_ptr:&PtrVWrap| {
            VWrap::new_with_val( OpZero::new(), ValType::F(0.) )
        })
    }
    fn adjoint(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, PtrVWrap, &PtrVWrap) -> Vec<PtrVWrap> > {
        unimplemented!();
    }
}

impl FWrap for OpConst {
    fn new() -> Box<dyn FWrap> where Self: Sized {
        Box::new( OpConst{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType,bool)>,Option<ValType>) -> ValType > {
        Box::new( move |_x:Vec<(ValType,bool)>,v:Option<ValType>| {
            v.expect("leaf value missing")
        } )
    }
    fn tangent(&self) -> Box<dyn FnMut(Vec<PtrVWrap>,&PtrVWrap) -> PtrVWrap > {
        Box::new( move |_args: Vec<PtrVWrap>,self_ptr:&PtrVWrap| {
            VWrap::new_with_val( OpZero::new(), ValType::F(0.) )
        })
    }
    fn adjoint(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, PtrVWrap, &PtrVWrap) -> Vec<PtrVWrap> > {
        unimplemented!();
    }
}

impl FWrap for OpOne {
    fn new() -> Box<dyn FWrap> where Self: Sized {
        Box::new( OpOne{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType,bool)>,Option<ValType>) -> ValType > {
        Box::new( move |_x:Vec<(ValType,bool)>,_v:Option<ValType>| {
            //todo
            ValType::F(1.)
        } )
    }
    fn tangent(&self) -> Box<dyn FnMut(Vec<PtrVWrap>,&PtrVWrap) -> PtrVWrap > {
        Box::new( move |_args: Vec<PtrVWrap>,self_ptr:&PtrVWrap| {
            VWrap::new_with_val( OpZero::new(), ValType::F(0.) )
        })
    }
    fn adjoint(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, PtrVWrap, &PtrVWrap) -> Vec<PtrVWrap> > {
        unimplemented!();
    }
}

impl FWrap for OpZero {
    fn new() -> Box<dyn FWrap> where Self: Sized {
        Box::new( OpZero{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType,bool)>,Option<ValType>) -> ValType > {
        Box::new( move |_x:Vec<(ValType,bool)>,_v:Option<ValType>| {
            //todo
            ValType::F(0.)
        } )
    }
    fn tangent(&self) -> Box<dyn FnMut(Vec<PtrVWrap>,&PtrVWrap) -> PtrVWrap > {
        Box::new( move |_args: Vec<PtrVWrap>,self_ptr:&PtrVWrap| {
            VWrap::new_with_val( OpZero::new(), ValType::F(0.) )
        })
    }
    fn adjoint(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, PtrVWrap, &PtrVWrap) -> Vec<PtrVWrap> > {
        unimplemented!();
    }
}

#[allow(dead_code)]
fn Mul( arg0: PtrVWrap, arg1: PtrVWrap ) -> PtrVWrap {
    let mut a = VWrap::new( OpMul::new() );
    a.set_inp( vec![ arg0, arg1 ] );
    a
}

#[allow(dead_code)]
fn Add( arg0: PtrVWrap, arg1: PtrVWrap ) -> PtrVWrap {
    let mut a = VWrap::new( OpAdd::new() );
    a.set_inp( vec![ arg0, arg1 ] );
    a
}

#[allow(dead_code)]
fn Leaf( arg0: ValType ) -> PtrVWrap {
    let a = VWrap::new_with_val( OpLeaf::new(), arg0 );
    a
}

#[test]
fn test(){
    
    // let mut l0 = Leaf( ValType::F(4.) ).active();
    // let mut l1 = Leaf( ValType::F(3.) ).active();
    // let mut a = Mul( l0.clone(), l1.clone() );

    // let mut b = a.fwd();
    
    // let c = b.apply_fwd();
    // dbg!(c);

    // l0.set_val( ValType::F(2.) );
    // let d = b.apply_fwd();
    // dbg!(d);

    let mut l0 = Leaf( ValType::F(2.) ).active();

    let mut l = l0.clone();
    for _ in 0..10 {
        l = Mul( l, Leaf( ValType::F(2.) ) );
    }

    dbg!(l.apply_fwd());
    
    let mut g = l.fwd();
    let h = g.apply_fwd();
    dbg!(h);
    
    // let mut l0 = Leaf( ValType::F(4.) );
    // let mut a = Mul( Mul( l0.clone(), l0.clone() ), l0.clone() );

    // let mut b = a.fwd().fwd().fwd();
    
    // let c = b.apply_fwd();
    // dbg!(c);

    //todo: reverse mode
}
