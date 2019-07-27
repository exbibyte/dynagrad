#![allow(non_snake_case)]
/// An implementation of dynamic automatic differentiation

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
        } ) ) )
    }

    fn new_with_input( f: Box<dyn FWrap>, v: Vec<PtrVWrap> ) -> PtrVWrap {
        PtrVWrap( Rc::new( RefCell::new( VWrap {
            inp: v,
            raw: f,
            val: None,
            id: get_id(),
            eval_g: false,
        } ) ) )
    }

    fn new_with_val( v: Box<dyn FWrap>, val: ValType ) -> PtrVWrap {
        PtrVWrap( Rc::new( RefCell::new( VWrap {
            inp: vec![],
            raw: v,
            val: Some(val),
            id: get_id(),
            eval_g: false,
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
    
    //create a gradient function wrt. to the current variable
    pub fn grad(&self) -> PtrVWrap {
        let mut g = self.0.borrow().raw.g();
        let ret = g( self.0.borrow().inp.clone(), self );
        ret
    }

    fn active(&mut self) -> Self {
        self.0.borrow_mut().eval_g = true;
        self.clone()
    }
    
    // pub fn deep_clone(&self) -> PtrVWrap {
    //     PtrVWrap( Rc::new( RefCell::new( VWrap {
    //         inp: vec![],
    //         raw: OpConst::new(),
    //         val: self.0.borrow().val,
    //         id: get_id(),
    //     } ) ) )
    // }
}

/// wrapper for function
trait FWrap : std::fmt::Debug {
    
    fn new() -> Box<dyn FWrap > where Self: Sized;

    ///creates a function to evaluate given values
    fn f(&self) -> Box<dyn FnMut(Vec<(ValType,bool)>, Option<ValType>) -> ValType >;

    ///creates gradient function with given input dependencies and returns wrapped variable
    fn g(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, &PtrVWrap) -> PtrVWrap >;

    // fn adjoint(&self) -> Box<dyn FnMut(Vec<PtrVWrap>, &PtrVWrap) -> PtrVWrap >;
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
    fn g(&self) -> Box<dyn FnMut(Vec<PtrVWrap>,&PtrVWrap) -> PtrVWrap > {
        Box::new( move |args:Vec<PtrVWrap>,_:&PtrVWrap| {
            
            assert!(args.len()==2);
            
            //apply chain rule: (xy)' = x'y + xy'

            let a_prime = args[0].grad();
            let m1 = VWrap::new_with_input( OpMul::new(),
                                            vec![ a_prime,
                                                  args[1].clone() ] );

            let b_prime = args[1].grad();
            let m2 = VWrap::new_with_input( OpMul::new(),
                                            vec![ args[0].clone(),
                                                  b_prime ] );

            VWrap::new_with_input( OpAdd::new(),
                                   vec![ m1, m2 ] )
        })
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
    fn g(&self) -> Box<dyn FnMut(Vec<PtrVWrap>,&PtrVWrap) -> PtrVWrap > {

        Box::new( move |args: Vec<PtrVWrap>,_:&PtrVWrap| {

            //apply rule: (a+b+c+...)' = a'+b'+c'+...
            
            let mut inp_grad = vec![];
            
            for i in args.iter() {
                let d = i.grad();
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
    fn g(&self) -> Box<dyn FnMut(Vec<PtrVWrap>,&PtrVWrap) -> PtrVWrap > {
        Box::new( move |_args: Vec<PtrVWrap>,self_ptr:&PtrVWrap| {
            VWrap::new_with_input( OpLink::new(), vec![self_ptr.clone()] )
        })
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
    fn g(&self) -> Box<dyn FnMut(Vec<PtrVWrap>,&PtrVWrap) -> PtrVWrap > {
        Box::new( move |_args: Vec<PtrVWrap>,self_ptr:&PtrVWrap| {
            VWrap::new_with_val( OpZero::new(), ValType::F(0.) )
        })
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
    fn g(&self) -> Box<dyn FnMut(Vec<PtrVWrap>,&PtrVWrap) -> PtrVWrap > {
        Box::new( move |_args: Vec<PtrVWrap>,self_ptr:&PtrVWrap| {
            VWrap::new_with_val( OpZero::new(), ValType::F(0.) )
        })
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
    fn g(&self) -> Box<dyn FnMut(Vec<PtrVWrap>,&PtrVWrap) -> PtrVWrap > {
        Box::new( move |_args: Vec<PtrVWrap>,self_ptr:&PtrVWrap| {
            VWrap::new_with_val( OpZero::new(), ValType::F(0.) )
        })
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
    fn g(&self) -> Box<dyn FnMut(Vec<PtrVWrap>,&PtrVWrap) -> PtrVWrap > {
        Box::new( move |_args: Vec<PtrVWrap>,self_ptr:&PtrVWrap| {
            VWrap::new_with_val( OpZero::new(), ValType::F(0.) )
        })
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

    // let mut b = a.grad();
    
    // let c = b.apply_fwd();
    // dbg!(c);

    // l0.set_val( ValType::F(2.) );
    // let d = b.apply_fwd();
    // dbg!(d);

    let mut l0 = Leaf( ValType::F(2.) ).active();

    let mut l = l0.clone();
    for _ in 0..10{
        l = Mul( l, Leaf( ValType::F(2.) ) );
    }

    dbg!(l.apply_fwd());
    
    let mut g = l.grad();
    let h = g.apply_fwd();
    dbg!(h);
    
    // let mut l0 = Leaf( ValType::F(4.) );
    // let mut a = Mul( Mul( l0.clone(), l0.clone() ), l0.clone() );

    // let mut b = a.grad().grad().grad();
    
    // let c = b.apply_fwd();
    // dbg!(c);

    //todo: reverse mode
}
