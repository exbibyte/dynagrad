/// An implementation of dynamic automatic differentiation

use std::rc::{Rc,Weak};
use std::cell::RefCell;

type PtrWrap = Rc<RefCell<Wrap>>;
type WeakWrap = Weak<RefCell<Wrap>>;

pub struct Wrap {
    pub inp: Vec<PtrWrap>,
    pub raw: Box<dyn Var>,
}

impl Wrap {
    fn new( v: Box<dyn Var> ) -> PtrWrap {
        Rc::new( RefCell::new( Wrap {
            inp: vec![],
            raw: v
        } ) )
    }
    fn set_inp( & mut self, v: Vec<PtrWrap> ) {
        self.inp = v;
    }
    fn eval(&self) -> f32 {
        let mut args = vec![];
        
        for i in self.inp.iter() {
            let w = i.borrow_mut();
            let val = w.eval();
            args.push(val);
        }
        
        self.raw.f()(args)
    }
    fn dual(&self) -> PtrWrap {
        self.raw.dual( self.inp.clone() )
    }
}

trait Var {
    fn new() -> Box<dyn Var > where Self: Sized;
    fn f(&self) -> Box<dyn FnMut(Vec<f32>) -> f32 >;
    fn primal(&self) -> Box<dyn Var>;
    fn dual(&self, args: Vec<PtrWrap> ) -> PtrWrap;
}

struct OpX {} //marker for indicating a variable for differentiation
struct OpConst { v: f32 }
struct OpSin {}
struct OpCos {}
struct OpMul {}
struct OpAdd {}

impl Var for OpX {
    fn new() -> Box<dyn Var> where Self: Sized {
        Box::new( OpX{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<f32>) -> f32 > {
        Box::new( move |x:Vec<f32>| 1. )
    }
    fn primal(&self) -> Box<dyn Var> {
        Box::new( OpX{} )
    }
    fn dual(&self, _inp: Vec<PtrWrap> ) -> PtrWrap {
        Const(1.)
   }
}

impl Var for OpConst {
    fn new() -> Box<dyn Var> where Self: Sized {
        Box::new( OpConst{ v: 0. } )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<f32>) -> f32 > {
        let val = self.v;
        Box::new( move |x:Vec<f32>| val )
    }
    fn primal(&self) -> Box<dyn Var> {
        Box::new( OpConst{ v: self.v } )
    }
    fn dual(&self, _inp: Vec<PtrWrap> ) -> PtrWrap {
        Const(0.)
   }
}

impl OpConst {
    fn new_with( v: f32 ) -> Box<dyn Var>{
        Box::new( OpConst{ v: v } )
    }
}

impl Var for OpSin {
    fn new() -> Box<dyn Var> where Self: Sized {
        Box::new( OpSin{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<f32>) -> f32 > {
        Box::new( |x:Vec<f32>| x[0].sin() )
    }
    fn primal(&self) -> Box<dyn Var>{
        Self::new()
    }
    fn dual(&self, inp: Vec<PtrWrap> ) -> PtrWrap {

        let mut inp_dual = vec![];
        
        for i in inp.iter() {
            let w = i.borrow_mut();
            let d = w.dual();
            inp_dual.push(d);
        }
        
        Rc::new( RefCell::new( Wrap{
            inp: inp_dual,
            raw: OpCos::new(),
        } ) )

    }
}

impl Var for OpCos {
    fn new() -> Box<dyn Var> where Self: Sized {
        Box::new( OpCos{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<f32>) -> f32 > {
        Box::new( |x:Vec<f32>| x[0].cos() )
    }
    fn primal(&self) -> Box<dyn Var>{
        Self::new()
    }
    fn dual(&self, inp: Vec<PtrWrap> ) -> PtrWrap {
        
        let mut inp_dual = vec![];
        
        for i in inp.iter() {
            let w = i.borrow_mut();
            let d = w.dual();
            inp_dual.push(d);
        }
        
        Rc::new( RefCell::new( Wrap{
            inp: inp_dual,
            raw: OpSin::new(),
        } ) )
    }
}

impl Var for OpMul {
    fn new() -> Box<dyn Var> where Self: Sized {
        Box::new( OpMul{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<f32>) -> f32 > {
        Box::new( |x:Vec<f32>| {
            assert!( x.len() > 1 );
            x[0]*x[1] } )
    }
    fn primal(&self) -> Box<dyn Var>{
        Self::new()
    }
    fn dual(&self, args: Vec<PtrWrap> ) -> PtrWrap {

        //only consider 2 input arguments

        assert!( args.len() >= 2 );

        //apply chain rule: (xy)' = x'y + xy'

        let a_prime = args[0].borrow_mut().dual();
        let m1 = Rc::new( RefCell::new(
            Wrap {
                inp: vec![ a_prime,
                           args[1].clone() ],
                raw: OpMul::new(),
            }
        ) );

        let b_prime = args[1].borrow_mut().dual();
        let m2 = Rc::new( RefCell::new(
            Wrap {
                inp: vec![ args[0].clone(),
                           b_prime ],
                raw: OpMul::new(),
            }
        ) );
        
        Rc::new( RefCell::new(
            Wrap {
                inp: vec![ m1, m2 ],
                raw: OpAdd::new(),
            }
        ) )
    }
}

impl Var for OpAdd {
    fn new() -> Box<dyn Var> where Self: Sized {
        Box::new( OpAdd{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<f32>) -> f32 > {
        Box::new( |x:Vec<f32>| {
            assert!( x.len() > 1 );
            x[0]+x[1]
        } )
    }
    fn primal(&self) -> Box<dyn Var>{
        Self::new()
    }
    fn dual(&self, args: Vec<PtrWrap> ) -> PtrWrap {

        
        //apply rule: (a+b+c+...)' = a'+b'+c'+...
        
        let mut inp_dual = vec![];
        
        for i in args.iter() {
            let w = i.borrow_mut();
            let d = w.dual();
            inp_dual.push(d);
        }

        assert!( inp_dual.len() > 0 );

        let count = inp_dual.len();
        
        if count > 1 {
            for i in 1..count {   
                let temp = Rc::new( RefCell::new( Wrap{
                    inp: vec![ inp_dual[i-1].clone(), inp_dual[i].clone() ],
                    raw: OpAdd::new(),
                } ) );

                inp_dual[i] = temp;
            }
        }
        inp_dual[count-1].clone()
    }
}

fn Const( inp: f32 ) -> PtrWrap {
    let a = Wrap::new( OpConst::new_with( inp ) );
    a
}

fn Sin( inp: PtrWrap ) -> PtrWrap {
    let a = Wrap::new( OpSin::new() );
    a.borrow_mut().set_inp( vec![ inp.clone() ] );
    a
}

fn Mul( arg0: PtrWrap, arg1: PtrWrap ) -> PtrWrap {
    let a = Wrap::new( OpMul::new() );
    a.borrow_mut().set_inp( vec![ arg0, arg1 ] );
    a
}

fn Add( arg0: PtrWrap, arg1: PtrWrap ) -> PtrWrap {
    let a = Wrap::new( OpAdd::new() );
    a.borrow_mut().set_inp( vec![ arg0, arg1 ] );
    a
}

fn X() -> PtrWrap {
    let a = Wrap::new( OpX::new() );
    a
}

fn test_fun_1( a: PtrWrap, b: PtrWrap ) -> PtrWrap {
    Mul(a.clone(),Add(a.clone(),b.clone()))
}
    
#[test]
fn test() {

    // let a = Const(3.);
    // let b = X();

    // let c = Const(5.);
    // let d = Const(6.);

    // let e = Mul(a,b);
    // let f = Mul(c,d);

    // let g = Const(2.);
    
    // let h = Add(e,f);
    
    // let i = Mul(h,g);
    
    // let y = i.borrow().eval();
    // let dy = i.borrow().dual().borrow().eval();
    // let ddy = i.borrow().dual().borrow().dual().borrow().eval();
    
    // dbg!(y);
    // dbg!(dy);
    // dbg!(ddy);

    let a = Const(3.);
    let b = X();
    let c = Mul(a,b);
    let d = Const(5.);
    let y = test_fun_1( d, c );
    let ans = y.borrow().eval();
    dbg!(ans);
    let ans2 = y.borrow().dual().borrow().eval();
    dbg!(ans2);

    let ans3 = y.borrow().dual().borrow().dual().borrow().eval();
    dbg!(ans3);
}
