#![allow(non_snake_case)]
/// An implementation of dynamic automatic differentiation

use std::rc::{Rc,/*Weak*/};
use std::cell::RefCell;

type PtrWrap = Rc<RefCell<Wrap>>;
//type WeakWrap = Weak<RefCell<Wrap>>;

#[derive(Debug)]
pub struct Wrap {
    pub inp: Vec<PtrWrap>,
    raw: Box<dyn Var>,
}

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

#[allow(dead_code)]
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
    fn eval(&self) -> ValType {
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

trait Var : std::fmt::Debug {
    fn new() -> Box<dyn Var > where Self: Sized;
    fn f(&self) -> Box<dyn FnMut(Vec<ValType>) -> ValType >;
    fn primal(&self) -> Box<dyn Var>;
    fn dual(&self, args: Vec<PtrWrap> ) -> PtrWrap;
}

#[derive(Debug,Clone,Copy)]
struct OpX {} //marker for indicating a variable for differentiation
#[derive(Debug,Clone,Copy)]
struct OpConst { v: f32 }
#[derive(Debug,Clone,Copy)]
struct OpConsti { v: i32 }
#[derive(Debug,Clone,Copy)]
struct OpSin {}
#[derive(Debug,Clone,Copy)]
struct OpCos {}
#[derive(Debug,Clone,Copy)]
struct OpMul {}
#[derive(Debug,Clone,Copy)]
struct OpAdd {}
#[derive(Debug,Clone,Copy)]
struct OpSub {}
#[derive(Debug,Clone,Copy)]
struct OpPowi {}

impl Var for OpX {
    fn new() -> Box<dyn Var> where Self: Sized {
        Box::new( OpX{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<ValType>) -> ValType > {
        Box::new( move |_x:Vec<ValType>| ValType::F(1.) )
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
    fn f(&self) -> Box<dyn FnMut(Vec<ValType>) -> ValType > {
        let val = self.v;
        Box::new( move |_x:Vec<ValType>| ValType::F(val) )
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

impl Var for OpConsti {
    fn new() -> Box<dyn Var> where Self: Sized {
        Box::new( OpConsti{ v: 0 } )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<ValType>) -> ValType > {
        let val = self.v;
        Box::new( move |_x:Vec<ValType>| ValType::I(val) )
    }
    fn primal(&self) -> Box<dyn Var> {
        Box::new( OpConsti{ v: self.v } )
    }
    fn dual(&self, _inp: Vec<PtrWrap> ) -> PtrWrap {
        Consti(0)
   }
}

impl OpConsti {
    fn new_with( v: i32 ) -> Box<dyn Var>{
        Box::new( OpConsti{ v: v } )
    }
}

impl Var for OpSin {
    fn new() -> Box<dyn Var> where Self: Sized {
        Box::new( OpSin{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<ValType>) -> ValType > {
        Box::new( |x:Vec<ValType>|
                   match x[0] {
                       ValType::F(v) => ValType::F(v.sin()),
                       _ => {panic!("type not supported");},
                   }
        )
    }
    fn primal(&self) -> Box<dyn Var>{
        Self::new()
    }
    fn dual(&self, inp: Vec<PtrWrap> ) -> PtrWrap {

        let mut inp_dual = vec![];

        //assume 1 input for now
        
        for i in inp.iter().take(1) {
            let w = i.borrow_mut();
            let d = w.dual();
            inp_dual.push(d);
        }

        //(sin(x))'=cos(x)x'
        
        let a = Rc::new( RefCell::new( Wrap{
            inp: inp,
            raw: OpCos::new(),
        } ) );

        let b = Rc::new( RefCell::new( Wrap{
            inp: vec![ a, inp_dual[0].clone() ],
            raw: OpMul::new(),
        } ) );

        b
    }
}

impl Var for OpCos {
    fn new() -> Box<dyn Var> where Self: Sized {
        Box::new( OpCos{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<ValType>) -> ValType > {
        Box::new( |x:Vec<ValType>|
                   match x[0] {
                       ValType::F(v) => ValType::F(v.cos()),
                       _ => {panic!("type not supported");},
                   }
        )
    }
    fn primal(&self) -> Box<dyn Var>{
        Self::new()
    }
    fn dual(&self, inp: Vec<PtrWrap> ) -> PtrWrap {

        //assume 1 input for now
        
        let mut inp_dual = vec![];
        
        for i in inp.iter().take(1) {
            let w = i.borrow_mut();
            let d = w.dual();
            inp_dual.push(d);
        }

        //(cos(x))'=-sin(x)x'
        
        let a = Rc::new( RefCell::new( Wrap{
            inp: inp,
            raw: OpSin::new(),
        } ) );
        
        let b = Rc::new( RefCell::new( Wrap{
            inp: vec![ a, inp_dual[0].clone() ],
            raw: OpMul::new(),
        } ) );

        let c = Rc::new( RefCell::new( Wrap{
            inp: vec![ Const(-1.), b ],
            raw: OpMul::new(),
        } ) );

        c
    }
}

impl Var for OpMul {
    fn new() -> Box<dyn Var> where Self: Sized {
        Box::new( OpMul{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<ValType>) -> ValType > {
        Box::new( |x:Vec<ValType>| {
            assert!( x.len() > 1 );
            match (x[0],x[1]) {
                (ValType::F(v0), ValType::F(v1)) => ValType::F(v0*v1),
                (ValType::I(v0), ValType::I(v1)) => ValType::I(v0*v1),
                (ValType::F(v0), ValType::I(v1)) => ValType::F(v0 * v1 as f32),
                (ValType::I(v0), ValType::F(v1)) => ValType::F(v0 as f32 * v1),
                _ => { panic!("type not supported"); },
            }
        } )
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
    fn f(&self) -> Box<dyn FnMut(Vec<ValType>) -> ValType > {
        Box::new( |x:Vec<ValType>| {
            assert!( x.len() > 1 );
            match (x[0],x[1]){
                (ValType::F(v0),ValType::F(v1)) => { ValType::F(v0+v1) },
                (ValType::I(v0),ValType::I(v1)) => { ValType::I(v0+v1) },
                _ => {panic!("type not supported");},
            }
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

impl Var for OpSub {
    fn new() -> Box<dyn Var> where Self: Sized {
        Box::new( OpSub{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<ValType>) -> ValType > {
        Box::new( |x:Vec<ValType>| {
            assert!( x.len() > 1 );
            match (x[0],x[1]){
                (ValType::F(v0),ValType::F(v1)) => { ValType::F(v0-v1) },
                (ValType::I(v0),ValType::I(v1)) => { ValType::I(v0-v1) },
                _ => {panic!("type not supported");},
            }
        } )
    }
    fn primal(&self) -> Box<dyn Var>{
        Self::new()
    }
    fn dual(&self, args: Vec<PtrWrap> ) -> PtrWrap {

        //apply rule: (a-b-c-...)' = a'-b'-c'+...
        
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
                    raw: OpSub::new(),
                } ) );

                inp_dual[i] = temp;
            }
        }
        inp_dual[count-1].clone()
    }
}

impl Var for OpPowi {
    fn new() -> Box<dyn Var> where Self: Sized {
        Box::new( OpPowi{} )
    }
    fn f(&self) -> Box<dyn FnMut(Vec<ValType>) -> ValType > {
        Box::new( |x:Vec<ValType>| {
            assert!( x.len() >= 2 );
            //x[0]^x[1]
            match (x[0],x[1]) {
                (ValType::F(v0), ValType::I(v1)) => ValType::F(v0.powi(v1)),
                _ => { panic!("type not supported"); }
            }
        } )
    }
    fn primal(&self) -> Box<dyn Var>{
        Self::new()
    }
    fn dual(&self, args: Vec<PtrWrap> ) -> PtrWrap {

        //apply rule: (a^b)'=(b)(a^(b-1))(a') where b is an integer
        //x[0]=a, x[1]=b

        assert!( args.len() >= 2 );
        
        let mut inp_dual = vec![];
        
        for i in args.iter() {
            let w = i.borrow_mut();
            let d = w.dual();
            inp_dual.push(d);
        }

        let a = args[0].clone();
        let a_prime = inp_dual[0].clone();
        let b = args[1].clone();
        
        let temp = Rc::new( RefCell::new( Wrap{
            inp: vec![a, Sub(b.clone(),Consti(1))],
            raw: OpPowi::new(),
        } ) );

        let temp2 = Mul(b, Mul(temp,a_prime));
        temp2
    }
}

#[allow(dead_code)]
fn Const( inp: f32 ) -> PtrWrap {
    let a = Wrap::new( OpConst::new_with( inp ) );
    a
}

#[allow(dead_code)]
fn Consti( inp: i32 ) -> PtrWrap {
    let a = Wrap::new( OpConsti::new_with( inp ) );
    a
}

#[allow(dead_code)]
fn Sin( inp: PtrWrap ) -> PtrWrap {
    let a = Wrap::new( OpSin::new() );
    a.borrow_mut().set_inp( vec![ inp.clone() ] );
    a
}

#[allow(dead_code)]
fn Cos( inp: PtrWrap ) -> PtrWrap {
    let a = Wrap::new( OpCos::new() );
    a.borrow_mut().set_inp( vec![ inp.clone() ] );
    a
}

#[allow(dead_code)]
fn Mul( arg0: PtrWrap, arg1: PtrWrap ) -> PtrWrap {
    let a = Wrap::new( OpMul::new() );
    a.borrow_mut().set_inp( vec![ arg0, arg1 ] );
    a
}

#[allow(dead_code)]
fn Add( arg0: PtrWrap, arg1: PtrWrap ) -> PtrWrap {
    let a = Wrap::new( OpAdd::new() );
    a.borrow_mut().set_inp( vec![ arg0, arg1 ] );
    a
}

#[allow(dead_code)]
fn Sub( arg0: PtrWrap, arg1: PtrWrap ) -> PtrWrap {
    let a = Wrap::new( OpSub::new() );
    a.borrow_mut().set_inp( vec![ arg0, arg1 ] );
    a
}

#[allow(dead_code)]
fn Powi( base: PtrWrap, exponent: PtrWrap ) -> PtrWrap {
    let a = Wrap::new( OpPowi::new() );
    a.borrow_mut().set_inp( vec![ base, exponent ] );
    a
}

#[allow(dead_code)]
fn X( inp: f32 ) -> PtrWrap {
    let a = Wrap::new( OpX::new() );
    let b = Mul(a,Const(inp));
    b
}

#[allow(dead_code)]
#[cfg(test)]
fn test_fun_1( a: PtrWrap, b: PtrWrap ) -> PtrWrap {
    Mul(a.clone(),Add(a.clone(),b.clone()))
}
    
#[test]
fn test_helper_function() {

    let mut c = X(3.);

    for i in 0..3 {
        c = test_fun_1( Const(3.), c );
    }

    let ans = c.borrow().eval();
    dbg!(&ans);

    let ans2 = c.borrow().dual().borrow().eval();
    dbg!(ans2);
}

#[test]
fn test_sin() {
    let a = X(2.);
    let c = Sin(a);
    let ans = c.borrow().dual().borrow().eval();
    dbg!(&ans);
    let ans2 = c.borrow().dual().borrow().dual().borrow().eval();
    dbg!(&ans2);
}

#[test]
fn test_powi() {
    let a = X(4.);
    let c = Powi(a,Consti(1));
    let dc = c.borrow().dual().borrow().eval();
    let ddc = c.borrow().dual().borrow().dual().borrow().eval();
    dbg!(&dc);
    dbg!(&ddc);
}
