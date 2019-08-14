extern crate dynagrad;

use dynagrad as dg;

fn eq_f32( a:f32, b:f32 ) -> bool {
    (a-b).abs() < 0.01
}

fn fwd_looping(){
    let mut l0 = dg::Leaf( dg::ValType::F(2.) ).active();

    let mut l = l0.clone();
    for _ in 0..10 {
        l = dg::Mul( l, dg::Leaf( dg::ValType::F(2.) ) );
    }

    let vl = l.apply_fwd();

    dbg!(vl);

    assert!(eq_f32(vl.into(),2048.));

    let mut g = l.fwd();
    let h = g.apply_fwd();

    dbg!(h);

    assert!(eq_f32(h.into(),1024.));
}

fn fwd(){
    let mut l0 = dg::Leaf( dg::ValType::F(4.) ).active(); 
    let mut l1 = dg::Leaf( dg::ValType::F(3.) );

    //(3x)' = 3
    
    let mut a = dg::Mul( l0.clone(), l1.clone() );

    let mut b = a.fwd();
    
    let c = b.apply_fwd();
    
    dbg!(c);

    assert!(eq_f32(c.into(),3.));
}

fn rev(){

    //(3x)' = 3
    let mut l0 = dg::Leaf( dg::ValType::F(4.) ).active();
    let mut l1 = dg::Leaf( dg::ValType::F(3.) );
    let mut a = dg::Mul( l0.clone(), l1.clone() );

    a.rev();
    
    // let ret = l0.apply_rev_for_adjoint();
    
//     dbg!(ret);

//     assert!(eq_f32(ret.into(),3.));
}

fn fwd_over_fwd(){

    //y=3*x^2 where x=4
    //compute y'' = (6x)' = 6
   
    let mut l0 = dg::Leaf( dg::ValType::F(4.) ).active();
    let mut l1 = dg::Leaf( dg::ValType::F(3.) );
    let mut a = dg::Mul( dg::Mul( l0.clone(), l0.clone()), l1.clone() );
    
    let mut gg = a.fwd().fwd();
    
    let ret = gg.apply_fwd();
    
    dbg!(&ret);

    assert!(eq_f32(ret.into(),6.));

    //change to (7x^2)''=(14x)'=14
    l1.set_val( dg::ValType::F(7.) );

    let ret2 = gg.apply_fwd();
    
    dbg!(&ret2);

    assert!(eq_f32(ret2.into(),14.));
}

fn rev_over_rev(){

    //(3x^2)'' = 6
    
    let mut l0 = dg::Leaf( dg::ValType::F(4.) ).active();
    let mut l1 = dg::Leaf( dg::ValType::F(3.) );
    let mut a = dg::Mul( dg::Mul( l0.clone(), l0.clone()), l1.clone() );

    a.rev();

    let mut adj = l0.adjoint().expect("adjoint empty");

    //todo: consider how to make temporary adjoint data reset possible automatically from user
    l0.reset_adjoint();
    l1.reset_adjoint();
    
    adj.rev();
    
    let ret = l0.apply_rev_for_adjoint();
    
    dbg!(&ret);

    assert!(eq_f32(ret.into(),6.));
    
    // //change to (7x^2)''=(14x)'=14
    // l1.set_val( dg::ValType::F(7.) );

    // let ret2 = l0.apply_rev_for_adjoint();
    
    // dbg!(&ret2);

    // assert!(eq_f32(ret2.into(),14.));
}

fn fwd_over_rev(){

    //y=x*3 where x=4
    //compute y'' = (3)' = 0
    
    let mut l0 = dg::Leaf( dg::ValType::F(4.) ).active();
    let mut l1 = dg::Leaf( dg::ValType::F(3.) );
    let mut a = dg::Mul( l0.clone(), l1.clone() );

    a.rev();
    
    let mut adj = l0.adjoint().expect("adjoint empty");

    let mut g = adj.fwd();
    
    let ret = g.apply_fwd();
    
    dbg!(&ret);

    assert!(eq_f32(ret.into(),0.));
}

fn rev_over_fwd(){

    //(3x^2)'' = 6
    
    let mut l0 = dg::Leaf( dg::ValType::F(4.) ).active();
    let mut l1 = dg::Leaf( dg::ValType::F(3.) );
    let mut a = dg::Mul( dg::Mul( l0.clone(), l0.clone()), l1.clone() );

    let mut g = a.fwd();
    
    g.rev();
        
    let ret = l0.apply_rev_for_adjoint();
    
    dbg!(&ret);

    assert!(eq_f32(ret.into(),6.));
}

fn main(){
    
    // fwd();
    // fwd_looping();
    // fwd_over_fwd();

    //todo:fix leaks
    rev();
    // rev_over_rev();
    // fwd_over_rev();
    // rev_over_fwd();
    
}

