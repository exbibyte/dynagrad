// #![feature(weak_into_raw)]
extern crate dynagrad;

use dynagrad as dg;

fn eq_f32(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.01
}

fn fwd() {
    let mut l0 = dg::Leaf(dg::ValType::F(4.)).active();
    let mut l1 = dg::Leaf(dg::ValType::F(3.));

    //(3x)' = 3

    let mut a = dg::Mul(l0.clone(), l1.clone());

    let mut b = a.fwd();

    let c = b.apply_fwd();

    dbg!(c);

    assert!(eq_f32(c.into(), 3.));
}

fn fwd_looping() {
    // (x*2^10) where {x=2} = 2^11
    // (x*2^10)' = 2^10
    // (x*2^10)'' = 0
    let mut l0 = dg::Leaf(dg::ValType::F(2.)).active();

    let mut l = l0.clone();
    for _ in 0..10 {
        l = dg::Mul(l, dg::Leaf(dg::ValType::F(2.)));
    }

    let vl = l.apply_fwd();

    dbg!(vl);

    assert!(eq_f32(vl.into(), 2048.));

    let mut g = l.fwd();
    let h = g.apply_fwd();

    dbg!(h);

    assert!(eq_f32(h.into(), 1024.));

    let mut gg = l.fwd().fwd();
    assert!(eq_f32(gg.apply_fwd().into(), 0.));
}

fn rev() {
    //(3x)' = 3
    let mut l0 = dg::Leaf(dg::ValType::F(4.));
    let mut l1 = dg::Leaf(dg::ValType::F(3.));
    let mut a = dg::Mul(l0.clone(), l1.clone());

    let mut adjoints = a.rev();

    let ret = adjoints
        .get_mut(&l0)
        .expect("l0 adjoint missing")
        .apply_rev();

    dbg!(ret);

    assert!(eq_f32(ret.into(), 3.));
}

fn fwd_over_fwd() {
    //y=3*x^2 where x=4
    //compute y'' = (6x)' = 6

    let mut l0 = dg::Leaf(dg::ValType::F(4.)).active();
    let mut l1 = dg::Leaf(dg::ValType::F(3.));
    let mut a = dg::Mul(dg::Mul(l0.clone(), l0.clone()), l1.clone());

    let mut gg = a.fwd().fwd();

    let ret = gg.apply_fwd();

    dbg!(&ret);

    assert!(eq_f32(ret.into(), 6.));

    //change to (7x^2)''=(14x)'=14
    l1.set_val(dg::ValType::F(7.));

    let ret2 = gg.apply_fwd();

    dbg!(&ret2);

    assert!(eq_f32(ret2.into(), 14.));
}

fn rev_over_rev() {
    //(3x^2)'' = 6

    let mut l0 = dg::Leaf(dg::ValType::F(4.));
    let mut l1 = dg::Leaf(dg::ValType::F(3.));
    let mut a = dg::Mul(dg::Mul(l0.clone(), l0.clone()), l1.clone());

    let mut l0_adj = a.rev().get_mut(&l0).expect("l0 adjoint missing").clone();

    assert!(eq_f32(l0_adj.apply_rev().into(), 24.));

    let mut l0_adj_2 = l0_adj
        .rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing")
        .clone();

    let ret = l0_adj_2.apply_rev();

    dbg!(&ret);

    assert!(eq_f32(ret.into(), 6.));

    //change to (7x^2)''=(14x)'=14

    l1.set_val(dg::ValType::F(7.));

    let ret2 = l0_adj_2.apply_rev();

    dbg!(&ret2);

    assert!(eq_f32(ret2.into(), 14.));
}

fn rev_over_rev_2() {
    //(3x^2)'' = 6

    let mut l0 = dg::Leaf(dg::ValType::F(4.));
    let mut l1 = dg::Leaf(dg::ValType::F(3.));
    let mut a = dg::Mul(dg::Mul(l0.clone(), l0.clone()), l1.clone());

    let mut ret = a
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

fn rev_over_rev_3() {
    //(3x^2)'' = 6

    let mut l0 = dg::Leaf(dg::ValType::F(4.));
    let mut l1 = dg::Leaf(dg::ValType::F(3.));
    let mut a = dg::Mul(dg::Mul(l0.clone(), l0.clone()), l1.clone());

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

    l1.set_val(dg::ValType::F(7.));

    let ret2 = gg.apply_rev();

    dbg!(&ret2);

    assert!(eq_f32(ret2.into(), 14.));
}

fn fwd_over_rev() {
    //y=x*3 where x=4
    //compute y'' = (3)' = 0

    let mut l0 = dg::Leaf(dg::ValType::F(4.)).active();
    let mut l1 = dg::Leaf(dg::ValType::F(3.));
    let mut a = dg::Mul(l0.clone(), l1.clone());

    let ret = a
        .rev()
        .get(&l0)
        .expect("l0 adjoint missing")
        .fwd()
        .apply_fwd();

    dbg!(&ret);

    assert!(eq_f32(ret.into(), 0.));
}

fn rev_over_fwd() {
    //(3x^2)'' = 6

    let mut l0 = dg::Leaf(dg::ValType::F(4.)).active();
    let mut l1 = dg::Leaf(dg::ValType::F(3.));
    let mut a = dg::Mul(dg::Mul(l0.clone(), l0.clone()), l1.clone());

    let ret = a
        .fwd()
        .rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing")
        .apply_rev();

    dbg!(&ret);

    assert!(eq_f32(ret.into(), 6.));
}

fn rev_rev_2nd_partial() {
    //x = 4
    //y = 3
    //f = x^2 * y^2
    //d^2/dxdy (f) = d(d(x^2 * y^2)/dx)/dy = d(2x*y^2)/dy = 2x*2y = 4*x*y = 48

    let mut l0 = dg::Leaf(dg::ValType::F(4.));
    let mut l1 = dg::Leaf(dg::ValType::F(3.));
    let mut a = dg::Mul(
        dg::Mul(l0.clone(), l0.clone()),
        dg::Mul(l1.clone(), l1.clone()),
    );

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


extern crate gnuplot;

use gnuplot::*;

///require gnuplot installation
fn plot() {
    
    //sin(x)' over [-2pi,2pi]
    
    let mut l0 = dg::Leaf(dg::ValType::F(0.));
    let mut a = dg::Sin(l0.clone());
    let mut dx = a.rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing")
        .clone();

    let mut ddx = dx.rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing").clone();

    let pi = std::f32::consts::PI;
    let count = 200;
    let delta = 4.*pi/count as f32;

    let mut xs = vec![];
    let mut dys = vec![];
    let mut ddys = vec![];
    
    for i in 0..count {
        let x = -2.*pi+delta*i as f32;
        l0.set_val(dg::ValType::F(x));
        let dy : f32 = dx.apply_rev().into();
        let ddy : f32 = ddx.apply_rev().into();
        xs.push(x);
        dys.push(dy);
        ddys.push(ddy);
    }

    let mut fg = Figure::new();
    fg.axes2d()
	.set_title("Sin(x)", &[])
	.set_legend(Graph(0.5), Graph(0.9), &[], &[])
	.set_x_label("x", &[])
	.set_y_label("y", &[])
        .set_x_range(Fix(-2.*pi as f64),Fix(2.*pi as f64))
	.lines(
            xs.as_slice(),
            dys.as_slice(),
            &[Caption("f'")],
	)
	.lines(
            xs.as_slice(),
            ddys.as_slice(),
            &[Caption("f''")],
	);
    
    fg.show().unwrap();
}

///require gnuplot installation
fn plot_2() {
    
    //tan(x)' over [-pi,pi]
    
    let mut l0 = dg::Leaf(dg::ValType::F(0.));
    let mut a = dg::Tan(l0.clone()).clone();
    let mut dx = a.rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing")
        .clone();

    let mut ddx = dx.rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing").clone();
    
    let mut dddx = ddx.rev()
        .get_mut(&l0)
        .expect("l0 adjoint missing").clone();

    let pi = std::f32::consts::PI;
    let count = 200;
    let delta = 2.*pi/count as f32;

    let mut xs = vec![];
    let mut ys = vec![];
    let mut dys = vec![];
    let mut ddys = vec![];
    let mut dddys = vec![];
    
    for i in 0..count {
        let x = -1.*pi+delta*i as f32;
        l0.set_val(dg::ValType::F(x));
        let y : f32 = a.apply_fwd().into();
        let dy : f32 = dx.apply_rev().into();
        let ddy : f32 = ddx.apply_rev().into();
        let dddy : f32 = dddx.apply_rev().into();
        xs.push(x);
        ys.push(y);
        dys.push(dy);
        ddys.push(ddy);
        dddys.push(dddy);
    }

    let mut fg = Figure::new();
    fg.axes2d()
	.set_title("Tan(x)", &[])
	.set_legend(Graph(0.5), Graph(0.9), &[], &[])
	.set_x_label("x", &[])
	.set_y_label("y", &[])
        .set_y_range(Fix(-30.),Fix(30.))
        .set_x_range(Fix(-pi as _),Fix(pi as _))
        .lines(
            xs.as_slice(),
            ys.as_slice(),
            &[Caption("f")],
	)
	.lines(
            xs.as_slice(),
            dys.as_slice(),
            &[Caption("f'")],
	)
	.lines(
            xs.as_slice(),
            ddys.as_slice(),
            &[Caption("f''")],
	)
    	.lines(
            xs.as_slice(),
            dddys.as_slice(),
            &[Caption("f'''")],
	);
    
    fg.show().unwrap();
}

fn main() {
    fwd();
    fwd_looping();
    fwd_over_fwd();

    rev();
    rev_over_rev();
    rev_over_rev_2();
    rev_over_rev_3();
    fwd_over_rev();
    rev_over_fwd();
    rev_rev_2nd_partial();
    
    //require gnuplot installation
    plot();
    plot_2();
}
