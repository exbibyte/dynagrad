//! An implementation of Ricci calculus use for
//! tensor manipulations and computing derivatives
//!
//! use uppercase for tensor entity
//! use lowercase for indices in superscript/subscript
//! no indices indicates a scalar
//! contraction happens for a matching superscript-subscript index
//!
//! valid expressions: A^ij_kl^mB^k_ij, A, A^i_j, A_ii, AB
//! invalid expressions: ^A, _A, i, i^A, A^B_ij
//!
//! Work in progress..

use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::mem;

type IndexLoc = usize;

///encode an index along with the subscript/superscript type
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
enum Index {
    SuperScript(char),
    SubScript(char),
}

impl Index {
    pub fn get(&self) -> char {
        match self {
            Self::SuperScript(x) => *x,
            Self::SubScript(x) => *x,
        }
    }
}

///used in recognition of superscript/subscript delimiters
enum StateIndex {
    SuperScript,
    SubScript,
}

///representation of a tensor/vector/covector
///may be simplified
#[derive(Clone, Debug)]
struct Entity {
    ///id of the entity (an Uppercase char)
    pub c: char,

    ///stores superscripts and subscripts of the current entity
    pub indices: Vec<Index>,

    ///all superscripts of an index
    pub indices_match_superscript: HashMap<char, Vec<IndexLoc>>,

    ///all subscripts of an index
    pub indices_match_subscript: HashMap<char, Vec<IndexLoc>>,

    ///contracted result and its orignal location
    pub indices_result: Vec<(Index, IndexLoc)>,

    ///contraction pairs of matching (superscript,subscript) indices
    pub contraction_pairs_loc: Vec<(IndexLoc, IndexLoc)>,
}

///a sequence of entities to be simplified
#[derive(Clone, Debug)]
struct Expr {
    pub original: Vec<Entity>,
    pub result: Vec<Entity>,
}

impl Entity {
    pub fn new(x: char) -> Self {
        Entity {
            c: x,
            indices: Default::default(),
            indices_match_superscript: Default::default(),
            indices_match_subscript: Default::default(),
            indices_result: Default::default(),
            contraction_pairs_loc: Default::default(),
        }
    }

    pub fn assign_index_loc(&mut self) {
        for (loc, i) in self.indices.iter().enumerate() {
            match i {
                Index::SuperScript(x) => {
                    let arr = self.indices_match_superscript.entry(*x).or_insert(vec![]);
                    arr.push(loc);
                }
                Index::SubScript(x) => {
                    let arr = self.indices_match_subscript.entry(*x).or_insert(vec![]);
                    arr.push(loc);
                }
            }
        }
    }

    ///todo: add tensor related code here
    pub fn check_indices_for_tensor(&self) -> Result<(), &'static str> {
        unimplemented!();
    }

    pub fn determine_contraction_single(&mut self) {
        //pairs of matching (superscript,subscript) indices
        let mut contraction_pairs_loc: Vec<(IndexLoc, IndexLoc)> = vec![];

        let mut indices_contraction = HashSet::new();

        for (key, arr1) in self.indices_match_superscript.iter() {
            match self.indices_match_subscript.get(key) {
                Some(arr2) => {
                    let items = arr1.iter().zip(arr2.iter());
                    for (a, b) in items {
                        contraction_pairs_loc.push((*a, *b));
                        indices_contraction.insert(*a);
                        indices_contraction.insert(*b);
                    }
                }
                _ => {}
            }
        }

        self.contraction_pairs_loc = contraction_pairs_loc;

        for i in 0..self.indices.len() {
            if !indices_contraction.contains(&i) {
                self.indices_result.push((self.indices[i], i));
            }
        }
    }
}

impl Expr {
    pub fn try_parse_original(s: &str) -> Result<Expr, &'static str> {
        let mut expr = Expr {
            original: Default::default(),
            result: Default::default(),
        };

        let mut entity = None;

        let mut state_index = None;

        for i in s.chars() {
            match i {
                '^' => {
                    if entity.is_none() {
                        return Err(&"entity missing");
                    }
                    state_index = Some(StateIndex::SuperScript);
                }
                '_' => {
                    if entity.is_none() {
                        return Err(&"entity missing");
                    }
                    state_index = Some(StateIndex::SubScript);
                }
                x => {
                    if x.is_ascii_alphanumeric() && x.is_uppercase() {
                        match entity.as_mut() {
                            Some(y) => {
                                let mut temp = Entity::new(x);
                                mem::swap(&mut temp, y);
                                expr.original.push(temp);
                                state_index = None;
                            }
                            _ => {
                                entity = Some(Entity::new(x));
                                state_index = None;
                            }
                        }
                    } else if x.is_ascii_alphanumeric() && x.is_lowercase() {
                        match entity.as_mut() {
                            Some(y) => match state_index {
                                None => return Err(&"index state missing"),
                                Some(StateIndex::SuperScript) => {
                                    let index = Index::SuperScript(x);
                                    y.indices.push(index);
                                }
                                Some(StateIndex::SubScript) => {
                                    let index = Index::SubScript(x);
                                    y.indices.push(index);
                                }
                            },
                            _ => return Err(&"entity missing"),
                        }
                    }
                }
            }
        }

        match entity {
            Some(y) => {
                expr.original.push(y);
            }
            _ => {}
        }

        Ok(expr)
    }
}

impl TryFrom<&str> for Expr {
    type Error = &'static str;

    fn try_from(s: &str) -> Result<Expr, Self::Error> {
        //get orders of indices
        //get types of indices
        //check dimensions and indices (defer to actual tensor instance)
        //check parity of index counts
        //categorize axes as contractables or uncontractables from matching index pairs

        let mut expr = Expr::try_parse_original(s)?;

        for i in expr.original.iter_mut() {
            i.assign_index_loc();
        }

        //determine possibility of single entity contraction
        for i in expr.original.iter_mut() {
            i.determine_contraction_single();
        }

        Ok(expr)
    }
}

pub fn ricci(expr: &str) {
    let expr = Expr::try_from(expr).unwrap();
}

#[test]
fn test_expr() {
    assert!(Expr::try_parse_original(&"_ijk^lmn_ab").is_err());
    assert!(Expr::try_parse_original(&"AB_i^j").is_ok());
    assert!(Expr::try_parse_original(&"A").is_ok());
    assert!(Expr::try_parse_original(&"ABC").is_ok());
    assert!(Expr::try_parse_original(&"A_ij_k").is_ok());
    assert!(Expr::try_parse_original(&"A^ijk^gh").is_ok());
    assert!(Expr::try_parse_original(&"A^ij_lm").is_ok());
    assert!(Expr::try_parse_original(&"A^ij_lmB_ab_gh").is_ok());
    assert!(Expr::try_parse_original(&"A_ij^lm_Bi^j").is_err());
    assert!(Expr::try_parse_original(&"A_ij^lm_B_i^j").is_ok());

    assert!(Expr::try_from("A_ij^lm_B_i^j").is_ok());
    assert!(Expr::try_parse_original(&"^_ijk^lmn_ab").is_err());
    assert!(Expr::try_parse_original(&"^_A").is_err());

    dbg!(Expr::try_from("A_ikj^jim").unwrap());
    dbg!(Expr::try_from("A_ikj^j").unwrap());
}
