use std::{fmt::Display, rc::Rc};

use ndarray::ArrayD;

use crate::operator::Operator;

/// Wrap `ArrayD` with `Rc` because it is hard to maintain both owned arrays and views in
/// `TensorStorage`.
pub type NdArray = Rc<ArrayD<f32>>;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct TensorId(pub(crate) usize);

impl Display for TensorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub(crate) struct TensorInternal {
    pub operator: Option<Box<dyn Operator>>,
    pub inputs: Vec<TensorId>,
    pub(crate) is_differentiable: bool,
}
