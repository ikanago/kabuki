use ndarray::{ArrayD, ArrayViewD};

pub type NdArray = ArrayD<f32>;

pub type NdArrayView<'v> = ArrayViewD<'v, f32>;

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct TensorId(pub(crate) usize);

