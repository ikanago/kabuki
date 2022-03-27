use ndarray::{ArrayD, ArrayViewD};

pub type NdArray = ArrayD<f32>;

pub type NdArrayView<'v> = ArrayViewD<'v, f32>;

/// Specifies the way to the tensor is stored in `TensorStorage`.
#[derive(Copy, Clone)]
pub(crate) enum TensorType {
    View,
}

/// Id to specify a tensor in computation graph.
#[derive(Copy, Clone)]
pub struct TensorId {
    ty: TensorType,
    // `id` alone is not unique in the graph.
    id: usize,
}

impl TensorId {
    pub(crate) fn new(ty: TensorType, id: usize) -> Self {
        Self { ty, id }
    }

    pub(crate) fn ty(&self) -> TensorType {
        self.ty
    }

    pub(crate) fn id(&self) -> usize {
        self.id
    }
}
