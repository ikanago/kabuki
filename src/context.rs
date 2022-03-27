use std::cell::RefCell;

use crate::tensor::{NdArrayView, TensorId, TensorType};

pub struct Context<'v> {
    storage: TensorStorage<'v>,
}

/// `TensorStorage` stores all of the tensors used in computation graph.
pub struct TensorStorage<'v> {
    inner: RefCell<TensorStorageInner<'v>>,
}

struct TensorStorageInner<'v> {
    views: Vec<NdArrayView<'v>>,
}

impl<'v> TensorStorage<'v> {
    pub fn new() -> Self {
        Self {
            inner: RefCell::new(TensorStorageInner { views: Vec::new() }),
        }
    }

    pub fn push_view(&mut self, view: NdArrayView<'v>) -> TensorId {
        let mut inner = self.inner.borrow_mut();
        let id = TensorId::new(TensorType::View, inner.views.len());
        inner.views.push(view);
        id
    }

    pub fn get(&mut self, id: TensorId) -> NdArrayView<'v> {
        match id.ty() {
            TensorType::View => self.inner.borrow().views[id.id()].clone(),
        }
    }
}

impl<'v> Default for TensorStorage<'v> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_rel_eq_arr;

    use super::*;

    use approx::assert_relative_eq;
    use ndarray::Array2;

    #[test]
    fn push_view() {
        let mut storage = TensorStorage::new();
        let array = Array2::<f32>::ones((2, 2)).into_dyn();
        let id = storage.push_view(array.view());
        assert_rel_eq_arr!(
            Array2::<f32>::ones((2, 2)).into_dyn().view(),
            storage.get(id)
        );
    }
}
