use std::{cell::RefCell, collections::HashMap};

use crate::tensor::{NdArrayView, TensorId};

pub struct Context<'v> {
    storage: TensorStorage<'v>,
}

/// Specifies the way to the tensor is stored in `TensorStorage`.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum StorageType {
    View,
}

/// Id to specify a tensor in computation graph.
/// `NdArray` of a placeholder is registered after its `TensorId` is created, so we cannot use it to
/// access `NdArray`.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct StorageId {
    pub(crate) ty: StorageType,
    // `id` alone is not unique in the graph.
    pub(crate) id: usize,
}

impl StorageId {
    pub(crate) fn new(ty: StorageType, id: usize) -> Self {
        Self { ty, id }
    }
}

/// `TensorStorage` stores all of the tensors used in computation graph.
pub struct TensorStorage<'v> {
    views: RefCell<Vec<NdArrayView<'v>>>,
    lookup_table: HashMap<TensorId, StorageId>,
}

impl<'v> TensorStorage<'v> {
    pub fn new() -> Self {
        Self {
            views: RefCell::new( Vec::new() ),
            lookup_table: HashMap::new(),
        }
    }

    pub fn push_view(&mut self, tensor_id: TensorId, view: NdArrayView<'v>) {
        let mut inner = self.views.borrow_mut();
        let id = StorageId::new(StorageType::View, inner.len());
        inner.push(view);
        self.lookup_table.insert(tensor_id, id);
    }

    pub fn get(&mut self, id: TensorId) -> Option<NdArrayView<'v>> {
        let storage_id = self.lookup_table.get(&id)?;
        match storage_id.ty {
            StorageType::View => Some(self.views.borrow()[storage_id.id].clone()),
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
        let id = TensorId(0);
        storage.push_view(id, array.view());
        assert_rel_eq_arr!(
            Array2::<f32>::ones((2, 2)).into_dyn().view(),
            storage.get(id).unwrap()
        );
    }
}
