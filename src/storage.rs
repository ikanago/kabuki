use std::{cell::RefCell, collections::HashMap};

use crate::tensor::{NdArray, TensorId};

/// `TensorStorage` stores all of the tensors used in computation graph.
pub struct TensorStorage {
    // Wrap with `RefCell`.
    // Without this, we cannot perform `insert` and `get` in the same scope because the former
    // requires `&mut self` and the latter requires `&self`, which occurs borrow checker errror.
    values: RefCell<HashMap<TensorId, NdArray>>,
}

impl TensorStorage {
    pub fn new() -> Self {
        Self {
            values: RefCell::new(HashMap::new()),
        }
    }

    /// Register a tensor value whose id is `id`.
    pub fn insert(&self, id: TensorId, value: NdArray) {
        self.values.borrow_mut().insert(id, value);
    }

    /// Retrieve a tensor corresponding to `id`.
    pub fn get(&self, id: &TensorId) -> Option<NdArray> {
        Some(self.values.borrow().get(id)?.clone())
    }
}

impl Default for TensorStorage {
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
    fn insert() {
        let storage = TensorStorage::new();
        let array = NdArray::new(Array2::<f32>::ones((2, 2)).into_dyn());
        let id = TensorId(0);
        storage.insert(id, array);
        let actual = storage.get(&id).unwrap();
        assert_rel_eq_arr!(actual.view(), Array2::<f32>::ones((2, 2)).into_dyn().view());

        let array = NdArray::new(Array2::<f32>::zeros((2, 2)).into_dyn());
        storage.insert(id, array);
        let actual = storage.get(&id).unwrap();
        assert_rel_eq_arr!(
            actual.view(),
            Array2::<f32>::zeros((2, 2)).into_dyn().view()
        );
    }
}
