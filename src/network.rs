use std::collections::HashMap;

use crate::{
    operator::{Addition, ComputeContext},
    storage::TensorStorage,
    tensor::{NdArray, TensorId, TensorInternal},
};

pub(crate) struct Feeder {
    feeds: Vec<(TensorId, NdArray)>,
}

impl Feeder {
    pub fn new() -> Self {
        Self { feeds: Vec::new() }
    }

    pub fn feed(&mut self, id: TensorId, view: NdArray) {
        self.feeds.push((id, view));
    }

    pub fn get(&self, id: TensorId) -> NdArray {
        // In most cases, there are only a few items, so linear search is not inefficent way.
        for feed in &self.feeds {
            if feed.0 == id {
                return feed.1.clone();
            }
        }

        panic!("Placeholder must be filled.");
    }
}

pub struct Network {
    tensors: HashMap<TensorId, TensorInternal>,
    placeholders: Vec<TensorId>,
    storage: TensorStorage,
    feeder: Feeder,
}

impl Network {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            placeholders: Vec::new(),
            storage: TensorStorage::new(),
            feeder: Feeder::new(),
        }
    }

    pub(crate) fn register_tensor(&mut self, tensor: TensorInternal) -> TensorId {
        let id = TensorId(self.tensors.len());
        self.tensors.insert(id, tensor);
        id
    }

    pub(crate) fn access_tensor(&self, id: TensorId) -> &TensorInternal {
        self.tensors
            .get(&id)
            .expect("Must not accessed by TensorId which has not registered.")
    }

    pub fn placeholder(&mut self) -> TensorId {
        let tensor = TensorInternal {
            operator: None,
            inputs: Vec::new(),
        };
        let id = self.register_tensor(tensor);
        self.placeholders.push(id);
        id
    }

    pub fn feed(&mut self, id: TensorId, array: NdArray) -> &mut Self {
        self.feeder.feed(id, array);
        self
    }

    pub fn forward(&self, tensor: TensorId) -> NdArray {
        // (ID, has_input_evaled)
        let mut dfs_stack = vec![(tensor, false)];

        // Perform postorder dfs to compute all of the node in the computation graph.
        while let Some((id, has_input_evaled)) = dfs_stack.pop() {
            let tensor = self.access_tensor(id);

            if has_input_evaled {
                // Since the inputs for the `id` node has been completed, compute a tensor for the
                // node.
                if let Some(ref operator) = tensor.operator {
                    let inputs = tensor
                        .inputs
                        .iter()
                        .map(|input_id| {
                            self.storage
                                .get(input_id)
                                .unwrap_or_else(|| panic!("NdArray for {} is not filled", input_id))
                        })
                        .collect();
                    let ctx = ComputeContext::new(inputs);
                    let result = operator.compute(&ctx);
                    self.storage.insert(id, result);
                } else {
                    let input = self.feeder.get(id);
                    self.storage.insert(id, input);
                }
            } else {
                // The inputs for the `id` node has not been computed.
                dfs_stack.push((id, true));
                for &input_id in &tensor.inputs {
                    dfs_stack.push((input_id, false));
                }
            }
        }

        self.storage
            .get(&tensor)
            .expect("Computation has not finished.")
    }
}

// Operations
impl Network {
    pub fn add(&mut self, lhs: TensorId, rhs: TensorId) -> TensorId {
        let tensor = TensorInternal {
            operator: Some(Box::new(Addition)),
            inputs: vec![lhs, rhs],
        };
        self.register_tensor(tensor)
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::arr2;

    use crate::assert_rel_eq_arr;

    use super::*;

    #[test]
    fn add() {
        let mut network = Network::new();
        let x = network.placeholder();
        let y = network.placeholder();
        let z = network.add(x, y);

        let x_data = NdArray::new(arr2(&[[1.0, 1.0], [2.0, 2.0]]).into_dyn());
        let y_data = NdArray::new(arr2(&[[3.0, 4.0], [3.0, 4.0]]).into_dyn());
        let result = network.feed(x, x_data).feed(y, y_data).forward(z);
        assert_rel_eq_arr!(arr2(&[[4.0, 5.0], [5.0, 6.0]]).into_dyn(), *result);
    }
}
