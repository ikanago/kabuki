use std::collections::HashMap;

use ndarray::ArrayD;

use crate::{
    operator::{Addition, BackwardContext, ForwardContext},
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
    tensor_info: HashMap<TensorId, TensorInternal>,
    placeholders: Vec<TensorId>,
    variables: Vec<TensorId>,
    tensors: TensorStorage,
    grads: TensorStorage,
}

impl Network {
    pub fn new() -> Self {
        Self {
            tensor_info: HashMap::new(),
            placeholders: Vec::new(),
            variables: Vec::new(),
            tensors: TensorStorage::new(),
            grads: TensorStorage::new(),
        }
    }

    pub(crate) fn register_tensor(&mut self, tensor: TensorInternal) -> TensorId {
        let id = TensorId(self.tensor_info.len());
        self.tensor_info.insert(id, tensor);
        id
    }

    pub(crate) fn access_tensor(&self, id: TensorId) -> &TensorInternal {
        self.tensor_info
            .get(&id)
            .expect("Must not accessed by TensorId which has not registered.")
    }

    pub(crate) fn get_grad(&self, id: TensorId) -> NdArray {
        self.grads
            .get(&id)
            .unwrap_or_else(|| panic!("Gradient of NdArray for {} is not initialized", id))
    }

    pub fn placeholder(&mut self) -> TensorId {
        let tensor = TensorInternal {
            operator: None,
            inputs: Vec::new(),
            is_differentiable: false,
        };
        let id = self.register_tensor(tensor);
        self.placeholders.push(id);
        id
    }

    pub fn variable(&mut self, array: NdArray) -> TensorId {
        let tensor = TensorInternal {
            operator: None,
            inputs: Vec::new(),
            is_differentiable: true,
        };
        let id = self.register_tensor(tensor);
        self.variables.push(id);
        self.tensors.insert(id, array);
        id
    }

    pub fn feed(&mut self, id: TensorId, array: NdArray) -> &mut Self {
        self.tensors.insert(id, array);
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
                            self.tensors
                                .get(input_id)
                                .unwrap_or_else(|| panic!("NdArray for {} is not filled", input_id))
                        })
                        .collect();
                    let ctx = ForwardContext::new(inputs);
                    let result = operator.forward(&ctx);
                    self.tensors.insert(id, result);
                }
            } else {
                // The inputs for the `id` node has not been computed.
                dfs_stack.push((id, true));
                for &input_id in &tensor.inputs {
                    dfs_stack.push((input_id, false));
                }
            }
        }

        self.tensors
            .get(&tensor)
            .expect("Computation has not finished.")
    }

    pub fn backward(&self, tensor: TensorId) {
        let mut dfs_stack = vec![tensor];

        let output = self.tensors.get(&tensor).unwrap();
        self.grads
            .insert(tensor, NdArray::new(ArrayD::ones(output.shape())));

        while let Some(id) = dfs_stack.pop() {
            let tensor = self.access_tensor(id);
            if !tensor.is_differentiable {
                continue;
            }

            if let Some(ref operator) = tensor.operator {
                let inputs = tensor
                    .inputs
                    .iter()
                    .map(|input_id| {
                        self.tensors
                            .get(input_id)
                            .unwrap_or_else(|| panic!("NdArray for {} is not filled", input_id))
                    })
                    .collect();
                let grad = self
                    .grads
                    .get(&id)
                    .unwrap_or_else(|| panic!("NdArray of gradient for {} is not initialized", id));
                let ctx = BackwardContext::new(grad, inputs);
            }
        }
    }
}

// Operations
impl Network {
    pub fn add(&mut self, lhs: TensorId, rhs: TensorId) -> TensorId {
        let tensor = TensorInternal {
            operator: Some(Box::new(Addition)),
            inputs: vec![lhs, rhs],
            is_differentiable: true,
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
        let x = network.variable(NdArray::new(arr2(&[[1.0, 1.0], [2.0, 2.0]]).into_dyn()));
        let y = network.placeholder();
        let z = network.add(x, y);

        let y_data = NdArray::new(arr2(&[[3.0, 4.0], [3.0, 4.0]]).into_dyn());
        let result = network.feed(y, y_data).forward(z);
        assert_rel_eq_arr!(arr2(&[[4.0, 5.0], [5.0, 6.0]]).into_dyn(), *result);

        network.backward(z);
        assert_rel_eq_arr!(
            arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn(),
            *network.get_grad(x)
        );
    }

    #[test]
    fn nested_add() {
        let mut network = Network::new();
        let x = network.variable(NdArray::new(arr2(&[[1.0, 1.0], [2.0, 2.0]]).into_dyn()));
        let y = network.placeholder();
        let z = network.add(x, y);
        let w = network.add(x, z);

        let y_data = NdArray::new(arr2(&[[3.0, 4.0], [-3.0, 4.0]]).into_dyn());
        let result = network.feed(y, y_data).forward(w);
        assert_rel_eq_arr!(arr2(&[[5.0, 6.0], [1.0, 8.0]]).into_dyn(), *result);

        network.backward(w);
        assert_rel_eq_arr!(
            arr2(&[[2.0, 2.0], [2.0, 2.0]]).into_dyn(),
            *network.get_grad(x)
        );
    }
}
