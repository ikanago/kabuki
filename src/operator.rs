use crate::tensor::NdArray;

pub trait Operator {
    fn compute(&self, ctx: &ComputeContext) -> NdArray;
}

pub struct ComputeContext {
    inputs: Vec<NdArray>,
}

impl ComputeContext {
    pub fn new(inputs: Vec<NdArray>) -> Self {
        Self { inputs }
    }

    pub fn input(&self, index: usize) -> NdArray {
        self.inputs[index].clone()
    }
}

pub struct Addition;

impl Operator for Addition {
    fn compute(&self, ctx: &ComputeContext) -> NdArray {
        let lhs = ctx.input(0);
        let rhs = ctx.input(1);
        let result = &*lhs + &*rhs;
        NdArray::new(result)
    }
}
