use crate::tensor::NdArray;

pub trait Operator {
    fn forward(&self, ctx: &ForwardContext) -> NdArray;

    fn backward(&self, ctx: BackwardContext) -> BackwardContext;
}

pub struct ForwardContext {
    inputs: Vec<NdArray>,
}

impl ForwardContext {
    pub fn new(inputs: Vec<NdArray>) -> Self {
        Self { inputs }
    }

    pub fn input(&self, index: usize) -> NdArray {
        self.inputs[index].clone()
    }
}

pub struct BackwardContext {
    output_grad: NdArray,
    inputs: Vec<NdArray>,
    input_grads: Vec<NdArray>,
}

impl BackwardContext {
    pub fn new(output_grad: NdArray, inputs: Vec<NdArray>) -> Self {
        Self {
            output_grad,
            inputs,
            input_grads: Vec::new(),
        }
    }

    pub fn append_input_grad(&mut self, input: NdArray) {
        self.input_grads.push(input);
    }
}

pub struct Addition;

impl Operator for Addition {
    fn forward(&self, ctx: &ForwardContext) -> NdArray {
        let lhs = ctx.input(0);
        let rhs = ctx.input(1);
        let result = &*lhs + &*rhs;
        NdArray::new(result)
    }

    fn backward(&self, mut ctx: BackwardContext) -> BackwardContext {
        // Is it ok to share Rc with output grad?
        ctx.append_input_grad(NdArray::clone(&ctx.output_grad));
        ctx.append_input_grad(NdArray::clone(&ctx.output_grad));
        ctx
    }
}
