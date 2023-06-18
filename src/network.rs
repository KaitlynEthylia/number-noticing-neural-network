use std::rc::Rc;

use rand::{thread_rng, Rng};

#[derive(Debug)]
struct Node {
    bias: f64,
    weights: Vec<f64>,
}

impl Node {
    fn random(in_nodes: usize) -> Self {
        let mut rand = thread_rng();
        let mut weights: Vec<f64> = Vec::with_capacity(in_nodes);
        for _ in 0..weights.capacity() {
            weights.push(rand.gen_range(-1.0..=1.0));
        }

        Self {
            bias: rand.gen_range(-1.0..=1.0),
            weights,
        }
    }

    fn compute(&self, inputs: Vec<f64>, activation: fn(f64) -> f64) -> f64 {
        let value = inputs
            .iter()
            .zip(self.weights.clone())
            .map(|(value, weight)| value * weight)
            .sum::<f64>()
            + self.bias;
        activation(value)
    }
}

#[derive(Debug)]
pub struct Layer {
    next: Option<Box<Layer>>,
    nodes: Vec<Node>,
    in_nodes: usize,
    activation: Rc<fn(f64) -> f64>,
}

impl Layer {
    pub fn last(activation: fn(f64) -> f64, num_nodes: usize, in_nodes: usize) -> Self {
        let mut nodes = Vec::with_capacity(num_nodes);
        for _ in 0..nodes.capacity() {
            nodes.push(Node::random(in_nodes));
        }

        Layer {
            next: None,
            nodes,
            in_nodes,
            activation: Rc::new(activation),
        }
    }

    pub fn prefix(self, in_nodes: usize) -> Self {
        let activation = self.activation.clone();
        let mut nodes = Vec::with_capacity(self.in_nodes);
        for _ in 0..nodes.capacity() {
            nodes.push(Node::random(in_nodes));
        }

        Layer {
            next: Some(Box::new(self)),
            nodes,
            in_nodes,
            activation,
        }
    }

    fn compute(&self, inputs: Vec<f64>) -> Vec<f64> {
        self.nodes
            .iter()
            .map(|node| node.compute(inputs.clone(), *self.activation))
            .collect::<Vec<_>>()
    }

    pub fn compute_flat(&self, inputs: Vec<f64>) -> Vec<f64> {
        let outputs = self.compute(inputs);
        match &self.next {
            Some(next) => next.compute_flat(outputs),
            None => outputs,
        }
    }

    pub fn cost(&self, inputs: Vec<f64>, value: usize) -> f64 {
        let outputs = self.compute_flat(inputs);
        1.0 - (outputs[value] / outputs.iter().sum::<f64>())
    }
}
