#![allow(dead_code)]
mod network;

use std::env::args;

use activation_functions::f64 as af;
use mnist::*;
use network::Layer;
use rand::seq::IteratorRandom;
use rand::thread_rng;

fn main() {
    let net = Layer::last(af::sigmoid, 10, 100)
        .prefix(200)
        .prefix(400)
        .prefix(784);
    let base_path = args().nth(1).unwrap();
    let Mnist {
        trn_lbl, trn_img, ..
    } = MnistBuilder::new()
        .base_path(&base_path)
        .training_set_length(100)
        .test_set_length(100)
        .finalize();

    let dataset = get_dataset(trn_img, trn_lbl);

    let result = cost(&net, &dataset, 100);
    println!("Cost: {result}");
}

fn get_dataset(dataset: Vec<u8>, labels: Vec<u8>) -> Vec<(Vec<u8>, u8)> {
    dataset
        .chunks(784)
        .map(|arr| arr.to_vec())
        .zip(labels)
        .collect::<Vec<_>>()
}

fn cost(network: &Layer, dataset: &Vec<(Vec<u8>, u8)>, sample_size: usize) -> f64 {
    let mut rand = thread_rng();
    let buffer = dataset.iter().choose_multiple(&mut rand, sample_size);

    let mut results = Vec::with_capacity(sample_size);
    for (image, value) in buffer {
        let image = image.iter().map(|i| (*i).into()).collect();
        let value = value.to_owned().into();
        results.push(network.cost(image, value));
    }
    println!("{results:#?}");

    results.iter().sum()
}
