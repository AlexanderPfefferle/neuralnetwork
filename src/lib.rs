pub mod matrix;
pub mod neuralnetwork;
pub mod xorshift;

use crate::matrix::Matrix;
use crate::neuralnetwork::NeuralNetwork;

use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn current_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

pub fn parse_csv(
    filename: &str,
    input_size: usize,
    output_size: usize,
) -> (Vec<Matrix>, Vec<Matrix>) {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let content = fs::read_to_string(filename).expect("Error: Can't open file!");
    let lines: Vec<&str> = content.lines().collect();
    for line_index in 0..lines.len() {
        let line = lines[line_index];
        let values: Vec<&str> = line.split(",").collect();
        let mut input_vector = Matrix::new(input_size, 1);
        let mut output_vector = Matrix::new(output_size, 1);
        for value_index in 0..values.len() {
            if value_index < input_size {
                input_vector[value_index][0] = values[value_index].parse::<f32>().unwrap();
            } else {
                output_vector[value_index - input_size][0] =
                    values[value_index].parse::<f32>().unwrap();
            }
        }
        inputs.push(input_vector);
        outputs.push(output_vector);
    }
    (inputs, outputs)
}

pub fn get_accuracy(nn: &NeuralNetwork, filename: &str) -> f32 {
    let (inputs, outputs) = parse_csv(filename, nn.input_nodes, nn.output_nodes);
    let mut num_right: usize = 0;
    for i in 0..inputs.len() {
        if nn.predict(&inputs[i]).index_of_max() == outputs[i].index_of_max() {
            num_right += 1;
        }
    }
    num_right as f32 / inputs.len() as f32
}

pub fn train_on_dataset(nn: &mut NeuralNetwork, filename: &str, epochs: u32) {
    let (inputs, outputs) = parse_csv(filename, nn.input_nodes, nn.output_nodes);
    let start_time = current_millis();
    for i in 0..epochs {
        for j in 0..inputs.len() {
            nn.train(&inputs[j], &outputs[j]);
        }
        print!("{} of {} epochs done\n", i + 1, epochs);
    }
    let end_time = (current_millis() - start_time) as f32 / 1000 as f32;
    print!("Training took {}s\n", end_time);
}
