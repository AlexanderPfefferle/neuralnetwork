# neuralnetwork
A small neural network lib written from scratch in rust.

### XOR Example:

Creates a net with 2 input nodes, 2 hidden layers which have 2 nodes each and 1 output node, using sigmoid as an activation function and learning rate set to 1.

Trains it for 10k epochs using stochastic gradient descent and mean square error as loss function.

```rust
use neuralnetwork::neuralnetwork::NeuralNetwork;
use neuralnetwork::{current_millis, parse_csv};

fn main() {
    let mut nn = NeuralNetwork::new(2, vec![2, 2], 1, 1.0, "sigmoid");
    let (inputs, outputs) = parse_csv("xor.csv", 2, 1);
    let start_time = current_millis();
    let epochs = 10_000;
    for _ in 0..epochs {
        for i in 0..inputs.len() {
            nn.train(&inputs[i], &outputs[i]);
        }
    }
    let end_time = (current_millis() - start_time) as f32 / 1000 as f32;
    println!("Training {} epochs took {}s", epochs, end_time);
    for i in 0..inputs.len() {
        println!(
            "Input {} {} Prediction {:.8} Goal {:.}",
            inputs[i][0][0],
            inputs[i][1][0],
            nn.predict(&inputs[i])[0][0],
            outputs[i][0][0]
        );
    }
}
```
Content of xor.csv:
```
0,0,0
0,1,1
1,0,1
1,1,0
```
