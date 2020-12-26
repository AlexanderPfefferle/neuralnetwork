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

### MNIST Example:
Creates a net with 784 input nodes, 2 hidden layers which have 8 nodes each and 10 output nodes, using sigmoid as an activation function and learning rate set to 0.1.

```rust
use neuralnetwork::neuralnetwork::NeuralNetwork;
use neuralnetwork::{get_accuracy, train_on_dataset};

fn main() {
    let mut nn = NeuralNetwork::new(784, vec![8, 8], 10, 0.1, "sigmoid");
    train_on_dataset(
        &mut nn,
        "mnist_train.csv",
        10,
    );
    print!(
        "Accuarcy: {}%\n",
        get_accuracy(&nn, "mnist_test.csv") * 100.0
    );
}
```

The dataset used is a modified version of mnist, where the first 784 values in each line are the inputs scaled down to the range [0,1],
and the last 10 represent the output, using one-hot encoding.

`train_on_dataset` takes a net, a path to a dataset and the number of epochs as input and trains the net via stochastic gradient descent.

`get_accuracy` calculates the accuracy on the test set, it assumes one-hot encoding and checks whether the node with the highest value is the right one.

It takes ~30s to train it for 10 epochs and it achieves an accuracy >90%.
