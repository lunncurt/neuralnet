# neuralnet
#### First attempt at a neural network implementation from scratch
- MNIST dataset for training/testing
- Using only Eigen for Matrix operations and SFML for graphical output

#### Features
- Achieved a 96% testing accuracy with 2 hidden layers, ReLu activation, and 3 full training epochs
    - Learning rate is tapered off by intial rate *= 0.1 for each epoch
- Supports both Sigmoid and ReLu activation functions for hidden layers (change in source files to implement)
- ***Users can test their own handwritten digits (see below)***

#### Video / Screenshots
- Handwritten Digit Test
![Digit Recognition](media/digit_rec.gif)
- Example CLI output/testing (slightly outdated)
![CLI Output](media/example.png)
