# neuralnet
#### First attempt at a neural network implementation with:
- MNIST dataset for training/testing
- Using only Eigen for Matrix operations and SFML for graphical output

#### Features
- Achieved a 93% accuracy with 2 hidden layers (275 and 125 neurons), ReLu/Softmax for activation
- Supports both Sigmoid and ReLu activation functions for hidden layers (change in source files to implement)
- ***Users can test their own handwritten digits (see below)***

#### Video / Screenshots
- Handwritten Digit Test
    - Boxes Identification
        - [0][1][2][3][4]  
          [5][6][7][8][9]
    - Green bar represents Network Confidence in result
        - 0 - 100%  
![Digit Recognition](media/digit_recognize.gif)
- Example CLI output/testing
![CLI Output](media/neuralnet_output.jpg)
