#pragma once

#include <Eigen/Dense>

class Layer {
public:
  Eigen::MatrixXd weights;
  Eigen::VectorXd biases;
  Eigen::VectorXd output;
  Eigen::VectorXd input;

  // Char value to determine layer type for bias initialization
  // (h = hidden, o = output)
  char layer_type;

  // Constructor
  Layer(int input_size, int output_size, char layer_type);

  // Forward Pass: Layer output
  Eigen::VectorXd forward(const Eigen::VectorXd &input);

  // Backward Pass: Update weights and biases
  Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output,
                           double learning_rate);
};
