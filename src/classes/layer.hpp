#pragma once

#include <Eigen/Dense>

class Layer {
public:
  Eigen::MatrixXd weights;
  Eigen::VectorXd biases;
  Eigen::VectorXd output;
  Eigen::VectorXd input;
  Eigen::MatrixXd g_weights;

  // Char value to determine layer type for bias initialization
  // (h = hidden, o = output)
  char layer_type;

  // Constructor
  Layer(int input_size, int output_size, char layer_type);

  // ReLu activation function
  Eigen::VectorXd relu(const Eigen::VectorXd &x) { return x.array().max(0); }

  // ReLu derivative function for backprop
  Eigen::VectorXd relu_derivative(const Eigen::VectorXd &x) {
    return (x.array() > 0).cast<double>();
  }

  // Forward Pass: Layer output
  Eigen::VectorXd forward(const Eigen::VectorXd &input);

  // Backward Pass: Update weights and biases
  Eigen::VectorXd backward(const Eigen::MatrixXd &nlayer_weights,
                           const Eigen::VectorXd &nlayer_gradients);
};
