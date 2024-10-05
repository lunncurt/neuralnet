#pragma once

#include <Eigen/Dense>

// Layer class to hold a given layers data and necessary functions
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

  // Sigmoid activation function 
  Eigen::VectorXd sigmoid(const Eigen::VectorXd &x) {
    return 1.0 / (1.0 + (-x.array()).exp());
  }

  // Sigmoid derivative function for backprop
  Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd &x) {
    Eigen::VectorXd s = sigmoid(x);
    return s.array() * (1 - s.array());
  }

  // Forward Pass: Layer output
  Eigen::VectorXd forward(const Eigen::VectorXd &input);

  // Backward Pass: Update weights and biases
  Eigen::VectorXd backward(const Eigen::MatrixXd &nlayer_weights,
                           const Eigen::VectorXd &nlayer_gradients);
};
