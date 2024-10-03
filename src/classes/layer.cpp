#include "layer.hpp"
#include <random>

Layer::Layer(int input_size, int output_size, const char type)
    : layer_type(type) {
  // Initialize random weights using mersenne twister into a normal distribution
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> he_dist(0, std::sqrt(2.0 / input_size));

  // fill the weights matrix
  weights =
      Eigen::MatrixXd::Zero(output_size, input_size).unaryExpr([&](double x) {
        return he_dist(gen);
      });

  // Initialize bias vectors based on layer type
  if (layer_type == 'h') {
    biases = Eigen::VectorXd::Constant(output_size, 0.1);
  } else { // output layer
    biases = Eigen::VectorXd::Constant(output_size, 0.0);
  }
}

Eigen::VectorXd Layer::forward(const Eigen::VectorXd &input) {
  // Check dimensions
  assert(weights.cols() == input.size() &&
         "Input size does not match the number of columns in weights!");
  assert(weights.rows() == biases.size() &&
         "Biases size does not match the number of rows in weights!");

  this->input = input;

  // Calculate weighted sum + biases
  Eigen::VectorXd wsum = (weights * input) + biases;

  // Apply each layers respective activation function
  if (layer_type == 'h') {
    // ReLU
    output = relu(wsum);
  } else if (layer_type == 'o') {
    // Softmax
    Eigen::VectorXd smax = (wsum.array() - wsum.maxCoeff()).exp();
    output = smax / smax.sum();
  }

  return output;
}

Eigen::VectorXd Layer::backward(const Eigen::MatrixXd &nlayer_weights,
                                const Eigen::VectorXd &nlayer_gradients) {
  // Step 1: Compute the gradient of the activation function
  Eigen::VectorXd activation_grad = relu_derivative(output);

  // Step 2: Compute the error signal for this layer (chain rule)
  Eigen::VectorXd delta = (nlayer_weights.transpose() * nlayer_gradients);
  delta = delta.cwiseProduct(activation_grad);

  // Step 3: Calculate the gradient for the weights
  g_weights = delta * input.transpose();

  // Step 4: Update biases (simply the delta)
  biases -= delta;

  return delta;
}
