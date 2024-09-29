#include "neural.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <vector>

Layer::Layer(int input_size, int output_size, const char type)
    : layer_type(type) {
  // Initialize random weights using mersenne twister into a normal distribution
  std::random_device rd;
  std::mt19937 gen(rd);
  std::normal_distribution<> he_dist(0, std::sqrt(2.0 / input_size));

  // fill the weights matrix
  weights =
      Eigen::MatrixXd::Zero(output_size, input_size).unaryExpr([&](double x) {
        return he_dist(gen);
      });

  // Initialize bias vectors based on layer type
  if (layer_type == 'h') {
    biases = Eigen::VectorXd::Constant(weights.cols(), 0.1);
  } else { // output layer
    biases = Eigen::VectorXd::Constant(weights.cols(), 0.0);
  }
}

Eigen::VectorXd Layer::forward(const Eigen::VectorXd &input) {
  // Calculate weighted sum + biases
  Eigen::VectorXd wsum = (weights * input) + biases;

  // Apply each layers respective activation function
  if (layer_type == 'h') {
    // ReLU
    output = wsum.unaryExpr([](double val) { return std::max(0.0, val); });
  } else if (layer_type == 'o') {
    // Softmax
    Eigen::VectorXd smax = (wsum.array() - wsum.maxCoeff()).exp();
    output = smax / smax.sum();
  }

  return output;
}

Network::Network(const std::vector<int> &topology) {
  for (size_t i = 1; i < topology.size() - 1; i++) {
    layers.push_back(Layer(topology[i], topology[i + 1], 'h'));
  }

  layers.push_back(
      Layer(topology[topology.size() - 2], topology[topology.size() - 1], 'o'));
}

Eigen::VectorXd Network::forward(const Eigen::VectorXd &input_data){
  Eigen::VectorXd output = input_data;

  for(auto &layer : layers){
    output = layer.forward(output);
  }

  return output;
}
