#include "neural.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <vector>

Network::Network(const std::vector<int> &topology) {
  // Ignore the first element as its only the input values and initialize the
  // hidden layers
  for (size_t i = 1; i < topology.size() - 1; i++) {
    layers.push_back(Layer(topology[i - 1], topology[i], 'h'));
  }
  // Last element becomes the output layer
  layers.push_back(
      Layer(topology[topology.size() - 2], topology[topology.size() - 1], 'o'));
}

Eigen::VectorXd Network::forward(const Eigen::VectorXd &input_data) {
  Eigen::VectorXd fpass_out = input_data;

  // Pass the input values through the different layers and apply the
  // weights/biases
  for (auto &layer : layers) {
    fpass_out = layer.forward(fpass_out);
  }

  return fpass_out;
}

double Network::compute_loss(const Eigen::VectorXd &output, int &label) {
  Eigen::VectorXd expected = Eigen::VectorXd::Zero(10);
  expected[label] = 1;

  double loss = 0.0;
  const double nonzero = 1e-15;

  // compute cross entropy loss calculation
  for(int i = 0; i < expected.size(); i++){
    loss -= expected[i] * std::log(output[i] + nonzero);
  }

  return loss;
}
