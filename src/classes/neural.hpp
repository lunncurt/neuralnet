#pragma once

#include "layer.hpp"
#include <Eigen/Dense>
#include <vector>

// Image struct to hold a given images correct label and data (rgb values)
struct Image {
  int label;
  Eigen::VectorXd data;

  Image() : label(0), data(Eigen::VectorXd(784)) {}
};

// Network class to handle network topology creation, training functions, and testing
class Network {
private:
  double learning_rate = 0.01;

public:
  std::vector<Layer> layers;

  // Default Constructor
  Network() = default;
  // Constructs a network with given topology
  Network(const std::vector<int> &topology);

  // Forward pass on the network
  void forward(const Eigen::VectorXd &input_data);
  // Backpropogate the network and update the weights/biases using gradient
  // descent
  void backprop(const Eigen::VectorXd &expected);

  // Compute loss of result
  double compute_loss(const int &label);

  // Runner to train network on given batch
  void train(const std::vector<Image> &input_batch, const int epochs);
  // Runner to test network on given batch
  void test(const std::vector<Image> &input_batch);

  // Coverts the network data (topology, weights, biases) into a string
  std::string network_data();

  // Saves network data (see above) to a file
  void save();
  // Loads network data from a file
  void load();
};
