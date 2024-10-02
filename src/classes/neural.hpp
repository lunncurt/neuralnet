#pragma once

#include "layer.hpp"
#include <Eigen/Dense>
#include <vector>

struct Image {
  int label;
  Eigen::VectorXd data;

  Image() : label(0), data(Eigen::VectorXd(784)) {}
};

// Network class to handle network topology creation, training functions, and
// testing
class Network {
private:
  std::vector<Layer> layers;

public:
  Network(const std::vector<int> &topology);

  Eigen::VectorXd forward(const Eigen::VectorXd &input_data);

  void backward(const Eigen::MatrixXd &loss_gradient, double learning_rate);

  void train(const std::vector<Image> input_batch);

  double compute_loss(const Eigen::VectorXd &output, int &label);
};
