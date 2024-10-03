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
  double learning_rate = 0.01;

public:
  std::vector<Layer> layers;

  Network() = default;
  Network(const std::vector<int> &topology);

  void forward(const Eigen::VectorXd &input_data);
  void backprop(const Eigen::VectorXd &expected);

  double compute_loss(int &label);

  void train(const std::vector<Image> &input_batch);
  void test(const std::vector<Image> &input_batch);

  std::string network_data();

  void save();
  void load();
};
