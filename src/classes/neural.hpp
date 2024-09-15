#pragma once

#include <Eigen/Dense>
#include <functional>
#include <vector>

class Layer {
public:
  Eigen::MatrixXd weights;
  Eigen::VectorXd biases;
  Eigen::MatrixXd activation;
  Eigen::VectorXd output;

  // Char value to determine layer type for bias initialization
  // (i = input, h = hidden, o = output)
  char layer_type;

  // Constructor
  Layer(int input_size, int outputsize, char layer_type);

  // Forward Pass: Layer output
  Eigen::MatrixXd forward(const Eigen::MatrixXd &input);

  // Backward Pass: Update weights and biases
  Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output,
                           double learning_rate);
};

// Network class to handle network topology creation, training functions, and
// testing
class Network {
private:
  std::vector<Layer> layers;

public:
  Network(const std::vector<int> &topology);

  Eigen::MatrixXd forward(const Eigen::MatrixXd &input_batch);

  void backward(const Eigen::MatrixXd &loss_gradient, double learning_rate);

  void train(const Eigen::MatrixXd &input_batch, const Eigen::MatrixXd &labels,
             double learning_rate, int epochs);

  double compute_loss(const Eigen::MatrixXd &output,
                      const Eigen::MatrixXd &labels);
};
