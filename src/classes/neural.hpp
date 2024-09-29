#pragma once

#include <Eigen/Dense>
#include <vector>

struct Values{
  int label;
  Eigen::VectorXd data;
};

class Layer {
public:
  Eigen::MatrixXd weights;
  Eigen::VectorXd biases;
  Eigen::VectorXd output;

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

// Network class to handle network topology creation, training functions, and
// testing
class Network {
private:
  std::vector<Layer> layers;

public:
  Network(const std::vector<int> &topology);

  Eigen::VectorXd forward(const Eigen::VectorXd &input_data);

  void backward(const Eigen::MatrixXd &loss_gradient, double learning_rate);

  void train(const Eigen::MatrixXd &input_batch, const Eigen::MatrixXd &labels,
             double learning_rate, int epochs);

  double compute_loss(const Eigen::VectorXd &output,
                      const Eigen::MatrixXd &labels);
};
