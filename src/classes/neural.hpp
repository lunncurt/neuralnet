#pragma once

#include <Eigen/Dense>
#include <vector>

class Neuron {
public:
  Neuron(int connections);

private:
  Eigen::VectorXf weights;
  double output_val;
};

// Network class to handle network topology creation, training functions, and
// testing
class Network {
public:
private:
};
