#include "classes/neural.hpp"
#include "classes/utils.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>

int main() {
  std::string fname = "mnist_test.csv";

  int amount = 5;

  std::vector<Image> test = read(amount, fname);
  std::vector<int> topology = {784, 200, 10};

  Network t(topology);

  for (int i = 0; i < test.size(); i++) {
    Eigen::VectorXd output = t.forward(test[i].data);

    for (int j = 0; j < 10; j++) {
      std::cout << j << ": " << output[j] << std::endl;
    }
    std::cout << "expected: " << test[i].label << std::endl;
    std::cout << "loss: " << t.compute_loss(output, test[i].label) << std::endl;
    std::cout << std::endl;
  }

  std::cout << std::endl;

  return 0;
}
