#include "classes/neural.hpp"
#include "classes/utils.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

int main() {
  std::string fname = "mnist_test.csv";

  int amount = 1;

  std::vector<Image> test = read(amount, fname);
  std::vector<int> topology = {400, 10};

  Network t(topology);

  std::cout << test[0].data.size();

//  Eigen::VectorXd output = t.forward(test[0].data);
//
//  for(int i = 0; i < 10; i++){
//    std::cout << output[i];
//  }
//
//  std::cout << std::endl;

  return 0;
}
