#include "classes/neural.hpp"
#include "classes/utils.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>

int main() {
  //std::string training_data = "../../mnist_train.csv";

  //std::cout << "loading training batch" << std::endl;
  //int training_amount = 60000;
  //std::vector<Image> training_batch = read(training_amount, training_data);

  std::vector<int> topology = {784, 275, 125, 10};

  Network model(topology);

  model.load();

  //std::cout << "starting training" << std::endl;
  //model.train(training_batch);

  std::string testing_data = "../../mnist_test.csv";

  int testing_amount = 10000;
  std::vector<Image> testing_batch = read(testing_amount, testing_data);

  std::cout << "starting testing" << std::endl;
  model.test(testing_batch);

  //std::cout << "Saving model" << std::endl;
  //model.save();

  return 0;
}
