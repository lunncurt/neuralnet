#include "neural.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
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

void Network::forward(const Eigen::VectorXd &input_data) {
  Eigen::VectorXd fpass_out = input_data;

  // Pass the input values through the different layers and apply the
  // weights/biases
  for (auto &layer : layers) {
    fpass_out = layer.forward(fpass_out);
  }
}

double Network::compute_loss(const int &label) {
  Eigen::VectorXd expected = Eigen::VectorXd::Zero(10);
  expected[label] = 1;

  double loss = 0.0;
  const double nonzero = 1e-15;

  // compute cross entropy loss calculation
  for (int i = 0; i < expected.size(); i++) {
    loss -= expected[i] * std::log(layers.back().output[i] + nonzero);
  }

  return loss;
}

void Network::backprop(const Eigen::VectorXd &expected) {
  // Compute the error at the output layer
  Layer &output_layer = layers.back();
  Eigen::VectorXd output_error = output_layer.output - expected;

  Eigen::VectorXd output_gradients = output_error;
  // output_layer.g_weights = output_layer.input * output_gradients.transpose();
  output_layer.g_weights = output_gradients * output_layer.input.transpose();

  Eigen::VectorXd gradients = output_error;
  Eigen::MatrixXd weights = output_layer.weights;

  // Start from the last hidden layer and go backward
  for (int i = layers.size() - 2; i >= 0; --i) {
    gradients = layers[i].backward(weights, gradients);
    weights = layers[i].weights;
  }

  // Update weights in all layers
  for (Layer &layer : layers) {
    layer.weights -= learning_rate * layer.g_weights;
  }
}

void Network::train(const std::vector<Image> &input_batch, const int epochs) {
  // Init clock to determing training time
  auto start = std::chrono::high_resolution_clock::now();

  // Loop through input batch and compute forward pass result(1), backprop
  // result and update(2)

  for (int i = 0; i < epochs; i++) {
    std::cout << "Starting epoch " << i + 1 << std::endl;
    if (i > 0) {
      this->learning_rate *= .1 * i;
    }

    for (int j = 0; j < input_batch.size(); j++) {
      forward(input_batch[j].data);

      Eigen::VectorXd expected = Eigen::VectorXd::Zero(10);
      expected[input_batch[j].label] = 1;

      backprop(expected);
    }
  }

  // End clock
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;

  // Display time taken and num of images tested on
  int mins = duration.count() / 60.0;
  double seconds = std::fmod(duration.count(), 60);

  if (mins > 0 && mins < 2) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Ran " << input_batch.size() << " passes in " << mins
              << " minute and " << seconds << " seconds" << std::endl;
  } else if (mins >= 2) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Ran " << input_batch.size() << " passes in " << mins
              << " minutes and " << seconds << " seconds" << std::endl;
  } else {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Ran " << input_batch.size() << " passes in " << seconds
              << " seconds" << std::endl;
  }
}

void Network::test(const std::vector<Image> &input_batch) {
  int success_count = 0;
  int answer_index;

  double loss_total;

  // Determine if forward pass was a success/compute loss
  for (int i = 0; i < input_batch.size(); i++) {
    forward(input_batch[i].data);

    loss_total += compute_loss(input_batch[i].label);

    layers.back().output.maxCoeff(&answer_index);

    if (answer_index == input_batch[i].label)
      success_count++;
  }

  // Convert accuracy to a percentage
  double result =
      (static_cast<double>(success_count) / input_batch.size()) * 100;

  std::cout << "Accuracy: " << result << "%" << std::endl;
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Average loss: " << loss_total / input_batch.size() << std::endl;
}

std::string Network::network_data() {
  std::ostringstream data;

  // Topology;
  data << "topology\n";

  data << layers[0].input.size() << " ";
  for (const auto &layer : layers) {
    data << layer.output.size();
    if (layer.layer_type != 'o') {
      data << " ";
    } else {
      data << "\n";
    }
  }

  // Weights
  data << "weights\n";

  for (const auto &layer : layers) {
    for (int i = 0; i < layer.weights.rows(); i++) {
      for (int j = 0; j < layer.weights.cols(); j++) {
        data << layer.weights(i, j);
        if (j <= layer.weights.cols() - 1) {
          data << " ";
        }
      }
    }
    data << "\n";
  }

  // Biases
  data << "biases\n";

  for (const auto &layer : layers) {
    for (const auto &bias : layer.biases) {
      data << bias;
      if (bias != layer.biases[layer.biases.size() - 1]) {
        data << " ";
      }
    }
    data << "\n";
  }

  return data.str();
}

void Network::save() {
  std::string save_file = "../../src/saved_models/model.txt";

  std::ofstream outfile(save_file);

  outfile << network_data();

  outfile.close();
}

void Network::load() {
  // Can alter to change models
  std::string load_file = "../../src/saved_models/model.txt";

  std::ifstream in(load_file);

  if (!in.is_open()) {
    throw std::invalid_argument("file does not exist");
  }

  std::string line;

  while (getline(in, line)) {
    if (line == "topology") {
      getline(in, line);
      std::istringstream iss(line);
      std::vector<int> topology;
      topology.clear();
      int val;
      while (iss >> val) {
        topology.push_back(val);
      }

      Network temp(topology);
      this->layers = temp.layers;

    } else if (line == "weights") {
      for (auto &layer : layers) {
        getline(in, line);
        std::istringstream iss(line);
        for (int i = 0; i < layer.weights.rows(); i++) {
          for (int j = 0; j < layer.weights.cols(); j++) {
            double val;
            iss >> val;

            layer.weights(i, j) = val;
          }
        }
      }
    } else if (line == "biases") {
      for (auto &layer : layers) {
        getline(in, line);
        std::istringstream iss(line);
        for (int i = 0; i < layer.biases.size(); i++) {
          double val;
          iss >> val;
          layer.biases[i] = val;
        }
      }
    }
  }

  in.close();
}
