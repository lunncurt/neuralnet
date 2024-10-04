#include "utils.hpp"
#include "neural.hpp"

#include <Eigen/Dense>
#include <SFML/Graphics.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

const int grid_size = 28;
const int cell_size = 20;

const int DRAWING_SIZE = grid_size * cell_size;
const int INFO_WIDTH = 200;
const int WINDOW_WIDTH = DRAWING_SIZE + INFO_WIDTH;
const int WINDOW_HEIGHT = DRAWING_SIZE;
// Maximum distance for intensity calculation
const float MAX_DISTANCE = 3.0f;

std::vector<Image> read(int num_data, std::string filename) {
  std::vector<Image> out(num_data);

  std::ifstream ip(filename);

  if (!ip.is_open()) {
    throw std::invalid_argument("file open error");
  }

  std::string line;
  int count = 0;

  for (int i = -1; getline(ip, line, '\n'); i++) {
    if (i == -1)
      continue;
    else if (count == num_data)
      break;

    std::stringstream line_stream(line);
    std::string val;

    for (int j = 0; getline(line_stream, val, ','); j++) {
      if (j == 0) {
        out[i].label = std::stoi(val);
      } else {
        out[i].data[j - 1] = std::stoi(val) / 255.0;
      }
    }

    count++;
  }

  return out;
}

bool fileExists(const std::string &filename) {
  std::ifstream ip(filename);
  return ip.is_open();
}

float calculateIntensity(float distance) {
  if (distance > MAX_DISTANCE)
    return 0.0f;
  return 1.0f - (distance / MAX_DISTANCE);
}

void window(Network &network) {
  sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT),
                          "Handwritten Digit Recognition");

  std::vector<std::vector<float>> grid(grid_size,
                                       std::vector<float>(grid_size, 0.0f));

  sf::RectangleShape cell(sf::Vector2f(cell_size - 1, cell_size - 1));

  bool isDrawing = false;
  double confidence_rating = 0.0;
  int guess = -1;

  // Create rectangles and text for displaying guess
  std::vector<sf::RectangleShape> guessRects;
  std::vector<sf::Text> guessTexts;
  sf::Font font;
  if (!font.loadFromFile("../../media/dogica.ttf")) {

    std::cout << "Error loading font" << std::endl;
    return;
  }

  sf::Text guessText("Guess:", font, 12);
  guessText.setPosition(DRAWING_SIZE + 10, 30);
  guessText.setFillColor(sf::Color::White);

  // Create rectangles for displaying guess
  for (int i = 0; i < 10; ++i) {
    sf::RectangleShape rect(sf::Vector2f(20, 20));
    rect.setPosition(DRAWING_SIZE + 10 + (i % 5) * 30, 50 + (i / 5) * 30);
    rect.setFillColor(sf::Color::White);
    rect.setOutlineColor(sf::Color::Red);
    rect.setOutlineThickness(1);
    guessRects.push_back(rect);

    sf::Text text(std::to_string(i), font, 12);
    text.setPosition(DRAWING_SIZE + 15 + (i % 5) * 30, 52 + (i / 5) * 30);
    text.setFillColor(sf::Color::Black);
    guessTexts.push_back(text);
  }

  // Create rectangle for confidence bar
  sf::RectangleShape confidenceBar(sf::Vector2f(180, 20));
  confidenceBar.setPosition(DRAWING_SIZE + 10, 150);
  confidenceBar.setFillColor(sf::Color::Green);

  // Create border for drawing area
  sf::RectangleShape drawingBorder(sf::Vector2f(DRAWING_SIZE, DRAWING_SIZE));
  drawingBorder.setFillColor(sf::Color::Transparent);
  drawingBorder.setOutlineColor(sf::Color::White);
  drawingBorder.setOutlineThickness(2);

  // Create border for confidence bar
  sf::RectangleShape confidenceBorder(sf::Vector2f(184, 24));
  confidenceBorder.setPosition(DRAWING_SIZE + 8, 148);
  confidenceBorder.setFillColor(sf::Color::Transparent);
  confidenceBorder.setOutlineColor(sf::Color::White);
  confidenceBorder.setOutlineThickness(2);

  // Confidence text
  sf::Text confidenceText("Confidence: 0%", font, 12);
  confidenceText.setPosition(DRAWING_SIZE + 10, 125);
  confidenceText.setFillColor(sf::Color::White);

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed)
        window.close();

      if (event.type == sf::Event::MouseButtonPressed &&
          event.mouseButton.button == sf::Mouse::Left)
        isDrawing = true;

      if (event.type == sf::Event::MouseButtonReleased &&
          event.mouseButton.button == sf::Mouse::Left)
        isDrawing = false;

      if (event.type == sf::Event::KeyPressed) {
        switch (event.key.code) {
        // Clear the drawing
        case sf::Keyboard::C:
          for (auto &row : grid) {
            std::fill(row.begin(), row.end(), 0.0f);
          }
          confidence_rating = 0.0;
          guess = -1;
          break;
        // Quit the progam
        case sf::Keyboard::Q:
          window.close();
          std::cout << "Program exited." << std::endl;
          break;
        }
      }
    }

    if (isDrawing) {
      sf::Vector2f mousePos =
          window.mapPixelToCoords(sf::Mouse::getPosition(window));
      float gridX = mousePos.x / cell_size;
      float gridY = mousePos.y / cell_size;

      if (gridX >= 0 && gridX < grid_size && gridY >= 0 && gridY < grid_size) {
        for (int y = std::max(0, static_cast<int>(gridY) - 1);
             y <= std::min(grid_size - 1, static_cast<int>(gridY) + 1); y++) {
          for (int x = std::max(0, static_cast<int>(gridX) - 1);
               x <= std::min(grid_size - 1, static_cast<int>(gridX) + 1); x++) {
            float distance =
                std::sqrt(std::pow(x - gridX, 2) + std::pow(y - gridY, 2));
            float intensity = calculateIntensity(distance);
            grid[y][x] = std::max(grid[y][x], intensity);
          }
        }
      }

      Eigen::VectorXd user_digit(784);
      int index = 0;
      for (const auto &row : grid) {
        for (float cellValue : row) {
          user_digit(index++) = cellValue;
        }
      }

      // Test the users drawing
      network.forward(user_digit);

      Eigen::VectorXd &output = network.layers.back().output;
      confidence_rating = output.maxCoeff();
      guess = -1;

      for (int i = 0; i < 10; i++) {
        if (output[i] == confidence_rating) {
          guess = i;
          break;
        }
      }

      confidenceText.setString(
          "Confidence: " +
          std::to_string(static_cast<int>(confidence_rating * 100)) + "%");
    }

    window.clear(sf::Color::Black);

    // Draw the grid
    for (int y = 0; y < grid_size; ++y) {
      for (int x = 0; x < grid_size; ++x) {
        cell.setPosition(x * cell_size, y * cell_size);
        sf::Uint8 colorValue = static_cast<sf::Uint8>(grid[y][x] * 255);
        cell.setFillColor(sf::Color(colorValue, colorValue, colorValue));
        window.draw(cell);
      }
    }

    // Draw borders
    window.draw(drawingBorder);
    window.draw(confidenceBorder);

    // Draw guess rectangles
    for (int i = 0; i < 10; ++i) {
      guessRects[i].setFillColor(i == guess ? sf::Color::Red
                                            : sf::Color::White);
      window.draw(guessRects[i]);
      window.draw(guessTexts[i]);
    }
    window.draw(guessText);

    // Draw confidence bar and text
    confidenceBar.setSize(sf::Vector2f(180 * confidence_rating, 20));
    window.draw(confidenceBar);
    window.draw(confidenceText);

    window.display();
  }
}

void runner() {
  std::cout << "Would you like to load a model (1), or train a new model (2): ";
  int answer;
  std::cin >> answer;
  std::cout << std::endl;

  Network model;

  if (answer == 1) {
    model.load();
  } else if (answer == 2) {
    std::cout
        << "How many hidden layers would you like (answer must be >= 1): ";
    int num_hidden;
    std::cin >> num_hidden;
    std::cout << std::endl;

    if (num_hidden < 1) {
      throw std::invalid_argument("Reponse must be >= 1");
    }

    std::vector<int> topology = {784};

    for (int i = 0; i < num_hidden; i++) {
      std::cout << "How many neurons would you like hidden layer " << i + 1
                << " to have: ";
      int amount_neurons;
      std::cin >> amount_neurons;
      std::cout << std::endl;

      if (amount_neurons < 1) {
        throw std::invalid_argument("Response must be >= 1");
      }

      topology.push_back(amount_neurons);
    }

    topology.push_back(10);

    Network temp(topology);

    model.layers = temp.layers;

    std::cout << "How many images would you like to train on: ";
    int img_amount;
    std::cin >> img_amount;
    std::cout << std::endl;

    std::cout << "Loading training batch" << std::endl;
    std::string training_data = "../../mnist_train.csv";
    std::vector<Image> training_batch = read(img_amount, training_data);
    std::cout << "Success" << std::endl;

    std::cout << "Starting training" << std::endl;
    model.train(training_batch);
  } else {
    std::cout << "Well that wasn't very nice" << std::endl;
    return;
  }

  while (true) {
    std::cout << "Would you like to test the model (1), save the model (2), or "
                 "exit (3): ";
    int response;
    std::cin >> response;
    std::cout << std::endl;

    if (response == 3) {
      return;
    } else if (response == 2) {
      std::cout << "Saving the model" << std::endl;
      model.save();
      std::cout << "Success" << std::endl;
    } else if (response == 1) {
      std::cout << "Would you like to test the model yourself (1), or test on "
                   "the MNIST database (2): ";
      int test_response;
      std::cin >> test_response;
      std::cout << std::endl;

      if (test_response == 1) {
        std::cout << "Press 'c' to clear window, 'q' to quit" << std::endl;
        std::cout << "Starting window" << std::endl;

        window(model);
      } else if (test_response == 2) {
        std::cout << "How many images would you like to test on: ";
        int test_amount;
        std::cin >> test_amount;
        std::cout << std::endl;

        if (test_amount < 1) {
          std::cout << "Response must be >= 1" << std::endl;
          continue;
        }

        std::string testing_data = "../../mnist_test.csv";
        std::vector<Image> testing_batch = read(test_amount, testing_data);
        std::cout << "Starting testing" << std::endl;
        model.test(testing_batch);
      } else {
        std::cout << "Response must be >= 1" << std::endl;
        continue;
      }
    } else {
      std::cout << "Invalid Reponse" << std::endl;
      continue;
    }
  }
}
