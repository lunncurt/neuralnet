#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

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

bool fileExists(const std::string& filename) {
  std::ifstream ip(filename);
  return ip.is_open();
}
