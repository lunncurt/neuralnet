#include "utils.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

std::vector<Image> read(int num_data, std::string filename) {
  std::vector<Image> out(num_data);

  std::ifstream ip(filename);

  if (!ip.is_open()) {
    std::cerr << "file open error" << std::endl;
    return out;
  }

  std::string line;
  int count = 0;

  for(int i = -1; getline(ip, line, '\n'); i++){
    if(i == -1) continue;
    else if(count == num_data) break;

    std::stringstream line_stream(line);
    std::string val;

    for(int j = 0; getline(line_stream, val, ','); j++){
      if(j == 0){
        out[i].label = std::stoi(val);
      }else{
        out[i].data[j - 1] = std::stoi(val);
      }
    }

    count++;
  }

  return out;
}
