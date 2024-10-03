#pragma once

#include "neural.hpp"
#include <SFML/Graphics.hpp>
#include <vector>

// Read data from the mnist datasets
std::vector<Image> read(int num_data, std::string filename);

// Check if a file exists
bool fileExists(const std::string& filename);
