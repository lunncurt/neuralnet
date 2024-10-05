#pragma once

#include "neural.hpp"
#include <SFML/Graphics.hpp>
#include <vector>

// Read data from the mnist datasets
std::vector<Image> read(int num_data, std::string filename);

// Uses distance to calcualte 'intensity' for square colors
float calculateIntensity(float distance);

// Window runner that allows users to test handwritten digits
void window(Network &network);

// Main runner
void runner();
