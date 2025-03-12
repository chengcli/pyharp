// C/C++
#include <fstream>
#include <iostream>
#include <sstream>

// harp
#include "fileio.hpp"
#include "read_data_tensor.hpp"

namespace harp {

// Function to read a 2D data table into a Torch tensor
torch::Tensor read_data_tensor(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  std::vector<std::vector<float>> data;
  std::string line;

  while (std::getline(file, line)) {
    // Skip commented lines
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::stringstream ss(line);
    std::vector<float> row;
    float value;

    while (ss >> value) {
      row.push_back(value);
    }

    if (!row.empty()) {
      data.push_back(row);
    }
  }

  file.close();

  if (data.empty()) {
    throw std::runtime_error("No valid data found in file.");
  }

  // Convert vector of vectors to a Torch tensor
  size_t rows = data.size();
  size_t cols = data[0].size();

  torch::Tensor result = torch::empty(
      {static_cast<long>(rows), static_cast<long>(cols)}, torch::kFloat64);

  for (size_t i = 0; i < rows; ++i) {
    if (data[i].size() != cols) {
      throw std::runtime_error("Inconsistent column size in data table.");
    }
    for (size_t j = 0; j < cols; ++j) {
      result[i][j] = data[i][j];
    }
  }

  return result;
}

}  // namespace harp
