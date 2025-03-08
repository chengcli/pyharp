
int main() {
  // Example usage

  // Coordinate arrays
  std::vector<torch::Tensor> coords = {
      torch::tensor({1.0, 2.0, 3.0}),    // X-coordinates
      torch::tensor({10.0, 20.0, 30.0})  // Y-coordinates
  };

  // Lookup data (3,3,2) where the last dimension represents 2 variables at each
  // (x,y)
  torch::Tensor lookup_data =
      torch::tensor({{{1.0, 10.0}, {2.0, 20.0}, {3.0, 30.0}},
                     {{4.0, 40.0}, {5.0, 50.0}, {6.0, 60.0}},
                     {{7.0, 70.0}, {8.0, 80.0}, {9.0, 90.0}}});

  // Query coordinates (1, 1, 2) representing (x,y) coordinates to interpolate
  torch::Tensor query_coords = torch::tensor({{{2.5, 15.0}}});

  // Perform interpolation
  torch::Tensor result =
      multilinear_interpolation(coords, lookup_data, query_coords);

  std::cout << "Interpolated Values:\n" << result << std::endl;
  return 0;
}
