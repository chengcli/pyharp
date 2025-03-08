// harp
#include "read_data_tensor.hpp"

#include "fileio.hpp"

namespace harp {

torch::Tensor read_data_tensor(std::string const& fname) {
  auto str_file = decomment_file(fname);

  // get number of rows and columns
  int rows = get_num_rows_str(str_file);
  int cols = get_num_cols_str(str_file);

  torch::Tensor result = torch::zeros({rows * cols}, torch::kFloat64);
  std::stringstream ss(str_file);

  auto data = result.accessor<double, 1>();
  int count = 0;
  while (ss >> data[count++] && count < rows * cols);

  return result.view({rows, cols});
}

}  // namespace harp
