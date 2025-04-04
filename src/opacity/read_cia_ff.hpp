// torch
#include <torch/torch.h>

//! The first array is the 2D data sheet of cross sections, the second vector is
//! the temperature axis, and the third is spectral axis

//! read cia reform format file on temperature vs. spectral wavelength 2D data
//! set.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> read_cia_reform(
    std::string filename);

//! read free-free absorption format file on temperature vs. spectral wavelength
//! 2D data set.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> read_freefree(
    std::string filename);
