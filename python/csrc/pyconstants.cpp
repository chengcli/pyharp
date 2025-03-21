// torch
#include <torch/extension.h>

// harp
#include <harp/constants.h>

namespace py = pybind11;
using namespace harp;

void bind_constants(py::module &m) {
  py::module_ c = m.def_submodule("constants", "Physical constants");

  c.attr("Rgas") = py::float_(constants::Rgas);
  c.attr("kBtolz") = py::float_(constants::kBoltz);
}
