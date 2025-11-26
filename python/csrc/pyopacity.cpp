// torch
#include <torch/extension.h>

// harp
#include <harp/opacity/fourcolumn.hpp>
#include <harp/opacity/jit_opacity.hpp>
#include <harp/opacity/multiband.hpp>
#include <harp/opacity/opacity_formatter.hpp>
#include <harp/opacity/opacity_options.hpp>
#include <harp/opacity/rfm.hpp>
#include <harp/opacity/wavetemp.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_opacity(py::module &parent) {
  auto m = parent.def_submodule("opacity", "Opacity module");

  auto pyOpacityOptions =
      py::class_<harp::OpacityOptionsImpl, harp::OpacityOptions>(
          m, "OpacityOptions");

  pyOpacityOptions.def(py::init<>())
      .def("__repr__",
           [](const harp::OpacityOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("OpacityOptions(\n{})", ss.str());
           })
      .def("query_wavenumber", &harp::OpacityOptionsImpl::query_wavenumber)
      .def("query_weight", &harp::OpacityOptionsImpl::query_weight)
      .ADD_OPTION(std::string, harp::OpacityOptionsImpl, type)
      .ADD_OPTION(std::string, harp::OpacityOptionsImpl, bname)
      .ADD_OPTION(std::vector<std::string>, harp::OpacityOptionsImpl,
                  opacity_files)
      .ADD_OPTION(std::vector<int>, harp::OpacityOptionsImpl, species_ids)
      .ADD_OPTION(std::vector<std::string>, harp::OpacityOptionsImpl,
                  jit_kwargs)
      .ADD_OPTION(std::vector<double>, harp::OpacityOptionsImpl, fractions);

  ADD_HARP_MODULE(JITOpacity, OpacityOptions, py::arg("conc"), py::arg("atm"));
  ADD_HARP_MODULE(WaveTemp, OpacityOptions, py::arg("conc"), py::arg("atm"));
  ADD_HARP_MODULE(MultiBand, OpacityOptions, py::arg("conc"), py::arg("atm"));
  ADD_HARP_MODULE(FourColumn, OpacityOptions, py::arg("conc"), py::arg("atm"));
  ADD_HARP_MODULE(RFM, OpacityOptions, py::arg("conc"), py::arg("atm"));
}
