// torch
#include <torch/extension.h>

// harp
#include <harp/opacity/attenuator_options.hpp>
#include <harp/opacity/fourcolumn.hpp>
#include <harp/opacity/jit_opacity.hpp>
#include <harp/opacity/multiband.hpp>
#include <harp/opacity/opacity_formatter.hpp>
#include <harp/opacity/rfm.hpp>
#include <harp/opacity/wavetemp.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_opacity(py::module &parent) {
  auto m = parent.def_submodule("opacity", "Opacity module");

  auto pyAttenuatorOptions =
      py::class_<harp::AttenuatorOptions>(m, "AttenuatorOptions");

  pyAttenuatorOptions
      .def(py::init<>())

      .def("__repr__",
           [](const harp::AttenuatorOptions &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("AttenuatorOptions(\n{})", ss.str());
           })

      .ADD_OPTION(std::string, harp::AttenuatorOptions, type)

      .ADD_OPTION(std::string, harp::AttenuatorOptions, bname)

      .ADD_OPTION(std::vector<std::string>, harp::AttenuatorOptions,
                  opacity_files)

      .ADD_OPTION(std::vector<int>, harp::AttenuatorOptions, species_ids)

      .ADD_OPTION(std::vector<std::string>, harp::AttenuatorOptions, jit_kwargs)

      .ADD_OPTION(std::vector<double>, harp::AttenuatorOptions, fractions);

  ADD_HARP_MODULE(JITOpacity, AttenuatorOptions,
                  py::arg("conc"), py::arg("kwargs"));

  ADD_HARP_MODULE(WaveTemp, AttenuatorOptions,
                  py::arg("conc"), py::arg("kwargs"));

  ADD_HARP_MODULE(MultiBand, AttenuatorOptions,
                  py::arg("conc"), py::arg("kwargs"));

  ADD_HARP_MODULE(FourColumn, AttenuatorOptions,
                  py::arg("conc"), py::arg("kwargs"));

  ADD_HARP_MODULE(RFM, AttenuatorOptions,
                  py::arg("conc"), py::arg("kwargs"));
}
