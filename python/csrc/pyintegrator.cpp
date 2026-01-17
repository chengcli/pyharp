// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// harp
#include <harp/integrator/integrator.hpp>
#include <harp/integrator/integrator_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_integrator(py::module &m) {
  auto pyIntegratorWeight =
      py::class_<harp::IntegratorWeight>(m, "IntegratorWeight");

  pyIntegratorWeight.def(py::init<>())
      .def("__repr__",
           [](const harp::IntegratorWeight &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("IntegratorWeight(\n{})", ss.str());
           })
      .ADD_OPTION(double, harp::IntegratorWeight, wght0)
      .ADD_OPTION(double, harp::IntegratorWeight, wght1)
      .ADD_OPTION(double, harp::IntegratorWeight, wght2);

  auto pyIntegratorOptions =
      py::class_<harp::IntegratorOptionsImpl, harp::IntegratorOptions>(
          m, "IntegratorOptions");

  pyIntegratorOptions.def(py::init<>())
      .def_static("from_yaml",
                  py::overload_cast<std::string const &, bool>(
                      &harp::IntegratorOptionsImpl::from_yaml),
                  py::arg("filename"), py::arg("verbose") = false)
      .def("__repr__",
           [](const harp::IntegratorOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("IntegratorOptions(\n{})", ss.str());
           })
      .ADD_OPTION(std::string, harp::IntegratorOptionsImpl, type)
      .ADD_OPTION(double, harp::IntegratorOptionsImpl, cfl)
      .ADD_OPTION(double, harp::IntegratorOptionsImpl, tlim)
      .ADD_OPTION(int, harp::IntegratorOptionsImpl, nlim)
      .ADD_OPTION(int, harp::IntegratorOptionsImpl, ncycle_out)
      .ADD_OPTION(int, harp::IntegratorOptionsImpl, max_redo);

  ADD_HARP_MODULE(Integrator, IntegratorOptions, py::arg("stage"),
                  py::arg("u0"), py::arg("u1"), py::arg("u2"))
      .def_readonly("stages", &harp::IntegratorImpl::stages)
      .def_readwrite("current_redo", &harp::IntegratorImpl::current_redo)
      .def("stop", &harp::IntegratorImpl::stop, py::arg("steps"),
           py::arg("current_time"));
}
