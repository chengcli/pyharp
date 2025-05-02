#pragma once

// C/C+
#include <string>

#define ADD_OPTION(T, st_name, op_name, doc)                                 \
  def(#op_name, (T const &(st_name::*)() const) & st_name::op_name,          \
      py::return_value_policy::reference, doc)                               \
      .def(#op_name, (st_name & (st_name::*)(const T &)) & st_name::op_name, \
           py::return_value_policy::reference, doc)

#define ADD_HARP_MODULE(m_name, op_name, doc)                  \
  torch::python::bind_module<harp::m_name##Impl>(m, #m_name)   \
      .def(py::init<>(), R"(Construct a new default module.)") \
      .def(py::init<harp::op_name>(), R"(                     \
        Construct a new module with options)")                 \
      .def_readonly("options", &harp::m_name##Impl::options)   \
      .def("__repr__",                                         \
           [](const harp::m_name##Impl &a) {                   \
             return fmt::format(#m_name "{}", a.options);      \
           })                                                  \
      .def("module",                                           \
           [](harp::m_name##Impl &self, std::string name) {    \
             return self.named_modules()[name];                \
           })                                                  \
      .def("buffer",                                           \
           [](harp::m_name##Impl &self, std::string name) {    \
             return self.named_buffers()[name];                \
           })                                                  \
      .def("forward", &harp::m_name##Impl::forward, doc)
