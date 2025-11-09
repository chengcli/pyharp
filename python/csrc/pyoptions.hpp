#pragma once

// C/C+
#include <string>

#define ADD_OPTION(T, st_name, op_name)                                      \
  def(#op_name, (T const &(st_name::*)() const) & st_name::op_name,          \
      py::return_value_policy::reference)                                    \
      .def(#op_name, (st_name & (st_name::*)(const T &)) & st_name::op_name, \
           py::return_value_policy::reference)

#define ADD_HARP_MODULE(m_name, op_name, args...)                       \
  torch::python::bind_module<harp::m_name##Impl>(m, #m_name)            \
      .def(py::init<>())                                                \
      .def(py::init<harp::op_name>(),                                   \
           py::arg("options"))                                          \
      .def_readonly("options", &harp::m_name##Impl::options)            \
      .def("__repr__",                                                  \
           [](const harp::m_name##Impl &a) {                            \
             return fmt::format(#m_name "{}", a.options);               \
           })                                                           \
      .def(                                                             \
          "module",                                                     \
          [](harp::m_name##Impl &self) {                                \
            return py::make_iterator(self.named_modules().begin(),      \
                                     self.named_modules().end());       \
          },                                                            \
          py::keep_alive<0, 1>())                                       \
      .def("get_module",                                                \
           [](harp::m_name##Impl &self, std::string name) {             \
             return self.named_modules()[name];                         \
           })                                                           \
      .def(                                                             \
          "buffer",                                                     \
          [](harp::m_name##Impl &self) {                                \
            return py::make_iterator(self.named_buffers().begin(),      \
                                     self.named_buffers().end());       \
          },                                                            \
          py::keep_alive<0, 1>())                                       \
      .def("get_buffer",                                                \
           [](harp::m_name##Impl &self, std::string name) {             \
             return self.named_buffers()[name];                         \
           })                                                           \
      .def("forward", &harp::m_name##Impl::forward, args)
