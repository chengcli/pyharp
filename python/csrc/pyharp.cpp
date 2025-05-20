// torch
#include <torch/extension.h>

// harp
#include <harp/radiation/radiation.hpp>
#include <harp/utils/find_resource.hpp>

namespace py = pybind11;

void bind_radiation(py::module &m);
void bind_opacity(py::module &m);
void bind_math(py::module &m);
void bind_constants(py::module &m);

PYBIND11_MODULE(pyharp, m) {
  m.attr("__name__") = "pyharp";
  m.doc() = R"(
  Python bindings for HARP (High-performance Atmospheric Radiation Package) Program"
  )";

  bind_opacity(m);
  bind_radiation(m);
  bind_math(m);
  bind_constants(m);

  m.def(
      "species_names",
      []() -> const std::vector<std::string> & { return harp::species_names; },
      R"doc(Retrieves the list of species names)doc");

  m.def(
      "species_weights",
      []() -> const std::vector<double> & { return harp::species_weights; },
      R"doc(Retrieves the list of species molecular weights [kg/mol])doc");

  m.def(
      "shared",
      []() {
        return py::make_iterator(harp::shared.begin(), harp::shared.end());
      },
      py::keep_alive<0, 1>(),
      R"doc(
`Pyharp` module deposits data -- tensors -- to a shared dictionary, which can be accessed by other modules.
This function returns an iterator over the shared data.

After running the forward method of the :class:`RadiationBand <pyharp.cpp.RadiationBand>`, the shared data with the following keys are available:

  .. list-table::
    :widths: 15 25
    :header-rows: 1

    * - Key
      - Description
    * - "radiation/<band_name>/total_flux"
      - total flux in a band

Yields:
  torch.Tensor: shared data of the pyharp module

Examples:
  .. code-block:: python

    >>> import pyharp
    >>> import torch

    # ... after calling the forward method

    # loop over the shared data
    >>> for data in pyharp.shared():
    >>>     print(type(data), data.size())  # prints the shared data
      )doc");

  m.def(
      "get_shared",
      [](const std::string &key) {
        auto it = harp::shared.find(key);
        if (it == harp::shared.end()) {
          throw std::runtime_error("Key not found in shared data");
        }
        return it->second;
      },
      R"doc(
Get the shared data by key.

Args:
  key (str): The key of the shared data.

Returns:
  torch.Tensor: The shared data.

Example:

  .. code-block:: python

    >>> import pyharp
    >>> import torch

    # ... after calling the forward method

    # get the shared data
    >>> data = pyharp.get_shared("radiation/band1/total_flux")
    >>> print(type(data), data.size())  # prints the shared data
      )doc",
      py::arg("key"));

  m.def(
      "set_search_paths",
      [](const std::string path) {
        strcpy(harp::search_paths, path.c_str());
        return harp::deserialize_search_paths(harp::search_paths);
      },
      R"doc(
Set the search paths for resource files.

Args:
  path (str): The search paths

Return:
  str: The search paths

Example:
  .. code-block:: python

    >>> import pyharp

    # set the search paths
    >>> pyharp.set_search_paths("/path/to/resource/files")
      )doc",
      py::arg("path"));

  m.def(
      "get_search_paths",
      []() { return harp::deserialize_search_paths(harp::search_paths); },
      R"doc(
Get the search paths for resource files.

Return:
  str: The search paths

Example:
  .. code-block:: python

    >>> import pyharp

    # get the search paths
    >>> pyharp.get_search_paths()
      )doc");

  m.def(
      "add_resource_directory",
      [](const std::string path, bool prepend) {
        harp::add_resource_directory(path, prepend);
        return harp::deserialize_search_paths(harp::search_paths);
      },
      R"doc(
Add a resource directory to the search paths.

Args:
  path (str): The resource directory to add.
  prepend (bool): If true, prepend the directory to the search paths. If false, append it.

Returns:
  str: The updated search paths.

Example:
  .. code-block:: python

    >>> import pyharp

    # add a resource directory
    >>> pyharp.add_resource_directory("/path/to/resource/files")
      )doc",
      py::arg("path"), py::arg("prepend") = true);

  m.def("find_resource", &harp::find_resource, R"doc(
Find a resource file from the search paths.

Args:
  filename (str): The name of the resource file.

Returns:
  str: The full path to the resource file.

Example:
  .. code-block:: python

    >>> import pyharp

    # find a resource file
    >>> path = pyharp.find_resource("example.txt")
    >>> print(path)  # /path/to/resource/files/example.txt
      )doc",
        py::arg("filename"));
}
