Pyharp Documentation
====================

Pyharp module provides a python interface to the C++ version of the HARP (High-performance Atmospheric Radiation Package) program [1]_.
It contains a minimum set of classes and subroutines for computing the radiative fluxes and radiances for a plane-parallel atmosphere.
Pyharp is designed with efficiency -- all heavy computations are implemented in C/C++ and wraped with a thin python interface.
It can be used standalone to compute multi-column radiative transfer or as a module in other larger programs, both in C/C++ and Python.
Significant changes have been made to the original `HARP <https://github.com/luminoctum/athena-harp>`_ code as the backend data structure has changed from ``Athena`` to ``PyTorch``.

Pyharp treats the radiative transfer problem as layers of computation, akin to a neural network, where the first layer -- the input -- is the atmospheric state; the second layer is the optical properties, and the final layer -- the output -- is the radiative fluxes or radiances.
Each layer is represented by a class, which contains one important method: ``forward``.
The ``forward`` method takes tensor data as inputs and also returns tensor data as outputs. This design allows for chaining multiple layers together to perform a sophisticated computation as well as easy integration with other modules, such as neural networks.

Having both input and output as tensors makes the data flow through the layers transparent, mimicking the paradigm of functional programming.

Automatic differentiation is available when the input tensor has the ``requires_grad`` attribute set to ``True``. This allows for the computation of gradients with respect to the input tensor, which is useful for optimization tasks.

The following example gives a practical demonstration of how to use the Pyharp module.

Example of calculating the radiative fluxes of Jupiter
------------------------------------------------------


References
----------
.. [1] Li, C., Le, T., Zhang, X., & Yung, Y. L. (2018). A high-performance atmospheric radiation package: With applications to the radiative energy budgets of giant planets. Journal of Quantitative Spectroscopy and Radiative Transfer, 217, 353-362.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    api
    opacity
