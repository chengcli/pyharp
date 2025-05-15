Pyharp Documentation
====================

.. toctree::
    :maxdepth: 2
    :caption: Contents:
    :glob:

    index
    api
    opacity

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

Radiation configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The program starts with writing a radiation configuration file in YAML format.
Radiative transfer is a physical calculation regarding the interaction of radiation with molecule and matter. We specify the species, their composition and radiation bands in a configuration file that looks like this:

.. code-block:: yaml

  species:
    - name: H2
      composition: {H: 1.9, He: 0.1}

  opacities:
    H2:
      type: multiband-ck
      data: ["sonora_2020_feh+000_co_100.data.196.pt"]
      species: [H2]

  bands: [sonora196]

  sonora196:
    range: [30.8, 38300.] # wavenumber
    opacities: [H2]
    solver: disort
    integration: weight
    flags: lamber,quiet,onlyfl,planck

`YAML` is a human-readable data serialization standard that is commonly used for configuration files.
The major fields can be specified in numbers, floats, strings, lists and dictionaries.

In this example, the Pyharp module has one species, `H2`, and the atomic composition of it is specified as a dictionary with the key as the name of the atom and the value as its abundance.
We use this entry to calculate its molecular weight.

Next, we specify the opacity sources, which are dictionary entries with the name of the opacity source as the key and its properties as the value.
In this case, we have one opacity source, `H2`, which is a multiband opacity source with the data file `sonora_2020_feh+000_co_100.data.196.pt` and the dependent species is `H2`.
We use these information to retrieve the opacity data from the file and calculate
species indices in a tensor data.

Next, we specify the radiation bands, which is a list of band names.
For each band, a dictionary entry is created with the band name as the key and its properties as the value.

Each band can have its own spectral range, opacity sources, solver, integration method and flags associated with radiative transfer computation.
In this example, we have one band, `sonora196`, which has a spectral range of 30.8 to 38300 cm :math:`^{-1}`; the opacity source is `H2`; the radiative transfer solver is `disort`; the integration method is `weight` (correlated-K); and the flags passed to `disort` are `lamber`, `quiet`, `onlyfl` and `planck`, which assumes lambertian surface, quiet mode, flux-only calculation and activate planck function for the radiation source function respectively.

Please see the `pydisort <https://pydisort.readthedocs.io/en/latest/>`_ documentation for more details on the available options of disort and their meanings.

Load modules
~~~~~~~~~~~~

Then, we create a python file and load system modules include `torch`, `numpy` and `os`.
These provide basic data structures and functions for numerical computation and file handling.

.. code-block:: python

  import torch
  import os
  import numpy as np

We will download the ``sonora2020`` opacity dataset and preprocess it to a format that can be used by the Pyharp module.
These are the submodule functions we need.

.. code-block:: python

  from pyharp.sonora import (
          load_sonora_data,
          load_sonora_window,
          save_sonora_multiband,
          )

Last, we load Pyharp classes and functions. Pyharp uses a minimal set of classes and functions that are used to configure the radiation model and run the radiative transfer calculation.

.. code-block:: python

  import pyharp
  from pyharp import (
          constants,
          RadiationOptions,
          Radiation,
          calc_dz_hypsometric,
          disort_config,
          )

That's all you need for the header of the program.

Pre-process sonora2020 data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``sonora2020`` data is a large dataset that contains the opacity data for hydrogen and helium atmospheres.
We first download a sample of it via the script

.. code-block:: bash

   fetch-sonora --feh=+000 --co=100

This downloads a file named `sonora_2020_feh+000_co_100.data.196.tar.gz` in the current directory. The ``feh`` argument specifies the metallicity of the atmosphere and the ``co`` argument specifies the carbon-to-oxygen ratio (solar).

.. code-block:: python

  def preprocess_sonora(fname: str):
      # returns dictionary of data
      data = load_sonora_data(fname)

      # save to ".pt" file
      save_sonora_multiband(fname, data, clean=False)

The above code block transforms the downloaded data into a ``.pt`` file, which is a binary format used by PyTorch for storing tensors and models.

Construct atmosphere
~~~~~~~~~~~~~~~~~~~~

The following function constructs a simple atmosphere model with a given pressure and temperature point in the troposphere. It constructs an adiabatic atmosphere with a temperature profile that decreases with height, and an isothermal atmosphere with a constant temperature at a specified value.

.. code-block:: python

    def construct_atm(pmax: float, pmin: float,
                      ncol: int = 1,
                      nlyr: int = 100) -> dict[str, torch.Tensor]:
        p1bar, T1bar, Tmin = 1.e5, 169., 135.
        pres = torch.logspace(np.log10(pmax), np.log10(pmin), nlyr + 1, dtype=torch.float64)

        # adibatic
        temp = T1bar * torch.pow(pres / p1bar, 2. / 7.)

        # isothermal
        temp.clip_(min=Tmin)

        atm = {
            'pres' : (pres[1:] * pres[:-1]).sqrt().unsqueeze(0).expand(ncol, nlyr),
            'temp' : (temp[1:] * temp[:-1]).sqrt().unsqueeze(0).expand(ncol, nlyr),
            'btemp0' : temp[0].unsqueeze(0).expand(ncol),
            'ttemp0' : temp[-1].unsqueeze(0).expand(ncol),
        }
        return atm

Configure radiation bands
~~~~~~~~~~~~~~~~~~~~~~~~~

This function is where we configure the entire radiation model.
Pyharp allows for multiple bands to be configured in a single radiation model.
So, generally, we need to loop over the bands and configure them one by one.

There is no rule for dividing or defining the bands. However, each band must have the same number of spectral points, absorbers, radiation flags and opacity sources.
The radiative transfer calculation is computed in parallel for every spectral point within each band, but sequentially over bands. So, there is a performance penalty when the number of bands is too large.

Since the ``sonora2020`` dataset is homogeneous despite that it contains opacities for 196 wavenumber bands, we can group all of them in a single band, `sonora196`, with a single opacity source, `H2`, that is of type `multiband-ck`.
This improves the computational efficiency as the 196 radiative transfer calculations are performed in parallel.

.. code-block:: python

  def configure_bands(config_file: str,
                      ncol: int = 1,
                      nlyr: int = 100,
                      nstr: int = 4) -> Radiation:
      rad_op = RadiationOptions.from_yaml(config_file)
      wmin, wmax = load_sonora_window()

      for [name, band] in rad_op.bands().items():
          if name == "sonora196":
              band.ww(band.query_weights())
              nwave = len(band.ww())
              ng = int(nwave / len(wmin))

              band.disort().accur(1.0e-4)
              disort_config(band.disort(), nstr, nlyr, ncol, nwave)

              data = [wmin] * ng
              band.disort().wave_lower([x for col in zip(*data) for x in col])

              data = [wmax] * ng
              band.disort().wave_upper([x for col in zip(*data) for x in col])
          else:
              raise ValueError(f"Unknown band: {name}")

      return Radiation(rad_op)

Run radiative transfer
~~~~~~~~~~~~~~~~~~~~~~

The last step is to run the radiative transfer model.
The crucial part of the code is to define the boundary conditions, `bc`, of the radiative transfer equations.
Specifically, we need to define the albedo, emissivity, temperature of the surfaces bounding the atmosphere.
Some boundary conditions are band-dependent, which is reflected in the keys of the `bc` dictionary.
Others are band-independent, such as the temperature of the surface and the top of the atmosphere.

Here comes the ``forward`` method of the `Radiation` class, which takes the concentration (in mol m :math:`^{-3}`), layer thickness (in m), boundary conditions and atmospheric state (pressure in pa and temperature in K) as inputs and returns the radiative fluxes.

Throughout the Pyharp code, we will be using SI units unless otherwise specified.

.. code-block:: python

  def run_rt(rad: Radiation, conc: torch.Tensor, dz: torch.Tensor,
             atm: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      ncol = conc.shape[0]
      bc = {}
      for [name, band] in rad.options.bands().items():
          nwave = len(band.ww())
          bc[name + "/albedo"] = torch.zeros((nwave, ncol), dtype=torch.float64)
          bc[name + "/temis"] = torch.ones((nwave, ncol), dtype=torch.float64)

      bc["btemp"] = atm['btemp0']
      bc["ttemp"] = atm['ttemp0']

      return rad.forward(conc, dz, bc, atm)

The main program
~~~~~~~~~~~~~~~~

This is the main program that calls the above functions and runs the radiative transfer model.
The ``run_rt`` function produces three radiative fluxes: 1) net flux at each atmospheric
layer, 2) downward flux to the surface and 3) upward flux at the top of the atmosphere.

The first two fluxes are important for forcing the atmospheric circulation and the
coupling between the atmosphere and the surface, while the last one is important for the
energy balance of the planet.

.. code-block:: python

  if __name__ == "__main__":
      # prepare sonora2020 opacity data
      fname = "sonora_2020_feh+000_co_100.data.196"
      if not os.path.exists(fname + ".pt"):
          preprocess_sonora(fname)

      # configure atmosphere model
      atm = configure_atm(100.e5, 10., ncol=1, nlyr=100)

      # configure radiation model
      config_file = "example_sonora_2020.yaml"
      rad = configure_bands(config_file, ncol=1,
                            nlyr=atm['pres'].shape[-1], nstr=8)
      print(rad.options)

      # calculate concentration and layer thickness
      mean_mol_weight = pyharp.species_weights()[0]
      print("mean mol weight = ", mean_mol_weight)
      grav = 24.8 # m/s^2
      dz = calc_dz_hypsometric(atm["pres"], atm["temp"],
                               torch.tensor(mean_mol_weight * grav / constants.Rgas)
                               )
      print("dz = ", dz)
      conc = atm["pres"] / (atm["temp"] * constants.Rgas)
      conc.unsqueeze_(-1)

      # run rt
      netflux, dnflux, upflux = run_rt(rad, conc, dz, atm)
      print("netflux = ", netflux)
      print("surface flux = ", dnflux)
      print("toa flux = ", upflux)


The complete python code can be downloaded from this `link <_static/example_sonora_2020_flux.py>`_ and the yaml configuration file from this `link <_static/example_sonora_2020.yaml>`_.


References
----------
.. [1] Li, C., Le, T., Zhang, X., & Yung, Y. L. (2018). A high-performance atmospheric radiation package: With applications to the radiative energy budgets of giant planets. Journal of Quantitative Spectroscopy and Radiative Transfer, 217, 353-362.
