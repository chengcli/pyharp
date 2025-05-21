How to set opacity sources
==========================

At the core of Pyharp is the way of managing and using different flavors of opacities.
We define a distinct opacity source by the file format of the data it uses to compute the
optical properties. If you have two opacity sources that share the same file format, they
belong to the same opacity source category (type).

It is also possible that user supplies a function that can compute and return the optical
properties without using any data file. This would be the case when the opacity source is
simple, e.g. a grey opacity. A user can do that by defining a :class:`torch.nn.Module` and passing it to
the :meth:`pyharp.opacity.AttenuatorOptions.user` field.

A complete list of supported opacity types is given in the table below.

.. _opacity_choices:

.. list-table:: List of supported opacity types
  :widths: 15 10 20
  :header-rows: 1

  * - Key
    - Format
    - Description
  * - 'user'
    - None
    - User-supplied opacity module. The ``user`` field
      must be set to use this option
  * - 'rfm-lbl'
    - NetCDF
    - Line-by-line absorption data computed by RFM
  * - 'rfm-ck'
    - NetCDF
    - Correlated-k absorption computed from line-by-line data
  * - 'multiband-ck'
    - '.pt' (saved by :func:`torch.jit.save`)
    - Three-dimensional correlated-k opacity lookup table. The axis are

      #. wave-mapped correlated-k axis (gaussian quadrature points),
      #. log pressure in [pa], 3) temperature [k]
  * - 'wavetemp'
    - '.pt' (saved by :func:`torch.jit.save`)
    - Two-dimensional continuum opacity lookup table. The axis are:

      #. wavelength [um] or wavenumber [cm^{-1}]
      #. temperature [K]
  * - 'fourcolumn'
    - Ascii
    - One-dimensional opacity lookup table. Each row is a data entry. Four columns are:

      #. wavelength [um] or wavenumber [cm^{-1}]
      #. mass extinction cross-section [m^2/kg]
      #. single scattering albedo
      #. Henyey-Greenstein asymmetry factor (g)

We give a few examples showing how to use these opacity class to load and compute optical properties.

Example 1. Compute Sonora2020 molecular opacities
-------------------------------------------------

Make sure you have download the Sonora2020 database using:

.. code-block:: bash

   fetch-sonora

and preprocess the data using:

.. code-block:: python

   from pyharp.sonora import (
      load_sonora_data, load_sonora_window,
   )
   fname = "sonora_2020_feh+000_co_100.data.196"

   # load sonora data into a dictionary
   data = load_sonora_data(fname)

   # save it to pt file
   save_sonora_multiband(fname, data, clean=False)

This will generate a file called
``sonora_2020_feh+000_co_100.data.196.pt`` in the current directory.

Then, we specify the fileds ``type``, ``opacity_files``, and ``species_ids``
of the :class:`pyharp.opacity.AttenuatorOptions` class.
``species_ids`` is a list of integers that specify the index of the
dependent species in a multi-dimensional concentration array.
We will talk about the concentration array later.

.. code-block:: python

   from pyharp import MultiBand, AttenuatorOptions

   # create sonora opacity
   op = AttenuatorOptions().type("multiband-ck")
   op.opacity_files(["sonora_2020_feh+000_co_100.data.196.pt"])
   op.species_ids([0])
   ab = MultiBand(op)

Optical properties are usually a function of temperature and pressure.
We set up an atmosphere grid of temperature and pressure values.
The opacity class will compute the optical properties for each grid point
of temperature and pressure.

.. code-block:: python

   # set up atmosphere grid
   temp = torch.tensor([100.0, 300.0, 600.0])
   pres = torch.tensor([0.1e5, 1.0e5, 10.0e5, 100.e5])
   X, Y = torch.meshgrid(temp, pres, indexing='ij')
   atm = {'temp': X, 'pres': Y}

The shape of `X` and `Y` is (3, 4), which is interpreted as having 3 columns
and 4 layers.
We use the convention that the last dimension for the temperature structure
is layers and the second to last dimension is columns.


Finally, we call the :meth:`forward <pyharp.opacity.cpp.MultiBand.forward>` method to compute the optical properties.

.. code-block:: python

   conc = torch.ones_like(atm['pres']).unsqueeze(-1)
   result = ab.forward(conc, atm)

The returned `result` is a :class:`torch.Tensor` with shape (1568, 3, 4, 1).
Pyharp is strict with the shape of the input and output tensors.
In this case, the first dimension of the result is the number of spectral grids (1568).
:mod:`pyharp.sonora` contains a database of 196 bands within each there are 8
correlated-k gaussian quadrature points. They multiply to 1568.
The second dimension is the number of columns of atmospheres (3).
The third dimension is the number of layers in each column (4).
The last dimension is the number of optical properties in the order of extinction coefficient,
single scattering albedo, and phase function moments.

The dimensions of result calculated by the ``forward`` method of an opacity class will alway be (waves, columns, layers, properties).
In the example above, the last dimension is degenerate because :mod:`pyharp.sonora` only treats the molecular absorption.

The ``forward`` method of an opacity class takes two arguments:

#. ``conc``: a tensor of shape (columns, layers, species), where the last dimension is the number of species.
   When an opacity source depends on the concentrations of species,
   the ``species_ids`` field supplies the indices of the species and
   the class object retrieves their concentrations from these indices
   mapped in the last dimension. In our case, the dependent species has an index of 0.
#. ``atm``: a dictionary of tensors that provides auxiliary data to the opacity
   calculation such as temperature and pressure.
   Different opacity classes may require different auxiliary data.
   In the case of molecular absorption, the absorption cross-section depends on temperature (``temp``) and pressure (``pres``).
   Pyharp will panic and throw an error message if the required auxiliary data is not provided.

By choosing different values for the `conc` argument, the ``forward`` method can be multi-purpose:

#. if `conc` is one, the returned `result` is interpreted as the extinction coefficient in m :math:`^2`/mol
#. if `conc` is concentration in mol/m :math:`^3`, the return `result` is interpreted as the extinction coefficient in 1/m.
   You can further multiply it by the layer thickness in m to get the optical thickness of the layer.

Example 2. Compute Hydrogen continuum opacities
-----------------------------------------------

Here is another demonstration of computing the hydrogen continuum opacities.
The data files have been preprocessed and are shipped when you install Pyharp.
To include them, use:

.. code-block:: python

  from pyharp import h2_cia_legacy

You can find out the absolute path of the data files using:

.. code-block:: python

  from pyharp import find_resource
  print(find_resource("H2-H2-eq.xiz.pt"))

Similar to the Example 1, we set up the opacity class first:

.. code-block:: python

  from pyharp import h2_cia_legacy
  from pyharp.opacity import AttenuatorOptions, WaveTemp
  op = AttenuatorOptions().type("wavetemp")
  op.opacity_files(["H2-H2-eq.xiz.pt", "H2-He-eq.xiz.pt"])
  op.fractions([0.9, 0.1])
  op.species_ids([0])
  ab = WaveTemp(op)

For this example, hydrogen-hydrogen continuum and hydrogen-helium continuum
are stored in two separate files. The fractions of the two molecules are 0.9 and 0.1 respectively within the species id 0.

The continumm absorption is a function of temperature and wavenumber.
Set these up like:

.. code-block:: python

  import torch
  atm = {
    'temp': torch.tensor([100.0, 300.0, 600.0]).unsqueeze(-1),
    'wavenumber': torch.logspace(np.log10(10), np.log10(10000), 10)
  }

Last, we call the ``forward`` method to compute the optical properties:

.. code-block:: python

  conc = torch.ones_like(atm['temp']).unsqueeze(-1)
  result = ab.forward(conc, atm)

Be aware that we have used the :meth:`torch.Tensor.unsqueeze` method to add back the degenerate dimension.

Details of opacity sources
--------------------------

Pyharp ships with a number of opacity sources that can be used to compute the
optical properties of the plantary atmosphere.

Sonora2020 molecular opacities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a database of pre-mixed Correlated-K hydrogen-helium opacities with abundances given by equilibrium chemistry
for each metallicity-C/O combination (version 3) [1]_.
It has been used for brown dwarf atmospheres [2]_.
View this `document <_static/sonora2020_opacity_references_20201.pdf>`_ for references of opacities included in the database.

Use the following script to checkout options and download the Sonora2020 database:

.. code-block:: bash

   fetch-sonora -h

By default, ``fetch-sonora`` downloads the database with [Fe/H] = 0.0 and C/O = 1 times solar abundances.

The following functions are available to process the original Sonora2020 opacities and
load/save them in the :mod:`torch`'s ``pt`` format.

.. automodule:: pyharp.sonora
   :members:
   :undoc-members:
   :imported-members:

Hydrogen and Helium continuum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pyharp ships with the following continuum opacity sources for H2 and He:

#. H2-H2-eq.xiz.pt
#. H2-He-eq.xiz.pt
#. H2-H2-nm.xiz.pt
#. H2-He-nm.xiz.pt
#. H2-H2-eq.orton.pt
#. H2-He-eq.orton.pt
#. H2-H2-nm.orton.pt
#. H2-He-nm.orton.pt

These are legacy files that have been used the original HARP publication [3]_.
They are used here in Pyharp to compute the collisional induced absorption (CIA) of H2 and He molecules.
The following functions have been used to process the legacy CIA data files:

.. automodule:: pyharp.h2_cia_legacy
   :members:
   :undoc-members:
   :imported-members:


Add a new opacity format
------------------------

References
----------
.. [1] Lupu, R., et al. "Correlated k coefficients for H2-He atmospheres; 196 spectral windows and 1460 pressure-temperature points." Zenodo, doi 0.5281/zenodo.5590988 (2021).
.. [2] Marley, Mark S., et al. "The Sonora brown dwarf atmosphere and evolution models. I. Model description and application to cloudless atmospheres in rainout chemical equilibrium." The Astrophysical Journal 920.2 (2021): 85.
.. [3] Li, C., Le, T., Zhang, X., & Yung, Y. L. (2018). A high-performance atmospheric radiation package: With applications to the radiative energy budgets of giant planets. Journal of Quantitative Spectroscopy and Radiative Transfer, 217, 353-362.
