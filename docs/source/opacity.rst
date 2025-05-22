Working with the opacity class
==============================

At the core of Pyharp is the way of managing and using different flavors of opacities.
We define a distinct opacity source by the file format of the data it uses to compute the
optical properties. If you have two opacity sources that share the same file format, they
belong to the same opacity source category (type).

It is also possible that user supplies a function that can compute and return the optical
properties without using any data file. See :ref:`new_opacity` for more details.

A complete list of built-in opacity types is given in the table below.

.. _opacity_choices:

.. list-table:: List of built-in opacity types
  :widths: 12 10 20
  :header-rows: 1

  * - Key
    - Format
    - Description
  * - 'jit'
    - '.pt' (saved by :func:`torch.jit.save`)
    - Just-In-Time scripted opacity model
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

.. _example_sonora:

Example 1. Compute Sonora2020 molecular opacities
-------------------------------------------------

Make sure you have download the Sonora2020 database using:

.. code-block:: bash

   fetch-sonora

and preprocess the data using:

.. code-block:: python

   from pyharp.sonora import load_sonora_data
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

#. if `conc` is one, the returned `result` is interpreted as the extinction coefficient in m^2/mol
#. if `conc` is concentration in mol/m^3, the return `result` is interpreted as the extinction coefficient in 1/m.
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

Similar to :ref:`Example 1 <example_sonora>`, we set up the opacity class first:

.. code-block:: python

  from pyharp.opacity import AttenuatorOptions, WaveTemp
  op = AttenuatorOptions().type("wavetemp")
  op.opacity_files(["H2-H2-eq.xiz.pt", "H2-He-eq.xiz.pt"])
  op.fractions([0.9, 0.1])
  op.species_ids([0])
  ab = WaveTemp(op)

For this example, hydrogen-hydrogen continuum and hydrogen-helium continuum
are stored in two separate files. The fractions of the two molecules are 0.9 and 0.1 respectively within the species id 0.

The continumm absorption is a function of temperature and wavenumber.
Set these fields up like:

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

.. _new_opacity:

Example 3. Add a new opacity
----------------------------

One of the most powerful features of Pyharp is the ability to add a new opacity source easily. This is fasciliated by the Just-In-Time (JIT) compilation feature of PyTorch.

JIT compilation scripts (compiles) a python module and saves the binary code to a file. The saved file can be loaded and used in the same way as the built-in opacity sources.

Let's define a grey opacity source that has 0.1 m^2/mol cross-section for all wavelengths:

.. code-block:: python

  import torch

  class GreyOpacity(torch.nn.Module):
      species_id = 0
      def forward(self, conc: torch.Tensor) -> torch.Tensor:
          return (0.1 * conc[species_id]).unsqueeze(-1)

Then we create a model, script it, and save it to a file:

.. code-block:: python

  model = GreyOpacity()
  scripted = torch.jit.script(model)
  scripted.save("grey_opacity.pt")

We use the :class:`pyharp.opacity.cpp.JITOpacity` class to load the JIT compiled model from the ``.pt`` file:

.. code-block:: python

  from pyharp.opacity import AttenuatorOptions, JITOpacity

  op = AttenuatorOptions().type("jit")
  op.opacity_files(["grey_opacity.pt"])

  ab = JITOpacity(op)

Finally, calculating the opacity is the same as before:

.. code-block:: python

  conc = torch.ones(3, 5)
  result = ab.forward(conc, {})


Summary
-------

From these examples, we can see that :class:`pyharp.opacity.AttenuatorOptions` is
the central class that manages the opacity source options.
This is a general structure of how classes in Pyharp are organized.
There is always an `Options` class that manages the parameters of a class.
The actual class that does the computation is initialized from the `Options` class.
All opacity classes within :ref:`opacity_classes` follow this pattern and :class:`pyharp.cpp.RadiationBand` and :class:`pyharp.cpp.Radiation` classes also follow this pattern.
