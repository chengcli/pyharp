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

Existing opacity sources
------------------------

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

.. automodule:: pyharp.sonora
   :members:
   :undoc-members:
   :imported-members:

Hydrogen and Helium continuum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pyharp ships with several continuum opacity sources for H2 and He. These are used to compute the collisional induced absorption (CIA) of H2 and He molecules.

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
