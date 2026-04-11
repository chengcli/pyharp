pyharp-dump CLI
================

``pyharp-dump`` writes spectroscopy products as NetCDF files. It supports
single-species molecular products, CIA pair products, and gas-mixture
products built from HITRAN line and CIA data.

Basic Usage
-----------

Choose one subcommand, then choose the target with ``--species``, ``--pair``,
or ``--composition`` where supported.

.. code-block:: bash

   pyharp-dump xsection --species H2O --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
   pyharp-dump xsection --pair H2-He --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
   pyharp-dump transmission --composition H2:0.9,He:0.1,H2O:0.002 --path-length-m 1000 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500

Without ``--output``, files are written under ``output/`` with names derived
from the target, product type, thermodynamic state, and wavenumber range.

Subcommands
-----------

``xsection``
~~~~~~~~~~~~

Write absorption cross-section products.

Supported selectors:

* ``--species`` for a molecular absorber
* ``--pair`` for a CIA pair
* ``--composition`` for a gas mixture

Examples:

.. code-block:: bash

   pyharp-dump xsection --species CO2 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
   pyharp-dump xsection --pair H2-He --temperature-k 300 --pressure-bar 1 --wn-range=20,10000
   pyharp-dump xsection --composition H2:0.9,He:0.1,CH4:0.004,H2O:0.002 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500

``transmission``
~~~~~~~~~~~~~~~~

Write transmission products over a fixed path length.

Supported selectors:

* ``--species`` for a molecular absorber
* ``--pair`` for a CIA pair
* ``--composition`` for a gas mixture

Examples:

.. code-block:: bash

   pyharp-dump transmission --species H2O --path-length-m 1 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
   pyharp-dump transmission --pair H2-He --path-length-m 1000 --temperature-k 300 --pressure-bar 1 --wn-range=20,10000
   pyharp-dump transmission --composition H2:0.9,He:0.1,H2O:0.002 --path-length-m 1000 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500

Shared Options
--------------

``--wn-range=min,max``
    Wavenumber bounds in ``cm^-1``. Repeat this option to write multiple bands
    into one NetCDF file.

``--resolution value``
    Wavenumber spacing in ``cm^-1``. The default is ``1``.

``--temperature-k value``
    Temperature in kelvin. The default is ``300``.

``--pressure-bar value``
    Pressure in bar. The default is ``1``.

``--path-length-m value``
    Propagation path length in meters. This option is required only for
    ``transmission`` and defaults to ``1``.

``--output path``
    Explicit NetCDF output path.

``--hitran-dir path``
    Directory used for cached HITRAN line and CIA files. The default is
    ``hitran``.

``--refresh-hitran`` and ``--refresh-cia``
    Re-download cached HITRAN line or CIA inputs.

``--broadening-composition BROADENER:FRACTION,...``
    Broadening gas composition for molecular line calculations. This affects
    line absorption only, not CIA-only ``--pair`` workflows.

Target Selection
----------------

Use ``--species`` for HITRAN line species such as ``CH4``, ``CO2``, ``H2``,
``H2O``, ``H2S``, ``N2``, and ``NH3``.

Use ``--pair`` for CIA pairs such as ``H2-H2`` and ``H2-He``.

Use ``--composition`` for gas mixtures:

.. code-block:: bash

   --composition H2:0.9,He:0.1,CH4:0.004,H2O:0.002,NH3:0.0003

Broadening Behavior
-------------------

If ``--broadening-composition`` is omitted for single-species molecular
products, pyharp defaults to ``self`` broadening. For mixture workflows,
pyharp defaults to the mixture composition itself as the requested line
broadening composition.

If a requested foreign broadener is not available in the HITRAN table for the
active absorber, pyharp falls back to ``air`` for that fraction and prints the
effective broadening summary.

Multi-band Output
-----------------

Repeat ``--wn-range`` to compute multiple bands in one run:

.. code-block:: bash

   pyharp-dump xsection --species H2O --temperature-k 300 --pressure-bar 1 \
       --wn-range=20,2500 --wn-range=2500,10000

Multi-band outputs contain:

* ``band`` and ``wavenumber`` dimensions
* ``wavenumber``
* ``band_size``
* ``band_wavenumber_min``
* ``band_wavenumber_max``

Per-band data variables are stored as ``(band, wavenumber)`` arrays.

NetCDF Naming Conventions
-------------------------

The dump CLI uses semantic field names and stores physical units in variable
attributes rather than in variable names.

Xsection outputs
~~~~~~~~~~~~~~~~

Single-species xsection dumps use names such as:

* ``sigma_line_h2o``
* ``sigma_continuum_h2o_continuum_mt_ckd``
* ``sigma_cia_h2o_h2o``
* ``binary_absorption_coefficient_h2o_h2o``
* ``sigma_total``

Pair xsection dumps use:

* ``binary_absorption_coefficient``

Composition xsection dumps contain one file with multiple component fields:

* ``sigma_line_<species>`` for unweighted line cross sections
* ``sigma_continuum_<source>`` for unweighted continuum cross sections
* ``binary_absorption_coefficient_<pair>`` for unweighted CIA binary coefficients
* ``sigma_total`` for the composition-weighted total cross section

For composition xsection dumps, component fields are not weighted by mole
fraction. Only ``sigma_total`` represents the weighted mixture result.

Transmission outputs
~~~~~~~~~~~~~~~~~~~~

Transmission dumps follow the same component naming pattern and add weighted
attenuation coefficients in ``m^-1``.

Examples:

* ``transmittance_line_h2o``
* ``attenuation_line_h2o``
* ``transmittance_continuum_h2o_continuum_mt_ckd``
* ``attenuation_continuum_h2o_continuum_mt_ckd``
* ``transmittance_cia_h2_he``
* ``attenuation_cia_h2_he``
* ``transmittance_total``
* ``attenuation_total``

Global Attributes
-----------------

Species and pair dumps include ``species_name`` plus the thermodynamic state.

Composition dumps include both:

* ``composition_input`` with the original input string
* ``species_name`` as a plain comma-separated species list such as
  ``H2,He,CH4,H2O,NH3``

Examples
--------

Single-species xsection with custom broadening:

.. code-block:: bash

   pyharp-dump xsection \
       --species H2O \
       --temperature-k 300 \
       --pressure-bar 1 \
       --broadening-composition H2:0.9,He:0.1 \
       --wn-range=20,2500

Multi-band xsection in one file:

.. code-block:: bash

   pyharp-dump xsection \
       --species H2O \
       --temperature-k 300 \
       --pressure-bar 1 \
       --wn-range=20,2500 \
       --wn-range=2500,10000

Composition transmission dump:

.. code-block:: bash

   pyharp-dump transmission \
       --composition H2:0.9,He:0.1,CH4:0.004,H2O:0.002 \
       --path-length-m 1000 \
       --temperature-k 300 \
       --pressure-bar 1 \
       --wn-range=20,2500
