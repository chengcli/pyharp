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
   pyharp-dump transmission --composition H2:0.9,He:0.1,H2O:0.002 --path-length-km 1 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500

Without ``--output``, files are written under ``--output-dir`` using names
derived from the target, product type, pressure, temperature, and wavenumber
range. Generated names end with ``cm1`` to indicate wavenumber bounds in
``cm^-1``. For state-grid dumps, the state portion of the filename is
``<min_temp>_<max_temp>K_<min_pres>_<max_pres>bar``.

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

   pyharp-dump transmission --species H2O --path-length-km 1 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
   pyharp-dump transmission --pair H2-He --path-length-km 1 --temperature-k 300 --pressure-bar 1 --wn-range=20,10000
   pyharp-dump transmission --composition H2:0.9,He:0.1,H2O:0.002 --path-length-km 1 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500

Shared Options
--------------

``--wn-range=min,max``
    Wavenumber bounds in ``cm^-1``. Repeat this option to write one NetCDF
    file per band. When ``--output`` is provided with multiple ranges,
    pyharp appends ``_<wnmin>_<wnmax>`` to the requested stem. Ranges are
    lower-inclusive and upper-exclusive, so ``--wn-range=20,22`` with
    ``--resolution 1`` samples ``20`` and ``21``.

``--resolution value``
    Wavenumber spacing in ``cm^-1``. The default is ``1``.

``--temperature-k value``
    Base temperature in kelvin. The default is ``300``. Use a comma-separated
    list such as ``300,400,500`` paired one-to-one with ``--pressure-bar``.

``--pressure-bar value``
    Pressure in bar. The default is ``1``. Use a comma-separated list paired
    one-to-one with ``--temperature-k``.

``--del-temperature-k value``
    Temperature anomalies in kelvin applied to each base ``(temperature,
    pressure)`` pair. The default is ``0``. For example, ``--temperature-k
    300 --pressure-bar 1 --del-temperature-k -10,-5,0,5,10`` evaluates the
    states ``290, 295, 300, 305, 310 K`` at ``1 bar``.

``--path-length-km value``
    Transmission path length in kilometers. This option is required only for
    ``transmission`` and defaults to ``1``.

``--output path``
    Explicit NetCDF output path.

``--output-dir path``
    Directory used for auto-generated NetCDF filenames. This is ignored when
    ``--output`` is provided.

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

This writes one file for ``[20, 2500)`` and one file for ``[2500, 10000)``.
Adjacent repeated ranges do not duplicate the boundary sample. If
``--output output/h2o.nc`` is provided, the generated files are
``output/h2o_20_2500.nc`` and ``output/h2o_2500_10000.nc``.

Auto-generated filenames follow the pattern
``<target>_<product>_<min_temp>_<max_temp>K_<min_pres>_<max_pres>bar_<wnmin>_<wnmax>cm1``.

State Grid Output
-----------------

Dump products are always written on the dimensions
``(del_temperature, pressure, wavenumber)``. The ``pressure`` coordinate
stores the paired ``--pressure-bar`` values converted to ``Pa``,
``del_temperature`` stores the requested anomalies, and the ``temperature``
data variable has dimensions ``(pressure,)`` with the base temperatures from
``--temperature-k``.

Even a single ``(temperature, pressure)`` pair is written with degenerate
``del_temperature`` and ``pressure`` dimensions of length one.

Parallel execution is flattened across all requested
``wn_range × pressure × del_temperature`` jobs.

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

Multi-band xsection in one run:

.. code-block:: bash

   pyharp-dump xsection \
       --species H2O \
       --temperature-k 300 \
       --pressure-bar 1 \
       --wn-range=20,2500 \
       --wn-range=2500,10000

Paired state grid with temperature anomalies:

.. code-block:: bash

   pyharp-dump xsection \
       --species H2O \
       --temperature-k 300,400 \
       --pressure-bar 1,10 \
       --del-temperature-k -10,-5,0,5,10 \
       --wn-range=20,2500

Composition transmission dump:

.. code-block:: bash

   pyharp-dump transmission \
       --composition H2:0.9,He:0.1,CH4:0.004,H2O:0.002 \
       --path-length-km 1 \
       --temperature-k 300 \
       --pressure-bar 1 \
       --wn-range=20,2500
