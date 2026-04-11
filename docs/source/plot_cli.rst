pyharp-plot CLI
================

``pyharp-plot`` is the unified command line entry point for spectroscopy
diagnostic figures. It can plot HITRAN collision-induced absorption (CIA)
binary coefficients, molecular cross sections, attenuation coefficients,
transmission, and multi-panel overview PDFs.

Basic Usage
-----------

Choose one subcommand, then choose the target data source with ``--pair``,
``--species``, or ``--composition`` where that subcommand supports it.

.. code-block:: bash

   pyharp-plot binary --pair H2-H2 --temperature-k 300 --wn-range=20,10000
   pyharp-plot xsection --species CO2 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
   pyharp-plot attenuation --species H2O --temperature-k 300 --pressure-bar 1 --wn-range=25,2500
   pyharp-plot transmission --composition H2O:0.1,H2:0.9 --temperature-k 300 --pressure-bar 1 --path-length-km 1 --wn-range=25,2500
   pyharp-plot overview --species H2O --temperature-k 300 --pressure-bar 1 --path-length-km 1 --wn-range=20,2500

Every command writes a figure. Use ``--output`` to select an explicit output
path. Without ``--output``, plots are written under ``--output-dir`` using a
name derived from the target, plot type, temperature, pressure, and
wavenumber range, such as
``output/co2_xsection_300K_1bar_20_2500.png``.

Subcommands
-----------

``binary``
~~~~~~~~~~

Plot a HITRAN CIA binary absorption coefficient spectrum. This subcommand
uses a CIA pair selected by ``--pair``.

.. code-block:: bash

   pyharp-plot binary --pair H2-H2 --temperature-k 300 --wn-range=20,10000
   pyharp-plot binary --pair H2-He --temperature-k 500 --resolution 5 --output output/h2_he_cia.png

``xsection``
~~~~~~~~~~~~

Plot a molecular absorption cross section at one pressure-temperature state.
This subcommand uses a molecule selected by ``--species``.

.. code-block:: bash

   pyharp-plot xsection --species CO2 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
   pyharp-plot xsection --species CH4 --temperature-k 700 --pressure-bar 0.1 --wn-range=1000,4000 --refresh-hitran

``attenuation``
~~~~~~~~~~~~~~~

Plot an attenuation coefficient in ``1/m``. Select exactly one target:

* ``--pair`` for a CIA pair.
* ``--species`` for a molecule.
* ``--composition`` for a gas mixture.

.. code-block:: bash

   pyharp-plot attenuation --pair H2-H2 --temperature-k 300 --pressure-bar 1 --wn-range=20,10000
   pyharp-plot attenuation --species CO2 --temperature-k 300 --pressure-bar 1 --wn-range=20,2500
   pyharp-plot attenuation --composition H2O:0.1,H2:0.9 --temperature-k 300 --pressure-bar 1 --wn-range=25,2500

``transmission``
~~~~~~~~~~~~~~~~

Plot transmission over a fixed path length. Select exactly one target with
``--pair``, ``--species``, or ``--composition``. The path length defaults to
``1`` km and can be changed with ``--path-length-km``.

.. code-block:: bash

   pyharp-plot transmission --pair H2-H2 --temperature-k 300 --pressure-bar 1 --path-length-km 1 --wn-range=20,10000
   pyharp-plot transmission --species CO2 --temperature-k 300 --pressure-bar 1 --path-length-km 0.5 --wn-range=20,2500
   pyharp-plot transmission --composition H2O:0.1,H2:0.9 --temperature-k 300 --pressure-bar 1 --path-length-km 1 --wn-range=25,2500

``overview``
~~~~~~~~~~~~

Generate a multi-panel PDF. With one species and one wavenumber range, the
output is a single molecule overview. With multiple species or multiple
``--wn-range`` values, the output is a combined multi-page PDF. With
``--composition``, the output is an atmospheric mixture overview PDF and a
manifest JSON file.

.. code-block:: bash

   pyharp-plot overview --species H2O --temperature-k 300 --pressure-bar 1 --path-length-km 1 --wn-range=20,2500
   pyharp-plot overview --species H2O CO2 CH4 --temperature-k 500 --pressure-bar 0.5 --wn-range=25,2500 --wn-range=2500,10000
   pyharp-plot overview --composition H2O:0.1,H2:0.9 --temperature-k 300 --pressure-bar 1 --wn-range=25,2500 --manifest output/h2o_h2_sources.json

Shared Options
--------------

``--wn-range=min,max``
    Wavenumber bounds in ``cm^-1``. CIA plots default to ``20,10000`` when it
    is omitted. Molecular and mixture plots default to ``20,2500``. The
    ``overview`` subcommand accepts this option more than once. Ranges are
    lower-inclusive and upper-exclusive, so ``--wn-range=20,22`` with
    ``--resolution 1`` samples ``20`` and ``21``.

``--resolution value``
    Wavenumber grid spacing in ``cm^-1``. The default is ``1``.

``--temperature-k value``
    Temperature in kelvin. The default is ``300``.

``--pressure-bar value``
    Pressure in bar. The default is ``1`` for subcommands that use pressure.

``--hitran-dir path``
    Directory used for downloaded HITRAN line and CIA data. The default is
    ``hitran``.

``--refresh-hitran`` and ``--refresh-cia``
    Re-download cached HITRAN line or CIA data.

``--cia-index-url url``
    HITRAN CIA index URL. The default is ``https://hitran.org/cia/``.

``--output path``
    Explicit output path. Use ``.png`` for single plots and ``.pdf`` for
    overview plots.

``--output-dir path``
    Directory used for auto-generated figure filenames. This is ignored when
    ``--output`` is provided.

``--broadening-composition BROADENER:FRACTION,...``
    Line-broadening gas composition for molecular line calculations. This
    option affects HITRAN line absorption only, not CIA-only ``--pair``
    workflows.

Target Selection
----------------

Use ``--pair`` for a CIA pair, for example ``H2-H2`` or ``H2-He``. Built-in
CIA pair resolution includes ``CH4-CH4``, ``CO2-CH4``, ``CO2-CO2``,
``CO2-H2``, ``H2-H2``, ``H2-He``, ``N2-CH4``, and ``N2-N2``.

Use ``--species`` for molecular line calculations. Built-in HITRAN line
species are ``CH4``, ``CO2``, ``H2``, ``H2O``, ``H2S``, ``N2``, and ``NH3``.

Use ``--composition`` for a gas mixture. The format is a comma-separated list
of ``species:fraction`` terms:

.. code-block:: bash

   --composition H2O:0.1,H2:0.9

For ``attenuation`` and ``transmission``, choose only one of ``--pair``,
``--species``, and ``--composition``. For ``overview``, choose only one of
``--species`` and ``--composition``.

Broadening Gases
----------------

The practical broadening gas keys supported by pyharp are:

* ``air`` for standard HITRAN air broadening
* ``self`` for absorber self broadening
* ``H2``
* ``He``
* ``CO2``

Examples:

.. code-block:: bash

   pyharp-plot xsection --species CO2 --broadening-composition air:0.8,self:0.2
   pyharp-plot transmission --species CH4 --broadening-composition H2:0.85,He:0.15 --path-length-km 1
   pyharp-plot overview --composition H2O:0.1,H2:0.9 --broadening-composition CO2:1 --wn-range=25,2500

Fallback Behavior
-----------------

If ``--broadening-composition`` is omitted, single-species molecular
workflows default to ``self`` broadening. Atmosphere-mixture workflows default
to the plotted gas mixture as the requested line-broadening composition.

If a requested foreign broadener cannot be found in the HITRAN line table for
the active absorber, pyharp falls back to ``air`` for that fraction. For
example, a requested broadening mixture of ``H2:0.85,He:0.15`` will be
remapped to ``air`` for any unavailable broadener component.

If fallback to ``air`` is needed but the line table also lacks ``air``
broadening parameters, pyharp raises an error instead of inventing
coefficients. Invalid broadening-composition syntax also raises a validation
error.

Output Examples
---------------

Default filenames are normalized for shells and filesystems:

.. code-block:: bash

   pyharp-plot xsection --species CO2 --temperature-k 275.5 --pressure-bar 0.25 --wn-range=25,30.5
   # writes output/co2_xsection_275p5K_0p25bar_25_30p5.png

   pyharp-plot overview --composition H2O:0.1,H2:0.9 --wn-range=25,2500
   # writes output/h2o_0p1_h2_0p9_overview_300K_1bar_25_2500.pdf
   # also writes output/h2o_0p1_h2_0p9_overview_300K_1bar_25_2500.manifest.json
