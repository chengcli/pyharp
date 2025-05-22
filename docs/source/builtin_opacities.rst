Built-in opacity sources
========================

Pyharp ships with a number of opacity sources that can be used to compute the
optical properties of the plantary atmosphere.

Sonora2020 molecular opacities
------------------------------

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
-----------------------------

Pyharp ships with the following continuum opacity sources for H2 and He:

* H2-H2-eq.xiz.pt
* H2-He-eq.xiz.pt
* H2-H2-nm.xiz.pt
* H2-He-nm.xiz.pt
* H2-H2-eq.orton.pt
* H2-He-eq.orton.pt
* H2-H2-nm.orton.pt
* H2-He-nm.orton.pt

These are legacy files that have been used the original HARP publication [3]_.
They are used here in Pyharp to compute the collisional induced absorption (CIA) of H2 and He molecules.
The following functions have been used to process the legacy CIA data files:

.. automodule:: pyharp.h2_cia_legacy
   :members:
   :undoc-members:
   :imported-members:

References
----------
.. [1] Lupu, R., et al. "Correlated k coefficients for H2-He atmospheres; 196 spectral windows and 1460 pressure-temperature points." Zenodo, doi 0.5281/zenodo.5590988 (2021).
.. [2] Marley, Mark S., et al. "The Sonora brown dwarf atmosphere and evolution models. I. Model description and application to cloudless atmospheres in rainout chemical equilibrium." The Astrophysical Journal 920.2 (2021): 85.
.. [3] Li, C., Le, T., Zhang, X., & Yung, Y. L. (2018). A high-performance atmospheric radiation package: With applications to the radiative energy budgets of giant planets. Journal of Quantitative Spectroscopy and Radiative Transfer, 217, 353-362.
