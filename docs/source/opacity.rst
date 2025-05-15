List of supported opacity sources
=================================

``Pyharp`` ships with a number of opacity sources that can be used to compute the
optical properties of the plantary atmosphere.

Sonora2020
----------

This is a database of pre-mixed Correlated-K hydrogen-helium opacities with abundances given by equilibrium chemistry
for each metallicity-C/O combination (version 3) [1]_.
It has been used for brown dwarf atmospheres [2]_.
View this `document <_static/sonora2020_opacity_references_20201.pdf>`_ for references of opacities included in the database.

Use the following script to checkout options and download the Sonora2020 database:

.. code-block:: bash

   fetch-sonora -h

By default, ``fetch-sonora`` downloads the database with [Fe/H] = 0.0 and C/O = 1 times solar abundances.

.. autofunction:: pyharp.sonora.load_sonora_atm
.. autofunction:: pyharp.sonora.load_sonora_window
.. autofunction:: pyharp.sonora.load_sonora_abundances
.. autofunction:: pyharp.sonora.load_sonora_data
.. autofunction:: pyharp.sonora.save_sonora_multiband

Add a new opacity format
------------------------

References
----------
.. [1] Lupu, R., et al. "Correlated k coefficients for H2-He atmospheres; 196 spectral windows and 1460 pressure-temperature points." Zenodo, doi 0.5281/zenodo.5590988 (2021).
.. [2] Marley, Mark S., et al. "The Sonora brown dwarf atmosphere and evolution models. I. Model description and application to cloudless atmospheres in rainout chemical equilibrium." The Astrophysical Journal 920.2 (2021): 85.
