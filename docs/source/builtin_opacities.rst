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

NetCDF dump-backed molecular opacities
--------------------------------------

Pyharp also supports NetCDF dump-backed molecular absorption through the
``molecule-line`` and ``cia`` opacity types. These readers consume
cross-section fields on ``(del_temperature, pressure, wavenumber)`` or any
equivalent variable-dimension ordering, convert them into the runtime units
used by the C++ radiative-transfer core, and interpolate on the requested
band grid.

Gas Rayleigh scattering
-----------------------

The ``rayleigh`` opacity type computes gas Rayleigh scattering analytically on
the active wavelength or wavenumber grid. It supports H2, He, H2O, CH4, N2, CO2,
and NH3, and returns conservative scattering with the Rayleigh Legendre moments,
and is mixed with line and CIA absorption by the radiation band before the
DISORT calculation.

Water-cloud optical properties
------------------------------

Pyharp provides separate optical-property models for liquid-water droplets
and water-ice particles, together with a temperature-partitioned model for a
single total-condensate field. All three models interpret the configured
species concentration as molar concentration in mol/m3 and return extinction
in 1/m, single-scattering albedo, and Legendre moments excluding the zeroth
moment.

Liquid-water clouds
~~~~~~~~~~~~~~~~~~~

The ``water-liquid-mie`` opacity type treats liquid-water droplets as
homogeneous, monodisperse spheres and evaluates the full Lorenz--Mie series.
The droplet radius ``re`` is supplied in um. By default, the wavelength-
dependent complex refractive index is interpolated from Segelstein (1981) [6]_
for liquid water at 25 C over 0.1--1000 um. The liquid-water density and both
parts of the refractive index can also be overridden at run time.

The Mie calculation supplies extinction, single-scattering albedo, and the
asymmetry factor ``g``. Until the full Mie phase function is projected
directly, the higher-order Legendre moments are represented by the
Henyey--Greenstein approximation, :math:`\beta_l = g^l`.

Water-ice clouds
~~~~~~~~~~~~~~~~

The ``water-ice-fu96-98`` opacity type uses the cirrus-cloud parameterizations
of Fu (1996) [7]_ from 0.25 um up to 4 um and Fu, Yang, and Sun (1998) [8]_
from 4 to 100 um. Optical properties are set to zero outside 0.25--100 um.
The required ``re`` is the ice effective radius in um; internally it is
converted to the generalized effective size used by the Fu tables,

.. math::

   D_{ge} = \frac{8 r_e}{3\sqrt{3}}.

The optional scalar flag ``fu_delta_scale`` applies the Fu (1996) delta
scaling. It is false by default because the Toon shortwave solver already
applies delta-Eddington scaling. The returned higher-order phase moments use
the Henyey--Greenstein representation of the parameterized asymmetry factor.

Temperature-partitioned water clouds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``water-cloud-temperature-switch`` opacity type is intended for models
that provide one total nonprecipitating water-condensate field instead of
separate liquid and ice concentrations. The liquid fraction is

.. math::

   \omega_l = \max\left[0,\min\left(1,
   \frac{T - 253.15\ \mathrm{K}}{273.15\ \mathrm{K} - 253.15\ \mathrm{K}}
   \right)\right].

Condensate is therefore entirely liquid at and above 273.15 K, entirely ice
at and below 253.15 K, and mixed phase between those temperatures. The model
requires ``temp``, ``liquid_re``, and ``ice_re`` at run time. It calls the Mie
liquid model and the Fu ice model separately, then combines their extinction,
single-scattering albedo, and phase moments using the appropriate extinction
and scattering weights.

The three opacity types are configured as follows. The named liquid, ice, or
total-condensate species must also be present in the top-level ``species``
list.

.. code-block:: yaml

   opacities:
     liquid-cloud:
       type: water-liquid-mie
       species: [H2O(l)]
       nmom: 4

     ice-cloud:
       type: water-ice-fu96-98
       species: [H2O(s)]
       nmom: 4

     mixed-phase-cloud:
       type: water-cloud-temperature-switch
       species: [H2O(s)]  # interpreted here as total condensed water
       nmom: 4

References
----------
.. [1] Lupu, R., et al. "Correlated k coefficients for H2-He atmospheres; 196 spectral windows and 1460 pressure-temperature points." Zenodo, doi 0.5281/zenodo.5590988 (2021).
.. [2] Marley, Mark S., et al. "The Sonora brown dwarf atmosphere and evolution models. I. Model description and application to cloudless atmospheres in rainout chemical equilibrium." The Astrophysical Journal 920.2 (2021): 85.
.. [3] Li, C., Le, T., Zhang, X., & Yung, Y. L. (2018). A high-performance atmospheric radiation package: With applications to the radiative energy budgets of giant planets. Journal of Quantitative Spectroscopy and Radiative Transfer, 217, 353-362.
.. [4] Dalgarno, A., & Williams, D. A. (1962). The scattering of light by molecular hydrogen. Astrophysical Journal, 136, 690.
.. [5] Pierrehumbert, R. T. (2010). Principles of planetary climate. Cambridge University Press.
.. [6] Segelstein, D. J. (1981). The complex refractive index of water. M.S. thesis, University of Missouri--Kansas City.
.. [7] Fu, Q. (1996). An accurate parameterization of the solar radiative properties of cirrus clouds for climate models. Journal of Climate, 9, 2058-2082.
.. [8] Fu, Q., Yang, P., & Sun, W. B. (1998). An accurate parameterization of the infrared radiative properties of cirrus clouds for climate models. Journal of Climate, 11, 2223-2237.