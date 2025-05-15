API Reference
=============

Global States
-------------

The following functions are used to set and get global states in the Pyharp package.

.. autofunction:: pyharp.species_names
.. autofunction:: pyharp.species_weights
.. autofunction:: pyharp.shared
.. autofunction:: pyharp.set_search_paths
.. autofunction:: pyharp.find_resource

Radiation Classes
-----------------

The following classes are used to configure and compute radiative transfer.

.. autoclass:: pyharp.RadiationOptions
   :members:

.. autoclass:: pyharp.RadiationBandOptions
   :members:

.. autoclass:: pyharp.cpp.RadiationBand
   :members: __init__, forward

.. autoclass:: pyharp.cpp.Radiation
   :members: __init__, forward

Opacity Classes
---------------

.. autoclass:: pyharp.AttenuatorOptions
   :members:

.. autoclass:: pyharp.cpp.S8Fuller
   :members: __init__, forward

.. autoclass:: pyharp.cpp.H2SO4Simple
   :members: __init__, forward

.. autoclass:: pyharp.cpp.RFM
   :members: __init__, forward

Helper Functions
----------------

The following functions are auxiliary (helper) functions frequently used in radiative transfer problems.

.. autofunction:: pyharp.bbflux_wavenumber
.. autofunction:: pyharp.bbflux_wavelength
.. autofunction:: pyharp.calc_dz_hypsometric
.. autofunction:: pyharp.interpn
