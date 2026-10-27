# ============================
# A collection of unit tests
# ============================

enable_testing()

setup_test(test_attenuator)
setup_test(test_band_accumulate)
setup_test(test_cia_autograd)
setup_test(test_interpn)
setup_test(test_hypsometric)
setup_test(test_trapez)
setup_test(test_bbflux)
setup_test(test_flux_utils)
setup_test(test_rayleigh)
setup_test(test_water_ice_fu96_98)
setup_test(test_water_liquid_mie)
setup_test(test_water_cloud_temperature)
setup_test(test_yaml_input)
setup_test(test_tridiag)
setup_test(test_toon)
setup_test(test_scattering_functions)
setup_test(test_composition)
setup_test(test_read_stellar_flux)
setup_test(test_read_yaml)
setup_test(test_mie_small_particle)

# fixtures of the C++ tests
configure_file(rayleigh_test.yaml rayleigh_test.yaml COPYONLY)
configure_file(example_infrared.yaml example_infrared.yaml COPYONLY)
configure_file(stellar_flux_test.txt stellar_flux_test.txt COPYONLY)
configure_file(stellar_wavelength_test.txt stellar_wavelength_test.txt COPYONLY)

# Python tests: each file runs under pytest with the interpreter that built the
# package (the installed pyharp must be importable, as in CI after `pip install`)

configure_file(water_ice_fu_test.yaml water_ice_fu_test.yaml COPYONLY)
configure_file(water_liquid_mie_test.yaml water_liquid_mie_test.yaml COPYONLY)
configure_file(water_cloud_temperature_test.yaml water_cloud_temperature_test.yaml COPYONLY)

file(GLOB pyfiles *.py)

foreach(pyfile ${pyfiles})
  get_filename_component(name ${pyfile} NAME)
  message(STATUS "Copying ${pyfile} to ${name}")
  configure_file(${pyfile} ${CMAKE_CURRENT_BINARY_DIR}/${name} @ONLY)
  add_test(NAME ${name} COMMAND ${Python3_EXECUTABLE} -m pytest -q -p no:cacheprovider ${name})
endforeach()
