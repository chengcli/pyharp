# ============================
# A collection of unit tests
# ============================

enable_testing()

setup_test(test_attenuator)
setup_test(test_interpn)
setup_test(test_hypsometric)
setup_test(test_trapez)
setup_test(test_bbflux)
setup_test(test_flux_utils)
setup_test(test_rayleigh)
setup_test(test_water_ice_fu96_98)
setup_test(test_water_liquid_mie)
#setup_test(test_yaml_input)
setup_test(test_tridiag)
setup_test(test_toon)

# Python tests

configure_file(rayleigh_test.yaml rayleigh_test.yaml COPYONLY)
configure_file(water_ice_fu_test.yaml water_ice_fu_test.yaml COPYONLY)
configure_file(water_liquid_mie_test.yaml water_liquid_mie_test.yaml COPYONLY)

file(GLOB pyfiles *.py)

foreach(pyfile ${pyfiles})
  get_filename_component(name ${pyfile} NAME)
  message(STATUS "Copying ${pyfile} to ${name}")
  configure_file(${pyfile} ${CMAKE_CURRENT_BINARY_DIR}/${name} @ONLY)
  #add_test(NAME ${name} COMMAND pytest ${name})
endforeach()
