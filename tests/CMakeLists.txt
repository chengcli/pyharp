# ============================
# A collection of unit tests
# ============================

enable_testing()

#setup_test(test_attenuator)
setup_test(test_interpn)
setup_test(test_hypsometric)
setup_test(test_trapez)
setup_test(test_bbflux)
setup_test(test_flux_utils)
#setup_test(test_yaml_input)
setup_test(test_tridiag)
setup_test(test_composition)

# Python tests

file(GLOB pyfiles *.py)

foreach(pyfile ${pyfiles})
  get_filename_component(name ${pyfile} NAME)
  message(STATUS "Copying ${pyfile} to ${name}")
  configure_file(${pyfile} ${CMAKE_CURRENT_BINARY_DIR}/${name} @ONLY)
  add_test(NAME ${name} COMMAND pytest ${name})
endforeach()
