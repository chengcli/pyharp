
# Legacy AMARS examples were removed with the RFM workflow.

if (${CMAKE_SOURCE_DIR} STREQUAL ${PROJECT_SOURCE_DIR})
  setup_problem(example_molecule_dump_opacity)

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/molecule_dump_opacity.yaml
    ${CMAKE_BINARY_DIR}/bin/molecule_dump_opacity.yaml
    COPYONLY)

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/h2_0p9_he_0p1_ch4_0p004_h2o_0p002_nh3_0p0003_xsection_120_730K_1_100bar_20_2500cm1.nc
    ${CMAKE_BINARY_DIR}/bin/h2_0p9_he_0p1_ch4_0p004_h2o_0p002_nh3_0p0003_xsection_120_730K_1_100bar_20_2500cm1.nc
    COPYONLY)
endif()
