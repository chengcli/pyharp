
setup_problem(amars_sw)
setup_problem(amars_lw)

# 4. Copy input files to run directory
file(GLOB inputs *.inp *.yaml a*.txt)
foreach(input ${inputs})
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/bin/${inp})
endforeach()
