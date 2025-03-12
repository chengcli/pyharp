
setup_problem(amars_sw)
setup_problem(amars_lw)
setup_problem(amars_rt_cli)

# 4. Copy input files to run directory
file(GLOB inputs *.inp *.yaml *.txt *.atm)
foreach(input ${inputs})
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/bin/${inp})
endforeach()
