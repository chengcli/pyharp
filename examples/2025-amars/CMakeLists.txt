
# setup_problem(amars_sw)
# setup_problem(amars_lw)
# setup_problem(amars_rt_cli)
# setup_problem(amars_rt)
# setup_problem(amars_rt_rk3)
setup_problem(uran_rt_rk3)

# 4. Copy input files to run directory
file(GLOB inputs *.inp *.yaml *.txt *.atm)
foreach(input ${inputs})
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/bin/${inp})
endforeach()
