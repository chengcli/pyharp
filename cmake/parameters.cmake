# define default parameters

# precision options
set_if_empty(PRECISION float)
if(${PRECISION} STREQUAL "float")
  set(PRECISION_FLOAT float)
  set(PRECISION_INT int32_t)
elseif(${PRECISION} STREQUAL "double")
  set(PRECISION_FLOAT double)
  set(PRECISION_INT int64_t)
else()
  message(FATAL_ERROR "PRECISION must be either float or double")
endif()

# logger options
set(LOGGER_LEVEL INFO)

# netcdf options
if(NOT NETCDF OR NOT DEFINED NETCDF)
  set(NETCDF_OPTION "NO_NETCDFOUTPUT")
else()
  set(NETCDF_OPTION "NETCDFOUTPUT")
  find_package(NetCDF REQUIRED)
endif()

# pnetcdf options
if(NOT PNETCDF OR NOT DEFINED PNETCDF)
  set(PNETCDF_OPTION "NO_PNETCDFOUTPUT")
else()
  set(PNETCDF_OPTION "PNETCDFOUTPUT")
  find_package(PNetCDF REQUIRED)
endif()

# fits options
if(NOT FITS OR NOT DEFINED FITS)
  set(FITS_OPTION "NO_FITSOUTPUT")
else()
  set(FITS_OPTION "FITSOUTPUT")
  find_package(cfitsio REQUIRED)
endif()

# hdf5 options
if(NOT HDF5 OR NOT DEFINED HDF5)
  set(HDF5_OPTION "NO_HDF5OUTPUT")
else()
  set(HDF5_OPTION "HDF5OUTPUT")
  find_package(HDF5 REQUIRED)
endif()