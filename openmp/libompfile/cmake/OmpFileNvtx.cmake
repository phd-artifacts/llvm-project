# NVTX for the ompfile/MPP trace layer (include/ompfile_trace.h).
#
# Included by libompfile and by the MPI plugin's EventSystem, so the origin
# plugin, the proxy tool and libompfile share one decision. NVTX v3 is
# header-only; its ranges are collected by Nsight Systems (nsys) and cost a
# no-op stub call when no collector is attached.
#
# The header is referenced by absolute path through OMPFILE_NVTX_HEADER rather
# than by adding its directory to the include path: nvToolsExt.h only uses
# relative quote includes, and putting a CUDA include directory on the plugin's
# search path could shadow the offload CUDA plugin's own cuda.h.
#
# Auto-disables when the header is absent, so a toolchain without CUDA headers
# still builds. (Both clusters currently build inside the ompc-base container,
# which has them, so in practice it is on.)
#
# Everything the function reads is a CACHE variable on purpose: libompfile and
# the plugin's event_system are sibling directory scopes, so a plain variable
# set by the first inclusion would be invisible to the second.

option(OMPFILE_ENABLE_NVTX
  "Emit NVTX ranges from libompfile and the MPI plugin (auto-off without nvtx3/nvToolsExt.h)"
  ON)

if(OMPFILE_ENABLE_NVTX)
  find_file(OMPFILE_NVTX_HEADER_PATH nvtx3/nvToolsExt.h
    HINTS "$ENV{CUDA_HOME}/include" "$ENV{CUDA_PATH}/include"
          /usr/local/cuda/include /usr/include
    NO_CMAKE_FIND_ROOT_PATH)
endif()

get_property(_ompfile_nvtx_reported GLOBAL PROPERTY OMPFILE_NVTX_REPORTED)
if(NOT _ompfile_nvtx_reported)
  set_property(GLOBAL PROPERTY OMPFILE_NVTX_REPORTED TRUE)
  if(NOT OMPFILE_ENABLE_NVTX)
    message(STATUS "ompfile NVTX tracing: OFF (OMPFILE_ENABLE_NVTX=OFF)")
  elseif(OMPFILE_NVTX_HEADER_PATH)
    message(STATUS "ompfile NVTX tracing: ON (${OMPFILE_NVTX_HEADER_PATH})")
  else()
    message(STATUS "ompfile NVTX tracing: OFF (nvtx3/nvToolsExt.h not found)")
  endif()
endif()

# ompfile_enable_nvtx(<target> <PUBLIC|PRIVATE|INTERFACE>)
function(ompfile_enable_nvtx target scope)
  if(OMPFILE_ENABLE_NVTX AND OMPFILE_NVTX_HEADER_PATH)
    target_compile_definitions(${target} ${scope}
      OMPFILE_ENABLE_NVTX=1
      OMPFILE_NVTX_HEADER="${OMPFILE_NVTX_HEADER_PATH}")
    # NVTX's lazy init dlopen()s the injection library nsys provides.
    target_link_libraries(${target} ${scope} ${CMAKE_DL_LIBS})
  endif()
endfunction()
