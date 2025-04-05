# Detects whether this is a top-level project
get_directory_property(HAS_PARENT PARENT_DIRECTORY)

if(HAS_PARENT)
  set(SCALABLE_CCD_TOPLEVEL_PROJECT OFF)
else()
  set(SCALABLE_CCD_TOPLEVEL_PROJECT ON)
endif()

# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.18.0")

if(SCALABLE_CCD_TOPLEVEL_PROJECT)
  cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})
else()
  # Don't use cmake_minimum_required here to avoid implicitly overriding parent policies
  if(${CMAKE_VERSION} VERSION_LESS ${REQUIRED_CMAKE_VERSION})
    message(FATAL_ERROR "CMake required version to build Scalable CCD is ${REQUIRED_CMAKE_VERSION}")
  endif()
endif()

# Include user-provided default options if available. We do that before the main
# `project()` so that we can define the C/C++ compilers from the option file.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ScalableCCDOptions.cmake)
  message(STATUS "Using local options file: ${CMAKE_CURRENT_SOURCE_DIR}/ScalableCCDOptions.cmake")
  include(${CMAKE_CURRENT_SOURCE_DIR}/ScalableCCDOptions.cmake)
endif()

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)

if(CCACHE_PROGRAM)
  option(SCALABLE_CCD_WITH_CCACHE "Enable ccache when building Scalable CCD" ${SCALABLE_CCD_TOPLEVEL_PROJECT})
else()
  option(SCALABLE_CCD_WITH_CCACHE "Enable ccache when building Scalable CCD" OFF)
endif()

if(SCALABLE_CCD_WITH_CCACHE AND CCACHE_PROGRAM)
  message(STATUS "Enabling Ccache support (${CCACHE_PROGRAM})")
  set(ccacheEnv
    CCACHE_BASEDIR=${CMAKE_BINARY_DIR}
    CCACHE_SLOPPINESS=clang_index_store,include_file_ctime,include_file_mtime,locale,pch_defines,time_macros
  )

  foreach(lang IN ITEMS C CXX)
    set(CMAKE_${lang}_COMPILER_LAUNCHER
      ${CMAKE_COMMAND} -E env ${ccacheEnv} ${CCACHE_PROGRAM}
    )
  endforeach()
endif()

# ==============================================================================
# CMake Policies
# ==============================================================================
cmake_policy(SET CMP0054 NEW) # Only interpret if() arguments as variables or keywords when unquoted.
cmake_policy(SET CMP0076 NEW) # target_sources() command converts relative paths to absolute.

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
  cmake_policy(SET CMP0135 NEW) # Set the timestamps of all extracted contents to the time of the extraction.
endif()

# ==============================================================================
project(ScalableCCD
  DESCRIPTION "Sweep and Tiniest Queue & Tight-Inclusion GPU CCD"
  LANGUAGES CXX
  VERSION "0.1.0")

option(SCALABLE_CCD_WITH_CUDA "Enable CUDA CCD" OFF)
option(SCALABLE_CCD_USE_DOUBLE "Use double precision calculation" ON)
option(SCALABLE_CCD_TOI_PER_QUERY "Output time of impact per query in JSON" OFF)
option(SCALABLE_CCD_BUILD_TESTS "Build Scalable CCD tests" ${SCALABLE_CCD_TOPLEVEL_PROJECT})
option(SCALABLE_CCD_WITH_PROFILER "Enable profiler" ${SCALABLE_CCD_TOPLEVEL_PROJECT})

# Set default minimum C++ standard
if(SCALABLE_CCD_TOPLEVEL_PROJECT)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# Configuration
set(SCALABLE_CCD_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/scalable_ccd")
set(SCALABLE_CCD_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/src")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/scalable_ccd/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/recipes/")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/find/")

# General CMake utils
include(scalable_ccd_cpm_cache)
include(scalable_ccd_use_colors)

# Generate position-independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==============================================================================
# Scalable CCD Library
# ==============================================================================

# Add an empty library and fill in the list of sources in `src/scalable_ccd/CMakeLists.txt`.
add_library(scalable_ccd)
add_library(scalable_ccd::scalable_ccd ALIAS scalable_ccd)

# Fill in configuration options
configure_file(
  "${SCALABLE_CCD_SOURCE_DIR}/config.hpp.in"
  "${SCALABLE_CCD_SOURCE_DIR}/config.hpp")

# Add source and header files to scalable_ccd
add_subdirectory("${SCALABLE_CCD_SOURCE_DIR}")

# Public include directory for Scalable CCD
target_include_directories(scalable_ccd PUBLIC "${SCALABLE_CCD_INCLUDE_DIR}")

# ==============================================================================
# Optional Definitions
# ==============================================================================

# For MSVC, do not use the min and max macros.
target_compile_definitions(scalable_ccd PUBLIC NOMINMAX)

# ==============================================================================
# Dependencies
# ==============================================================================

# Eigen
include(eigen)
target_link_libraries(scalable_ccd PUBLIC Eigen3::Eigen)

# TBB
include(onetbb)
target_link_libraries(scalable_ccd PRIVATE TBB::tbb)

# Logger
include(spdlog)
target_link_libraries(scalable_ccd PUBLIC spdlog::spdlog)

# JSON
if(SCALABLE_CCD_WITH_PROFILER)
  include(json)
  target_link_libraries(scalable_ccd PRIVATE nlohmann_json::nlohmann_json)
endif()

# Extra warnings (link last for highest priority)
include(scalable_ccd_warnings)
target_link_libraries(scalable_ccd PRIVATE scalable_ccd::warnings)

# ==============================================================================
# Compiler options
# ==============================================================================

# Use C++17
target_compile_features(scalable_ccd PUBLIC cxx_std_17)

# ==============================================================================
# CUDA
# ==============================================================================
if(SCALABLE_CCD_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)

  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "No CUDA support found!")
  endif()

  # We need to explicitly state that we need all CUDA files in the particle
  # library to be built with -dc as the member functions could be called by
  # other libraries and executables.
  set_target_properties(scalable_ccd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  if(DEFINED SCALABLE_CCD_CUDA_ARCHITECTURES)
    message(STATUS "CUDA_ARCHITECTURES was specified, skipping auto-detection")
    set(CMAKE_CUDA_ARCHITECTURES ${SCALABLE_CCD_CUDA_ARCHITECTURES})
  elseif(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
    set(CMAKE_CUDA_ARCHITECTURES "native")
  else()
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(CUDA_ARCH_LIST)
    string(STRIP "${CUDA_ARCH_LIST}" CUDA_ARCH_LIST)
    string(REPLACE " " ";" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  endif()
  message(STATUS "Setting CUDA_ARCHITECTURES to \"${CMAKE_CUDA_ARCHITECTURES}\"")

  set_target_properties(scalable_ccd PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

  if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET scalable_ccd
      PROPERTY
      BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()

  target_compile_options(scalable_ccd PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --generate-line-info
      --use_fast_math
      --relocatable-device-code=true
    >
  )
  target_compile_options(scalable_ccd PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:
      --expt-relaxed-constexpr
    >
  )

  find_package(CUDAToolkit)
  target_link_libraries(scalable_ccd PUBLIC CUDA::cudart)
endif()

# ==============================================================================
# Tests
# ==============================================================================

# Enable unit testing at the root level
if(SCALABLE_CCD_TOPLEVEL_PROJECT AND SCALABLE_CCD_BUILD_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(tests)
endif()
