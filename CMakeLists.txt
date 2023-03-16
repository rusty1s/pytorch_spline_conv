cmake_minimum_required(VERSION 3.0)
project(torchsplineconv)
set(CMAKE_CXX_STANDARD 14)
set(TORCHSPLINECONV_VERSION 1.2.2)

option(WITH_CUDA "Enable CUDA support" OFF)

if(WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
  add_definitions(-DWITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

find_package(Python3 COMPONENTS Development)
find_package(Torch REQUIRED)

file(GLOB HEADERS csrc/spline_conv.h)
file(GLOB OPERATOR_SOURCES csrc/cpu/*.h csrc/cpu/*.cpp csrc/*.cpp)
if(WITH_CUDA)
  file(GLOB OPERATOR_SOURCES ${OPERATOR_SOURCES} csrc/cuda/*.h csrc/cuda/*.cu)
endif()

add_library(${PROJECT_NAME} SHARED ${OPERATOR_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} Python3::Python)
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME TorchSplineConv)

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${HEADERS}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(TORCHSPLINECONV_CMAKECONFIG_INSTALL_DIR "share/cmake/TorchSplineConv" CACHE STRING "install path for TorchSplineConvConfig.cmake")

configure_package_config_file(cmake/TorchSplineConvConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/TorchSplineConvConfig.cmake"
  INSTALL_DESTINATION ${TORCHSPLINECONV_CMAKECONFIG_INSTALL_DIR})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/TorchSplineConvConfigVersion.cmake
  VERSION ${TORCHSPLINECONV_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TorchSplineConvConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/TorchSplineConvConfigVersion.cmake
  DESTINATION ${TORCHSPLINECONV_CMAKECONFIG_INSTALL_DIR})

install(TARGETS ${PROJECT_NAME}
  EXPORT TorchSplineConvTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install(EXPORT TorchSplineConvTargets
  NAMESPACE TorchSplineConv::
  DESTINATION ${TORCHSPLINECONV_CMAKECONFIG_INSTALL_DIR})

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
install(FILES
  csrc/cpu/basis_cpu.h
  csrc/cpu/weighting_cpu.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cpu)
if(WITH_CUDA)
  install(FILES
    csrc/cuda/basis_cuda.h
    csrc/cuda/weighting_cuda.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cuda)
endif()

if(WITH_CUDA)
  set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
  set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
endif()
