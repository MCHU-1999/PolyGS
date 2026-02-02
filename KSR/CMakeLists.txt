cmake_minimum_required(VERSION 3.16)
project(PolyGS-KSR)

# Set build type
set(CMAKE_BUILD_TYPE "Release")

# Use appropriate flags for MSVC
if (MSVC)
  add_definitions(/W4 /O2)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -std=c++17")
endif()

# Find CGAL
find_package(CGAL QUIET COMPONENTS)
if (CGAL_FOUND)
  message(STATUS "CGAL found")
else()
  message(SEND_ERROR "CGAL library is required")
  return()
endif()

# Find Eigen3 (required for CGAL kinetic surface / Eigen types)
find_package(Eigen3 REQUIRED)
include(CGAL_Eigen3_support)

# Export compile commands for editor tooling (e.g., VS Code C/C++ IntelliSense)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add executable for ksr_buildings
add_executable(ksr_buildings ksr_buildings.cpp)

# Link CGAL and Eigen support to the executable
target_link_libraries(ksr_buildings PRIVATE CGAL::CGAL)
if(TARGET CGAL::Eigen3_support)
  target_link_libraries(ksr_buildings PRIVATE CGAL::Eigen3_support)
endif()

# Ensure Eigen include/linking is available for headers like <Eigen/Dense>
target_link_libraries(ksr_buildings PRIVATE Eigen3::Eigen)

# Ensure C++17 and place built files into the KSR/ folder
set_target_properties(ksr_buildings PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build
)
