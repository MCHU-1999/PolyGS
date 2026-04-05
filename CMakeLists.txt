cmake_minimum_required(VERSION 3.12)
project(PolyGS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Common compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find common dependencies that might be used by subdirectories
find_package(CGAL QUIET)
if(CGAL_FOUND)
    message(STATUS "Found CGAL: ${CGAL_VERSION}")
else()
    message(WARNING "CGAL not found - some targets may not build")
endif()

find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${Eigen3_VERSION}")
    include(CGAL_Eigen3_support)
else()
    message(WARNING "Eigen3 not found - some targets may not build")
endif()

# Export compile commands for editor tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add subdirectories (this will create the executables)
message(STATUS "Adding subdirectory: GS2Plane")
add_subdirectory(GS2Plane)

message(STATUS "Adding subdirectory: KSR")
add_subdirectory(KSR)

message(STATUS "Adding subdirectory: PC2Mesh")
add_subdirectory(PC2Mesh)

# The default "all" target will now build everything automatically
# No need for a custom PolyGS target - just use the default behavior

# Print build summary
message(STATUS "")
message(STATUS "PolyGS Build Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CGAL: ${CGAL_FOUND}")
message(STATUS "  Eigen3: ${Eigen3_FOUND}")
message(STATUS "  Auto-building all .cpp files as executables")
message(STATUS "  Use 'make' or 'cmake --build .' to build all")
message(STATUS "")