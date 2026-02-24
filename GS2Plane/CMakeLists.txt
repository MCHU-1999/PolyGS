# GS2Plane subproject - don't set minimum version or build type
# (inherited from parent)
# Compiler flags inherited from parent project
project(GS2Plane)

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

# --- Executable 1: gs_to_rectangles ---
add_executable(gs_to_rectangles gs_to_rectangles.cpp)

target_link_libraries(gs_to_rectangles PRIVATE CGAL::CGAL)
if(TARGET CGAL::Eigen3_support)
    target_link_libraries(gs_to_rectangles PRIVATE CGAL::Eigen3_support)
endif()
target_link_libraries(gs_to_rectangles PRIVATE Eigen3::Eigen)

set_target_properties(gs_to_rectangles PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

# --- Executable 2: gs_to_superpoints ---
set(CP_SRC_DIR "${CMAKE_SOURCE_DIR}/pycut_pursuit/src")

# Add the source files to the executable target
add_executable(gs_to_superpoints 
    gs_to_superpoints.cpp
    "${CP_SRC_DIR}/cp_d0_dist.cpp"
    "${CP_SRC_DIR}/cut_pursuit_d0.cpp"
    "${CP_SRC_DIR}/cut_pursuit_d1.cpp"
    "${CP_SRC_DIR}/cut_pursuit.cpp"
    "${CP_SRC_DIR}/maxflow.cpp"
)

target_compile_definitions(gs_to_superpoints PRIVATE COMP_T_ON_32_BITS)
target_include_directories(gs_to_superpoints PRIVATE "${CMAKE_SOURCE_DIR}/pycut_pursuit/include")

target_link_libraries(gs_to_superpoints PRIVATE CGAL::CGAL)
if(TARGET CGAL::Eigen3_support)
    target_link_libraries(gs_to_superpoints PRIVATE CGAL::Eigen3_support)
endif()
target_link_libraries(gs_to_superpoints PRIVATE Eigen3::Eigen)

set_target_properties(gs_to_superpoints PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

message(STATUS "GS2Plane: Created executables gs_to_rectangles and gs_to_superpoints")
