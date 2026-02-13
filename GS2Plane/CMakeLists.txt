# GS2Plane subproject - don't set minimum version or build type
# (inherited from parent)
project(GS2Plane)

# Compiler flags inherited from parent project

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

# Find all .cpp files in current directory
file(GLOB CPP_FILES "*.cpp")

# Create executable for each .cpp file
foreach(CPP_FILE ${CPP_FILES})
    # Get filename without extension for executable name
    get_filename_component(EXEC_NAME ${CPP_FILE} NAME_WE)
    
    # Add executable
    add_executable(${EXEC_NAME} ${CPP_FILE})
    
    # Link CGAL and Eigen support to the executable
    target_link_libraries(${EXEC_NAME} PRIVATE CGAL::CGAL)
    if(TARGET CGAL::Eigen3_support)
        target_link_libraries(${EXEC_NAME} PRIVATE CGAL::Eigen3_support)
    endif()
    
    # Ensure Eigen include/linking is available for headers like <Eigen/Dense>
    target_link_libraries(${EXEC_NAME} PRIVATE Eigen3::Eigen)
    
    # C++17 is inherited from parent project
    set_target_properties(${EXEC_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )
    
    message(STATUS "GS2Plane: Created executable ${EXEC_NAME} from ${CPP_FILE}")
endforeach()
