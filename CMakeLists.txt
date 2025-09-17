cmake_minimum_required(VERSION 3.5)
project(mesh_nav)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pinocchio REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(
  include
  /usr/local/include
  /usr/include/eigen3
  ${CMAKE_CURRENT_SOURCE_DIR}/external/ligigl/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external/tinyply/source
)

link_directories(/usr/local/lib)

# Options for libigl
option(LIBIGL_WITH_EIGEN ON)
option(LIBIGL_WITH_PYTHON OFF)
option(LIBIGL_WITH_TETGEN OFF)
option(LIBIGL_WITH_TRIANGLE OFF)
option(LIBIGL_WITH_CGAL OFF)
option(LIBIGL_WITH_COMISO OFF)

###########
## Build ##
###########

# Library for robot kinematics
add_library(RobotKinematics
  src/mesh_nav_ros/RobotKinematics.cpp
)
set_target_properties(RobotKinematics PROPERTIES POSITION_INDEPENDENT_CODE ON)
ament_target_dependencies(RobotKinematics rclcpp std_msgs pinocchio)

# Library for SDF from PLY
add_library(SDFfromPLY
  src/mesh_nav_ros/SDFfromPLY.cpp
)
set_target_properties(SDFfromPLY PROPERTIES POSITION_INDEPENDENT_CODE ON)
ament_target_dependencies(SDFfromPLY rclcpp std_msgs)

# Library for voxel-based SDF
add_library(VoxelSDF
  src/mesh_nav_ros/VoxelSDF.cpp
)
set_target_properties(VoxelSDF PROPERTIES POSITION_INDEPENDENT_CODE ON)
ament_target_dependencies(VoxelSDF rclcpp std_msgs)

# Executable for SDF computing
add_executable(SDFcomputing
  src/mesh_nav_ros/SDFcomputing.cpp
)
ament_target_dependencies(SDFcomputing rclcpp std_msgs)
target_link_libraries(SDFcomputing SDFfromPLY OpenMP::OpenMP_CXX)

# Core planner library
add_library(elastic_band_planner_core SHARED
    src/mesh_nav_ros/ElasticBandPlanner.cpp
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(elastic_band_planner_core
    OpenMP::OpenMP_CXX
    VoxelSDF
    RobotKinematics
  )
  target_compile_definitions(elastic_band_planner_core PUBLIC EBP_USE_OPENMP=1)
else()
  target_link_libraries(elastic_band_planner_core
    VoxelSDF
    RobotKinematics
  )
endif()
target_include_directories(elastic_band_planner_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(elastic_band_planner_core rclcpp std_msgs)

# Python bindings
pybind11_add_module(elastic_band_planner_cpp
  bindings/ElasticBandBinding.cpp
)
target_link_libraries(elastic_band_planner_cpp PRIVATE
  elastic_band_planner_core
  VoxelSDF
  RobotKinematics
)
# Note: ament_target_dependencies not compatible with pybind11 keyword signature
target_include_directories(elastic_band_planner_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Installation
install(TARGETS
  RobotKinematics
  SDFfromPLY
  VoxelSDF
  elastic_band_planner_core
  SDFcomputing
  elastic_band_planner_cpp
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

# Debug options
foreach(tgt RobotKinematics VoxelSDF elastic_band_planner_core elastic_band_planner_cpp)
  target_compile_options(${tgt} PRIVATE -g -O0 -fno-omit-frame-pointer)
endforeach()

ament_package()
