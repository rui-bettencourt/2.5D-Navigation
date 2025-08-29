cmake_minimum_required(VERSION 3.0.2)
project(mesh_nav)

find_package(catkin REQUIRED COMPONENTS
              roscpp
              std_msgs)
              # Find Pinocchio
find_package(pinocchio REQUIRED)
# Tell CMake where to find custom FindOpenVDB.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/usr/local/lib/cmake/OpenVDB")
find_package(OpenVDB REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED)
find_package(pinocchio REQUIRED)


# include_directories(/usr/local/include)
# link_directories(/usr/local/lib)

include_directories(
include
${catkin_INCLUDE_DIRS}
/usr/local/include
)
# # Add CasADi library directory
link_directories(/usr/local/lib)

# Add its include directories
include_directories(/usr/local/include/eigen3)
# include_directories(${PINOCCHIO_INCLUDE_DIRS})
# link_directories(${PINOCCHIO_LIBRARY_DIRS})

# Add include directory
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/TriangleMeshDistance)
# Include mesh based signed distance field
option(LIBIGL_WITH_EIGEN ON)
option(LIBIGL_WITH_PYTHON OFF)
option(LIBIGL_WITH_TETGEN OFF)
option(LIBIGL_WITH_TRIANGLE OFF)
option(LIBIGL_WITH_CGAL OFF)
option(LIBIGL_WITH_COMISO OFF)
# add_subdirectory(external/libigl)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/ligigl/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/tinyply/source)
include_directories(/usr/local/include/openvdb)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES mesh_nav
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

### Create a cpp API to use Pinocchio with Tiago
add_library(RobotKinematics
  src/mesh_nav_ros/RobotKinematics.cpp
)
target_link_libraries(RobotKinematics
  ${catkin_LIBRARIES}
  pinocchio::pinocchio
  # ${PINOCCHIO_LIBRARIES}  # if using Pinocchio
)

add_library(SDFfromPLY
  src/mesh_nav_ros/SDFfromPLY.cpp
)

target_link_libraries(SDFfromPLY PUBLIC
  ${catkin_LIBRARIES}
)

add_library(VoxelSDF
  src/mesh_nav_ros/VoxelSDF.cpp
)

target_link_libraries(VoxelSDF PUBLIC
  ${catkin_LIBRARIES}
)


### Tester for Pinocchio
#add_executable(${PROJECT_NAME}_node src/mesh_nav_ros/mpc/mpc.cpp)
add_executable(RobotKinematicTester
  src/mesh_nav_ros/tests/RobotKinematicTester.cpp
  # Add other cpp files here
)
## Specify libraries to link a library or executable target against
target_link_libraries(RobotKinematicTester
  ${catkin_LIBRARIES}
  RobotKinematics
)

### Code to convert meshes to signed distance field
add_executable(SDFfromPLY_IGL
  src/mesh_nav_ros/SDFfromPLY_IGL.cpp
)
target_link_libraries(SDFfromPLY_IGL ${catkin_LIBRARIES} SDFfromPLY)

# add_executable(SDFfromPLY_OpenVDB
#   src/mesh_nav_ros/SDFfromPLY_OpenVDB.cpp
#   external/tinyply/source/tinyply.cpp
# )
# target_link_libraries(SDFfromPLY_OpenVDB openvdb tbb ${catkin_LIBRARIES})

add_executable(SDFcomputing
  src/mesh_nav_ros/SDFcomputing.cpp
  # Add other cpp files here
)
target_link_libraries(SDFcomputing
  ${catkin_LIBRARIES} SDFfromPLY OpenMP::OpenMP_CXX
)
# add_compile_options(-O3 -march=native)

add_executable(VoxelSDFtester
  src/mesh_nav_ros/tests/VoxelSDFtester.cpp
  # Add other cpp files here
)
target_link_libraries(VoxelSDFtester
  ${catkin_LIBRARIES} VoxelSDF SDFfromPLY
)

# library target for planner core
add_library(elastic_band_planner_core SHARED
    src/mesh_nav_ros/ElasticBandPlanner.cpp
)

if (OpenMP_CXX_FOUND)
  target_link_libraries(elastic_band_planner_core PUBLIC OpenMP::OpenMP_CXX
    VoxelSDF
    RobotKinematics
    ${catkin_LIBRARIES}
  )
  target_compile_definitions(elastic_band_planner_core PUBLIC EBP_USE_OPENMP=1)
else()
  target_link_libraries(elastic_band_planner_core
    VoxelSDF
    RobotKinematics
    ${catkin_LIBRARIES}
  )
endif()

target_include_directories(elastic_band_planner_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)



# python bindings module
# pybind11_add_module(elastic_band_planner_cpp bindings/ElasticBandBinding.cpp)
# target_include_directories(elastic_band_planner_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# target_link_libraries(elastic_band_planner_cpp PRIVATE elastic_band_planner_core)

# Python bindings
# pybind11_add_module(elastic_band_planner_cpp
#   bindings/ElasticBandBinding.cpp
# )

# target_link_libraries(elastic_band_planner_cpp PRIVATE
#   ${catkin_LIBRARIES}
# )

pybind11_add_module(elastic_band_planner_cpp
  bindings/ElasticBandBinding.cpp
)

target_link_libraries(elastic_band_planner_cpp PRIVATE
  elastic_band_planner_core       # <--- link your core planner library
  VoxelSDF
  RobotKinematics
  ${catkin_LIBRARIES}
)

install(TARGETS elastic_band_planner_cpp
  LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)

add_executable(test_tls_kinematics
  src/mesh_nav_ros/tests/test_tls_kinematics.cpp
)
target_link_libraries(test_tls_kinematics
  RobotKinematics           # your RK lib target
  ${catkin_LIBRARIES}
)
add_dependencies(test_tls_kinematics ${catkin_EXPORTED_TARGETS})

# put this after the targets are created
# foreach(tgt RobotKinematics VoxelSDF elastic_band_planner_core elastic_band_planner_cpp)
#   target_compile_options(${tgt} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -O1)
#   target_link_options(${tgt}    PRIVATE -fsanitize=address,undefined)
# endforeach()
# In your CMakeLists.txt, after targets are created:
foreach(tgt RobotKinematics VoxelSDF elastic_band_planner_core elastic_band_planner_cpp)
  target_compile_options(${tgt} PRIVATE -g -O0 -fno-omit-frame-pointer)
endforeach()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
foreach(dir src)
    install(DIRECTORY ${dir}/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir}
        USE_SOURCE_PERMISSIONS)
endforeach(dir)