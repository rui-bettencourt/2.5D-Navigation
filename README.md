# mesh_nav

`mesh_nav` is a ROS package for navigation on 3D mesh environments. It provides tools for path planning and robot kinematics using mesh representations, enabling robots to traverse complex terrains.

## Features

- Mesh-based navigation and path planning
- Integration with Pinocchio for forward kinematics
- Support for URDF robot descriptions

## Installation

1. Clone the repository into your ROS workspace:
    ```bash
    cd ~/(your_workspace))/src
    git clone --recursive https://github.com/rui-bettencourt/2.5D-Navigation.git
    ```

2. Install dependencies:
    You need:
    ROS Noetic
    open3d=0.18.0
    python-fcl=0.7.0
    octomap-python=1.8.0
    numpy

    pinocchio (https://github.com/stack-of-tasks/pinocchio?tab=readme-ov-file#installation) COMPILE USING RELEASE TO BE FAST!!!: using ros: catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
    ```bash
    sudo apt-get install ros-$ROS_DISTRO-kdl-parser
    pip install ikpy pykdl
    ```
    If you encounter an error while installing pykdl with pip, try installing it using apt instead:
    ```bash
    sudo apt install python3-pykdl
    ```
    (Make sure you have [kdl_parser](https://github.com/ros/kdl_parser) and `pykdl` installed.)
    

3. Build the workspace:
    ```bash
    cd ~/(your_workspace)
    catkin build
    source devel/setup.bash
    ```

## Usage



## License

This project is licensed under the MIT License.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Tips

If your robot only has a xacro file, you need to convert your robot's xacro file to URDF:
    ```bash
    rosrun xacro xacro /path/to/your/robot_description.urdf.xacro -o /path/to/expanded_robot_description.urdf
    ```
    Remove unnecessary links and joints from the URDF. Then use this urdf so pinocchio can perform forward kinematics.
