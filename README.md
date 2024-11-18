IKpy can't load xacros, you first need to convert your robot xacro to urdf like this:
```
rosrun xacro xacro /path/to/your/robot_description.urdf.xacro -o /path/to/expanded_robot_description.urdf

remove unnecessary links and joints```

Depends on git@github.com:ros/kdl_parser.git and pykdl.