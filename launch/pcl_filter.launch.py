from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    passthrough = Node(
        package='pcl_ros',
        executable='filter_node',
        name='passthrough',
        output='screen',
        parameters=[{
            'filter_field_name': 'x',
            'filter_limit_min': 0.1,
            'filter_limit_max': 30.0,
            'filter_limit_negative': False,
        }],
        remappings=[
            ('input', '/unitree/cloud'),
            ('output', '/unitree/cloud/filtered'),
        ],
    )

    return LaunchDescription([
        passthrough,
    ])
