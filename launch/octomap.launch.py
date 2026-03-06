from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    octomap_server = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap_server',
        output='screen',
        parameters=[{
            'resolution': 0.1,
        }],
        remappings=[
            ('cloud_in', '/unitree/cloud'),
        ],
    )

    return LaunchDescription([
        octomap_server,
    ])
