from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    fake_localization = Node(
        package='fake_localization',
        executable='fake_localization',
        name='fake_localization',
        output='screen',
        parameters=[{
            'odom_frame_id': 'odom',
            'base_frame_id': 'base_link',
            'global_frame_id': 'map',
        }],
    )

    map_to_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_odom_broadcaster',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    )

    return LaunchDescription([
        fake_localization,
        map_to_odom_tf,
    ])
