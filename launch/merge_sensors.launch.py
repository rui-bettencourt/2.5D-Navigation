from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    target_frame = LaunchConfiguration('target_frame')
    cloud_in1 = LaunchConfiguration('cloud_in1')
    cloud_in2 = LaunchConfiguration('cloud_in2')
    cloud_out = LaunchConfiguration('cloud_out')

    pointcloud_concat = Node(
        package='pointcloud_concatenate',
        executable='pointcloud_concatenate_node',
        name='pc_concat',
        output='screen',
        parameters=[{
            'target_frame': target_frame,
            'clouds': 2,
            'hz': 10,
            'filter_zs': False,
            'filter_zs_height': -0.01,
            'project_negatives_to_zero': False,
            'downsample': False,
            'downsample_after_filtering_z': False,
            'downsample_resolution': 0.02,
        }],
        remappings=[
            ('cloud_in1', cloud_in1),
            ('cloud_in2', cloud_in2),
            ('cloud_out', cloud_out),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('target_frame', default_value='base_footprint'),
        DeclareLaunchArgument('cloud_in1', default_value='/unilidar/points'),
        DeclareLaunchArgument('cloud_in2', default_value='/xtion/depth/points'),
        DeclareLaunchArgument('cloud_out', default_value='/tiago/points'),
        pointcloud_concat,
    ])
