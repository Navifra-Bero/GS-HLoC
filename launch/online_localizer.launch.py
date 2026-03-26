from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory('render_loc')

    return LaunchDescription([
        DeclareLaunchArgument('config_file',
            default_value=os.path.join(pkg, 'config', 'render_loc.yaml')),
        DeclareLaunchArgument('database_path',
            default_value='data/image_db.bin'),
        DeclareLaunchArgument('rgb_topic',
            default_value='/camera/color/image_raw'),
        DeclareLaunchArgument('depth_topic',
            default_value='/camera/depth/image_raw'),
        DeclareLaunchArgument('use_depth', default_value='false'),

        Node(
            package='render_loc',
            executable='online_localizer_node',
            name='renderloc',
            output='screen',
            parameters=[{
                'config_file': LaunchConfiguration('config_file'),
                'database_path': LaunchConfiguration('database_path'),
                'rgb_topic': LaunchConfiguration('rgb_topic'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'use_depth': LaunchConfiguration('use_depth'),
                'publish_tf': True,
                'map_frame': 'map',
                'camera_frame': 'camera_link',
                'rate_hz': 5.0,
            }],
        ),
    ])
