"""실시간 ROS2 Gaussian-Map Localization 런치.

구성:
  - ros_localizer            : 이미지 토픽 → step5~7 → pose(map frame)
  - gaussian_ply_publisher   : 정렬 가우시안 PLY → PointCloud2(map)
  - (use_rviz)  rviz2        : rviz/online_localizer.rviz
  - (use_optical_tf) static  : base_optical → base_link (OpenCV optical → REP-103 body)

bag 재생은 별도 터미널에서:
  ros2 bag play /home/park/Downloads/bero_test1/bero_test1_bag --rate 0.2

실행 전:
  conda activate render_loc && source /opt/ros/humble/setup.bash &&
  source install/setup.bash
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("render_loc")
    default_params = os.path.join(pkg_share, "config", "ros_localizer.yaml")
    default_rviz = os.path.join(pkg_share, "rviz", "online_localizer.rviz")

    params_file = LaunchConfiguration("params_file")
    repo_root = LaunchConfiguration("repo_root")
    use_rviz = LaunchConfiguration("use_rviz")
    use_optical_tf = LaunchConfiguration("use_optical_tf")
    rviz_config = LaunchConfiguration("rviz_config")

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params,
            description="ros_localizer / gaussian_ply_publisher 파라미터 yaml"),
        DeclareLaunchArgument("repo_root", default_value="",
            description="render_loc 소스 루트(상대경로 기준). 비우면 자동탐색"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("use_optical_tf", default_value="true",
            description="base_optical→base_link(REP-103) static TF publish"),
        DeclareLaunchArgument("rviz_config", default_value=default_rviz),

        Node(
            package="render_loc",
            executable="ros_localizer_node.py",
            name="ros_localizer",
            output="screen",
            parameters=[params_file, {"repo_root": repo_root}],
        ),
        Node(
            package="render_loc",
            executable="gaussian_ply_publisher.py",
            name="gaussian_ply_publisher",
            output="screen",
            parameters=[params_file],
        ),

        # OpenCV optical(x右 y下 z前) → REP-103 base_link(x前 y左 z上)
        # parent=base_optical, child=base_link 쿼터니언 (xyzw)
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="optical_to_base",
            condition=IfCondition(use_optical_tf),
            arguments=[
                "--frame-id", "base_optical",
                "--child-frame-id", "base_link",
                "--x", "0", "--y", "0", "--z", "0",
                "--qx", "0.5", "--qy", "-0.5", "--qz", "0.5", "--qw", "0.5",
            ],
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            condition=IfCondition(use_rviz),
            arguments=["-d", rviz_config],
            output="screen",
        ),
    ])
