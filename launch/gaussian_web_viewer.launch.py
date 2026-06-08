"""Standalone Gaussian splat web viewer for ROS2 pose topics.

구성:
  - web_pose_bridge: PoseStamped/Odometry topic → SSE + WebGL viewer
  - optional splat generation: aligned_map.ply → gaussian_map.splat

사용:
  ros2 launch render_loc gaussian_web_viewer.launch.py \
    repo_root:=/home/park/loc_ws/src/render_loc \
    aligned_ply:=/home/park/loc_ws/src/render_loc/output/gs_sdf_omni_2/aligned_map.ply \
    splat_path:=/home/park/loc_ws/src/render_loc/output/gs_sdf_omni_2/gaussian_map.splat \
    pose_topic:=/vps/current_pose

브라우저:
  http://localhost:8080
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("render_loc")
    repo_root_default = "/home/park/loc_ws/src/render_loc"
    default_web = os.path.join(pkg_share, "web")
    # In a source workspace, prefer the source web dir because lib/*.js is there
    # during development and splat files can be large runtime artifacts.
    if os.path.isdir(os.path.join(repo_root_default, "web")):
        default_web = os.path.join(repo_root_default, "web")

    repo_root = LaunchConfiguration("repo_root")
    web_dir = LaunchConfiguration("web_dir")
    aligned_ply = LaunchConfiguration("aligned_ply")
    splat_path = LaunchConfiguration("splat_path")
    pose_topic = LaunchConfiguration("pose_topic")
    pose_topic_type = LaunchConfiguration("pose_topic_type")
    gen_splat = LaunchConfiguration("gen_splat")
    force_splat = LaunchConfiguration("force_splat")
    port = LaunchConfiguration("port")
    gen_topdown = LaunchConfiguration("gen_topdown")
    topdown_z_min = LaunchConfiguration("topdown_z_min")
    topdown_z_max = LaunchConfiguration("topdown_z_max")
    image_topic = LaunchConfiguration("image_topic")
    image_topic_type = LaunchConfiguration("image_topic_type")

    return LaunchDescription([
        DeclareLaunchArgument("repo_root", default_value=repo_root_default),
        DeclareLaunchArgument("web_dir", default_value=default_web),
        DeclareLaunchArgument(
            "aligned_ply",
            default_value=[repo_root, "/output/gs_sdf_omni/aligned_map.ply"]),
        DeclareLaunchArgument(
            "splat_path",
            default_value=[repo_root, "/output/gs_sdf_omni/gaussian_map.splat"]),
        DeclareLaunchArgument("pose_topic", default_value="/vps/current_pose"),
        DeclareLaunchArgument("pose_topic_type", default_value="auto",
                              description="auto | pose | odom"),
        DeclareLaunchArgument("gen_splat", default_value="true"),
        DeclareLaunchArgument("force_splat", default_value="false"),
        DeclareLaunchArgument("port", default_value="8080"),
        DeclareLaunchArgument("gen_topdown", default_value="true",
                              description="top-down 뷰용 바닥 슬라이스 splat 생성"),
        DeclareLaunchArgument("topdown_z_min", default_value="-1.0",
                              description="top-down 슬라이스 하한 Z (바닥=0)"),
        DeclareLaunchArgument("topdown_z_max", default_value="3.0",
                              description="top-down 슬라이스 상한 Z (바닥+N m)"),
        DeclareLaunchArgument("image_topic",
                              default_value="/cam0/image_raw/compressed",
                              description="카메라 패널용 image 토픽 (빈 값이면 비활성)"),
        DeclareLaunchArgument("image_topic_type", default_value="auto",
                              description="auto | raw | compressed"),

        ExecuteProcess(
            condition=IfCondition(gen_splat),
            cmd=["bash", "-lc", [
                "if [ '", force_splat, "' = 'true' ] || [ ! -f '", splat_path, "' ]; then ",
                "mkdir -p \"$(dirname '", splat_path, "')\" && ",
                "python3 '", repo_root, "/scripts/ros/ply_to_splat.py' '",
                aligned_ply, "' '", splat_path, "'; ",
                "else echo 'splat exists: ", splat_path, "'; fi",
            ]],
            output="screen",
        ),

        # top-down 뷰용: 바닥(z=0)부터 z_max 까지 슬라이스한 splat.
        # 파일명은 splat_path 의 .splat → _topdown.splat (bridge 도 동일 규칙으로 서빙).
        ExecuteProcess(
            condition=IfCondition(gen_topdown),
            cmd=["bash", "-lc", [
                "td='", splat_path, "'; td=\"${td%.splat}_topdown.splat\"; ",
                "if [ '", force_splat, "' = 'true' ] || [ ! -f \"$td\" ]; then ",
                "mkdir -p \"$(dirname \"$td\")\" && ",
                "python3 '", repo_root, "/scripts/ros/ply_to_splat.py' '",
                aligned_ply, "' \"$td\" --z-min '", topdown_z_min,
                "' --z-max '", topdown_z_max, "'; ",
                "else echo \"topdown splat exists: $td\"; fi",
            ]],
            output="screen",
        ),

        Node(
            package="render_loc",
            executable="web_pose_bridge.py",
            name="web_pose_bridge",
            output="screen",
            parameters=[{
                "web_dir": web_dir,
                "splat_path": splat_path,
                "pose_topic": pose_topic,
                "pose_topic_type": pose_topic_type,
                "image_topic": image_topic,
                "image_topic_type": image_topic_type,
                "port": port,
            }],
        ),
    ])
