"""실시간 localization → 독립 웹 가우시안 뷰어 (Foxglove 불필요).

구성:
  - ros_localizer     : 이미지 토픽 → step5~7 → /vps/current_pose (PoseStamped)
  - web_pose_bridge   : /vps/current_pose → JSON(SSE) + 웹페이지/splat 서빙 (http://localhost:8080)

    splat 파일(splat_path)이 없으면 launch 시 자동 생성한다
    (aligned_map.ply → ply_to_splat.py).

실행:
  conda activate render_loc && source /opt/ros/humble/setup.bash && source install/setup.bash
  ros2 launch render_loc web_localizer.launch.py
  # 다른 터미널에서 이미지 재생
  ros2 bag play /home/park/Downloads/bero_test1/bero_test1_bag --rate 0.15
  # 브라우저: http://localhost:8080
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("render_loc")
    default_params = os.path.join(pkg_share, "config", "ros_localizer.yaml")
    # 소스 트리 web/ (splat 31MB 는 소스 web/ 에 둔다)
    repo_root_default = "/home/park/loc_ws/src/render_loc"

    params_file = LaunchConfiguration("params_file")
    repo_root = LaunchConfiguration("repo_root")
    web_dir = LaunchConfiguration("web_dir")
    aligned_ply = LaunchConfiguration("aligned_ply")
    splat_path = LaunchConfiguration("splat_path")
    gen_splat = LaunchConfiguration("gen_splat")
    force_splat = LaunchConfiguration("force_splat")
    port = LaunchConfiguration("port")

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("repo_root", default_value=repo_root_default),
        DeclareLaunchArgument("web_dir",
            default_value=[repo_root, "/web"]),
        DeclareLaunchArgument("aligned_ply",
            default_value=[repo_root, "/output/gs_sdf_omni/aligned_map.ply"]),
        DeclareLaunchArgument("splat_path",
            default_value=[repo_root, "/output/gs_sdf_omni/gaussian_map.splat"]),
        DeclareLaunchArgument("gen_splat", default_value="true",
            description="splat_path 없으면 생성"),
        DeclareLaunchArgument("force_splat", default_value="false",
            description="true면 기존 splat_path가 있어도 재생성"),
        DeclareLaunchArgument("port", default_value="8080"),

        # splat 생성 (이미 있으면 건너뜀)
        ExecuteProcess(
            condition=IfCondition(gen_splat),
            cmd=["bash", "-c", [
                "if [ '", force_splat, "' = 'true' ] || [ ! -f '", splat_path, "' ]; then ",
                "mkdir -p \"$(dirname '", splat_path, "')\" && ",
                "python3 ", repo_root, "/scripts/ros/ply_to_splat.py ",
                aligned_ply, " ", splat_path, "; ",
                "else echo 'splat exists: ", splat_path, "'; fi",
            ]],
            output="screen",
        ),

        Node(
            package="render_loc",
            executable="ros_localizer_node.py",
            name="ros_localizer",
            output="screen",
            parameters=[params_file, {"repo_root": repo_root}],
        ),
        Node(
            package="render_loc",
            executable="web_pose_bridge.py",
            name="web_pose_bridge",
            output="screen",
            parameters=[params_file, {
                "web_dir": web_dir,
                "splat_path": splat_path,
                "port": port,
            }],
        ),
    ])
