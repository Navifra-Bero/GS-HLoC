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
  ros2 bag play /home/park/Downloads/bero_test1/ch/test/test.db3
  # 브라우저: http://localhost:8081
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
    topdown_map_size = LaunchConfiguration("topdown_map_size")
    topdown_z_min = LaunchConfiguration("topdown_z_min")
    topdown_z_max = LaunchConfiguration("topdown_z_max")
    trajectory_json = LaunchConfiguration("trajectory_json")
    test_bag_path = LaunchConfiguration("test_bag_path")
    image_topic = LaunchConfiguration("image_topic")
    image_topic_type = LaunchConfiguration("image_topic_type")
    camera_stream_enabled = LaunchConfiguration("camera_stream_enabled")
    gen_splat = LaunchConfiguration("gen_splat")
    force_splat = LaunchConfiguration("force_splat")
    port = LaunchConfiguration("port")

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("repo_root", default_value=repo_root_default),
        DeclareLaunchArgument("web_dir",
            default_value=[repo_root, "/web"]),
        DeclareLaunchArgument("aligned_ply",
            default_value=[repo_root, "/output/gs_sdf_omni_2/aligned_map.ply"]),
        DeclareLaunchArgument("splat_path",
            default_value=[repo_root, "/output/gs_sdf_omni_2/gaussian_map.splat"]),
        DeclareLaunchArgument("topdown_map_size", default_value="1024"),
        DeclareLaunchArgument("topdown_z_min", default_value="-1.0"),
        DeclareLaunchArgument("topdown_z_max", default_value="3.0"),
        DeclareLaunchArgument("trajectory_json",
            default_value="/tmp/render_loc_no_trajectory.json",
            description="실시간 pose 모드에서는 존재하지 않는 경로로 두면 trajectory 자동탐색이 꺼짐"),
        DeclareLaunchArgument("test_bag_path",
            default_value="/home/park/Downloads/bero_test1/ch/test/test.db3",
            description=", 키로 재생할 테스트 rosbag2 디렉토리 또는 .db3 파일"),
        DeclareLaunchArgument("image_topic",
            default_value="/cam_0/image_raw/compressed"),
        DeclareLaunchArgument("image_topic_type", default_value="compressed"),
        DeclareLaunchArgument("camera_stream_enabled", default_value="true",
            description="false면 web_pose_bridge가 image topic을 구독하지 않음"),
        DeclareLaunchArgument("gen_splat", default_value="true",
            description="splat_path 없으면 생성"),
        DeclareLaunchArgument("force_splat", default_value="false",
            description="true면 기존 splat_path가 있어도 재생성"),
        DeclareLaunchArgument("port", default_value="8081"),

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
                "aligned_ply": aligned_ply,
                "splat_path": splat_path,
                "topdown_map_size": topdown_map_size,
                "topdown_z_min": topdown_z_min,
                "topdown_z_max": topdown_z_max,
                "trajectory_json": trajectory_json,
                "test_bag_path": test_bag_path,
                "image_topic": image_topic,
                "image_topic_type": image_topic_type,
                "camera_stream_enabled": camera_stream_enabled,
                "port": port,
            }],
        ),
    ])
