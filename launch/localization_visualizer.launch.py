"""
localization_visualizer.launch.py
==================================
PLY 맵 로컬라이제이션 시각화 + RViz2 실행

사용법:
  ros2 launch render_loc localization_visualizer.launch.py \
      ply_path:=/path/to/map.ply \
      db_path:=/path/to/step4_database.pkl

선택 인자:
  config_path:=...           YAML 설정 파일 (기본: render_loc.yaml)
  image_topic:=/cam_3/image  구독할 이미지 토픽
  frustum_depth:=5.0         카메라 프러스텀 깊이 (m)
  retrieval_every_n:=1       N 프레임마다 리트리벌 실행 (무거우면 높임)
  no_rviz:=false             RViz2 실행 안 하려면 true
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory("render_loc")
    scripts_dir = os.path.join(pkg, "scripts")
    rviz_cfg    = os.path.join(pkg, "config", "localization_visualizer.rviz")
    default_cfg = os.path.join(pkg, "config", "render_loc.yaml")

    # ── Launch Arguments ─────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument("ply_path",    default_value="/home/park/loc_ws/src/render_loc/output/step_by_step/aligned_map.ply",
                              description="PLY 맵 파일 경로 (필수)"),
        DeclareLaunchArgument("db_path",     default_value="/home/park/loc_ws/src/render_loc/output/step_by_step/step4_database.pkl",
                              description="step4_database.pkl 경로 (필수)"),
        DeclareLaunchArgument("config_path", default_value=default_cfg,
                              description="render_loc YAML 설정 파일"),
        DeclareLaunchArgument("image_topic", default_value="/cam_3/image",
                              description="구독할 이미지 토픽"),
        DeclareLaunchArgument("frustum_depth",      default_value="5.0",
                              description="카메라 프러스텀 깊이 (m)"),
        DeclareLaunchArgument("retrieval_every_n",  default_value="1",
                              description="N 프레임마다 리트리벌 (무거우면 3~5)"),
        DeclareLaunchArgument("no_rviz",     default_value="false",
                              description="RViz2 실행 생략 여부"),
    ]

    # ── Localization Visualizer Node (Python) ────────────────────────────
    visualizer_node = ExecuteProcess(
        cmd=[
            "python3",
            os.path.join(scripts_dir, "localization_visualizer.py"),
            "--ros-args",
            "-p", ["ply_path:=",    LaunchConfiguration("ply_path")],
            "-p", ["db_path:=",     LaunchConfiguration("db_path")],
            "-p", ["config_path:=", LaunchConfiguration("config_path")],
            "-p", ["image_topic:=", LaunchConfiguration("image_topic")],
            "-p", ["frustum_depth:=", LaunchConfiguration("frustum_depth")],
            "-p", ["retrieval_every_n:=", LaunchConfiguration("retrieval_every_n")],
        ],
        output="screen",
    )

    # ── RViz2 ────────────────────────────────────────────────────────────
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_cfg],
        output="screen",
        condition=UnlessCondition(LaunchConfiguration("no_rviz")),
    )

    return LaunchDescription(args + [visualizer_node, rviz_node])
