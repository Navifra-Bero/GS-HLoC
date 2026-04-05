"""
kidnap_localizer_viewer.launch.py
==================================
Usage:
  ros2 launch render_loc kidnap_localizer_viewer.launch.py
  ros2 launch render_loc kidnap_localizer_viewer.launch.py \\
      test_results_dir:=output/MegaLoc/test_results/cam_3

선택 인자:
  aligned_ply:=...         전체 맵 PLY
  vis_ply:=...             0.1m 다운샘플 PLY (가시 영역용)
  bg_ply:=...              0.4m 다운샘플 PLY (배경 맵용)
  test_results_dir:=...    test_results/cam_X 폴더
  map_voxel:=0.4           bg_ply 없을 때 다운샘플 크기
  depth_max:=20.0
  frustum_depth:=2.0
  fx/fy/cx/cy/img_width/img_height  카메라 내부 파라미터
  no_rviz:=false           RViz2 실행 생략
  no_panel:=false          PyQt5 패널 실행 생략
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg      = get_package_share_directory('render_loc')
    scripts  = os.path.join(pkg, 'scripts')
    rviz_cfg = os.path.join(pkg, 'config', 'map_localizer_viewer.rviz')

    base = '/home/park/loc_ws/src/render_loc'
    out  = os.path.join(base, 'output/step_by_step')

    args = [
        DeclareLaunchArgument('aligned_ply',
            default_value=os.path.join(out, 'aligned_map.ply')),
        DeclareLaunchArgument('vis_ply',
            default_value=os.path.join(out, 'aligned_map_01.ply')),
        DeclareLaunchArgument('bg_ply',
            default_value=os.path.join(out, 'aligned_map_01.ply')),
        DeclareLaunchArgument('top_down_ply',
            default_value=os.path.join(out, 'aligned_map_01_2.ply'),
            description='top-down 오버뷰용 PLY → /top_down_cloud (latching)'),
        DeclareLaunchArgument('test_results_dir',
            default_value=os.path.join(out, 'test_results/cam_3')),
        DeclareLaunchArgument('map_voxel',       default_value='0.4'),
        DeclareLaunchArgument('depth_max',        default_value='20.0'),
        DeclareLaunchArgument('frustum_depth',    default_value='2.0'),
        DeclareLaunchArgument('frustum_grid_w',   default_value='64'),
        DeclareLaunchArgument('frustum_grid_h',   default_value='40'),
        DeclareLaunchArgument('match_line_alpha',  default_value='0.2'),
        # 카메라 내부 파라미터 (cam_3 기준)
        DeclareLaunchArgument('fx',           default_value='1039.045981'),
        DeclareLaunchArgument('fy',           default_value='1041.496942'),
        DeclareLaunchArgument('cx',           default_value='937.044077'),
        DeclareLaunchArgument('cy',           default_value='560.826738'),
        DeclareLaunchArgument('img_width',    default_value='1920'),
        DeclareLaunchArgument('img_height',   default_value='1200'),
        DeclareLaunchArgument('no_rviz',      default_value='false'),
        DeclareLaunchArgument('no_panel',     default_value='false'),
    ]

    # ── kidnap_localizer_viewer 노드 ──────────────────────────────────
    viewer_node = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(scripts, 'kidnap_localizer_viewer.py'),
            '--ros-args',
            '-p', ['aligned_ply:=',       LaunchConfiguration('aligned_ply')],
            '-p', ['vis_ply:=',           LaunchConfiguration('vis_ply')],
            '-p', ['bg_ply:=',            LaunchConfiguration('bg_ply')],
            '-p', ['top_down_ply:=',      LaunchConfiguration('top_down_ply')],
            '-p', ['test_results_dir:=',  LaunchConfiguration('test_results_dir')],
            '-p', ['map_voxel:=',         LaunchConfiguration('map_voxel')],
            '-p', ['depth_max:=',         LaunchConfiguration('depth_max')],
            '-p', ['frustum_depth:=',     LaunchConfiguration('frustum_depth')],
            '-p', ['frustum_grid_w:=',    LaunchConfiguration('frustum_grid_w')],
            '-p', ['frustum_grid_h:=',    LaunchConfiguration('frustum_grid_h')],
            '-p', ['match_line_alpha:=',  LaunchConfiguration('match_line_alpha')],
            '-p', ['fx:=',                LaunchConfiguration('fx')],
            '-p', ['fy:=',                LaunchConfiguration('fy')],
            '-p', ['cx:=',                LaunchConfiguration('cx')],
            '-p', ['cy:=',                LaunchConfiguration('cy')],
            '-p', ['img_width:=',         LaunchConfiguration('img_width')],
            '-p', ['img_height:=',        LaunchConfiguration('img_height')],
        ],
        output='screen',
    )

    # ── RViz2 ─────────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('no_rviz')),
    )

    # ── PyQt5 인덱스 입력 패널 ────────────────────────────────────────
    panel_node = ExecuteProcess(
        cmd=['python3', os.path.join(scripts, 'kidnap_panel.py')],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('no_panel')),
    )

    return LaunchDescription(args + [viewer_node, rviz_node, panel_node])
