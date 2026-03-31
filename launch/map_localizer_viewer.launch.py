"""
map_localizer_viewer.launch.py
==============================
Usage:
  ros2 launch render_loc map_localizer_viewer.launch.py

선택 인자:
  aligned_ply:=...       output/step_by_step/aligned_map.ply 경로
  trajectory_json:=...   output/step_by_step/trajectory_poses.json 경로
  images_dir:=...        카메라 이미지 디렉터리
  fps:=5.0               재생 속도 (프레임/초)
  map_voxel:=0.1         PLY 다운샘플 voxel 크기 (m)
  depth_max:=15.0        가시 영역 최대 거리 (m)
  frustum_depth:=3.0     frustum 시각화 깊이 (m)
  frustum_grid_w:=80     frustum 이미지 가로 격자 수
  frustum_grid_h:=50     frustum 이미지 세로 격자 수
  fx/fy/cx/cy:=...       카메라 내부 파라미터
  img_width/img_height:= 카메라 해상도
  no_rviz:=false         RViz2 실행 생략
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
            default_value=os.path.join(out, 'aligned_map_01_2.ply')),   # 0.1m pre-downsampled
        DeclareLaunchArgument('bg_ply',
            default_value=os.path.join(out, 'aligned_map_04_2.ply')),   # 0.4m pre-downsampled
        DeclareLaunchArgument('trajectory_json',
            default_value=os.path.join(out, 'test_results/cam_3/trajectory_poses.json')),
        DeclareLaunchArgument('images_dir',
            default_value=os.path.join(base, 'test_data/cam_3/images')),
        DeclareLaunchArgument('fps',           default_value='5.0'),
        DeclareLaunchArgument('map_voxel',     default_value='0.4'),
        DeclareLaunchArgument('depth_max',     default_value='20.0'),
        DeclareLaunchArgument('frustum_depth', default_value='2.0'),
        DeclareLaunchArgument('frustum_grid_w', default_value='64'),
        DeclareLaunchArgument('frustum_grid_h', default_value='40'),
        # # 카메라 내부 파라미터 (cam_0 기준)
        # DeclareLaunchArgument('fx', default_value='1039.045981'),
        # DeclareLaunchArgument('fy', default_value='1041.496942'),
        # DeclareLaunchArgument('cx', default_value='937.044077'),
        # DeclareLaunchArgument('cy', default_value='560.826738'),
        # DeclareLaunchArgument('img_width', default_value='1920'),
        # DeclareLaunchArgument('img_height', default_value='1200'),
        # # 카메라 내부 파라미터 (cam_1 기준)
        # DeclareLaunchArgument('fx', default_value='1039.045981'),
        # DeclareLaunchArgument('fy', default_value='1041.496942'),
        # DeclareLaunchArgument('cx', default_value='937.044077'),
        # DeclareLaunchArgument('cy', default_value='560.826738'),
        # DeclareLaunchArgument('img_width', default_value='1920'),
        # DeclareLaunchArgument('img_height', default_value='1200'),
        # # 카메라 내부 파라미터 (cam_2 기준)
        # DeclareLaunchArgument('fx', default_value='1039.045981'),
        # DeclareLaunchArgument('fy', default_value='1041.496942'),
        # DeclareLaunchArgument('cx', default_value='937.044077'),
        # DeclareLaunchArgument('cy', default_value='560.826738'),
        # DeclareLaunchArgument('img_width', default_value='1920'),
        # DeclareLaunchArgument('img_height', default_value='1200'),
        # 카메라 내부 파라미터 (cam_3 기준)
        DeclareLaunchArgument('fx', default_value='1039.045981'),
        DeclareLaunchArgument('fy', default_value='1041.496942'),
        DeclareLaunchArgument('cx', default_value='937.044077'),
        DeclareLaunchArgument('cy', default_value='560.826738'),
        DeclareLaunchArgument('img_width', default_value='1920'),
        DeclareLaunchArgument('img_height', default_value='1200'),
        # Femto 카메라
        # DeclareLaunchArgument('fx', default_value='2256.627197'),
        # DeclareLaunchArgument('fy', default_value='2254.400635'),
        # DeclareLaunchArgument('cx', default_value='1891.352783'),
        # DeclareLaunchArgument('cy', default_value='1087.097656'),
        # DeclareLaunchArgument('img_width', default_value='3840'),
        # DeclareLaunchArgument('img_height', default_value='2160'),
        DeclareLaunchArgument('no_rviz',    default_value='false'),
        DeclareLaunchArgument('step6_results_dir',
            default_value=os.path.join(out, 'test_results/cam_3')),
    ]

    viewer_node = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(scripts, 'map_localizer_viewer.py'),
            '--ros-args',
            '-p', ['aligned_ply:=',      LaunchConfiguration('aligned_ply')],
            '-p', ['vis_ply:=',          LaunchConfiguration('vis_ply')],
            '-p', ['bg_ply:=',           LaunchConfiguration('bg_ply')],
            '-p', ['trajectory_json:=',  LaunchConfiguration('trajectory_json')],
            '-p', ['images_dir:=',       LaunchConfiguration('images_dir')],
            '-p', ['fps:=',              LaunchConfiguration('fps')],
            '-p', ['map_voxel:=',        LaunchConfiguration('map_voxel')],
            '-p', ['depth_max:=',        LaunchConfiguration('depth_max')],
            '-p', ['frustum_depth:=',    LaunchConfiguration('frustum_depth')],
            '-p', ['frustum_grid_w:=',   LaunchConfiguration('frustum_grid_w')],
            '-p', ['frustum_grid_h:=',   LaunchConfiguration('frustum_grid_h')],
            '-p', ['fx:=',               LaunchConfiguration('fx')],
            '-p', ['fy:=',               LaunchConfiguration('fy')],
            '-p', ['cx:=',               LaunchConfiguration('cx')],
            '-p', ['cy:=',               LaunchConfiguration('cy')],
            '-p', ['img_width:=',        LaunchConfiguration('img_width')],
            '-p', ['img_height:=',       LaunchConfiguration('img_height')],
            '-p', ['step6_results_dir:=', LaunchConfiguration('step6_results_dir')],
        ],
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('no_rviz')),
    )

    return LaunchDescription(args + [viewer_node, rviz_node])
