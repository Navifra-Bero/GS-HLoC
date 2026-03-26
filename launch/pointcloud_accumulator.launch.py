"""
pointcloud_accumulator.launch.py
==================================
Usage:
  ros2 launch render_loc pointcloud_accumulator.launch.py

선택 인자:
  trajectory_json:=...   trajectory_poses.json 경로
  images_dir:=...        cam_3/images 디렉터리
  depths_dir:=...        cam_3/depths 디렉터리
  fps:=5.0               재생 속도 (프레임/초)
  stride:=4              depth 샘플링 stride (클수록 빠름, 포인트 적음)
  voxel_size:=0.05       voxel 다운샘플 크기 (0=비활성)
  frustum_depth:=3.0     frustum 시각화 깊이 (m)
  depth_max:=10.0        depth 최대값 (m)
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
    pkg        = get_package_share_directory('render_loc')
    scripts    = os.path.join(pkg, 'scripts')
    rviz_cfg   = os.path.join(pkg, 'config', 'pointcloud_accumulator.rviz')

    base = '/home/park/loc_ws/src/render_loc'

    args = [
        DeclareLaunchArgument('trajectory_json',
            default_value=os.path.join(base, 'output/step_by_step/test_results/cam_3/trajectory_poses.json')),
        DeclareLaunchArgument('images_dir',
            default_value=os.path.join(base, 'test_data/cam_3/images')),
        DeclareLaunchArgument('depths_dir',
            default_value=os.path.join(base, 'test_data/cam_3/depths')),
        DeclareLaunchArgument('fps',           default_value='10.0'),
        DeclareLaunchArgument('stride',        default_value='4'),
        DeclareLaunchArgument('voxel_size',    default_value='0.01'),
        DeclareLaunchArgument('frustum_depth', default_value='3.0'),
        DeclareLaunchArgument('depth_max',     default_value='15.0'),
        # 카메라 내부 파라미터 (cam_3 기준으로 수정)
        DeclareLaunchArgument('fx', default_value='1039.045981'),
        DeclareLaunchArgument('fy', default_value='1041.496942'),
        DeclareLaunchArgument('cx', default_value='937.044077'),
        DeclareLaunchArgument('cy', default_value='560.826738'),
        DeclareLaunchArgument('no_rviz', default_value='false'),
    ]

    accumulator_node = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(scripts, 'pointcloud_accumulator.py'),
            '--ros-args',
            '-p', ['trajectory_json:=', LaunchConfiguration('trajectory_json')],
            '-p', ['images_dir:=',      LaunchConfiguration('images_dir')],
            '-p', ['depths_dir:=',      LaunchConfiguration('depths_dir')],
            '-p', ['fps:=',             LaunchConfiguration('fps')],
            '-p', ['stride:=',          LaunchConfiguration('stride')],
            '-p', ['voxel_size:=',      LaunchConfiguration('voxel_size')],
            '-p', ['frustum_depth:=',   LaunchConfiguration('frustum_depth')],
            '-p', ['depth_max:=',       LaunchConfiguration('depth_max')],
            '-p', ['fx:=',              LaunchConfiguration('fx')],
            '-p', ['fy:=',              LaunchConfiguration('fy')],
            '-p', ['cx:=',              LaunchConfiguration('cx')],
            '-p', ['cy:=',              LaunchConfiguration('cy')],
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

    return LaunchDescription(args + [accumulator_node, rviz_node])
