cmake_minimum_required(VERSION 3.14)
project(render_loc VERSION 0.1.0)

# ROS2 ament's CMake helpers execute Python modules such as catkin_pkg.
# When this repo is built inside the render_loc conda env, CMake may pick the
# conda Python, which usually does not contain those ROS build modules. Prefer
# the system ROS Python for configure-time ament scripts.
if(EXISTS "/usr/bin/python3")
  set(Python3_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH
      "Python interpreter used by ament CMake configure scripts" FORCE)
endif()

option(BUILD_ROS2 "Install ROS2 Python nodes and assets" ON)

# =============================================================================
# ROS2 (ament) — always required when BUILD_ROS2 is ON
# =============================================================================
if(BUILD_ROS2)
  find_package(ament_cmake REQUIRED)
endif()

# =============================================================================
# ROS2 install targets
# =============================================================================
if(BUILD_ROS2)
  # Python scripts, launch files, config, and web assets.
  install(DIRECTORY launch/  DESTINATION share/${PROJECT_NAME}/launch)
  install(DIRECTORY config/  DESTINATION share/${PROJECT_NAME}/config)
  install(DIRECTORY scripts/ DESTINATION share/${PROJECT_NAME}/scripts)
  # 웹 뷰어 정적 파일 (index.html, viewer.js, lib/*). splat 은 런타임 생성.
  install(DIRECTORY web/ DESTINATION share/${PROJECT_NAME}/web
    PATTERN "*.splat" EXCLUDE)

  # Python ROS2 노드는 ros2 run / launch 가 lib/<pkg> 에서 찾는다.
  install(PROGRAMS
    scripts/ros/ros_localizer_node.py
    scripts/ros/web_pose_bridge.py
    scripts/ros/ply_to_splat.py
    DESTINATION lib/${PROJECT_NAME})

  ament_package()
endif()
