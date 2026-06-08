cmake_minimum_required(VERSION 3.14)
project(render_loc VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ROS2 ament's CMake helpers execute Python modules such as catkin_pkg.
# When this repo is built inside the render_loc conda env, CMake may pick the
# conda Python, which usually does not contain those ROS build modules. Prefer
# the system ROS Python for configure-time ament scripts.
if(EXISTS "/usr/bin/python3")
  set(Python3_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH
      "Python interpreter used by ament CMake configure scripts" FORCE)
endif()

option(BUILD_ROS2       "Build ROS2 nodes"           ON)
option(USE_CUDA         "Enable CUDA acceleration"    ON)
option(BUILD_CPP_NODES  "Build C++ core + nodes"      OFF)

# =============================================================================
# ROS2 (ament) — always required when BUILD_ROS2 is ON
# =============================================================================
if(BUILD_ROS2)
  find_package(ament_cmake REQUIRED)
  if(BUILD_CPP_NODES)
    # rclcpp pulls these in through the ROS2 CMake config chain on Humble.
    # In lightweight Python-only use, missing system dev packages should not
    # block installing launch/config/web assets.
    find_package(spdlog QUIET)
    find_library(LIBYAML_LIBRARY yaml)
    if(NOT spdlog_FOUND OR NOT LIBYAML_LIBRARY)
      message(WARNING
        "render_loc: BUILD_CPP_NODES=ON but ROS2 C++ support deps are missing "
        "(spdlog=${spdlog_FOUND}, yaml=${LIBYAML_LIBRARY}) — C++ nodes will NOT be built.\n"
        "Install libspdlog-dev/libyaml-dev or pass -DBUILD_CPP_NODES=ON after "
        "installing them if you need online_localizer_node.")
      set(BUILD_CPP_NODES OFF CACHE BOOL "Build C++ core + nodes" FORCE)
    else()
      find_package(rclcpp REQUIRED)
      find_package(sensor_msgs REQUIRED)
      find_package(geometry_msgs REQUIRED)
      find_package(nav_msgs REQUIRED)
      find_package(cv_bridge REQUIRED)
      find_package(message_filters REQUIRED)
      find_package(tf2_ros REQUIRED)
    endif()
  endif()
endif()

# =============================================================================
# Optional C++ dependencies — skip C++ targets if any are missing
# =============================================================================
if(BUILD_CPP_NODES)
  find_package(Eigen3 3.3 QUIET)
  find_package(OpenCV 4.5 QUIET COMPONENTS core imgproc imgcodecs calib3d)
  find_package(yaml-cpp QUIET)
  find_package(Torch QUIET)
  # PCL: VTK cmake파일의 MPI 버그를 피하기 위해 VTK 탐색 비활성화
  set(CMAKE_DISABLE_FIND_PACKAGE_VTK ON)
  find_package(PCL 1.12 QUIET COMPONENTS common io filters kdtree registration)

  if(NOT Eigen3_FOUND OR NOT OpenCV_FOUND OR NOT yaml-cpp_FOUND OR NOT PCL_FOUND)
    message(WARNING
      "render_loc: Missing C++ dependencies (Eigen3=${Eigen3_FOUND} "
      "OpenCV=${OpenCV_FOUND} yaml-cpp=${yaml-cpp_FOUND} PCL=${PCL_FOUND} "
      "Torch=${Torch_FOUND}) — C++ nodes will NOT be built.\n"
      "Python scripts (scripts/) are unaffected.")
    set(BUILD_CPP_NODES OFF)
  endif()
endif()

# =============================================================================
# Core Library  (only when all deps are present)
# =============================================================================
if(BUILD_CPP_NODES)
  add_library(renderloc_core SHARED
    src/core/types.cpp
    src/core/config.cpp
    src/offline/depth_backprojector.cpp
    src/offline/feature_extractor.cpp
    src/online/feature_matcher.cpp
    src/online/pnp_solver.cpp
    src/online/icp_refiner.cpp
    src/online/localizer.cpp
  )

  target_include_directories(renderloc_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
  )

  target_link_libraries(renderloc_core PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
    $<$<BOOL:${Torch_FOUND}>:${TORCH_LIBRARIES}>
    ${PCL_LIBRARIES}
    yaml-cpp
  )
endif()

# =============================================================================
# ROS2 install targets
# =============================================================================
if(BUILD_ROS2)
  if(BUILD_CPP_NODES)
    add_executable(online_localizer_node
      src/online/online_localizer_node.cpp
    )
    target_link_libraries(online_localizer_node renderloc_core)
    ament_target_dependencies(online_localizer_node
      rclcpp sensor_msgs geometry_msgs nav_msgs
      cv_bridge message_filters tf2_ros
    )
    install(TARGETS renderloc_core online_localizer_node
      DESTINATION lib/${PROJECT_NAME})
    install(DIRECTORY include/ DESTINATION include)
  endif()

  # Python scripts, launch files, config, rviz — always installed
  install(DIRECTORY launch/  DESTINATION share/${PROJECT_NAME}/launch)
  install(DIRECTORY config/  DESTINATION share/${PROJECT_NAME}/config)
  install(DIRECTORY rviz/    DESTINATION share/${PROJECT_NAME}/rviz)
  install(DIRECTORY scripts/ DESTINATION share/${PROJECT_NAME}/scripts)
  # 웹 뷰어 정적 파일 (index.html, viewer.js, lib/*). splat 은 런타임 생성.
  install(DIRECTORY web/ DESTINATION share/${PROJECT_NAME}/web
    PATTERN "*.splat" EXCLUDE)

  # Python ROS2 노드는 ros2 run / launch 가 lib/<pkg> 에서 찾는다.
  install(PROGRAMS
    scripts/ros/ros_localizer_node.py
    scripts/ros/gaussian_ply_publisher.py
    scripts/ros/web_pose_bridge.py
    scripts/ros/ply_to_splat.py
    DESTINATION lib/${PROJECT_NAME})

  ament_package()
endif()
