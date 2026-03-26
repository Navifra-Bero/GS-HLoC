cmake_minimum_required(VERSION 3.14)
project(render_loc VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_ROS2       "Build ROS2 nodes"           ON)
option(USE_CUDA         "Enable CUDA acceleration"    ON)
option(BUILD_CPP_NODES  "Build C++ core + nodes"      ON)

# =============================================================================
# ROS2 (ament) — always required when BUILD_ROS2 is ON
# =============================================================================
if(BUILD_ROS2)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(cv_bridge REQUIRED)
  find_package(message_filters REQUIRED)
  find_package(tf2_ros REQUIRED)
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

  # Python scripts, launch files, config — always installed
  install(DIRECTORY launch/  DESTINATION share/${PROJECT_NAME}/launch)
  install(DIRECTORY config/  DESTINATION share/${PROJECT_NAME}/config)
  install(DIRECTORY scripts/ DESTINATION share/${PROJECT_NAME}/scripts)

  ament_package()
endif()
