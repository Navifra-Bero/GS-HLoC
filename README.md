# RenderLoc

**Single-image 6DoF visual localization in LiDAR point cloud maps**

Render synthetic images from your PLY map offline, then localize any query image in real-time using feature matching + PnP.

## How it works

```
OFFLINE (once, Python)                    ONLINE (per frame, C++/ROS2)
─────────────────────                     ──────────────────────────────
PLY Map                                   Query RGB image
  ↓                                         ↓
Viewpoint Sampling (grid + yaw)           NetVLAD → retrieve top-K DB images
  ↓                                         ↓
Open3D Render → RGB + Depth               SuperGlue → match query ↔ DB
  ↓                                         ↓
SuperPoint → keypoints + descriptors      2D-3D correspondences
  ↓                                         ↓
NetVLAD → global descriptors              PnP + RANSAC → 6DoF pose
  ↓                                         ↓
Depth backproject → keypoint 3D coords    (optional) ICP refinement → refined pose
  ↓
Save binary database
```

## Quick Start

### Step 1: Build offline database (Python)

```bash
pip install open3d torch torchvision opencv-python pyyaml

python scripts/build_database.py \
  --ply_map /path/to/omnilivo_map.ply \
  --config config/render_loc.yaml \
  --output data/image_db.bin \
  --render_dir data/rendered
```

This will:
1. Sample viewpoints inside your map (grid + multiple yaw angles)
2. Render RGB + Depth at each viewpoint using Open3D
3. Extract SuperPoint keypoints and NetVLAD global descriptors
4. Back-project keypoints to 3D using rendered depth
5. Save everything as a compact binary database

### Step 2: Run online localization (C++ / ROS2)

```bash
# Build
cd renderloc_ws
colcon build --packages-select render_loc

# Run
ros2 launch render_loc online_localizer.launch.py \
  database_path:=/path/to/image_db.bin \
  rgb_topic:=/camera/color/image_raw

# With depth for ICP refinement:
ros2 launch render_loc online_localizer.launch.py \
  database_path:=/path/to/image_db.bin \
  rgb_topic:=/camera/color/image_raw \
  depth_topic:=/camera/depth/image_raw \
  use_depth:=true
```

### Output topics
- `/renderloc/pose` — `geometry_msgs/PoseStamped` (6DoF in map frame)
- `/renderloc/odom` — `nav_msgs/Odometry`
- TF: `map → camera_link`

## Package Structure

```
render_loc/
├── CMakeLists.txt / package.xml
├── config/render_loc.yaml
├── launch/online_localizer.launch.py
├── scripts/
│   └── build_database.py         ← Offline pipeline (Python)
├── include/render_loc/
│   ├── core/
│   │   ├── types.h               ← Data structures
│   │   └── config.h              ← YAML config
│   ├── offline/
│   │   ├── viewpoint_sampler.h   ← Viewpoint generation
│   │   ├── map_renderer.h        ← Open3D rendering
│   │   ├── feature_extractor.h   ← SuperPoint + NetVLAD
│   │   ├── depth_backprojector.h ← Keypoint → 3D
│   │   └── database_builder.h    ← Orchestrator
│   ├── online/
│   │   ├── feature_matcher.h     ← SuperGlue
│   │   ├── pnp_solver.h          ← PnP + RANSAC
│   │   ├── icp_refiner.h         ← Optional depth refinement
│   │   └── localizer.h           ← Main pipeline
│   └── utils/timer.h
└── src/                           ← C++ implementations
```

## Dependencies

| Library | Purpose | Install |
|---------|---------|---------|
| Open3D | PLY rendering (offline) | `pip install open3d` |
| PyTorch | Feature extraction | `pip install torch` |
| LibTorch | C++ inference | cmake find_package(Torch) |
| OpenCV | Image + PnP | apt / cmake |
| PCL | ICP refinement | apt / cmake |
| Eigen3 | Linear algebra | apt |
| SuperPoint | Local features | [pretrained .pt](https://github.com/magicleap/SuperPointPretrainedNetwork) |
| SuperGlue | Feature matching | [pretrained .pt](https://github.com/magicleap/SuperGluePretrainedNetwork) |
| NetVLAD | Image retrieval | [pretrained .pt](https://github.com/Nanne/pytorch-NetVlad) |

## Configuration Tips

- **grid_spacing**: 0.5m for small rooms, 1.0m for larger spaces
- **num_yaw_angles**: 8 (45° increments) covers most cases; use 12 for narrow corridors
- **top_k_retrieval**: 5 is a good balance; increase to 10 if map has repetitive structures
- **ransac_reproj_threshold**: 8px default; lower (4px) for higher precision
- **ICP**: Enable when you have depth and need sub-centimeter accuracy

## Comparison with other approaches

| Aspect | HLoc (SfM-based) | RenderLoc (Ours) | VOLoc |
|--------|------------------|------------------|-------|
| Map input | Image collection | PLY point cloud | PLY point cloud |
| Query | Single image | Single image | Image sequence |
| 3D map | SfM reconstruction | Rendered from PLY | Compressed segments |
| Output | 6DoF pose | 6DoF pose | Coarse segment |
| Training | None | None | Required |
| Key advantage | Mature pipeline | Direct PLY use | Compressed storage |

## Real-time ROS2 localization (online)

검증된 오프라인 파이프라인 step5(retrieval)→step6(match)→step7(pnp)를 그대로
재사용해, 들어오는 이미지 토픽(main/sub 2대)으로 실시간 localization을 수행하고
추정 위치를 정렬(Z-up) 가우시안 지도 위에 rviz2로 표시한다.

추가 노드/파일:
- `scripts/ros/ros_localizer_node.py` — 이미지 토픽 → `localize_single` → pose(map frame).
  fisheye는 `camera_info`로 원본 K 유지 undistort 후 파이프라인에 투입.
- `scripts/ros/gaussian_ply_publisher.py` — 가우시안 PLY(center+SH DC 색) → 라치드 PointCloud2.
- `launch/ros_localizer.launch.py`, `config/ros_localizer.yaml`, `rviz/online_localizer.rviz`.

cam 매핑: 토픽↔`cam_ids`, `main_cam`/`sub_cams`는 전부 `config/ros_localizer.yaml`에서
변경한다(기본 bag `cam0`→`cam_0`(main), `cam1`→`cam_1`(sub)).

### 실행

```bash
# 환경 (torch+cuda / plyfile / rclpy 한 env)
conda activate render_loc
source /opt/ros/humble/setup.bash
source install/setup.bash      # colcon build --packages-select render_loc 이후

# 터미널 A: localizer + 가우시안 지도 + rviz
ros2 launch render_loc ros_localizer.launch.py use_rviz:=true

# 터미널 B: 이미지 토픽 재생 (step5~7가 GPU에서 프레임당 ~1–3s이므로 느리게)
ros2 bag play /home/park/Downloads/bero_test1/bero_test1_bag --rate 0.15
```

토픽: `/ros_localizer/pose`(PoseStamped), `/ros_localizer/path`(Path),
`/gaussian_ply_publisher/cloud`(PointCloud2), TF `map`→`base_optical`(→`base_link`).

참고: 노드는 동기된 **최신 프레임만 처리하고 나머지는 드롭**(`rate_hz` 상한)하므로
~0.5–1Hz로 pose를 갱신한다. 로봇/서버 분리 시 `ROS_DOMAIN_ID`만 맞추면 서버에서 그대로 동작.
