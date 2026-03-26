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
