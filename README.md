# RenderLoc

Gaussian/PLY map 기반 visual localization 파이프라인과 ROS2 웹 뷰어.

<p align="center">
  <img width="80%" alt="RenderLoc overview" src="https://github.com/user-attachments/assets/bd38e3ce-e19e-44b2-a509-460ae281f4c5" />
</p>

## Pipeline

<p align="center">
  <img width="1657" height="911" alt="Image" src="https://github.com/user-attachments/assets/edb0745f-a403-40d2-9a13-e6426526af4d" />
</p>

## 현재 흐름

```text
오프라인:
  step0 맵 정렬
  step1 viewpoint 샘플링
  step2 렌더링
  step3 global descriptor 추출
  step4 database 생성

온라인 / 테스트:
  step5 retrieval
  step6 feature matching
  step7 PnP
  /vps/current_pose publish
  웹 뷰어가 pose를 따라감
```

## 요구사항

핵심 실행 환경:

- Ubuntu 22.04 / Linux 계열
- ROS2 Humble
- Mambaforge 또는 Miniforge + `mamba`
- Python 3.10
- CUDA 지원 NVIDIA GPU
- 현재 RTX 50xx / Blackwell 세팅 기준 CUDA Toolkit 12.8
- PyTorch CUDA 12.8 build
- ROS2 workspace 빌드를 위한 `colcon`

주요 Python/CUDA 의존성은 `setup_env.sh`에서 설치합니다. 핵심 패키지는 PyTorch, OpenCV, Open3D, NumPy, SciPy, scikit-learn, Kornia, gsplat, plyfile, timm, torch-scatter, Scaffold-GS rasterizer입니다.

현재 설정은 아래 파일/디렉터리가 준비되어 있다고 가정합니다:

- `models/mixvpr_resnet50_4096.ckpt`
- `third_party/MixVPR`
- `third_party/JamMa`
- `third_party/JamMa/weights/jamma.ckpt` 또는 최초 실행 시 JamMa weight 다운로드 가능한 네트워크
- `test_rectified/cameras.txt`, `test_rectified/rigs.txt` 이는 테스트용 데이터로 camera intrinsic과 camera간의 extrinsic입니다.

## 설치 방법

ROS2 workspace의 source 디렉터리에서 시작합니다:

```bash
cd ~/loc_ws/src/render_loc
```

써드파티 코드를 준비합니다. `setup_env.sh`는 MixVPR이 없으면 자동으로 clone하지만, 현재 matcher가 `features.matcher_name: "jamma"`이므로 JamMa는 미리 준비해두는 것이 좋습니다.

```bash
mkdir -p third_party

# submodule이 설정된 checkout이면 실행합니다.
git submodule update --init --recursive

# 현재 step6 matcher에 필요합니다.
test -d third_party/JamMa || git clone https://github.com/leoluxxx/JamMa third_party/JamMa
```

mamba 환경을 만들고 CUDA/Python 의존성을 설치합니다:

```bash
bash setup_env.sh render_loc
mamba activate render_loc
```

RTX 50xx GPU가 아니라면 `setup_env.sh`의 `CUDA_ARCHES`를 먼저 수정하세요. 현재 기본값은 Blackwell용 `12.0`입니다.


## 실행 방법
우선 Map과 DB를 필요로 합니다.

Map은 Colmap SFM/OmniLIVO를 이용해 각 이미지들의 pose와 pointcloud 정보를 가지고 사용합니다.

이번 레포에서는 GS-SDF를 이용해서 가우시안 맵을 생성했습니다. 

https://github.com/hku-mars/GS-SDF.git

해당 방식을 이용해 학습 폴더가 하나 나왔다면 아래의 경로로 폴더를 옮긴다
```bash
output/gs_sdf_omni/gs_sdf_result/gs_sdf_scene1 #폴더 이름은 마음대로
```

이후 step0~4과정을 통해 DB를 생성합니다

```bash
#0 Map Align
python3 scripts/main.py \
  --ply_map output/gs_sdf_omni/gs_sdf_result/gs_sdf_scene1/model/gs.ply  \
  --config config/render_loc_multi_cam.yaml \
  --output_dir output/gs_sdf_omni \
  --step 0_align
  
#1 Viewpoints
python3 scripts/main.py \
  --ply_map output/gs_sdf_omni/gs_sdf_result/gs_sdf_scene1/model/gs.ply \
  --config config/render_loc_multi_cam.yaml \
  --output_dir output/gs_sdf_omni \
  --step 1_viewpoints
  
#2 Render
python3 scripts/main.py \
  --ply_map output/gs_sdf_omni/gs_sdf_result/gs_sdf_scene1/model/gs.ply \
  --config config/render_loc_multi_cam.yaml \
  --output_dir output/gs_sdf_omni \
  --step 2_render \
  --render_mode gaussian_ply

#3 Global desc
# 리트리벌에 사용할 수 있게 descriptor로 저장한다. 이떄 모델은 MixVPR을 사용한다.
# 이때 step2에서 렌더링 된 이미지 뿐만 아니라, 가우시안 맵을 만들 떄 사용했던 이미지들도 같이 저장한다.
python3 scripts/main.py \
	--config config/render_loc_multi_cam.yaml \
	--ply_map output/gs_sdf_omni/gs_sdf_result/gs_sdf_scene2/result/result_gs.ply \
	--output_dir output/gs_sdf_omni \
	--step 3_global_desc
	
#4 build db
python3 scripts/main.py \
	--config config/render_loc_multi_cam.yaml \
	--ply_map output/gs_sdf_omni/gs_sdf_result/gs_sdf_scene2/result/result_gs.ply \
	--output_dir output/gs_sdf_omni \
	--step 4_build_db
```

DB가 생성이 되었다면 실시간 localization을 위해 ROS2 패키지를 빌드합니다:

```bash
cd ~/loc_ws
source /opt/ros/humble/setup.zsh
colcon build --packages-select render_loc
source install/setup.zsh 
# or source install/render_loc/share/render_loc/local_setup.zsh
```

실시간 웹 로컬라이저를 실행합니다:

```bash
ros2 launch render_loc web_localizer.launch.py
```

브라우저에서 아래 주소로 접속합니다:

```text
http://localhost:8081
```

내장 테스트 bag 흐름에서는 웹 뷰어에서 `P`를 눌러 bag playback을 시작하고, `[`를 눌러 realtime localization을 켭니다.

## 설정

주요 config 파일은 두 개입니다. 각 항목은 접어서 볼 수 있게 정리했습니다.

<details>
<summary><code>config/ros_localizer.yaml</code> - ROS 토픽, 실시간 로컬라이저, 웹 뷰어 설정</summary>

- `output_dir`: localization DB/output root입니다. 현재 기본값은 `output/gs_sdf_omni_2`입니다.
- `config_file`: ROS 노드가 로드하는 step0~7 파이프라인 설정 파일입니다.
- `cam_topics`: 입력 compressed image topic입니다. 현재는 `/cam_0/image_raw/compressed`, `/cam_2/image_raw/compressed`를 사용합니다.
- `cam_ids`, `main_cam`, `sub_cams`: ROS 토픽과 파이프라인 camera ID의 매핑입니다.
- `static_camera_infos`: bag에 `CameraInfo`가 없을 때 사용하는 camera intrinsic/distortion입니다.
- `undistort`: fisheye image를 localization 전에 rectification할지 결정합니다.
- `lidar_topic`: data-ready 상태 확인에 사용하는 LiDAR topic입니다.
- `sync_slop`: 멀티 카메라 입력 timestamp 동기화 허용 범위입니다.
- `publish_view_cam`: 웹 뷰어에서 보여줄 기준 camera frame입니다. 보통 `cam_0`입니다.
- `pose_topic`, `path_topic`: localization 결과 pose/path 토픽입니다. 현재 웹 뷰어는 `/vps/current_pose`를 따라갑니다.
- `test_bag_path`: 웹 뷰어에서 `P` 키를 눌렀을 때 재생할 bag 경로입니다.
- `splat_path`, `aligned_ply`: 웹 뷰어가 서빙하는 맵 파일입니다.
- `host`, `port`: 웹 서버 bind 주소와 포트입니다. 다른 PC에서 접속하려면 `host: "0.0.0.0"`을 사용합니다.
- `camera_stream_enabled`: image streaming이 localization 속도를 떨어뜨리면 `false`로 끌 수 있습니다.
- `camera_stream_hz`: 카메라 패널 streaming 최대 Hz입니다.
- `topdown_map_size`, `topdown_z_min`, `topdown_z_max`: navigation mini-map 생성/표시 설정입니다.

</details>

<details>
<summary><code>config/render_loc_multi_cam.yaml</code> - step0~7 localization 파이프라인 설정</summary>

- `camera`: 오프라인/테스트 렌더링과 matching에 사용하는 기본 camera intrinsic입니다.
- `alignment.up_axis`, `alignment.yaw_deg`, `alignment.crop`: 맵 정렬, top-down 방향, crop 설정입니다.
- `sampling.path_spacing`, `sampling.num_yaw_angles`, `sampling.height_offsets_m`: database viewpoint 밀도와 카메라 높이 설정입니다.
- `rendering.gaussian_backend`: Gaussian 렌더링 backend입니다. 현재 기본값은 `2dgs`입니다.
- `features.global_desc_method`: global retrieval descriptor입니다. 현재 기본값은 `mixvpr`입니다.
- `features.mixvpr_ckpt`, `features.mixvpr_out_dim`: MixVPR checkpoint와 descriptor dimension입니다.
- `features.matcher_name`: local feature matcher입니다. 현재 기본값은 `jamma`입니다.
- `features.jamma_max_dim`: JamMa 입력 resize limit입니다. 낮추면 빠르고, 높이면 matching 품질이 좋아질 수 있습니다.
- `features.sold2_enable`: SOLD2 line matching과 line refinement 사용 여부입니다.
- `matching.top_k_retrieval`: feature matching 전에 가져올 retrieval 후보 수입니다.
- `multi_cam.retrieval_type`: 멀티 카메라 retrieval 방식입니다. 현재는 보통 `type2`를 사용합니다.
- `multi_cam.main_cam`, `multi_cam.sub_cams`, `multi_cam.cam_ids`: 현재 rig camera 구성입니다.
- `multi_cam.exclude_real_entries`: `true`면 real DB entry를 제외하고 Gaussian/rendered entry만 retrieval합니다.
- `multi_cam.include_camless_entries`: cam ID가 없는 Gaussian entry를 모든 camera 후보로 허용합니다.
- `multi_cam.match_top_k`: feature matching으로 넘길 최종 후보 수입니다.
- `multi_cam.kapture_dir`: `cameras.txt`, `rigs.txt`가 있는 디렉터리입니다.
- `pnp.solver`, `pnp.reproj_threshold`, `pnp.min_inliers`: PnP solver와 pose accept threshold입니다.
- `pnp.multi_cam_pose_selection`: 멀티 카메라 pose 선택/fusion 방식입니다.

</details>

## 주요 파일

- `scripts/main.py`: 오프라인/테스트 파이프라인 진입점
- `scripts/pipeline/`: step0~7 구현
- `scripts/ros/ros_localizer_node.py`: image topic을 받아 localization을 수행하고 `/vps/current_pose` publish
- `scripts/ros/web_pose_bridge.py`: pose/image/status를 브라우저 뷰어로 전달
- `scripts/ros/ply_to_splat.py`: Gaussian PLY를 `.splat`으로 변환
- `launch/web_localizer.launch.py`: 실시간 localizer + 웹 뷰어 실행
- `launch/gaussian_web_viewer.launch.py`: pose topic 뷰어만 실행
- `config/render_loc_multi_cam.yaml`: 파이프라인 설정 파일
- `config/ros_localizer.yaml`: 현재 ROS/web localizer 설정 파일
- `web/`: 브라우저 뷰어

## 오프라인 localzation 테스트 예시

```bash
python3 scripts/main.py \
  --config config/render_loc_multi_cam.yaml \
  --ply_map output/gs_sdf_omni_2/gs_sdf_result/gs_sdf_scene1/model/gs_best.ply \
  --output_dir output/gs_sdf_omni_2 \
  --step test \
  --test_dir ./test_rectified/cam_0/images/
```

## 웹 로컬라이저 실행

```bash
conda activate render_loc
source /opt/ros/humble/setup.zsh
source install/setup.zsh

ros2 launch render_loc web_localizer.launch.py
```

기본 뷰어 주소:

```text
http://localhost:8081
```

주요 키보드 조작은 뷰어 overlay에 표시됩니다.

## 참고

- 현재 live pose 토픽은 `/vps/current_pose`입니다.
- 현재 path 토픽은 `/vps/pred_path`입니다.
- 현재 test bag 경로는 `config/ros_localizer.yaml`에서 설정합니다.
