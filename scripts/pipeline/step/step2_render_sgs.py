"""
STEP 2: Rendering — Scaffold-GS mode (submodule wrapper).

원본 Scaffold-GS (https://github.com/city-super/Scaffold-GS) 를 submodule로 사용.
  third_party/scaffold_gs/train.py  — 학습
  third_party/scaffold_gs/render_custom.py — 커스텀 viewpoint 렌더링

Pipeline:
  1. kapture sensors → COLMAP sparse/0/{cameras,images,points3D}.txt + images/
  2. train.py subprocess 호출
  3. render_custom.py subprocess 호출 → PNG 저장
  4. step2_data.pkl 생성

Prerequisites (한 번만):
  cd third_party/scaffold_gs/submodules/diff-gaussian-rasterization
  CUDA_HOME=/usr/local/cuda-12.8 pip install . --no-build-isolation
  cd ../simple-knn
  CUDA_HOME=/usr/local/cuda-12.8 pip install . --no-build-isolation
"""
import os, sys, json, pickle, subprocess
import numpy as np
import open3d as o3d
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .step2_render import (
    _gs_parse_sensors,
    _gs_parse_trajectories,
    _gs_parse_records,
)

# Scaffold-GS 루트 경로
_SGS_ROOT = os.path.join(
    os.path.dirname(__file__),           # scripts/pipeline/step/
    "..", "..", "..",                    # render_loc/
    "third_party", "scaffold_gs"
)
_SGS_ROOT = os.path.normpath(_SGS_ROOT)


# =============================================================================
# kapture → COLMAP 변환
# =============================================================================

def _rotmat_to_quat_wxyz(R):
    """3x3 rotation matrix → (qw, qx, qy, qz)"""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return w, x, y, z


def kapture_to_colmap(kapture_dir, ply_path, colmap_dir):
    """
    kapture sensors 디렉토리를 COLMAP 포맷으로 변환.

    출력 구조:
      colmap_dir/
        images/          ← 이미지 심링크
        sparse/0/
          cameras.txt
          images.txt
          points3D.ply   ← ply_path 복사
    """
    os.makedirs(os.path.join(colmap_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(colmap_dir, "sparse", "0"), exist_ok=True)

    sensors_txt = os.path.join(kapture_dir, "sensors.txt")
    traj_txt    = os.path.join(kapture_dir, "trajectories.txt")
    records_txt = os.path.join(kapture_dir, "records_camera.txt")

    sensor_params = _gs_parse_sensors(sensors_txt)
    traj          = _gs_parse_trajectories(traj_txt)
    records       = _gs_parse_records(records_txt)

    # ── cameras.txt ─────────────────────────────────────────────────────────
    # COLMAP camera ID는 정수. cam_0 → 1, cam_1 → 2, ...
    cam_id_map = {}   # "cam_0" → 1
    cameras_lines = ["# Camera list with one line of data per camera:\n",
                     "#   CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"]
    for i, (sid, p) in enumerate(sorted(sensor_params.items()), start=1):
        cam_id_map[sid] = i
        cameras_lines.append(
            f"{i} PINHOLE {p['w']} {p['h']} "
            f"{p['fx']:.6f} {p['fy']:.6f} {p['cx']:.6f} {p['cy']:.6f}\n"
        )
    with open(os.path.join(colmap_dir, "sparse", "0", "cameras.txt"), "w") as f:
        f.writelines(cameras_lines)

    # ── images.txt ──────────────────────────────────────────────────────────
    # COLMAP 포맷: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    # Kapture pose = c2w, COLMAP = w2c
    images_lines = ["# Image list with two lines of data per image:\n",
                    "#   IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n",
                    "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"]
    img_id = 1
    records_data_dir = os.path.join(kapture_dir, "records_data")
    colmap_images_dir = os.path.join(colmap_dir, "images")

    for ts, dev_id, rel_path in records:
        key = (ts, dev_id)
        if key not in traj:
            continue
        if dev_id not in cam_id_map:
            continue

        T_c2w = traj[key]
        R_c2w = T_c2w[:3, :3]
        t_c2w = T_c2w[:3,  3]

        # w2c
        R_w2c = R_c2w.T
        t_w2c = -R_c2w.T @ t_c2w
        qw, qx, qy, qz = _rotmat_to_quat_wxyz(R_w2c)

        colmap_cam_id = cam_id_map[dev_id]
        rel_path_s = rel_path.strip()   # e.g. "cam_0/images/1771910567999219.jpg"
        ext = os.path.splitext(rel_path_s)[1]  # .jpg
        stem = os.path.splitext(os.path.basename(rel_path_s))[0]  # 1771910567999219
        # Scaffold-GS는 os.path.basename(name)으로 images/ 아래를 flat 검색하므로
        # 카메라별로 고유한 이름이 필요: cam_0_1771910567999219.jpg
        flat_name = f"{dev_id}_{stem}{ext}"

        images_lines.append(
            f"{img_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
            f"{t_w2c[0]:.9f} {t_w2c[1]:.9f} {t_w2c[2]:.9f} "
            f"{colmap_cam_id} {flat_name}\n"
        )
        images_lines.append("\n")   # empty 2D points line

        # 이미지 심링크 (flat: sgs_colmap/images/cam_0_1771910567999219.jpg)
        src = os.path.join(records_data_dir, rel_path_s)
        dst = os.path.join(colmap_images_dir, flat_name)
        if not os.path.exists(dst):
            if os.path.exists(src):
                os.symlink(src, dst)
            else:
                print(f"  [WARN] Image not found: {src}")

        img_id += 1

    with open(os.path.join(colmap_dir, "sparse", "0", "images.txt"), "w") as f:
        f.writelines(images_lines)

    # ── points3D.ply ────────────────────────────────────────────────────────
    # Scaffold-GS는 sparse/0/points3D.ply 를 초기 점군으로 사용
    dst_ply = os.path.join(colmap_dir, "sparse", "0", "points3D.ply")
    if not os.path.exists(dst_ply):
        pcd = o3d.io.read_point_cloud(ply_path)
        o3d.io.write_point_cloud(dst_ply, pcd)
        print(f"  Copied PLY: {len(pcd.points):,} pts → {dst_ply}")

    # empty points3D.txt (Scaffold-GS binary loader fallback 방지)
    pt_txt = os.path.join(colmap_dir, "sparse", "0", "points3D.txt")
    if not os.path.exists(pt_txt):
        with open(pt_txt, "w") as f:
            f.write("# 3D point list\n")

    print(f"  COLMAP data ready: {img_id-1} images, "
          f"{len(cam_id_map)} cameras → {colmap_dir}")
    return sensor_params, cam_id_map


# =============================================================================
# viewpoints → JSON (render_custom.py 입력)
# =============================================================================

def viewpoints_to_json(viewpoints, sensor_params, config, json_path):
    """
    Pipeline viewpoints를 render_custom.py 가 읽는 JSON으로 변환.
    각 viewpoint의 카메라 파라미터는 config의 cam_cfg에서 읽음.
    """
    cam_cfg = config.get("camera", {})

    # 첫 번째 센서 파라미터를 기본값으로 사용
    first = next(iter(sensor_params.values()))
    fx = cam_cfg.get("fx", first["fx"])
    fy = cam_cfg.get("fy", first["fy"])
    cx = cam_cfg.get("cx", first["cx"])
    cy = cam_cfg.get("cy", first["cy"])
    w  = cam_cfg.get("width",  first["w"])
    h  = cam_cfg.get("height", first["h"])

    out = []
    for vp in viewpoints:
        pose = vp["pose"]
        if isinstance(pose, np.ndarray):
            pose_list = pose.tolist()
        else:
            pose_list = pose
        out.append({
            "name":   f"{vp['id']:05d}",
            "pose":   pose_list,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": w, "height": h,
        })

    with open(json_path, "w") as f:
        json.dump(out, f)
    print(f"  Viewpoints JSON: {len(out)} entries → {json_path}")


# =============================================================================
# subprocess 헬퍼
# =============================================================================

def _run(cmd, cwd=None, env=None):
    """명령어 실행. 실패 시 예외 발생."""
    import os as _os, resource
    run_env = _os.environ.copy()
    if env:
        run_env.update(env)
    print(f"  $ {' '.join(cmd)}")

    def _set_limits():
        # 파일 디스크립터 한도 상향 (이미지 수가 많을 때 필요)
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
        except Exception:
            pass

    ret = subprocess.run(cmd, cwd=cwd, env=run_env, preexec_fn=_set_limits)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed (code {ret.returncode}): {' '.join(cmd)}")


# =============================================================================
# 메인 렌더링 함수
# =============================================================================

def _render_sgs(ply_path, viewpoints, config, output_dir,
                step0_data=None,
                kapture_dir="kapture/sensors",
                gs_epochs=150,
                subsample=1,
                voxel_size=0.05,
                train_img_size=1920,
                accum_steps=1,
                use_ppisp=False,
                **kwargs):

    sgs_root   = _SGS_ROOT
    colmap_dir = os.path.abspath(os.path.join(output_dir, "sgs_colmap"))
    model_dir  = os.path.abspath(os.path.join(output_dir, "sgs_model"))
    renders_dir = os.path.abspath(os.path.join(output_dir, "rendered", "rgb"))
    depth_dir   = os.path.abspath(os.path.join(output_dir, "rendered", "depth"))
    os.makedirs(renders_dir, exist_ok=True)
    os.makedirs(depth_dir,   exist_ok=True)

    # kapture 경로 절대화
    if not os.path.isabs(kapture_dir):
        kapture_dir = os.path.join(os.getcwd(), kapture_dir)

    # ── 1. kapture → COLMAP ─────────────────────────────────────────────────
    print("\n[SGS] Step 1: kapture → COLMAP")
    sensor_params, _ = kapture_to_colmap(kapture_dir, ply_path, colmap_dir)

    # Scaffold-GS voxel_size: 우리 파이프라인 값 전달
    # iterations: epochs × (이미지 수) 에 가깝게 설정
    # images.txt: 이미지당 pose 1줄 + 빈 줄 1줄. 빈 줄은 strip()으로 필터되므로 그대로가 실제 이미지 수
    n_real_images = sum(1 for line in open(
        os.path.join(colmap_dir, "sparse", "0", "images.txt"))
        if line.strip() and not line.startswith("#"))
    iterations = max(gs_epochs * max(n_real_images, 1), 30000)
    iterations = min(iterations, 100_000)
    print(f"  Images: {n_real_images}, Epochs target: {gs_epochs} → iterations: {iterations}")

    # ── 2. Scaffold-GS 학습 ─────────────────────────────────────────────────
    print("\n[SGS] Step 2: Train Scaffold-GS")
    cuda_env = {"CUDA_HOME": "/usr/local/cuda-12.8",
                "PATH": f"/usr/local/cuda-12.8/bin:{os.environ.get('PATH','')}"}

    train_cmd = [
        sys.executable, os.path.join(sgs_root, "train.py"),
        "--source_path",  colmap_dir,
        "--model_path",   model_dir,
        "--iterations",   str(iterations),
        "--voxel_size",   str(voxel_size),
        "--save_iterations", str(iterations),
        "--test_iterations", str(iterations),
        "--resolution",   "-1",
    ]
    _run(train_cmd, cwd=sgs_root, env=cuda_env)

    # ── 3. viewpoints JSON 생성 ──────────────────────────────────────────────
    print("\n[SGS] Step 3: Prepare viewpoints for rendering")
    vp_json = os.path.abspath(os.path.join(output_dir, "sgs_viewpoints.json"))
    viewpoints_to_json(viewpoints, sensor_params, config, vp_json)

    # ── 4. render_custom.py 실행 ─────────────────────────────────────────────
    print("\n[SGS] Step 4: Render viewpoints")
    render_cmd = [
        sys.executable, os.path.join(sgs_root, "render_custom.py"),
        "--model_path",      model_dir,
        "--source_path",     colmap_dir,
        "--viewpoints_json", vp_json,
        "--output_dir",      renders_dir,
        "--iteration",       str(iterations),
    ]
    _run(render_cmd, cwd=sgs_root, env=cuda_env)

    # ── 5. 결과 수집 ─────────────────────────────────────────────────────────
    print("\n[SGS] Step 5: Collect results")
    rendered = []
    for vp in viewpoints:
        name  = f"{vp['id']:05d}"
        rp_   = os.path.join(renders_dir, f"{name}.png")
        dp_np = os.path.join(depth_dir, f"{name}.npy")
        # depth: render_custom이 renders_dir/depth/ 에 저장
        dp_src = os.path.join(renders_dir, "depth", f"{name}.npy")
        if os.path.exists(dp_src) and not os.path.exists(dp_np):
            import shutil
            shutil.move(dp_src, dp_np)

        if not os.path.exists(rp_):
            print(f"  [WARN] Missing render: {rp_}")
            continue

        rendered.append({
            "id":         vp["id"],
            "pose":       vp["pose"],
            "floor":      vp.get("floor", 0),
            "yaw":        vp.get("yaw", 0.0),
            "rgb_path":   rp_,
            "depth_path": dp_np if os.path.exists(dp_np) else "",
        })

    # ── 6. 시각화 ─────────────────────────────────────────────────────────────
    if rendered:
        ns  = min(8, len(rendered))
        idx = np.linspace(0, len(rendered)-1, ns, dtype=int)
        fig, axes = plt.subplots(1, ns, figsize=(4*ns, 4))
        if ns == 1:
            axes = [axes]
        for c, ii in enumerate(idx):
            r = rendered[ii]
            img = cv2.cvtColor(cv2.imread(r["rgb_path"]), cv2.COLOR_BGR2RGB)
            axes[c].imshow(img)
            axes[c].set_title(f"SGS #{r['id']}", fontsize=9)
            axes[c].axis("off")
        fig.suptitle(f"Step 2-SGS: {len(rendered)} renders", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "step2_sgs_rendered.png"), dpi=150)
        plt.close()
        print(f"  Saved: step2_sgs_rendered.png")

    pkl_path = os.path.join(output_dir, "step2_sgs_data.pkl")
    pickle.dump(rendered, open(pkl_path, "wb"))
    pickle.dump(rendered, open(os.path.join(output_dir, "step2_data.pkl"), "wb"))
    print(f"  Saved: {pkl_path}  ({len(rendered)} entries)")
    return rendered


def step2_render_sgs(ply_path, viewpoints, config, output_dir,
                     step0_data=None, **kwargs):
    """Scaffold-GS 렌더링 entry point (submodule 방식)."""
    return _render_sgs(ply_path, viewpoints, config, output_dir,
                       step0_data=step0_data, **kwargs)
