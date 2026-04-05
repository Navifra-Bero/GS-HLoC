#!/usr/bin/env python3
"""
RenderLoc Step-by-Step Pipeline — LoFTR Edition
================================================
Offline (DB 구축):
  python3 step_by_step.py --ply_map /path/to/map.ply --step all
  python3 step_by_step.py --ply_map /path/to/map.ply --step 0_align
  python3 step_by_step.py --ply_map /path/to/map.ply --step 1_viewpoints
  python3 step_by_step.py --ply_map /path/to/map.ply --step 2_render
  python3 step_by_step.py --ply_map /path/to/map.ply --step 3_global_desc
  python3 step_by_step.py --ply_map /path/to/map.ply --step 4_build_db

Online (로컬라이제이션):
  python3 step_by_step.py --ply_map /path/to/map.ply --step online
  python3 step_by_step.py --ply_map /path/to/map.ply --step online --query_image /path/to/query.png

Batch test:
  python3 step_by_step.py --ply_map /path/to/map.ply --step test --test_dir /path/to/imgs [--gt_poses /path.json]
"""
import argparse, os, pickle, glob
import numpy as np
import open3d as o3d
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import (binary_dilation, binary_erosion, distance_transform_edt,
                           gaussian_filter, maximum_filter)
from skimage.morphology import skeletonize
import yaml


def load_config(p):
    with open(p) as f:
        return yaml.safe_load(f)


def default_config():
    return {
        #cam_0
        # "camera": {
        #     "fx": 1027.659153, "fy": 1030.215807, "cx": 966.827228, "cy": 585.037643,
        #     "width": 1920, "height": 1080,
        #     "depth_scale": 1000, "depth_min": 0.3, "depth_max": 20.0,
        # },
        # #cam_1
        # "camera": {
        #     "fx": 1028.487260, "fy": 1030.283620, "cx": 949.476264, "cy": 597.274302,
        #     "width": 1920, "height": 1080,
        #     "depth_scale": 1000, "depth_min": 0.3, "depth_max": 20.0,
        # },
        # #cam_2
        # "camera": {
        #     "fx": 1041.454577, "fy": 1044.072979, "cx": 945.363937, "cy": 610.455165,
        #     "width": 1920, "height": 1080,
        #     "depth_scale": 1000, "depth_min": 0.3, "depth_max": 20.0,
        # },
        # #cam_3
        # "camera": {
        #     "fx": 1039.04598063, "fy": 1041.49694151, "cx": 937.04407689, "cy": 560.82673816,
        #     "width": 1920, "height": 1080,
        #     "depth_scale": 1000, "depth_min": 0.3, "depth_max": 20.0,
        # },
        #femto
        "camera": {
            "fx": 2256.627197, "fy": 2254.400635, "cx": 1891.352783, "cy": 1087.097656,
            "width": 3840, "height": 2160,
            "depth_scale": 1000, "depth_min": 0.3, "depth_max": 20.0,
        },
        "alignment": {
            "normal_threshold": 0.8, "ransac_distance": 0.05,
            "ransac_n": 3, "ransac_iterations": 1000, "pre_flip_x": True,
        },
        "sampling": {
            "grid_resolution": 0.05, "path_spacing": 0.3,
            "height_above_floor": 1.2, "num_yaw_angles": 4,
            "pitch_deg": 0.0, "morph_kernel_size": 5,
            "distance_thresh_ratio": 0.3, "min_floor_points": 100,
            "max_floors": 1, "min_floor_gap": 2.5,
            "max_floor_height": 5.0, "floor_band": 0.3,
            # "skeleton": 기존 1-pixel 중심선만 샘플링
            # "corridor": 벽에서 min_wall_dist_m 이상 떨어진 전체 자유 공간 샘플링 (더 넓은 커버리지)
            "sample_mode": "skeleton",
            "min_wall_dist_m": 0.3,    # corridor 모드: 벽 최소 거리 (m)
            # skeleton 보강: corridor 내부에 격자선 추가 (0=비활성)
            # 예) 3.0 → 3m 간격 수평+수직선을 skeleton에 union → 넓은 방 커버리지 향상
            "skel_grid_spacing_m": 0.5,
        },
        "rendering": {
            "point_size": 1.0,
            "brightness_scale": 1.0,
            "ceiling_clip": True,
            "ceiling_margin": 0.3,
            # hole filling: 근거리(< near_dist m)는 near_fill_radius px, 원거리는 fill_radius px
            "near_dist": 3.0,
            "near_fill_radius": 6.0,
            "fill_radius": 2.0,
        },
        "features": {
            # global descriptor 방법 선택: "anyloc" (DINOv2+VLAD) 또는 "megaloc"
            "global_desc_method": "megaloc",
            # DINOv2 + VLAD global descriptor (anyloc)
            "dino_model": "dinov2_vitb14",   # vitb14(768d) / vits14(384d) / vitl14(1024d)
            "dino_img_size": 322,            # 14의 배수 (322=23×14, 224=16×14)
            "vlad_clusters": 64,             # VLAD cluster 수 (AnyLoc default)
            "vlad_pca_dim": 4096,            # PCA whitening 후 최종 dim
            # MegaLoc (SOTA retrieval)
            "megaloc_dim": 8448,             # 기본 descriptor 차원
            # EfficientLoFTR checkpoint path (자동 탐색)
            "eloftr_ckpt": "",   # 비워두면 EfficientLoFTR/weights/... 자동 탐색
            "eloftr_opt": False,  # True=빠른 추론(opt_default_cfg), False=최고 품질
            "eloftr_max_dim": 840,   # 입력 최대 해상도 (OOM 방지, 32 배수 권장)
            # confidence threshold for EfficientLoFTR matches
            "match_conf_thresh": 0.2,
        },
        "online": {
            "top_k": 5,
            "reprojection_error": 8.0,
            "pnp_iterations": 1000,
            "pnp_confidence": 0.99,
        },
    }


# =============================================================================
# STEP 0: Floor Plane Detection & Gravity Alignment
# =============================================================================
def step0_align(ply_path, config, output_dir):
    """
    PLY 맵의 바닥 평면을 RANSAC으로 찾아서 중력 방향(Z-up)으로 정렬.

    1) Normal 계산
    2) RANSAC plane fitting → 가장 큰 평면 = 바닥
    3) 바닥 법선 → Z-up 회전행렬 계산 (Rodrigues)
    4) 전체 포인트클라우드 회전 + 바닥 z=0 이동
    5) 정렬된 PLY 저장

    시각화: before/after 비교 (top-down, side view, Z histogram)
    """
    print("\n" + "="*60)
    print("STEP 0: Floor plane detection & gravity alignment")
    print("="*60)
    os.makedirs(output_dir, exist_ok=True)
    align_cfg = config.get("alignment", {})

    pcd = o3d.io.read_point_cloud(ply_path)
    points_orig = np.asarray(pcd.points).copy()
    has_color = pcd.has_colors()
    print(f"  Loaded: {len(points_orig)} points, color={has_color}")

    # PLY가 down-top view인 경우 X축 기준 180° 사전 회전 (Y, Z 반전)
    if align_cfg.get("pre_flip_x", False):
        R_flip = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
        pts = np.asarray(pcd.points)
        pcd.points = o3d.utility.Vector3dVector((R_flip @ pts.T).T)
        if pcd.has_normals():
            nrm = np.asarray(pcd.normals)
            pcd.normals = o3d.utility.Vector3dVector((R_flip @ nrm.T).T)
        print("  Pre-flip: 180° around X axis applied (down-top → top-down)")

    if not pcd.has_normals():
        print("  Computing normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    normals = np.asarray(pcd.normals)

    # RANSAC plane fitting — 바닥 포인트만 대상으로 (하단 40% Z 범위)
    rd = align_cfg.get("ransac_distance", 0.05)
    rn = align_cfg.get("ransac_n", 3)
    ri = align_cfg.get("ransac_iterations", 1000)
    z_vals = np.asarray(pcd.points)[:, 2]
    z_low  = np.percentile(z_vals, 40)
    floor_mask = z_vals <= z_low
    pcd_floor = pcd.select_by_index(np.where(floor_mask)[0])
    print(f"  RANSAC on bottom-40% points ({floor_mask.sum()}) (dist={rd}, iter={ri})...")
    plane_model, inliers_sub = pcd_floor.segment_plane(
        distance_threshold=rd, ransac_n=rn, num_iterations=ri)
    orig_indices = np.where(floor_mask)[0]
    inliers = orig_indices[inliers_sub]
    a, b, c, d = plane_model
    floor_normal = np.array([a, b, c])
    floor_normal /= np.linalg.norm(floor_normal)
    print(f"  Plane: {a:.4f}x+{b:.4f}y+{c:.4f}z+{d:.4f}=0")
    print(f"  Normal: {floor_normal}, Inliers: {len(inliers)}/{len(points_orig)}")

    if floor_normal[2] < 0:
        floor_normal = -floor_normal
        print("  Flipped normal to point upward (Z<0 detected)")

    z_up = np.array([0.0, 0.0, 1.0])
    v = np.cross(floor_normal, z_up)
    s = np.linalg.norm(v)
    c_val = np.dot(floor_normal, z_up)

    if s < 1e-6:
        R = np.eye(3) if c_val > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c_val) / (s * s))
    angle_deg = np.degrees(np.arccos(np.clip(c_val, -1, 1)))
    print(f"  Rotation: {angle_deg:.2f}°")

    points = np.asarray(pcd.points)
    points_rotated = (R @ points.T).T
    floor_z_after = np.median(points_rotated[inliers, 2])
    points_rotated[:, 2] -= floor_z_after
    print(f"  Floor z shifted: {floor_z_after:.4f} → 0")

    pcd_aligned = o3d.geometry.PointCloud()
    pcd_aligned.points = o3d.utility.Vector3dVector(points_rotated)
    if has_color:
        pcd_aligned.colors = pcd.colors
    # normals은 floor 검출용으로만 사용 — aligned PLY에는 저장 안 함 (용량 절약)

    aligned_path = os.path.join(output_dir, "aligned_map.ply")
    o3d.io.write_point_cloud(aligned_path, pcd_aligned, write_ascii=False, compressed=False)
    size_gb = os.path.getsize(aligned_path) / 1e9
    print(f"  Saved: {aligned_path} ({size_gb:.2f} GB)")

    if align_cfg.get("pre_flip_x", False):
        R_flip = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
        R_total = R @ R_flip
    else:
        R_total = R
    T_align = np.eye(4); T_align[:3,:3] = R_total; T_align[2,3] = -floor_z_after

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    sub = max(1, len(points_orig) // 30000)
    floor_pts = points_orig[inliers]
    sub_f = max(1, len(floor_pts) // 5000)

    axes[0,0].scatter(points_orig[::sub,0], points_orig[::sub,1],
                      c=points_orig[::sub,2], s=0.5, cmap="viridis")
    axes[0,0].set_title("Before: Top-down (X-Y)"); axes[0,0].set_aspect("equal")
    axes[0,1].scatter(points_orig[::sub,0], points_orig[::sub,2], c="gray", s=0.5, alpha=0.3)
    axes[0,1].scatter(floor_pts[::sub_f,0], floor_pts[::sub_f,2], c="red", s=1, alpha=0.5)
    axes[0,1].set_title("Before: Side (X-Z) + floor (red)")
    axes[0,2].scatter(points_orig[::sub,1], points_orig[::sub,2], c="gray", s=0.5, alpha=0.3)
    axes[0,2].scatter(floor_pts[::sub_f,1], floor_pts[::sub_f,2], c="red", s=1, alpha=0.5)
    axes[0,2].set_title("Before: Side (Y-Z) + floor (red)")

    axes[1,0].scatter(points_rotated[::sub,0], points_rotated[::sub,1],
                      c=points_rotated[::sub,2], s=0.5, cmap="viridis")
    axes[1,0].set_title("After: Top-down (X-Y)"); axes[1,0].set_aspect("equal")
    axes[1,1].scatter(points_rotated[::sub,0], points_rotated[::sub,2], c="gray", s=0.5, alpha=0.3)
    axes[1,1].axhline(y=0, color="green", linewidth=2, label="Floor z=0")
    axes[1,1].set_title("After: Side (X-Z)"); axes[1,1].legend()
    axes[1,2].hist(points_rotated[:,2], bins=200, orientation="horizontal",
                   color="steelblue", alpha=0.7)
    axes[1,2].axhline(y=0, color="green", linewidth=2, label="Floor z=0")
    axes[1,2].set_title("After: Z distribution"); axes[1,2].legend()

    fig.suptitle(f"Step 0: Floor alignment — {angle_deg:.1f}° rotation", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "step0_alignment.png"), dpi=150); plt.close()
    print(f"  Saved: step0_alignment.png")

    data = {
        "aligned_ply_path": aligned_path, "T_align": T_align, "R": R,
        "floor_normal_orig": floor_normal, "floor_z_shift": floor_z_after, "inliers": inliers,
    }
    pickle.dump(data, open(os.path.join(output_dir, "step0_data.pkl"), "wb"))
    return data


# =============================================================================
# STEP 1: Free Path Corridor Viewpoint Sampling
# =============================================================================
def step1_viewpoints(ply_path, config, output_dir, step0_data=None):
    """
    논문의 Free Path Corridor 방식으로 viewpoint 생성.

    1) 정렬된 PLY 로드 + normal 계산
    2) Normal이 위를 향하는 점 필터링 (dot > threshold)
    3) 높이 히스토그램 → peak detection → 층(floor level) 분리
    4) 각 층의 바닥 점을 top-down occupancy image로 변환
    5) Morphology (dilation → erosion) → 구멍 메우기
    6) Distance transform → 자유 공간 중심
    7) Threshold + Gaussian blur → 메인 통로
    8) Skeletonize → 1-pixel 경로 (walkable path)
    9) Skeleton 위에서 greedy sampling (path_spacing 간격)
    10) 각 위치 × num_yaw_angles 방향 → viewpoint 생성

    시각화: 6-column (occupancy→morph→dist→corridor→skeleton→viewpoints)
    """
    print("\n" + "="*60)
    print("STEP 1: Free Path Corridor viewpoint sampling")
    print("="*60)
    os.makedirs(output_dir, exist_ok=True)
    samp = config.get("sampling", {})
    align_cfg = config.get("alignment", {})

    if step0_data and "aligned_ply_path" in step0_data:
        ap = step0_data["aligned_ply_path"]
    else:
        ap = os.path.join(output_dir, "aligned_map.ply")
        if not os.path.exists(ap): ap = ply_path
    pcd = o3d.io.read_point_cloud(ap)
    points = np.asarray(pcd.points)
    print(f"  Loaded: {len(points)} points")

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    normals = np.asarray(pcd.normals)

    nth = align_cfg.get("normal_threshold", 0.8)
    dots = normals @ np.array([0.0, 0.0, 1.0])
    up_mask = dots > nth
    up_pts = points[up_mask]
    print(f"  Upward-facing: {len(up_pts)} ({100*len(up_pts)/len(points):.1f}%)")

    z_vals = up_pts[:, 2]
    max_floors = samp.get("max_floors", 1)
    min_gap = samp.get("min_floor_gap", 2.5)
    max_fh = samp.get("max_floor_height", 5.0)
    min_fp = samp.get("min_floor_points", 100)

    bins = np.arange(z_vals.min()-0.1, z_vals.max()+0.1, 0.05)
    hc, he = np.histogram(z_vals, bins=bins)
    hs = gaussian_filter(hc.astype(float), sigma=2)
    lm = (hs == maximum_filter(hs, size=20))

    candidates = []
    for i in range(len(hs)):
        if lm[i] and hs[i] > min_fp:
            z_center = (he[i]+he[i+1])/2
            if z_center <= max_fh:
                candidates.append((z_center, hs[i]))

    candidates.sort(key=lambda x: -x[1])

    floors = []
    for z_c, count in candidates:
        too_close = any(abs(z_c - ez) < min_gap for ez in floors)
        if not too_close:
            floors.append(z_c)
        if len(floors) >= max_floors:
            break

    floors.sort()
    if not floors: floors = [np.median(z_vals)]
    print(f"  Floors: {[f'{z:.2f}' for z in floors]} "
          f"(max_floors={max_floors}, min_gap={min_gap}m, max_h={max_fh}m)")

    gr = samp.get("grid_resolution", 0.05)
    ps = samp.get("path_spacing", 0.5)
    ch = samp.get("height_above_floor", 1.2)
    ny = samp.get("num_yaw_angles", 6)
    pitch = np.radians(samp.get("pitch_deg", 0.0))
    mk = samp.get("morph_kernel_size", 5)
    dr = samp.get("distance_thresh_ratio", 0.3)
    sample_mode = samp.get("sample_mode", "skeleton")   # "corridor" | "skeleton"
    min_wall_dist_px = samp.get("min_wall_dist_m", 0.3) / gr  # corridor 모드 벽 최소 거리(px)
    skel_grid_spacing_m = samp.get("skel_grid_spacing_m", 2.0)  # 0=비활성
    all_vp = []; debug_imgs = {}; vid = 0

    for fi, fz in enumerate(floors):
        print(f"\n  --- Floor {fi}: z={fz:.2f}m ---")
        band = samp.get("floor_band", 0.3)
        fm = up_mask & (points[:,2] > fz-band) & (points[:,2] < fz+band)
        fp = points[fm]
        print(f"    Floor points: {len(fp)}")
        if len(fp) < 50: print("    Skip"); continue

        margin = 1.0
        xn, yn = fp[:,0].min()-margin, fp[:,1].min()-margin
        xx, yx = fp[:,0].max()+margin, fp[:,1].max()+margin
        iw = int(np.ceil((xx-xn)/gr)); ih = int(np.ceil((yx-yn)/gr))
        print(f"    Image: {iw}x{ih}px (res={gr}m/px)")

        occ = np.zeros((ih, iw), dtype=np.uint8)
        px = np.clip(((fp[:,0]-xn)/gr).astype(int), 0, iw-1)
        py = np.clip(((fp[:,1]-yn)/gr).astype(int), 0, ih-1)
        occ[py, px] = 1

        kern = np.ones((mk, mk), dtype=np.uint8)
        closed = binary_dilation(occ, structure=kern, iterations=2).astype(np.uint8)
        closed = binary_erosion(closed, structure=kern, iterations=2).astype(np.uint8)

        dm = distance_transform_edt(closed)
        dmx = dm.max()
        print(f"    Dist max: {dmx:.1f}px ({dmx*gr:.2f}m)")

        dn = dm/dmx if dmx > 0 else dm
        corr = (dn > dr).astype(np.uint8)
        cs = gaussian_filter(corr.astype(float), sigma=3)
        cb = (cs > 0.3).astype(np.uint8)

        # 시각화용 skeleton (항상 계산)
        skel = skeletonize(cb > 0).astype(np.uint8)

        # ── skeleton 보강: corridor 내부에 격자선으로 대체 ──────────────
        if skel_grid_spacing_m > 0:
            step_px = max(1, int(round(skel_grid_spacing_m / gr)))
            grid = np.zeros_like(skel)
            for row in range(step_px // 2, ih, step_px):
                grid[row, :] = 1
            for col in range(step_px // 2, iw, step_px):
                grid[:, col] = 1
            # corridor 내부만 남김 (기존 skeleton 제거, 격자선만 사용)
            skel = (grid & cb).astype(np.uint8)
            print(f"    Grid lines only (spacing={skel_grid_spacing_m}m, step={step_px}px)")

        skel_px_all = np.argwhere(skel > 0)
        print(f"    Skeleton (grid): {len(skel_px_all)} px")
        # ── 샘플링 후보 선택 ──────────────────────────────────────────
        if sample_mode == "corridor":
            # 벽에서 min_wall_dist_m 이상 떨어진 전체 자유 공간 샘플링
            cand_px = np.argwhere(dm > min_wall_dist_px)
            print(f"    Corridor candidates: {len(cand_px)} px "
                  f"(wall_dist>={samp.get('min_wall_dist_m',0.3):.2f}m)")
        else:
            # 기존 skeleton 모드
            cand_px = skel_px_all
            if len(cand_px) == 0:
                cand_px = np.argwhere(dm > (dmx * 0.5))

        if len(cand_px) == 0: print("    No positions"); continue

        swx = cand_px[:,1]*gr + xn
        swy = cand_px[:,0]*gr + yn

        # Greedy spacing 샘플링 (path_spacing 간격 유지)
        sel = []
        rem = list(range(len(cand_px)))
        np.random.seed(42)
        np.random.shuffle(rem)
        sel_xy = []
        for idx in rem:
            pos = np.array([swx[idx], swy[idx]])
            if all(np.linalg.norm(pos - q) >= ps for q in sel_xy):
                sel.append(idx)
                sel_xy.append(pos)
        print(f"    Sampled: {len(sel)} positions (mode={sample_mode}, spacing={ps}m)")

        cz = fz + ch
        for si in sel:
            for yi in range(ny):
                yaw = 2*np.pi*yi/ny
                forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
                up = np.array([0.0, 0.0, 1.0])
                right = np.cross(forward, up)
                right = right / np.linalg.norm(right)
                # Camera axes: cam_x=right, cam_y=down(-up), cam_z=forward
                R_cam = np.column_stack([right, -up, forward])
                if abs(pitch) > 1e-6:
                    Rx = np.array([[1,0,0],
                                   [0,np.cos(pitch),-np.sin(pitch)],
                                   [0,np.sin(pitch), np.cos(pitch)]])
                    R_cam = R_cam @ Rx
                T = np.eye(4)
                T[:3,:3] = R_cam
                T[:3,3] = [swx[si], swy[si], cz]
                all_vp.append({"id": vid, "pose": T, "floor": fi, "yaw": yaw})
                vid += 1

        debug_imgs[fi] = {
            "occupancy": occ, "closed": closed, "dist_map": dm, "corridor": cb,
            "skeleton": skel, "selected_px": cand_px[sel] if sel else np.array([]),
            "xn": xn, "yn": yn, "gr": gr, "fz": fz,
        }

    print(f"\n  Total viewpoints: {len(all_vp)}")

    nf = len(debug_imgs)
    if nf > 0:
        fig, axes = plt.subplots(nf, 6, figsize=(30, 5*nf))
        if nf == 1: axes = axes.reshape(1, -1)
        titles = ["1.Occupancy","2.Morphology","3.Dist transform","4.Corridor","5.Skeleton","6.Viewpoints"]
        for fi, dbg in debug_imgs.items():
            r = fi
            axes[r,0].imshow(dbg["occupancy"], cmap="gray", origin="lower")
            axes[r,1].imshow(dbg["closed"], cmap="gray", origin="lower")
            axes[r,2].imshow(dbg["dist_map"], cmap="hot", origin="lower")
            axes[r,3].imshow(dbg["corridor"], cmap="gray", origin="lower")
            vs = dbg["closed"].astype(float)*0.3; vs[dbg["skeleton"]>0] = 1.0
            axes[r,4].imshow(vs, cmap="gray", origin="lower")
            axes[r,5].imshow(dbg["occupancy"], cmap="gray", origin="lower", alpha=0.5)
            sy = np.argwhere(dbg["skeleton"]>0)
            if len(sy)>0: axes[r,5].scatter(sy[:,1],sy[:,0],c="cyan",s=0.3,alpha=0.3)
            sp = dbg["selected_px"]
            if len(sp)>0:
                axes[r,5].scatter(sp[:,1],sp[:,0],c="red",s=30,marker="x")
                for pt in sp:
                    for yi in range(ny):
                        ya = 2*np.pi*yi/ny
                        axes[r,5].arrow(pt[1],pt[0],8*np.cos(ya),8*np.sin(ya),
                                        head_width=2, head_length=1,
                                        fc="orange", ec="orange", alpha=0.6)
            for ci in range(6):
                axes[r,ci].set_title(titles[ci], fontsize=9)
                axes[r,ci].set_xticks([]); axes[r,ci].set_yticks([])
            axes[r,0].set_ylabel(f"Floor {fi}\nz={dbg['fz']:.2f}m", fontsize=11, fontweight="bold")
        fig.suptitle(f"Step 1: Free Path Corridor — {len(all_vp)} viewpoints", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "step1_viewpoints.png"), dpi=150); plt.close()

    fig2, ax2 = plt.subplots(1, 2, figsize=(16, 8))
    sub = max(1, len(points)//30000)
    ax2[0].scatter(points[::sub,0],points[::sub,1],c=points[::sub,2],s=0.3,cmap="viridis",alpha=0.3)
    vpp = np.array([v["pose"][:3,3] for v in all_vp])
    if len(vpp)>0:
        up_vp = vpp[::ny]
        ax2[0].scatter(up_vp[:,0],up_vp[:,1],c="red",s=15,marker="x",label=f"{len(up_vp)} pos")
    ax2[0].set_title("Top-down: map + viewpoints"); ax2[0].set_aspect("equal"); ax2[0].legend()
    ax2[1].scatter(points[::sub,0],points[::sub,2],c="gray",s=0.3,alpha=0.3)
    if len(vpp)>0: ax2[1].scatter(vpp[::ny,0],vpp[::ny,2],c="red",s=15,marker="x")
    for fz in floors:
        ax2[1].axhline(y=fz,   color="green", ls="--", alpha=0.5)
        ax2[1].axhline(y=fz+ch,color="blue",  ls="--", alpha=0.5)
    ax2[1].set_title("Side view"); fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "step1_viewpoints_3d.png"), dpi=150); plt.close()
    print(f"  Saved: step1_viewpoints.png, step1_viewpoints_3d.png")

    data = {"viewpoints": all_vp, "floor_levels": floors, "debug_images": debug_imgs}
    pickle.dump(data, open(os.path.join(output_dir, "step1_data.pkl"), "wb"))
    return all_vp


# =============================================================================
# Pkl 절약 헬퍼
# =============================================================================
_HEAVY_KEYS = ("rgb", "depth")

def _slim(rendered):
    """rendered 리스트에서 대용량 배열 키를 제거한 경량 복사본을 반환."""
    return [{k: v for k, v in r.items() if k not in _HEAVY_KEYS}
            for r in rendered]


# =============================================================================
# STEP 2: Render (O3D split-render)
# =============================================================================
def _fill_holes(rgb, dep, near_dist, near_fill_radius, fill_radius):
    """
    O3D 렌더 결과의 빈 픽셀(depth==0)을 가장 가까운 채워진 픽셀로 채움.
    - depth < near_dist 인 가까운 픽셀: near_fill_radius (px) 이내 채움
    - 그 외 먼 픽셀: fill_radius (px) 이내 채움
    """
    if near_fill_radius <= 0 and fill_radius <= 0:
        return rgb, dep

    from scipy.ndimage import distance_transform_edt
    dep_f = dep.astype(np.float32)
    empty = dep_f == 0
    if not np.any(empty) or np.all(empty):
        return rgb, dep

    dist, (ny, nx) = distance_transform_edt(empty, return_indices=True)
    nearest_depth = dep_f[ny, nx]                                   # 가장 가까운 채워진 픽셀의 depth
    max_fill = np.where(nearest_depth < near_dist,
                        near_fill_radius, fill_radius).astype(np.float32)
    fill = empty & (dist <= max_fill)

    rgb_out = rgb.copy()
    dep_out = dep_f.copy()
    rgb_out[fill] = rgb[ny[fill], nx[fill]]
    dep_out[fill] = nearest_depth[fill]
    return rgb_out, dep_out


def step2_render(ply_path, viewpoints, config, output_dir, step0_data=None):
    """
    Split-render: O3D OffscreenRenderer로 하단/상단을 순차 렌더링 후 합성.
    천장 포인트가 카메라를 가리는 문제를 해결하면서 천장도 이미지에 포함.

    ceiling_clip=True (기본):
      - pcd_lo: z < cam_z + ceiling_margin  (바닥~카메라 부근)
      - pcd_hi: z >= cam_z + ceiling_margin (천장 이상)
      - 두 이미지를 depth 기준으로 합성 (가까운 쪽 우선)
    ceiling_clip=False: 단일 렌더링

    hole filling (렌더 후 후처리):
      - near_dist (m) 이내 빈 픽셀: near_fill_radius px 이내 채움
      - 그 외 빈 픽셀: fill_radius px 이내 채움
    """
    print("\n" + "="*60 + "\nSTEP 2: Rendering (split-render)\n" + "="*60)
    if step0_data and "aligned_ply_path" in step0_data:
        rp = step0_data["aligned_ply_path"]
    else:
        rp = os.path.join(output_dir, "aligned_map.ply")
        if not os.path.exists(rp): rp = ply_path
    print(f"  From: {rp}")

    cam = config["camera"]
    rd  = os.path.join(output_dir, "rendered")
    os.makedirs(os.path.join(rd, "rgb"),   exist_ok=True)
    os.makedirs(os.path.join(rd, "depth"), exist_ok=True)

    pcd_full    = o3d.io.read_point_cloud(rp)
    points_full = np.asarray(pcd_full.points)
    has_color   = pcd_full.has_colors()
    colors_full = np.asarray(pcd_full.colors) if has_color else None
    print(f"  Full cloud: {len(points_full)} pts, color={has_color}")

    w, h = cam["width"], cam["height"]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    rend_cfg         = config.get("rendering", {})
    ceiling_margin   = rend_cfg.get("ceiling_margin", 0.5)
    enable_split     = rend_cfg.get("ceiling_clip", True)
    bscale           = rend_cfg.get("brightness_scale", 0.8)
    near_dist        = float(rend_cfg.get("near_dist",        3.0))
    near_fill_radius = float(rend_cfg.get("near_fill_radius", 5.0))
    fill_radius      = float(rend_cfg.get("fill_radius",      2.0))
    print(f"  Split-render: {enable_split}, margin={ceiling_margin}m")
    print(f"  Hole-fill: near_dist={near_dist}m  near_fill={near_fill_radius}px  fill={fill_radius}px")

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.point_size = rend_cfg.get("point_size", 1.0)
    mat.shader     = "defaultUnlit"

    # 렌더러는 clip_z가 바뀔 때만 새로 생성 (단일 층이면 전체 루프에서 1번)
    # remove_geometry 호출 없이 del → 새 인스턴스로 교체 → Filament 리소스 안전
    prev_clip_z = None
    ren_single = None
    ren_lo = None
    ren_hi = None

    rendered = []
    for i, vp in enumerate(viewpoints):
        pose      = vp["pose"]
        cam_z     = pose[2, 3]
        extrinsic = np.linalg.inv(pose)

        if not enable_split:
            if ren_single is None:
                ren_single = o3d.visualization.rendering.OffscreenRenderer(w, h)
                ren_single.scene.set_background([0.0, 0.0, 0.0, 1.0])
                ren_single.scene.add_geometry("map", pcd_full, mat)
            ren_single.setup_camera(intr, extrinsic)
            rgb = np.asarray(ren_single.render_to_image()).copy()
            dep = np.asarray(ren_single.render_to_depth_image(z_in_view_space=True)).copy()
        else:
            clip_z = cam_z + ceiling_margin

            # clip_z 변경 시: 기존 렌더러 del 후 새로 생성
            if prev_clip_z is None or abs(clip_z - prev_clip_z) > 0.01:
                if ren_lo is not None: del ren_lo
                if ren_hi is not None: del ren_hi

                mask_lo = points_full[:, 2] < clip_z
                mask_hi = ~mask_lo

                pcd_lo = o3d.geometry.PointCloud()
                pcd_lo.points = o3d.utility.Vector3dVector(points_full[mask_lo])
                if has_color:
                    pcd_lo.colors = o3d.utility.Vector3dVector(colors_full[mask_lo])

                pcd_hi = o3d.geometry.PointCloud()
                pcd_hi.points = o3d.utility.Vector3dVector(points_full[mask_hi])
                if has_color:
                    pcd_hi.colors = o3d.utility.Vector3dVector(colors_full[mask_hi])

                ren_lo = o3d.visualization.rendering.OffscreenRenderer(w, h)
                ren_lo.scene.set_background([0.0, 0.0, 0.0, 1.0])
                ren_lo.scene.add_geometry("map", pcd_lo, mat)

                ren_hi = o3d.visualization.rendering.OffscreenRenderer(w, h)
                ren_hi.scene.set_background([0.0, 0.0, 0.0, 1.0])
                ren_hi.scene.add_geometry("map", pcd_hi, mat)

                prev_clip_z = clip_z
                print(f"  [split] clip_z={clip_z:.2f}: lo={mask_lo.sum()}, hi={mask_hi.sum()}")

            # 하단 렌더링
            ren_lo.setup_camera(intr, extrinsic)
            rgb_lo = np.asarray(ren_lo.render_to_image()).copy()
            dep_lo = np.asarray(ren_lo.render_to_depth_image(z_in_view_space=True)).copy()

            # 상단 렌더링
            ren_hi.setup_camera(intr, extrinsic)
            rgb_hi = np.asarray(ren_hi.render_to_image()).copy()
            dep_hi = np.asarray(ren_hi.render_to_depth_image(z_in_view_space=True)).copy()

            if rgb_lo.dtype != np.uint8:
                rgb_lo = (np.clip(rgb_lo.astype(np.float32), 0, 1) * 255).astype(np.uint8)
            if rgb_hi.dtype != np.uint8:
                rgb_hi = (np.clip(rgb_hi.astype(np.float32), 0, 1) * 255).astype(np.uint8)

            dep_lo_f = dep_lo.astype(np.float32)
            dep_hi_f = dep_hi.astype(np.float32)

            INF = 1e6
            dep_lo_f[(dep_lo_f <= 0) | (dep_lo_f > 100)] = INF
            dep_hi_f[(dep_hi_f <= 0) | (dep_hi_f > 100)] = INF

            # 합성: depth가 가까운 쪽 우선
            use_hi = dep_hi_f < dep_lo_f
            rgb = rgb_lo.copy()
            rgb[use_hi] = rgb_hi[use_hi]

            # 하단이 배경(검정)이면 상단으로 채움
            lo_bg = (rgb_lo.max(axis=2) < 5)
            hi_ok = (rgb_hi.max(axis=2) > 5)
            rgb[lo_bg & hi_ok] = rgb_hi[lo_bg & hi_ok]

            dep = np.minimum(dep_lo_f, dep_hi_f)
            dep[dep >= INF] = 0

        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb.astype(np.float32), 0, 1) * 255).astype(np.uint8)
        if bscale != 1.0:
            rgb = np.clip(rgb.astype(np.float32) * bscale, 0, 255).astype(np.uint8)

        # Hole filling: 근거리는 크게, 원거리는 작게
        rgb, dep = _fill_holes(rgb, dep, near_dist, near_fill_radius, fill_radius)

        if i == 0:
            print(f"  [debug] rgb mean={rgb.mean():.1f}  dep nonzero={np.count_nonzero(dep)}/{dep.size}")

        rp_ = os.path.join(rd, "rgb",   f"{vp['id']:06d}.png")
        dp_ = os.path.join(rd, "depth", f"{vp['id']:06d}.npy")
        cv2.imwrite(rp_, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(dp_, dep.astype(np.float32))
        # rgb/depth 배열은 디스크에 저장됐으므로 메모리에 쌓지 않음 (크래시 방지)
        rendered.append({
            "id": vp["id"], "pose": pose,
            "rgb_path": rp_, "depth_path": dp_,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(viewpoints)}")

    print(f"  Rendered: {len(rendered)}")
    ns = min(8, len(rendered))
    idx = np.linspace(0, len(rendered)-1, ns, dtype=int)
    fig, ax = plt.subplots(2, ns, figsize=(4*ns, 8))
    if ns == 1: ax = ax.reshape(2, 1)
    for c, ii in enumerate(idx):
        r = rendered[ii]
        rgb_v = cv2.cvtColor(cv2.imread(r["rgb_path"]), cv2.COLOR_BGR2RGB)
        ax[0,c].imshow(rgb_v)
        ax[0,c].set_title(f"RGB #{r['id']}", fontsize=9); ax[0,c].axis("off")
        dv = np.load(r["depth_path"])
        dv[dv > cam.get("depth_max", 10)] = 0
        ax[1,c].imshow(dv, cmap="plasma")
        ax[1,c].set_title("Depth", fontsize=9); ax[1,c].axis("off")
    fig.suptitle(f"Step 2: Rendered — {len(rendered)} images (split={enable_split})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "step2_rendered.png"), dpi=150); plt.close()
    print(f"  Saved: step2_rendered.png")
    # rendered는 이미 경량 (rgb/depth 배열 없음) — _slim 불필요하지만 호환성 유지
    pickle.dump(rendered, open(os.path.join(output_dir, "step2_data.pkl"), "wb"))
    return rendered


# =============================================================================
# Regional Descriptor 공통 헬퍼
# =============================================================================
def build_regional_descriptor(rgb_img, extract_fn, grid_rows, grid_cols):
    """
    이미지를 grid_rows × grid_cols 로 분할하여 각 셀 + 전체 이미지의
    descriptor를 concatenate → 공간 정보 보존 descriptor 반환.

    최종 dim = base_dim × (grid_rows × grid_cols + 1)

    Args:
        rgb_img    : (H, W, 3) uint8 RGB
        extract_fn : rgb_img(H,W,3) → 1D numpy array (base descriptor)
        grid_rows  : 수직 분할 수
        grid_cols  : 수평 분할 수 (좌/우 구분에는 cols=3 권장)

    Returns:
        (N,) float32, L2-normalized
    """
    H, W = rgb_img.shape[:2]

    # 전체 이미지 descriptor
    parts = [extract_fn(rgb_img)]

    # 그리드 셀 descriptor
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0 = int(H * r / grid_rows)
            y1 = int(H * (r + 1) / grid_rows)
            x0 = int(W * c / grid_cols)
            x1 = int(W * (c + 1) / grid_cols)
            cell = rgb_img[y0:y1, x0:x1]
            parts.append(extract_fn(cell))

    desc = np.concatenate(parts).astype(np.float32)
    norm = np.linalg.norm(desc)
    return desc / (norm + 1e-8)


# =============================================================================
# STEP 3: Global Descriptors  (DINOv2 + VLAD, AnyLoc style)
# =============================================================================
def _dino_preprocess(img_rgb, img_size):
    """RGB uint8 → normalized tensor (1,3,H,W)"""
    import torch
    t = cv2.resize(img_rgb, (img_size, img_size)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    t = (t - mean) / std
    return torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0)


def _extract_dino_patches(img_rgb, model, dev, img_size):
    """DINOv2 patch feature 추출: (n_patches, feat_dim) float32"""
    import torch
    t = _dino_preprocess(img_rgb, img_size).to(dev)
    with torch.no_grad():
        feats = model.get_intermediate_layers(t, n=1)[0]  # (1, P, D)
    return feats[0].cpu().numpy().astype(np.float32)       # (P, D)


def _compute_vlad(patch_feats, centers):
    """
    VLAD descriptor 계산 (intra-normalized + L2 global)
    patch_feats: (P, D), centers: (K, D)
    """
    K, D = centers.shape
    # L2 distance → nearest cluster
    diff = patch_feats[:, None, :] - centers[None, :, :]   # (P, K, D)
    dists = np.linalg.norm(diff, axis=2)                    # (P, K)
    labels = dists.argmin(axis=1)                           # (P,)

    vlad = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        mask = labels == k
        if mask.any():
            vlad[k] = (patch_feats[mask] - centers[k]).sum(axis=0)

    # intra-normalize per cluster
    norms = np.linalg.norm(vlad, axis=1, keepdims=True) + 1e-8
    vlad /= norms

    # flatten + L2 normalize
    vlad = vlad.flatten()
    vlad /= (np.linalg.norm(vlad) + 1e-8)
    return vlad


def _megaloc_preprocess(img_rgb, resize=518):
    """MegaLoc 입력 전처리: (H,W,3) uint8 → [1,3,resize,resize] float tensor (ImageNet norm)
    resize=518: MegaLoc 학습 해상도 (DINOv2 ViT-B/14, 37×14=518). 다른 크기는 품질 저하."""
    import torch, torchvision.transforms as T
    tf = T.Compose([
        T.ToPILImage(),
        T.Resize((resize, resize)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(img_rgb).unsqueeze(0)   # [1, 3, resize, resize]


def _extract_megaloc_desc(img_rgb, model, dev):
    """MegaLoc 전역 디스크립터 추출: (megaloc_dim,) float32, L2-normalized"""
    import torch
    t = _megaloc_preprocess(img_rgb).to(dev)
    with torch.no_grad():
        desc = model(t).cpu().numpy().flatten()
    return desc.astype(np.float32)


def step3_global_desc(rendered, config, output_dir):
    """
    DINOv2 patch features → VLAD aggregation (AnyLoc 방식).
    1) 모든 렌더링 이미지의 patch features 수집
    2) MiniBatchKMeans로 VLAD vocabulary (cluster centers) 학습
    3) 각 이미지에 대해 VLAD descriptor 계산
    최종 dim = vlad_clusters × dino_feat_dim
    """
    import torch
    fc     = config["features"]
    method = fc.get("global_desc_method", "megaloc")
    dev    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── MegaLoc 분기 ─────────────────────────────────────────────────────
    if method == "megaloc":
        print("\n" + "="*60 + "\nSTEP 3: Global descriptors (MegaLoc)\n" + "="*60)
        print("  Loading MegaLoc from torch.hub …")
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        model.eval().to(dev)
        print(f"  MegaLoc loaded  →  descriptor dim={fc.get('megaloc_dim', 8448)}")

        for i, r in enumerate(rendered):
            img_bgr = cv2.imread(r["rgb_path"])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            r["global_descriptor"] = _extract_megaloc_desc(img_rgb, model, dev)
            r["global_desc_method"] = "megaloc"
            if (i + 1) % 50 == 0 or i == 0:
                print(f"    {i+1}/{len(rendered)}")

        # 시각화: similarity matrix
        nv = min(100, len(rendered))
        if nv >= 2:
            idx = np.linspace(0, len(rendered)-1, nv, dtype=int)
            dm  = np.array([rendered[i]["global_descriptor"] for i in idx], dtype=np.float32)
            sim = dm @ dm.T
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            ax.imshow(sim, cmap="hot", vmin=0, vmax=1)
            ax.set_title(f"Step 3: MegaLoc similarity  (dim={dm.shape[1]})", fontsize=13)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "step3_global_desc.png"), dpi=150); plt.close()
            print(f"  Saved: step3_global_desc.png")

        slim = _slim(rendered)
        pickle.dump({"rendered": slim},
                    open(os.path.join(output_dir, "step3_data.pkl"), "wb"))
        return rendered, None   # centers=None (VLAD vocabulary 불필요)

    # ── AnyLoc 분기 (기존 DINOv2+VLAD) ─────────────────────────────────
    print("\n" + "="*60 + "\nSTEP 3: Global descriptors (DINOv2 + VLAD)\n" + "="*60)
    from sklearn.cluster import MiniBatchKMeans

    dino_name = fc.get("dino_model", "dinov2_vitb14")
    img_size  = int(fc.get("dino_img_size", 322))
    n_clusters = int(fc.get("vlad_clusters", 64))

    # ── DINOv2 로드 ───────────────────────────────────────────────────
    print(f"  Loading DINOv2: {dino_name}  (img_size={img_size})")
    model = torch.hub.load("facebookresearch/dinov2", dino_name, pretrained=True)
    model.eval().to(dev)
    feat_dim = model.embed_dim
    vlad_dim = n_clusters * feat_dim
    print(f"  feat_dim={feat_dim}, clusters={n_clusters}  →  VLAD dim={vlad_dim}")

    # ── 1. Patch features 전체 수집 ───────────────────────────────────
    print(f"  Extracting patch features from {len(rendered)} images …")
    all_patches = []
    per_image_patches = []
    for i, r in enumerate(rendered):
        rgb = r["rgb"] if isinstance(r.get("rgb"), np.ndarray) \
              else cv2.cvtColor(cv2.imread(r["rgb_path"]), cv2.COLOR_BGR2RGB)
        pf = _extract_dino_patches(rgb, model, dev, img_size)
        per_image_patches.append(pf)
        all_patches.append(pf)
        if (i+1) % 100 == 0:
            print(f"    {i+1}/{len(rendered)}")

    # ── 2. VLAD vocabulary 학습 ───────────────────────────────────────
    all_feats = np.vstack(all_patches)
    max_sample = 200_000
    if len(all_feats) > max_sample:
        idx = np.random.choice(len(all_feats), max_sample, replace=False)
        sample = all_feats[idx]
    else:
        sample = all_feats
    print(f"  Fitting VLAD vocabulary: {n_clusters} clusters on {len(sample)} patches …")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                             n_init=5, batch_size=4096, max_iter=300)
    kmeans.fit(sample)
    centers = kmeans.cluster_centers_.astype(np.float32)
    print(f"  Vocabulary fitted.")

    # ── 3. VLAD descriptor 계산 ───────────────────────────────────────
    for i, (r, pf) in enumerate(zip(rendered, per_image_patches)):
        r["global_descriptor"] = _compute_vlad(pf, centers)
        r["vlad_vocab"] = centers   # step4에서 db에 저장하기 위해 첫 entry에만 있으면 됨
        r["global_desc_method"] = "anyloc"

    # ── 시각화: similarity matrix + PCA ───────────────────────────────
    nv  = min(100, len(rendered))
    idx = np.linspace(0, len(rendered)-1, nv, dtype=int)
    dm  = np.array([rendered[i]["global_descriptor"] for i in idx], dtype=np.float32)
    dm /= np.linalg.norm(dm, axis=1, keepdims=True) + 1e-8

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    im = ax[0].imshow(dm @ dm.T, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax[0].set_title(f"Similarity {nv}×{nv}")
    plt.colorbar(im, ax=ax[0], fraction=0.046)

    from sklearn.decomposition import PCA
    c2  = PCA(n_components=2).fit_transform(dm)
    ps_ = np.array([rendered[i]["pose"][:3, 3] for i in idx])
    ax[1].scatter(c2[:, 0], c2[:, 1], c=ps_[:, 0], cmap="viridis", s=10)
    ax[1].set_title("PCA (color=X)")

    fig.suptitle(f"Step 3: Global descriptors (DINOv2+VLAD, K={n_clusters}, dim={vlad_dim})",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "step3_global_desc.png"), dpi=150); plt.close()
    print(f"  Saved: step3_global_desc.png")

    slim = _slim(rendered)
    pickle.dump({"rendered": slim, "vlad_centers": centers},
                open(os.path.join(output_dir, "step3_data.pkl"), "wb"))
    return rendered, centers


# =============================================================================
# STEP 4: Build Database  (KDTree indexing)
# =============================================================================
def step4_build_db(rendered, output_dir):
    """
    오프라인 마지막 단계: global descriptors를 KDTree로 인덱싱.

    저장 내용:
      - global_descs: (N, D) float32, L2 정규화 완료
      - kdtree: scipy.spatial.KDTree
      - entries: list of dict  (id, pose, rgb_path, depth_path)
    """
    from scipy.spatial import KDTree

    print("\n" + "="*60 + "\nSTEP 4: Build database (KDTree)\n" + "="*60)

    global_descs = []
    entries = []

    for r in rendered:
        gd = r.get("global_descriptor")
        if gd is None:
            print(f"  WARNING: #{r['id']} global_descriptor 없음, 건너뜀")
            continue
        global_descs.append(gd)
        entries.append({
            "id":         r["id"],
            "pose":       r["pose"],
            "rgb_path":   r["rgb_path"],
            "depth_path": r["depth_path"],
        })

    global_descs = np.array(global_descs, dtype=np.float32)
    norms = np.linalg.norm(global_descs, axis=1, keepdims=True) + 1e-8
    global_descs_normed = (global_descs / norms).astype(np.float32)

    kdtree = KDTree(global_descs_normed)

    # VLAD vocabulary + PCA model: rendered[0]에서 꺼냄 (anyloc 방식만 존재)
    vlad_centers = None
    pca_model    = None
    global_desc_method = "anyloc"
    for r in rendered:
        if r.get("global_desc_method"):
            global_desc_method = r["global_desc_method"]
        if r.get("vlad_vocab") is not None:
            vlad_centers = r["vlad_vocab"]
            pca_model    = r.get("pca_model")
            break

    db = {"global_descs": global_descs_normed, "kdtree": kdtree,
          "entries": entries, "vlad_centers": vlad_centers, "pca_model": pca_model,
          "global_desc_method": global_desc_method}
    db_pkl = os.path.join(output_dir, "step4_database.pkl")
    db_npz = os.path.join(output_dir, "step4_database.npz")
    pickle.dump(db, open(db_pkl, "wb"))
    np.savez(db_npz, global_descs=global_descs_normed,
             poses=np.array([e["pose"] for e in entries]))

    print(f"  DB entries : {len(entries)}")
    print(f"  Descriptor : shape={global_descs_normed.shape}")
    print(f"  KDTree     : {kdtree.n} nodes built")
    print(f"  Saved: {db_pkl}")

    nv = min(200, len(entries))
    idx = np.linspace(0, len(entries)-1, nv, dtype=int)
    sub_d = global_descs_normed[idx]
    sim_mat = sub_d @ sub_d.T
    poses_sub = np.array([entries[i]["pose"][:3,3] for i in idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im = axes[0].imshow(sim_mat, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[0].set_title(f"Global desc cosine similarity ({nv}×{nv})")
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].scatter(poses_sub[:,0], poses_sub[:,1],
                    c=np.arange(nv), cmap="viridis", s=10)
    axes[1].set_title("DB viewpoint positions (top-down)"); axes[1].set_aspect("equal")
    fig.suptitle(f"Step 4: Database — {len(entries)} entries, KDTree ready", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "step4_database.png"), dpi=150); plt.close()
    print(f"  Saved: step4_database.png")
    return db


# =============================================================================
# STEP 5: Global Retrieval  (query → KDTree → top-K candidates)
# =============================================================================
def step5_retrieval(query_image_path, db, config, output_dir):
    """
    1) Query 이미지 → global descriptor
    2) KDTree.query() → Top-K nearest neighbors (cosine similarity)
    3) 후보 리스트를 step6_match에 전달
    """
    import torch
    print("\n" + "="*60 + "\nSTEP 5: Global retrieval\n" + "="*60)
    fc    = config["features"]
    onl   = config.get("online", {})
    top_k = onl.get("top_k", 5)
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Query 이미지 로드 ──────────────────────────────────────────────
    if query_image_path and os.path.exists(query_image_path):
        query_rgb = cv2.cvtColor(cv2.imread(query_image_path), cv2.COLOR_BGR2RGB)
        gt_entry  = None
        print(f"  Query: {query_image_path} ({query_rgb.shape[1]}×{query_rgb.shape[0]})")
    else:
        qi = len(db["entries"]) // 3
        gt_entry  = db["entries"][qi]
        query_rgb = cv2.cvtColor(cv2.imread(gt_entry["rgb_path"]), cv2.COLOR_BGR2RGB)
        print(f"  Query: DB entry #{gt_entry['id']} (self-test)")

    # ── Query descriptor (방법은 DB에 저장된 global_desc_method 기준) ──
    _cache = step5_retrieval.__dict__
    global_desc_method = db.get("global_desc_method", "anyloc")

    if global_desc_method == "megaloc":
        print(f"  Method: MegaLoc  (dim={fc.get('megaloc_dim', 8448)})")
        if "_megaloc_model" not in _cache:
            print("  Loading MegaLoc …")
            model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
            model.eval().to(dev)
            _cache["_megaloc_model"] = model
        model = _cache["_megaloc_model"]
        q_gd_norm = _extract_megaloc_desc(query_rgb, model, dev)
        q_gd_norm = q_gd_norm / (np.linalg.norm(q_gd_norm) + 1e-8)

    else:   # anyloc (DINOv2 + VLAD)
        vlad_centers = db.get("vlad_centers")
        if vlad_centers is None:
            raise RuntimeError("DB에 vlad_centers가 없습니다. step3→step4를 재실행하세요.")
        dino_name  = fc.get("dino_model", "dinov2_vitb14")
        img_size   = int(fc.get("dino_img_size", 322))
        n_clusters = vlad_centers.shape[0]
        feat_dim   = vlad_centers.shape[1]
        print(f"  Method: AnyLoc  DINOv2={dino_name}  VLAD K={n_clusters}  dim={n_clusters*feat_dim}")
        if "_dino_model" not in _cache or _cache.get("_dino_name") != dino_name:
            model = torch.hub.load("facebookresearch/dinov2", dino_name, pretrained=True)
            model.eval().to(dev)
            _cache["_dino_model"] = model
            _cache["_dino_name"]  = dino_name
            print(f"  Loaded DINOv2: {dino_name}")
        model = _cache["_dino_model"]
        q_patches = _extract_dino_patches(query_rgb, model, dev, img_size)
        q_gd_norm = _compute_vlad(q_patches, vlad_centers)

    # ── KDTree retrieval ───────────────────────────────────────────────
    dists, idxs = db["kdtree"].query(q_gd_norm, k=top_k)
    cos_sims    = 1.0 - dists**2 / 2.0
    candidates  = [db["entries"][i] for i in idxs]

    print(f"  Top-{top_k} results:")
    for rank, (cand, sim) in enumerate(zip(candidates, cos_sims)):
        gt_str = ""
        if gt_entry:
            d = np.linalg.norm(np.array(cand["pose"])[:3,3]
                               - np.array(gt_entry["pose"])[:3,3])
            gt_str = f"  GT_dist={d:.2f}m"
        print(f"    Rank{rank+1}: #{cand['id']}  sim={sim:.4f}{gt_str}")

    # ── 시각화: 1행(RGB) ──────────────────────────────────────────────
    n_show  = min(top_k, 5) + 1
    fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 4))
    if n_show == 1: axes = [axes]

    # ── Query 열 ──────────────────────────────────────────────────────
    axes[0].imshow(query_rgb)
    axes[0].set_title("QUERY", color="blue", fontsize=10)
    axes[0].axis("off")

    # ── Candidate 열 ──────────────────────────────────────────────────
    for rank, (cand, sim) in enumerate(zip(candidates[:n_show-1], cos_sims[:n_show-1])):
        col = "green" if rank == 0 else "orange"
        ref_rgb = cv2.cvtColor(cv2.imread(cand["rgb_path"]), cv2.COLOR_BGR2RGB)
        p = np.array(cand["pose"])[:3, 3]
        axes[rank+1].imshow(ref_rgb)
        axes[rank+1].set_title(
            f"Rank{rank+1} #{cand['id']}  sim={sim:.3f}\n"
            f"({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})",
            color=col, fontsize=8)
        axes[rank+1].axis("off")

    fig.suptitle(f"Step 5: KDTree Retrieval — Top-{top_k}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,"step5_retrieval.png"), dpi=150); plt.close()
    print(f"  Saved: step5_retrieval.png")

    data = {
        "query_rgb":        query_rgb,
        "query_gd_norm":    q_gd_norm,
        "candidates":       candidates,
        "cos_sims":         cos_sims.tolist(),
        "gt_entry":         gt_entry,
        "query_image_path": query_image_path,
    }
    pickle.dump(data, open(os.path.join(output_dir,"step5_data.pkl"),"wb"))
    return data


# =============================================================================
# STEP 6: LoFTR Matching  (detector-free, top-K candidates → best match)
# =============================================================================
def step6_match(step5_data, config, output_dir):
    """
    EfficientLoFTR detector-free matching.

    - query image vs top-K candidates 각각 EfficientLoFTR 수행
    - confidence threshold 필터링
    - 가장 많은 inlier를 가진 candidate를 best match로 선택
    - 출력: matched_q_kps, matched_r_kps, confs, best_ref_entry

    EfficientLoFTR (CVPR 2024):
      - 입력: (1,1,H,W) grayscale [0,1], H/W는 32의 배수
      - 출력: mkpts0_f (N,2), mkpts1_f (N,2), mconf (N,)
    """
    import sys, torch

    # EfficientLoFTR repo를 path에 추가
    eloftr_root = os.path.join(os.path.dirname(__file__), "..", "third_party", "EfficientLoFTR")
    eloftr_root = os.path.abspath(eloftr_root)
    if eloftr_root not in sys.path:
        sys.path.insert(0, eloftr_root)

    from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

    print("\n" + "="*60 + "\nSTEP 6: EfficientLoFTR matching\n" + "="*60)

    fc          = config["features"]
    onl         = config.get("online", {})
    dev         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_thresh = float(fc.get("match_conf_thresh", 0.2))

    query_rgb  = step5_data["query_rgb"]
    candidates = step5_data["candidates"]

    # ── EfficientLoFTR 로드 ───────────────────────────────────────────
    ckpt_path = fc.get("eloftr_ckpt",
        os.path.join(eloftr_root, "weights", "ELoFTR", "weights", "eloftr_outdoor.ckpt"))
    use_opt   = fc.get("eloftr_opt", False)   # True=빠른 추론, False=최고 품질

    _cfg = (opt_default_cfg if use_opt else full_default_cfg).copy()
    matcher = LoFTR(config=_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    matcher.load_state_dict(state["state_dict"])
    matcher = reparameter(matcher).eval().to(dev)
    print(f"  EfficientLoFTR: ckpt={os.path.basename(ckpt_path)}, "
          f"mode={'opt' if use_opt else 'full'}, device={dev}")

    # 이전 step 잔여 GPU 메모리 해제
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    max_dim = int(fc.get("eloftr_max_dim", 840))  # 리사이즈 최대 해상도 (32 배수 권장)

    def to_gray_tensor(rgb_img):
        """RGB ndarray → (1,1,H,W) float32 [0,1], max_dim 이하로 리사이즈 후 32 배수 패딩
        Returns: (tensor, scale_x, scale_y)
          scale_x/y: 리사이즈 공간 → 원본 공간 변환 배율 (kp_orig = kp_resized * scale)
        """
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        orig_h, orig_w = gray.shape
        h, w = orig_h, orig_w
        # max_dim 초과 시 비율 유지하며 축소
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            nw = int(w * s); nh = int(h * s)
            gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
            h, w = gray.shape
        scale_x = orig_w / w  # 리사이즈 → 원본 배율
        scale_y = orig_h / h
        # 32의 배수로 패딩
        ph = ((h + 31) // 32) * 32
        pw = ((w + 31) // 32) * 32
        if ph != h or pw != w:
            padded = np.zeros((ph, pw), dtype=np.float32)
            padded[:h, :w] = gray
            gray = padded
        return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(dev), scale_x, scale_y

    q_tensor, q_scale_x, q_scale_y = to_gray_tensor(query_rgb)

    best_mkpts_q = np.zeros((0, 2))
    best_mkpts_r = np.zeros((0, 2))
    best_confs   = np.array([])
    best_cand    = candidates[0]
    best_ref_rgb = cv2.cvtColor(cv2.imread(candidates[0]["rgb_path"]), cv2.COLOR_BGR2RGB)
    best_n       = -1

    all_match_counts = []

    for rank, cand in enumerate(candidates):
        ref_rgb  = cv2.cvtColor(cv2.imread(cand["rgb_path"]), cv2.COLOR_BGR2RGB)
        r_tensor, r_scale_x, r_scale_y = to_gray_tensor(ref_rgb)

        try:
            batch = {"image0": q_tensor, "image1": r_tensor}
            with torch.no_grad():
                matcher(batch)
            mkpts0 = batch["mkpts0_f"].cpu().numpy()  # (N, 2) — 리사이즈 공간
            mkpts1 = batch["mkpts1_f"].cpu().numpy()  # (N, 2) — 리사이즈 공간
            confs  = batch["mconf"].cpu().numpy()      # (N,)
        except Exception as e:
            print(f"  Rank{rank+1} ELoFTR failed: {e}")
            all_match_counts.append(0)
            continue

        # ── 원본 해상도 좌표로 역변환 ─────────────────────────────────
        mkpts0 = mkpts0 * np.array([q_scale_x, q_scale_y], dtype=np.float32)
        mkpts1 = mkpts1 * np.array([r_scale_x, r_scale_y], dtype=np.float32)

        mask   = confs >= conf_thresh
        n_good = int(mask.sum())
        all_match_counts.append(n_good)
        print(f"  Rank{rank+1} #{cand['id']}: {len(mkpts0)} raw → {n_good} conf≥{conf_thresh}")

        if n_good > best_n:
            best_n       = n_good
            best_mkpts_q = mkpts0[mask]
            best_mkpts_r = mkpts1[mask]
            best_confs   = confs[mask]
            best_cand    = cand
            best_ref_rgb = ref_rgb

    print(f"\n  Best: #{best_cand['id']}  matches={best_n}")

    # ── 시각화: query + best_ref matching lines ────────────────────────
    h1, w1 = query_rgb.shape[:2]; h2, w2 = best_ref_rgb.shape[:2]
    th = max(h1, h2)
    sq, sr = 1.0, 1.0
    if h1 != th:
        sq = th / h1
        query_rgb    = cv2.resize(query_rgb,    (int(w1*sq), th))
    if h2 != th:
        sr = th / h2
        best_ref_rgb = cv2.resize(best_ref_rgb, (int(w2*sr), th))
    h1, w1 = query_rgb.shape[:2]; h2, w2 = best_ref_rgb.shape[:2]
    canvas = np.concatenate([query_rgb, best_ref_rgb], axis=1)
    mkpts_q_v = best_mkpts_q * sq
    mkpts_r_v = best_mkpts_r * sr

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.imshow(canvas)
    if len(mkpts_q_v) > 0:
        cmap_v = plt.cm.RdYlGn(best_confs / (best_confs.max() + 1e-8))
        step_v = max(1, len(mkpts_q_v) // 200)
        for i in range(0, len(mkpts_q_v), step_v):
            ax.plot([mkpts_q_v[i,0], mkpts_r_v[i,0]+w1],
                    [mkpts_q_v[i,1], mkpts_r_v[i,1]],
                    c=cmap_v[i], alpha=0.5, linewidth=0.8)
        ax.scatter(mkpts_q_v[:,0], mkpts_q_v[:,1], c="cyan",   s=8, zorder=3)
        ax.scatter(mkpts_r_v[:,0]+w1, mkpts_r_v[:,1], c="yellow", s=8, zorder=3)

    match_summary = "  |  ".join(
        f"R{r+1}:{n}" for r, n in enumerate(all_match_counts))
    ax.set_title(f"Step 6: EfficientLoFTR — best=#{best_cand['id']} ({best_n} matches)\n"
                 f"{match_summary}")
    ax.axis("off"); fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "step6_matching.png"), dpi=150); plt.close()
    print(f"  Saved: step6_matching.png")

    # ── match counts bar chart ─────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors_bar = ["green" if c["id"]==best_cand["id"] else "steelblue"
                  for c in candidates[:len(all_match_counts)]]
    ax2.bar([f"R{r+1}\n#{c['id']}" for r, c in enumerate(candidates[:len(all_match_counts)])],
            all_match_counts, color=colors_bar)
    ax2.set_ylabel("Matches (conf≥{:.2f})".format(conf_thresh))
    ax2.set_title("EfficientLoFTR match counts per candidate")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "step6_match_counts.png"), dpi=150); plt.close()

    data = {
        "matched_q_kps":    best_mkpts_q,
        "matched_r_kps":    best_mkpts_r,
        "confidences":      best_confs,
        "best_cand":        best_cand,
        "query_rgb":        query_rgb,
        "ref_rgb":          best_ref_rgb,
        "all_match_counts": all_match_counts,
        "candidates":       candidates,
    }
    pickle.dump(data, open(os.path.join(output_dir,"step6_data.pkl"),"wb"))
    return data


# =============================================================================
# STEP 6a: 배치 retrieval + matching 시각화
#   query_dir 내 모든 이미지 → 각각 step5(retrieval) + LoFTR → best match PNG
#   결과는 output_dir/step6a_results/ 에 저장
# =============================================================================
def step6a_match_viz(query_dir, db, config, output_dir):
    """
    query_dir 내 모든 이미지에 대해:
      1) MegaLoc/AnyLoc retrieval → top-K candidates
      2) EfficientLoFTR matching → best candidate 선택
      3) query + best_ref match line 이미지를 step6a_results/<query_stem>.png 로 저장
    """
    import sys, torch

    eloftr_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "third_party", "EfficientLoFTR"))
    if eloftr_root not in sys.path:
        sys.path.insert(0, eloftr_root)
    from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

    print("\n" + "="*60 + "\nSTEP 6a: Batch retrieval + match viz\n" + "="*60)
    print(f"  Query dir : {query_dir}")

    fc          = config["features"]
    dev         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_thresh = float(fc.get("match_conf_thresh", 0.2))
    max_dim     = int(fc.get("eloftr_max_dim", 840))
    onl         = config.get("online", {})
    top_k       = onl.get("top_k", 5)

    # ── 쿼리 이미지 목록 ──────────────────────────────────────────────
    query_files = sorted(
        glob.glob(os.path.join(query_dir, "*.jpg")) +
        glob.glob(os.path.join(query_dir, "*.png")) +
        glob.glob(os.path.join(query_dir, "*.jpeg"))
    )
    if not query_files:
        print(f"  ERROR: {query_dir} 에 이미지 없음"); return
    print(f"  {len(query_files)} query images found")

    # ── Retrieval 모델 로드 ───────────────────────────────────────────
    global_desc_method = db.get("global_desc_method", "anyloc")
    retr_model = None
    if global_desc_method == "megaloc":
        print("  Loading MegaLoc …")
        retr_model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        retr_model.eval().to(dev)

    # ── EfficientLoFTR 로드 ───────────────────────────────────────────
    ckpt_path = fc.get("eloftr_ckpt",
        os.path.join(eloftr_root, "weights", "ELoFTR", "weights", "eloftr_outdoor.ckpt"))
    use_opt = fc.get("eloftr_opt", False)
    _cfg = (opt_default_cfg if use_opt else full_default_cfg).copy()
    matcher = LoFTR(config=_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    matcher.load_state_dict(state["state_dict"])
    matcher = reparameter(matcher).eval().to(dev)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  EfficientLoFTR loaded  (device={dev})")

    def to_gray_tensor(rgb_img):
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        oh, ow = gray.shape
        h, w = oh, ow
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            gray = cv2.resize(gray, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
            h, w = gray.shape
        sx, sy = ow / w, oh / h
        ph = ((h+31)//32)*32; pw = ((w+31)//32)*32
        if ph != h or pw != w:
            pad = np.zeros((ph, pw), dtype=np.float32); pad[:h, :w] = gray; gray = pad
        return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(dev), sx, sy

    save_dir = os.path.join(output_dir, "step6a_results")
    os.makedirs(save_dir, exist_ok=True)

    for qi, qpath in enumerate(query_files):
        stem = os.path.splitext(os.path.basename(qpath))[0]
        print(f"\n  [{qi+1}/{len(query_files)}] {stem}")

        query_rgb = cv2.cvtColor(cv2.imread(qpath), cv2.COLOR_BGR2RGB)

        # ── Retrieval ─────────────────────────────────────────────────
        if global_desc_method == "megaloc":
            q_gd = _extract_megaloc_desc(query_rgb, retr_model, dev)
        else:
            # AnyLoc
            vlad_centers = db.get("vlad_centers")
            img_size = int(fc.get("dino_img_size", 322))
            dino_name = fc.get("dino_model", "dinov2_vitb14")
            if "_dino_model" not in step5_retrieval.__dict__:
                import torch
                step5_retrieval.__dict__["_dino_model"] = torch.hub.load(
                    "facebookresearch/dinov2", dino_name).eval().to(dev)
            dino = step5_retrieval.__dict__["_dino_model"]
            q_patches = _extract_dino_patches(query_rgb, dino, dev, img_size)
            q_gd = _compute_vlad(q_patches, vlad_centers)

        q_gd = q_gd / (np.linalg.norm(q_gd) + 1e-8)
        dists, idxs = db["kdtree"].query(q_gd, k=top_k)
        cos_sims  = 1.0 - dists**2 / 2.0
        candidates = [db["entries"][i] for i in idxs]
        print(f"    Retrieval top1: #{candidates[0]['id']}  sim={cos_sims[0]:.4f}")

        # ── LoFTR matching on all candidates → best ───────────────────
        q_tensor, q_sx, q_sy = to_gray_tensor(query_rgb)
        best_n = -1; best_mkpts_q = best_mkpts_r = best_confs = None
        best_cand = candidates[0]; best_ref_rgb = None

        for rank, cand in enumerate(candidates):
            ref_rgb = cv2.cvtColor(cv2.imread(cand["rgb_path"]), cv2.COLOR_BGR2RGB)
            r_tensor, r_sx, r_sy = to_gray_tensor(ref_rgb)
            try:
                batch = {"image0": q_tensor, "image1": r_tensor}
                with torch.no_grad():
                    matcher(batch)
                mkpts0 = batch["mkpts0_f"].cpu().numpy() * np.array([q_sx, q_sy])
                mkpts1 = batch["mkpts1_f"].cpu().numpy() * np.array([r_sx, r_sy])
                confs  = batch["mconf"].cpu().numpy()
            except Exception as e:
                print(f"    Rank{rank+1} LoFTR failed: {e}"); continue

            mask   = confs >= conf_thresh
            n_good = int(mask.sum())
            print(f"    Rank{rank+1} #{cand['id']}: {n_good} matches")
            if n_good > best_n:
                best_n = n_good
                best_mkpts_q = mkpts0[mask]; best_mkpts_r = mkpts1[mask]
                best_confs = confs[mask]; best_cand = cand; best_ref_rgb = ref_rgb

        # ── 시각화 저장 ───────────────────────────────────────────────
        if best_ref_rgb is None:
            best_ref_rgb = cv2.cvtColor(cv2.imread(candidates[0]["rgb_path"]), cv2.COLOR_BGR2RGB)

        h1, w1 = query_rgb.shape[:2]; h2, w2 = best_ref_rgb.shape[:2]
        th = max(h1, h2)
        sq, sr = 1.0, 1.0
        if h1 != th:
            sq = th / h1
            query_rgb    = cv2.resize(query_rgb,    (int(w1*sq), th))
        if h2 != th:
            sr = th / h2
            best_ref_rgb = cv2.resize(best_ref_rgb, (int(w2*sr), th))
        h1, w1 = query_rgb.shape[:2]; h2, w2 = best_ref_rgb.shape[:2]
        canvas = np.concatenate([query_rgb, best_ref_rgb], axis=1)
        mkpts_q_v = best_mkpts_q * sq if best_mkpts_q is not None else None
        mkpts_r_v = best_mkpts_r * sr if best_mkpts_r is not None else None

        fig, ax = plt.subplots(1, 1, figsize=(18, 6))
        ax.imshow(canvas)
        if best_n > 0 and mkpts_q_v is not None:
            cmap_v = plt.cm.RdYlGn(best_confs / (best_confs.max() + 1e-8))
            step_v = max(1, best_n // 300)
            for i in range(0, best_n, step_v):
                ax.plot([mkpts_q_v[i,0], mkpts_r_v[i,0]+w1],
                        [mkpts_q_v[i,1], mkpts_r_v[i,1]],
                        c=cmap_v[i], alpha=0.5, linewidth=0.8)
            ax.scatter(mkpts_q_v[:,0], mkpts_q_v[:,1], c="cyan",   s=6, zorder=3)
            ax.scatter(mkpts_r_v[:,0]+w1, mkpts_r_v[:,1], c="yellow", s=6, zorder=3)

        ax.set_title(f"{stem}  →  best=#{best_cand['id']}  ({best_n} matches, conf≥{conf_thresh})")
        ax.axis("off"); fig.tight_layout()
        out_png = os.path.join(save_dir, f"{stem}.png")
        fig.savefig(out_png, dpi=120); plt.close()
        print(f"    Saved: step6a_results/{stem}.png")

    print(f"\n  Done. {len(query_files)} results in {save_dir}/")


# =============================================================================
# STEP 7: 2D-3D Correspondence + PnP  (Algorithm 1: T_WQ = T_WR × T_QR⁻¹)
# =============================================================================
def step7_pnp(step6_data, step5_data, config, output_dir):
    """
    논문 Algorithm 1:

    for each matched pair (q_uv, r_uv):
        Pz ← D[r_v][r_u]               # reference depth
        Px ← (r_u − cx) * Pz / fx
        Py ← (r_v − cy) * Pz / fy
        pts3d_refcam ← [Px, Py, Pz]    # reference camera frame
        pts2d_query  ← [q_u, q_v]

    PnP: T_QR (ref_cam → query_cam)
    T_WQ = T_WR × T_QR⁻¹
    """
    print("\n" + "="*60 + "\nSTEP 7: 2D-3D correspondence + PnP\n" + "="*60)
    cam      = config["camera"]
    onl      = config.get("online", {})
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    K        = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    dist_c   = np.zeros(4)
    dmin     = cam.get("depth_min", 0.3)
    dmax     = cam.get("depth_max", 20.0)

    matched_q = step6_data["matched_q_kps"]   # (N,2) query image 2D
    matched_r = step6_data["matched_r_kps"]   # (N,2) reference image 2D
    ref_entry = step6_data["best_cand"]
    query_rgb = step6_data["query_rgb"]
    ref_rgb   = step6_data["ref_rgb"]
    gt_entry  = step5_data.get("gt_entry")

    dep  = np.load(ref_entry["depth_path"])
    T_WR = np.array(ref_entry["pose"], dtype=np.float64)

    # ── 2D-3D backproject (Algorithm 1) ──────────────────────────────
    pts2d = []; pts3d = []; invalid = 0
    for i in range(len(matched_q)):
        qu, qv = matched_q[i]
        ru, rv = matched_r[i]
        ri, rj = int(round(rv)), int(round(ru))
        if not (0 <= ri < dep.shape[0] and 0 <= rj < dep.shape[1]):
            invalid += 1; continue
        pz = float(dep[ri, rj])
        if not np.isfinite(pz) or pz < dmin or pz > dmax:
            invalid += 1; continue
        pts2d.append([qu, qv])
        pts3d.append([(ru-cx)*pz/fx, (rv-cy)*pz/fy, pz])   # ref camera frame

    pts2d = np.array(pts2d, dtype=np.float64)
    pts3d = np.array(pts3d, dtype=np.float64)

    print(f"  Matched pairs  : {len(matched_q)}")
    print(f"  Valid (depth OK): {len(pts2d)}  (depth invalid: {invalid})")
    print(f"  Ref pose        : {T_WR[:3,3].round(3)}")

    # ── PnP ───────────────────────────────────────────────────────────
    estimated_pose = None; inlier_count = 0; T_QR = None

    if len(pts2d) >= 6:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d, K, dist_c,
            iterationsCount=onl.get("pnp_iterations", 1000),
            reprojectionError=onl.get("reprojection_error", 8.0),
            confidence=onl.get("pnp_confidence", 0.99),
            flags=cv2.SOLVEPNP_EPNP,
        )
        if success and inliers is not None:
            inlier_count = len(inliers)
            R_qr, _ = cv2.Rodrigues(rvec)
            T_QR = np.eye(4); T_QR[:3,:3] = R_qr; T_QR[:3,3] = tvec.flatten()
            T_WQ = T_WR @ np.linalg.inv(T_QR)
            estimated_pose = T_WQ
            print(f"  PnP SUCCESS: {inlier_count}/{len(pts2d)} inliers")
            print(f"  T_WQ position : {T_WQ[:3,3].round(4)}")
            if gt_entry is not None:
                err = np.linalg.norm(T_WQ[:3,3] - np.array(gt_entry["pose"])[:3,3])
                print(f"  GT  position  : {np.array(gt_entry['pose'])[:3,3].round(4)}")
                print(f"  Position error: {err:.4f} m")
        else:
            print("  PnP FAILED: insufficient inliers")
    else:
        print(f"  PnP SKIPPED: {len(pts2d)} < 6 correspondences")

    # ── 시각화 ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

    ax = fig.add_subplot(gs[0,0])
    ax.imshow(query_rgb)
    if len(pts2d) > 0:
        ax.scatter(pts2d[:,0], pts2d[:,1], c="red", s=10, marker="x", alpha=0.7)
    ax.set_title(f"Query + {len(pts2d)} 2D corr", fontsize=9); ax.axis("off")

    ax = fig.add_subplot(gs[0,1])
    ax.imshow(ref_rgb)
    ax.set_title(f"Reference #{ref_entry['id']}", fontsize=9); ax.axis("off")

    ax = fig.add_subplot(gs[0,2])
    if len(pts3d) > 0:
        sc = ax.scatter(pts3d[:,0], pts3d[:,2], c=pts3d[:,1], cmap="RdYlBu", s=20)
        plt.colorbar(sc, ax=ax, fraction=0.046, label="Y (m)")
    ax.set_title("3D pts: X vs Z (ref cam frame)")
    ax.set_xlabel("X"); ax.set_ylabel("Z")

    ax = fig.add_subplot(gs[1,:])
    cands = step5_data["candidates"]
    all_pos = np.array([e["pose"][:3,3] for e in cands])
    ax.scatter(all_pos[:,0], all_pos[:,1], c="lightgray", s=5, alpha=0.4, label="Candidates")
    ref_pos = np.array(ref_entry["pose"])[:3,3]
    ax.scatter(ref_pos[0], ref_pos[1], c="orange", s=200, marker="D",
               zorder=5, label=f"Best Ref #{ref_entry['id']}")
    if gt_entry is not None:
        gp = np.array(gt_entry["pose"])[:3,3]
        ax.scatter(gp[0], gp[1], c="blue", s=300, marker="*", zorder=6, label="GT query")
    if estimated_pose is not None:
        ep = estimated_pose[:3,3]
        ax.scatter(ep[0], ep[1], c="red", s=300, marker="^", zorder=7, label="Estimated")
        ax.plot([ref_pos[0], ep[0]], [ref_pos[1], ep[1]], "r--", alpha=0.5)
        if gt_entry is not None:
            gp = np.array(gt_entry["pose"])[:3,3]
            err = np.linalg.norm(ep - gp)
            ax.plot([ep[0],gp[0]], [ep[1],gp[1]], "b:", alpha=0.7,
                    label=f"err={err:.3f}m")
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="best")
    ax.set_title("Top-down: Ref (orange◆), GT (blue★), Estimated (red▲)")

    status  = "SUCCESS" if estimated_pose is not None else "FAILED"
    err_str = ""
    if gt_entry is not None and estimated_pose is not None:
        err_str = f" | err={np.linalg.norm(estimated_pose[:3,3]-np.array(gt_entry['pose'])[:3,3]):.3f}m"
    fig.suptitle(f"Step 7: PnP — {status} ({inlier_count} inliers, {len(pts2d)} corr){err_str}",
                 fontsize=12)
    fig.savefig(os.path.join(output_dir,"step7_pnp.png"), dpi=150); plt.close()
    print(f"  Saved: step7_pnp.png")

    result = {
        "estimated_pose":    estimated_pose,
        "T_WR":              T_WR,
        "T_QR":              T_QR,
        "inlier_count":      inlier_count,
        "n_correspondences": len(pts2d),
        "best_ref_id":       ref_entry["id"],
        "gt_entry":          gt_entry,
    }
    pickle.dump(result, open(os.path.join(output_dir,"step7_data.pkl"),"wb"))
    return result


# =============================================================================
# Batch Test Runner
# =============================================================================
def localize_single(query_image_path, db, config, work_dir):
    """
    단일 쿼리 이미지에 대해 step5→step7 파이프라인을 실행.
    estimated_pose (4×4) or None 반환.
    """
    os.makedirs(work_dir, exist_ok=True)
    try:
        s5 = step5_retrieval(query_image_path, db, config, work_dir)
        s6 = step6_match(s5, config, work_dir)
        result = step7_pnp(s6, s5, config, work_dir)
        return result.get("estimated_pose") if result else None
    except Exception as e:
        print(f"    localize_single error: {e}")
        return None


def run_test_batch(test_dir, db, config, output_dir, gt_poses_path=None):
    """
    test_dir 안의 모든 이미지에 대해 온라인 파이프라인 실행.
    추론된 경로와 GT 경로를 비교한 plot을 저장.

    gt_poses_path (JSON, optional):
      {filename: [[4×4 rows]]}  또는  {filename: [x, y, z]}
    """
    import json, re
    print("\n" + "="*60)
    print("BATCH TEST: Online localization on test set")
    print("="*60)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if os.path.isfile(test_dir):
        img_paths = [test_dir]
    else:
        img_paths = sorted([
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])
    if not img_paths:
        print(f"  ERROR: {test_dir} 에 이미지가 없습니다.")
        return

    print(f"  Test images: {len(img_paths)} ({test_dir})")

    gt_map = {}
    if gt_poses_path and os.path.exists(gt_poses_path):
        raw = json.load(open(gt_poses_path))
        for fname, val in raw.items():
            arr = np.array(val)
            if arr.shape == (4, 4):
                gt_map[fname] = arr[:3, 3]
            elif arr.ndim == 1 and len(arr) >= 3:
                gt_map[fname] = arr[:3]
        print(f"  GT poses loaded: {len(gt_map)} entries")
    else:
        print("  GT poses: 없음 (추정 경로만 표시)")

    # test_dir 경로에서 cam_# 추출 (예: test_data/cam_0/ → cam_0)
    cam_match = re.search(r'cam_\d+', os.path.abspath(test_dir))
    cam_subdir = cam_match.group(0) if cam_match else "cam_unknown"
    test_out = os.path.join(output_dir, "test_results", cam_subdir)
    print(f"  Output dir: {test_out}")
    os.makedirs(test_out, exist_ok=True)

    results = []
    for i, img_path in enumerate(img_paths):
        fname    = os.path.basename(img_path)
        work_dir = os.path.join(test_out, os.path.splitext(fname)[0])
        print(f"\n  [{i+1}/{len(img_paths)}] {fname}")

        est_pose = localize_single(img_path, db, config, work_dir)
        est_xyz  = est_pose[:3, 3] if est_pose is not None else None
        gt_xyz   = gt_map.get(fname)
        err      = float(np.linalg.norm(est_xyz - gt_xyz)) \
                   if (est_xyz is not None and gt_xyz is not None) else None

        results.append({
            "fname":    fname,
            "success":  est_pose is not None,
            "est_pose": est_pose,
            "est_xyz":  est_xyz,
            "gt_xyz":   gt_xyz,
            "error_m":  err,
        })
        status  = "OK" if est_pose is not None else "FAIL"
        err_str = f"  err={err:.3f}m" if err is not None else ""
        print(f"    → {status}{err_str}")

    n_ok    = sum(r["success"] for r in results)
    n_total = len(results)
    errors  = [r["error_m"] for r in results if r["error_m"] is not None]
    print(f"\n  Success rate: {n_ok}/{n_total} ({100*n_ok/n_total:.1f}%)")
    if errors:
        print(f"  Error  mean={np.mean(errors):.3f}m  "
              f"median={np.median(errors):.3f}m  "
              f"max={np.max(errors):.3f}m")

    # ── Plot ──────────────────────────────────────────────────────────
    has_gt  = any(r["gt_xyz"]  is not None for r in results)
    has_est = any(r["est_xyz"] is not None for r in results)
    ncols   = 3 if (has_gt and has_est and errors) else 2

    fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 7))

    ax = axes[0]
    db_poses = np.array([e["pose"][:3, 3] for e in db["entries"]])
    ax.scatter(db_poses[:,0], db_poses[:,1], c="lightgray", s=2, alpha=0.4, label="DB viewpoints")

    if has_gt:
        gt_pts = np.array([r["gt_xyz"] for r in results if r["gt_xyz"] is not None])
        ax.plot(gt_pts[:,0], gt_pts[:,1], "b-o", lw=1.5, ms=4, label="GT path", zorder=3)

    if has_est:
        est_ok = [(r["est_xyz"], r["gt_xyz"]) for r in results if r["est_xyz"] is not None]
        ep = np.array([x[0] for x in est_ok])
        ax.plot(ep[:,0], ep[:,1], "r-^", lw=1.5, ms=5, label="Estimated", zorder=4)
        if has_gt:
            for r in results:
                if r["est_xyz"] is not None and r["gt_xyz"] is not None:
                    ax.plot([r["gt_xyz"][0], r["est_xyz"][0]],
                            [r["gt_xyz"][1], r["est_xyz"][1]],
                            "k-", lw=0.5, alpha=0.4)
        fail_gt = [r["gt_xyz"] for r in results if not r["success"] and r["gt_xyz"] is not None]
        if fail_gt:
            fp = np.array(fail_gt)
            ax.scatter(fp[:,0], fp[:,1], c="red", marker="x", s=60, zorder=5, label="Failed")

    ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.set_title(f"Top-down: GT vs Estimated\n({n_ok}/{n_total} success)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    ax2 = axes[1]
    colors_b = []
    vals     = []
    xlabels  = []
    for r in results:
        xlabels.append(os.path.splitext(r["fname"])[0][-12:])
        if not r["success"]:
            vals.append(0); colors_b.append("red")
        elif r["error_m"] is not None:
            vals.append(r["error_m"]); colors_b.append("steelblue")
        else:
            vals.append(0); colors_b.append("orange")

    xpos = np.arange(len(vals))
    ax2.bar(xpos, vals, color=colors_b)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(xlabels, rotation=60, fontsize=6, ha="right")
    ax2.set_ylabel("Position error (m)")
    ax2.set_title("Per-image error\n(red=failed, orange=no GT, blue=error)")
    if errors:
        ax2.axhline(np.mean(errors), color="black", ls="--", lw=1,
                    label=f"mean={np.mean(errors):.3f}m")
        ax2.axhline(np.median(errors), color="gray", ls=":", lw=1,
                    label=f"median={np.median(errors):.3f}m")
        ax2.legend(fontsize=7)

    if ncols == 3:
        ax3 = axes[2]
        errs_sorted = np.sort(errors)
        cdf = np.arange(1, len(errs_sorted)+1) / len(errs_sorted)
        ax3.plot(errs_sorted, cdf*100, "b-", lw=2)
        for thresh in [0.25, 0.5, 1.0, 2.0]:
            pct = 100 * np.mean(np.array(errors) <= thresh)
            ax3.axvline(thresh, color="gray", ls="--", lw=0.8, alpha=0.7)
            ax3.text(thresh, pct+2, f"{pct:.0f}%\n@{thresh}m", fontsize=7, ha="center")
        ax3.set_xlabel("Error threshold (m)"); ax3.set_ylabel("Recall (%)")
        ax3.set_title("Error CDF"); ax3.set_ylim(0, 105); ax3.grid(True, alpha=0.3)

    title = (f"Batch Test — {n_ok}/{n_total} success"
             + (f" | mean err={np.mean(errors):.3f}m" if errors else ""))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    out_png = os.path.join(test_out, "test_trajectory.png")
    fig.savefig(out_png, dpi=150); plt.close()
    print(f"\n  Saved: {out_png}")

    pickle.dump(results, open(os.path.join(test_out, "batch_results.pkl"), "wb"))
    print(f"  Saved: {os.path.join(test_out, 'batch_results.pkl')}")

    # ── Trajectory 파일 저장 ───────────────────────────────────────────
    # TUM format: timestamp tx ty tz qx qy qz qw
    import json as _json
    from scipy.spatial.transform import Rotation

    tum_lines = []
    csv_rows  = [["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "success"]]
    traj_json = {}

    for r in results:
        stem = os.path.splitext(r["fname"])[0]
        # 파일명이 microsecond timestamp면 그대로, 아니면 인덱스 사용
        try:
            ts = float(stem) / 1e6   # microsec → sec
        except ValueError:
            ts = results.index(r)

        if r["est_pose"] is not None:
            T = r["est_pose"]
            tx, ty, tz = T[:3, 3]
            quat = Rotation.from_matrix(T[:3, :3]).as_quat()  # [qx,qy,qz,qw]
            qx, qy, qz, qw = quat
            tum_lines.append(f"{ts:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                             f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")
            csv_rows.append([f"{ts:.6f}", f"{tx:.6f}", f"{ty:.6f}", f"{tz:.6f}",
                             f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}", f"{qw:.6f}", "1"])
            traj_json[r["fname"]] = T.tolist()
        else:
            csv_rows.append([f"{ts:.6f}", "", "", "", "", "", "", "", "0"])

    # TUM trajectory (.txt)
    tum_path = os.path.join(test_out, "trajectory_tum.txt")
    with open(tum_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        f.write("\n".join(tum_lines))
    print(f"  Saved: {tum_path}  ({len(tum_lines)} poses, TUM format)")

    # CSV trajectory
    csv_path = os.path.join(test_out, "trajectory.csv")
    with open(csv_path, "w") as f:
        for row in csv_rows:
            f.write(",".join(row) + "\n")
    print(f"  Saved: {csv_path}")

    # JSON (4×4 matrix per frame)
    json_path = os.path.join(test_out, "trajectory_poses.json")
    with open(json_path, "w") as f:
        _json.dump(traj_json, f, indent=2)
    print(f"  Saved: {json_path}  (4×4 matrices)")

    return results


# =============================================================================
# Main
# =============================================================================
OFFLINE_STEPS = ["0_align", "1_viewpoints", "2_render", "3_global_desc", "4_build_db"]
ONLINE_STEPS  = ["5_retrieval", "6_match", "6a_match_viz", "7_pnp"]
STEPS = OFFLINE_STEPS + ONLINE_STEPS


def _load(output_dir, name):
    p = os.path.join(output_dir, name)
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_map",    required=True)
    parser.add_argument("--config",     default="config/render_loc.yaml")
    parser.add_argument("--output_dir", default="output/MegaLoc")
    parser.add_argument("--step",       default="all",
                        choices=["all","offline","online","test"] + STEPS)
    parser.add_argument("--query_image", default=None,
                        help="Query 이미지 경로 (없으면 DB 내 self-test)")
    parser.add_argument("--query_dir",  default=None,
                        help="step 6a_match_viz: 배치 쿼리 이미지 폴더")
    parser.add_argument("--test_dir",    default=None,
                        help="배치 테스트용 이미지 디렉토리 또는 단일 파일")
    parser.add_argument("--gt_poses",   default=None,
                        help='GT poses JSON. 형식: {"filename": [[4x4]] or [x,y,z]}')
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else default_config()
    os.makedirs(args.output_dir, exist_ok=True)
    run_offline = args.step in ("all", "offline")
    run_online  = args.step in ("all", "online")
    run_test    = args.step == "test"

    # ── Offline ───────────────────────────────────────────────────────
    s0 = None
    if run_offline or args.step == "0_align":
        s0 = step0_align(args.ply_map, config, args.output_dir)
    else:
        s0 = _load(args.output_dir, "step0_data.pkl")

    if run_offline or args.step == "1_viewpoints":
        vps = step1_viewpoints(args.ply_map, config, args.output_dir, s0)
    else:
        d = _load(args.output_dir, "step1_data.pkl")
        vps = d["viewpoints"] if d else []

    if run_offline or args.step == "2_render":
        rendered = step2_render(args.ply_map, vps, config, args.output_dir, s0)
    else:
        rendered = _load(args.output_dir, "step2_data.pkl") or []
        if not rendered:
            # step2_data.pkl 없으면 rendered/ 폴더의 실사 이미지로 대체
            rgb_dir   = os.path.join(args.output_dir, "rendered", "rgb")
            depth_dir = os.path.join(args.output_dir, "rendered", "depth")
            if os.path.isdir(rgb_dir):
                rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")) +
                                   glob.glob(os.path.join(rgb_dir, "*.png")))
                rendered = []
                for i, rgb_path in enumerate(rgb_files):
                    stem = os.path.splitext(os.path.basename(rgb_path))[0]
                    depth_path = os.path.join(depth_dir, stem + ".depth")
                    if not os.path.exists(depth_path):
                        depth_path = os.path.join(depth_dir, stem + ".png")
                    rendered.append({
                        "id":         i,
                        "rgb_path":   rgb_path,
                        "depth_path": depth_path if os.path.exists(depth_path) else "",
                        "pose":       np.eye(4),   # 포즈 없음 → step6 PnP 불가
                        "floor":      0,
                        "yaw":        0.0,
                    })
                print(f"  [step2 fallback] {len(rendered)} real images loaded from {rgb_dir}")

    need_gd = run_offline or args.step in ("3_global_desc", "4_build_db")
    if run_offline or args.step == "3_global_desc":
        rendered, _ = step3_global_desc(rendered, config, args.output_dir)
    elif need_gd:
        _s3 = _load(args.output_dir, "step3_data.pkl")
        if isinstance(_s3, dict):
            rendered = _s3.get("rendered") or rendered
        elif _s3:
            rendered = _s3

    db = None
    if run_offline or args.step == "4_build_db":
        db = step4_build_db(rendered, args.output_dir)

    # ── DB 로드 (online 단독 실행 시) ────────────────────────────────
    need_db = run_online or run_test or args.step in ONLINE_STEPS
    if need_db and db is None:
        db = _load(args.output_dir, "step4_database.pkl")
        if db is None:
            print("ERROR: step4_database.pkl not found. Run --step 4_build_db first.")
            return

    # ── Batch test ────────────────────────────────────────────────────
    if run_test:
        if not args.test_dir:
            print("ERROR: --step test 사용 시 --test_dir 을 지정하세요.")
            return
        import re as _re
        _cam = _re.search(r'cam_\d+', os.path.abspath(args.test_dir))
        _cam_sub = _cam.group(0) if _cam else "cam_unknown"
        run_test_batch(args.test_dir, db, config, args.output_dir, args.gt_poses)
        print(f"\n=== Done === Results in: {args.output_dir}/test_results/{_cam_sub}/")
        return

    # ── Online single query ───────────────────────────────────────────
    s5 = None
    if run_online or args.step == "5_retrieval":
        s5 = step5_retrieval(args.query_image, db, config, args.output_dir)
    elif args.step in ("6_match", "7_pnp"):
        s5 = _load(args.output_dir, "step5_data.pkl")
        if s5 is None:
            print("ERROR: step5_data.pkl 없음. 5_retrieval 먼저 실행.")
            return

    if args.step == "6a_match_viz":
        if not args.query_dir:
            print("ERROR: --query_dir 를 지정하세요."); return
        step6a_match_viz(args.query_dir, db, config, args.output_dir)
        return

    s6 = None
    if run_online or args.step == "6_match":
        s6 = step6_match(s5, config, args.output_dir)
    elif args.step == "7_pnp":
        s6 = _load(args.output_dir, "step6_data.pkl")
        if s6 is None:
            print("ERROR: step6_data.pkl 없음. 6_match 먼저 실행.")
            return

    if run_online or args.step == "7_pnp":
        step7_pnp(s6, s5, config, args.output_dir)

    print(f"\n=== Done === Results in: {args.output_dir}/")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".png"):
            print(f"  {f}")


if __name__ == "__main__":
    main()