import os, pickle, json, math
import numpy as np
import open3d as o3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import (binary_dilation, binary_erosion, distance_transform_edt,
                           gaussian_filter, maximum_filter, label)
from collections import deque
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from plyfile import PlyData


def _estimate_xy_spacing(xy, max_samples=5000):
    """Median nearest-neighbor spacing in XY, robust enough for sparse COLMAP maps."""
    if len(xy) < 2:
        return None
    if len(xy) > max_samples:
        rng = np.random.default_rng(42)
        xy = xy[rng.choice(len(xy), size=max_samples, replace=False)]
    tree = cKDTree(xy)
    dists, _ = tree.query(xy, k=2)
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    if len(nn) == 0:
        return None
    return float(np.median(nn))


def _splat_pixels(occ, px, py, radius_px):
    """Rasterize each sparse floor point as a small disk instead of one pixel."""
    if radius_px <= 0:
        occ[py, px] = 1
        return occ

    yy, xx = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    disk = (xx * xx + yy * yy) <= radius_px * radius_px
    dys, dxs = np.nonzero(disk)
    dys = dys - radius_px
    dxs = dxs - radius_px

    h, w = occ.shape
    for dy, dx in zip(dys, dxs):
        y = np.clip(py + dy, 0, h - 1)
        x = np.clip(px + dx, 0, w - 1)
        occ[y, x] = 1
    return occ


def _splat_pixels_variable_radius(occ, px, py, radius_px):
    """Rasterize points with per-point disk radii."""
    radius_px = np.asarray(radius_px, dtype=np.int32)
    for r in np.unique(radius_px):
        m = radius_px == r
        occ = _splat_pixels(occ, px[m], py[m], int(r))
    return occ


def _quat_wxyz_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)


def _read_gaussian_vertex_fields(ply_path):
    """Return Gaussian scale/rotation/opacity fields if this is a Gaussian PLY."""
    try:
        vertex = PlyData.read(ply_path)["vertex"]
    except Exception:
        return None
    names = {p.name for p in vertex.properties}
    required = {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    if not required.issubset(names):
        return None
    return {
        "opacity_raw": np.asarray(vertex["opacity"], dtype=np.float64),
        "scale_log": np.stack([np.asarray(vertex[f"scale_{i}"], dtype=np.float64) for i in range(3)], axis=1),
        "quat_wxyz": np.stack([np.asarray(vertex[f"rot_{i}"], dtype=np.float64) for i in range(4)], axis=1),
    }


def _sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _outside_flood_mask(blocked):
    """Return pixels connected to the image border without crossing blocked cells."""
    h, w = blocked.shape
    outside = np.zeros_like(blocked, dtype=bool)
    q = deque()

    def push(y, x):
        if 0 <= y < h and 0 <= x < w and not blocked[y, x] and not outside[y, x]:
            outside[y, x] = True
            q.append((y, x))

    for x in range(w):
        push(0, x)
        push(h - 1, x)
    for y in range(h):
        push(y, 0)
        push(y, w - 1)

    while q:
        y, x = q.popleft()
        push(y - 1, x)
        push(y + 1, x)
        push(y, x - 1)
        push(y, x + 1)
    return outside


def _gaussian_floor_support(points, gaussian_fields, floor_z, gr, samp):
    """Select/splat Gaussians whose ellipsoid intersects the step0 floor plane."""
    scales = np.exp(np.clip(gaussian_fields["scale_log"], -10.0, 5.0))
    rot = _quat_wxyz_to_rotmat(gaussian_fields["quat_wxyz"])
    opacity = _sigmoid(gaussian_fields["opacity_raw"])

    sigma_x = np.sqrt(np.sum((rot[:, 0, :] * scales) ** 2, axis=1))
    sigma_y = np.sqrt(np.sum((rot[:, 1, :] * scales) ** 2, axis=1))
    sigma_z = np.sqrt(np.sum((rot[:, 2, :] * scales) ** 2, axis=1))

    opacity_min = float(samp.get("gaussian_floor_opacity_min", 0.01))
    z_margin = float(samp.get("gaussian_floor_z_margin_m", 0.15))
    z_sigma_scale = float(samp.get("gaussian_floor_z_sigma_scale", 2.5))
    center_z_max = float(samp.get("gaussian_floor_center_z_max_m", 0.6))
    radius_scale = float(samp.get("gaussian_floor_radius_scale", 2.0))
    radius_min = float(samp.get("gaussian_floor_min_radius_m", gr))
    radius_max = float(samp.get("gaussian_floor_max_radius_m", 0.8))

    z_dist = np.abs(points[:, 2] - floor_z)
    floor_mask = ((opacity >= opacity_min) &
                  (z_dist <= center_z_max) &
                  (z_dist <= (z_margin + z_sigma_scale * sigma_z)))
    radius_m = np.maximum(sigma_x, sigma_y) * radius_scale
    radius_m = np.clip(radius_m, radius_min, radius_max)
    radius_px = np.ceil(radius_m / gr).astype(np.int32)
    radius_px = np.maximum(radius_px, 1)
    return floor_mask, radius_px


def _opencv_c2w_to_opengl(c2w_cv):
    """
    OpenCV (Y-down, Z-forward) → OpenGL (Y-up, Z-back) 카메라 좌표계 변환.
    카메라 로컬 frame만 회전 (월드 좌표계는 그대로).

    OpenCV camera frame:  X=right, Y=down,  Z=forward(into scene)
    OpenGL camera frame:  X=right, Y=up,    Z=back   (out of scene)

    → 카메라 로컬에서 Y, Z 축 부호 반전.
    """
    flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
    return c2w_cv @ flip


def _apply_dataparser_transform(c2w, dp_transform=None, dp_scale=1.0):
    """
    nerfstudio dataparser_transforms.json (있으면) 적용.

    학습 시 auto-orient/center/scale이 활성화되었다면 학습된 가우시안은
    변환된 좌표계에 있음. 입력 c2w(world frame)에 같은 변환을 적용해야 함.

    Args:
        c2w: (4, 4) world c2w
        dp_transform: (3, 4) or (4, 4) transform (T) 또는 None
        dp_scale: float (s)
    Returns:
        (4, 4) 변환된 c2w
    """
    if dp_transform is None:
        return c2w
    T = np.eye(4)
    dp_arr = np.asarray(dp_transform, dtype=np.float64)
    if dp_arr.shape == (3, 4):
        T[:3, :4] = dp_arr
    elif dp_arr.shape == (4, 4):
        T = dp_arr
    else:
        raise ValueError(f"Unexpected dataparser transform shape: {dp_arr.shape}")
    out = T @ c2w
    out[:3, 3] *= float(dp_scale)
    return out


def _load_dataparser_transform(path):
    """nerfstudio dataparser_transforms.json 로드. 없으면 None 반환."""
    if not path or not os.path.exists(path):
        return None, 1.0
    try:
        with open(path) as f:
            d = json.load(f)
        T = d.get("transform", None)
        s = d.get("scale", 1.0)
        if T is not None:
            T = np.asarray(T, dtype=np.float64)
        return T, float(s)
    except Exception as e:
        print(f"  [warn] dataparser transform 로드 실패: {e}")
        return None, 1.0


def _export_nerfstudio_camera_path(viewpoints, config, output_dir,
                                    dataparser_transform_path=None,
                                    T_align=None,
                                    out_filename="step1_camera_path.json",
                                    invert_align=True,
                                    invert_dataparser=False,
                                    flip_camera_frame=True,
                                    debug_print_first=True):
    """
    Step1 viewpoints를 nerfstudio `ns-render camera-path`용 JSON으로 export.

    좌표계 변환 chain:
      viewpoints[i]["pose"]   (align된 world, OpenCV camera frame)
        ↓ [선택] T_align 처리   (invert_align=True면 T_align⁻¹ 적용,
                                False면 그대로 둠 = step1 pose가 이미 학습 좌표계라고 가정)
        ↓ [선택] OpenCV→OpenGL (flip_camera_frame=True; nerfstudio는 OpenGL 사용)
        ↓ [선택] dataparser_T  (invert_dataparser로 방향 토글 가능)
      → ns-render가 기대하는 좌표

    Args:
        invert_align: True면 T_align⁻¹ 적용 (step1 pose가 align 좌표계라고 가정).
                      False면 step1 pose가 이미 학습 좌표계라고 가정 (변환 안 함).
        invert_dataparser: True면 dataparser_T⁻¹ 적용. 기본은 False (T_dp @ c2w).
        flip_camera_frame: True면 OpenCV→OpenGL camera frame 변환.
        debug_print_first: True면 첫 번째 viewpoint의 변환 단계별 좌표 출력.
    """
    cam = config["camera"]
    W = int(cam["width"])
    H = int(cam["height"])
    fy = float(cam["fy"])
    vfov_deg = math.degrees(2.0 * math.atan(H / (2.0 * fy)))
    aspect = W / H

    # step0 align 처리 준비
    T_align_arr = None
    if T_align is not None:
        T_align_arr = np.asarray(T_align, dtype=np.float64)
        if T_align_arr.shape != (4, 4):
            print(f"  [warn] T_align shape이 (4,4)가 아님: {T_align_arr.shape}, 무시함")
            T_align_arr = None

    if T_align_arr is not None:
        if invert_align:
            T_align_apply = np.linalg.inv(T_align_arr)
            print(f"  [camera-path] step0 align 역변환 적용 (T_align⁻¹)")
        else:
            T_align_apply = T_align_arr
            print(f"  [camera-path] step0 align 정변환 적용 (T_align)")
    else:
        T_align_apply = None
        print(f"  [camera-path] step0 align 변환 없음 (identity)")

    dp_T, dp_s = _load_dataparser_transform(dataparser_transform_path)
    if dp_T is not None:
        direction = "inverse" if invert_dataparser else "forward"
        print(f"  [camera-path] dataparser transform 적용 ({direction}): "
              f"scale={dp_s:.4f}")
    else:
        print(f"  [camera-path] dataparser transform 없음 (identity)")

    if flip_camera_frame:
        print(f"  [camera-path] camera frame: OpenCV → OpenGL flip 적용")
    else:
        print(f"  [camera-path] camera frame: flip 안 함 (입력 pose 그대로)")

    cam_list = []
    for vi, vp in enumerate(viewpoints):
        c2w_in = np.asarray(vp["pose"], dtype=np.float64)
        c2w = c2w_in.copy()

        # 1) align 변환
        if T_align_apply is not None:
            c2w_after_align = T_align_apply @ c2w
        else:
            c2w_after_align = c2w

        # 2) OpenCV → OpenGL (camera frame)
        if flip_camera_frame:
            c2w_after_flip = _opencv_c2w_to_opengl(c2w_after_align)
        else:
            c2w_after_flip = c2w_after_align

        # 3) dataparser transform
        if dp_T is not None and invert_dataparser:
            # 역변환: T_dp⁻¹ 적용
            T_dp_full = np.eye(4)
            dp_arr = np.asarray(dp_T, dtype=np.float64)
            if dp_arr.shape == (3, 4):
                T_dp_full[:3, :4] = dp_arr
            elif dp_arr.shape == (4, 4):
                T_dp_full = dp_arr
            T_dp_inv = np.linalg.inv(T_dp_full)
            c2w_final = T_dp_inv @ c2w_after_flip
            c2w_final[:3, 3] /= dp_s if dp_s != 0 else 1.0
        else:
            c2w_final = _apply_dataparser_transform(c2w_after_flip, dp_T, dp_s)

        # 디버그: 첫 viewpoint의 단계별 좌표
        if debug_print_first and vi == 0:
            np.set_printoptions(precision=4, suppress=True)
            print(f"  [debug] viewpoint[0] 변환 단계별:")
            print(f"    [in]    pos={c2w_in[:3,3]}, fwd(-Z)={-c2w_in[:3,2]}")
            print(f"    [align] pos={c2w_after_align[:3,3]}, fwd(-Z)={-c2w_after_align[:3,2]}")
            print(f"    [flip]  pos={c2w_after_flip[:3,3]}, fwd(-Z)={-c2w_after_flip[:3,2]}")
            print(f"    [final] pos={c2w_final[:3,3]}, fwd(-Z)={-c2w_final[:3,2]}")

        cam_list.append({
            "camera_to_world": c2w_final.flatten().tolist(),
            "fov": vfov_deg,
            "aspect": aspect,
        })

    n = len(cam_list)
    out = {
        "camera_type": "perspective",
        "render_height": H,
        "render_width": W,
        "fps": 1,
        "seconds": max(1, n),
        "is_cycle": False,
        "smoothness_value": 0.0,
        "camera_path": cam_list,
        "keyframes": [
            {
                "matrix": c["camera_to_world"],
                "fov": c["fov"],
                "aspect": c["aspect"],
                "override_transition_enabled": False,
                "override_transition_sec": None,
            }
            for c in cam_list
        ],
        "default_fov": vfov_deg,
        "default_transition_sec": 1.0,
    }

    out_path = os.path.join(output_dir, out_filename)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  [camera-path] Saved: {out_path}  ({n} cameras, vFOV={vfov_deg:.2f}°, "
          f"{W}×{H})")
    return out_path


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
    gaussian_fields = _read_gaussian_vertex_fields(ap)
    if gaussian_fields is not None and len(gaussian_fields["opacity_raw"]) == len(points):
        print("  Input type: Gaussian PLY (scale/rotation/opacity available)")
    else:
        gaussian_fields = None

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    normals = np.asarray(pcd.normals)

    nth = align_cfg.get("normal_threshold", 0.8)
    dots = normals @ np.array([0.0, 0.0, 1.0])
    up_mask = dots > nth
    up_pts = points[up_mask]
    print(f"  Upward-facing: {len(up_pts)} ({100*len(up_pts)/len(points):.1f}%)")

    max_floors = samp.get("max_floors", 1)
    min_gap = samp.get("min_floor_gap", 2.5)
    max_fh = samp.get("max_floor_height", 5.0)
    min_fp = samp.get("min_floor_points", 100)

    use_step0_floor = bool(samp.get("use_step0_floor", True))
    if use_step0_floor and step0_data is not None:
        floors = [float(samp.get("step0_floor_z", 0.0))]
        print("  Floor source: step0 aligned floor z=0.00")
    else:
        z_vals = up_pts[:, 2]
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
    ch = samp.get("height_above_floor", 1.0)
    ny = samp.get("num_yaw_angles", 4)
    # height_offsets_m: 기준 높이(ch)에서의 오프셋 목록 (미터)
    # e.g. [0.0, -0.2, 0.2] → 기준 / 20cm 아래 / 20cm 위
    height_offsets = list(samp.get("height_offsets_m", [0.0]))
    print(f"  Height offsets: {height_offsets} m")
    occupancy_mode = str(samp.get("occupancy_mode", "height_slice_obstacle")).lower()
    slice_min_height_m = float(samp.get("slice_min_height_m", 0.5))
    slice_max_height_m = float(samp.get("slice_max_height_m", 1.5))
    slice_splat_radius_m = float(samp.get("slice_splat_radius_m", 0.25))
    slice_wall_close_m = float(samp.get("slice_wall_close_m", 1.0))
    slice_obstacle_inflate_m = float(samp.get("slice_obstacle_inflate_m", 0.45))
    slice_margin_m = float(samp.get("slice_margin_m", 2.0))
    free_min_height_m = float(samp.get("free_evidence_min_height_m", -0.2))
    free_max_height_m = float(samp.get("free_evidence_max_height_m", 2.5))
    free_splat_radius_m = float(samp.get("free_evidence_splat_radius_m", 0.9))
    mk = samp.get("morph_kernel_size", 3)
    mi = samp.get("morph_iterations", 4)
    dr = samp.get("distance_thresh_ratio", 0.2)
    distance_thresh_m = float(samp.get("distance_thresh_m", 0.0))
    corridor_blur_sigma = float(samp.get("corridor_blur_sigma", 3.0))
    corridor_blur_threshold = float(samp.get("corridor_blur_threshold", 0.3))
    kl = samp.get("keep_largest_component", True)
    sample_mode = samp.get("sample_mode", "skeleton")   # "skeleton" or "grid"
    skel_grid_spacing_m = samp.get("skel_grid_spacing_m", 0.8)
    oversample_sparse_floor = samp.get("oversample_sparse_floor", True)
    sparse_fill_ratio = samp.get("sparse_floor_fill_ratio", 0.08)
    floor_splat_radius_m = float(samp.get("floor_splat_radius_m", 0.0))
    floor_splat_radius_scale = float(samp.get("floor_splat_radius_scale", 0.75))
    floor_splat_min_radius_m = float(samp.get("floor_splat_min_radius_m", 0.0))
    floor_splat_max_radius_m = float(samp.get("floor_splat_max_radius_m", 0.8))
    if occupancy_mode in ("obstacle_free", "free_from_obstacles"):
        subtract_obstacles = samp.get("subtract_obstacles_from_free", True)
    else:
        subtract_obstacles = samp.get("subtract_obstacles_from_floor", False)
    obstacle_min_height_m = float(samp.get("obstacle_min_height_m", 0.25))
    obstacle_max_height_m = float(samp.get("obstacle_max_height_m", 2.2))
    obstacle_dilate_m = float(samp.get("obstacle_dilate_m", 0.25))
    all_vp = []; debug_imgs = {}; vid = 0

    for fi, fz in enumerate(floors):
        print(f"\n  --- Floor {fi}: z={fz:.2f}m ---")
        margin = slice_margin_m
        if occupancy_mode in ("height_slice_obstacle", "slice_obstacle", "height_slice"):
            sm = ((points[:, 2] >= fz + slice_min_height_m) &
                  (points[:, 2] <= fz + slice_max_height_m))
            sp = points[sm]
            print(f"    Height-slice obstacle points: {len(sp)} "
                  f"(z +{slice_min_height_m:.2f}~+{slice_max_height_m:.2f}m)")
            if len(sp) < 50:
                print("    Skip")
                continue

            xn, yn = sp[:,0].min()-margin, sp[:,1].min()-margin
            xx, yx = sp[:,0].max()+margin, sp[:,1].max()+margin
            iw = int(np.ceil((xx-xn)/gr)); ih = int(np.ceil((yx-yn)/gr))
            print(f"    Image: {iw}x{ih}px (res={gr}m/px)")

            px = np.clip(((sp[:,0]-xn)/gr).astype(int), 0, iw-1)
            py = np.clip(((sp[:,1]-yn)/gr).astype(int), 0, ih-1)
            obstacle = np.zeros((ih, iw), dtype=np.uint8)
            obstacle[py, px] = 1
            raw_obstacle_fill = obstacle.sum() / max(1, obstacle.size)

            point_radius_px = max(0, int(np.ceil(slice_splat_radius_m / gr)))
            obstacle = np.zeros((ih, iw), dtype=np.uint8)
            obstacle = _splat_pixels(obstacle, px, py, point_radius_px)

            wall_close_px = max(0, int(np.ceil(slice_wall_close_m / gr)))
            if wall_close_px > 0:
                wk = np.ones((2 * wall_close_px + 1, 2 * wall_close_px + 1), dtype=np.uint8)
                wall = binary_dilation(obstacle, structure=wk, iterations=1).astype(np.uint8)
                wall = binary_erosion(wall, structure=wk, iterations=1).astype(np.uint8)
            else:
                wall = obstacle.copy()

            outside = _outside_flood_mask(wall > 0)
            known_area = (~outside).astype(np.uint8)
            if known_area.sum() == 0:
                known_area = binary_dilation(obstacle, structure=np.ones((mk, mk), dtype=np.uint8),
                                             iterations=max(1, mi)).astype(np.uint8)
                print("    Known-area flood failed -> using dilated slice support fallback")

            inflate_px = max(0, int(np.ceil(slice_obstacle_inflate_m / gr)))
            if inflate_px > 0:
                ik = np.ones((2 * inflate_px + 1, 2 * inflate_px + 1), dtype=np.uint8)
                obstacle_inflated = binary_dilation(obstacle, structure=ik, iterations=1).astype(np.uint8)
            else:
                obstacle_inflated = obstacle

            occ = obstacle
            closed = (known_area > 0).astype(np.uint8)
            closed[obstacle_inflated > 0] = 0
            print(f"    Slice obstacle fill: raw={100*raw_obstacle_fill:.2f}% "
                  f"splat={100*obstacle.mean():.2f}% wall={100*wall.mean():.2f}% "
                  f"known={100*known_area.mean():.2f}% free={100*closed.mean():.2f}%")
        elif occupancy_mode in ("gaussian_floor", "gaussian", "gaussian_support") and gaussian_fields is not None:
            floor_mask, radius_px_all = _gaussian_floor_support(points, gaussian_fields, fz, gr, samp)
            fp = points[floor_mask]
            radius_px = radius_px_all[floor_mask]
            print(f"    Gaussian floor support: {len(fp)} / {len(points)} "
                  f"(radius_px median={np.median(radius_px) if len(radius_px) else 0:.1f})")
            if len(fp) < 50:
                print("    Skip")
                continue

            xn, yn = fp[:,0].min()-margin, fp[:,1].min()-margin
            xx, yx = fp[:,0].max()+margin, fp[:,1].max()+margin
            iw = int(np.ceil((xx-xn)/gr)); ih = int(np.ceil((yx-yn)/gr))
            print(f"    Image: {iw}x{ih}px (res={gr}m/px)")

            occ = np.zeros((ih, iw), dtype=np.uint8)
            px = np.clip(((fp[:,0]-xn)/gr).astype(int), 0, iw-1)
            py = np.clip(((fp[:,1]-yn)/gr).astype(int), 0, ih-1)
            raw_occ = np.zeros_like(occ)
            raw_occ[py, px] = 1
            raw_fill = raw_occ.sum() / max(1, raw_occ.size)
            occ = _splat_pixels_variable_radius(occ, px, py, radius_px)
            fill = occ.sum() / max(1, occ.size)
            print(f"    Gaussian floor splat: fill {100*raw_fill:.2f}% -> {100*fill:.2f}%")
        elif occupancy_mode in ("obstacle_free", "free_from_obstacles"):
            fm = ((points[:, 2] > fz + free_min_height_m) &
                  (points[:, 2] < fz + free_max_height_m))
            fp = points[fm]
            print(f"    Free-space evidence points: {len(fp)} "
                  f"(z band: {free_min_height_m:+.2f}~{free_max_height_m:+.2f}m)")
            if len(fp) < 50:
                print("    Skip")
                continue

            xn, yn = fp[:,0].min()-margin, fp[:,1].min()-margin
            xx, yx = fp[:,0].max()+margin, fp[:,1].max()+margin
            iw = int(np.ceil((xx-xn)/gr)); ih = int(np.ceil((yx-yn)/gr))
            print(f"    Image: {iw}x{ih}px (res={gr}m/px)")

            occ = np.zeros((ih, iw), dtype=np.uint8)
            px = np.clip(((fp[:,0]-xn)/gr).astype(int), 0, iw-1)
            py = np.clip(((fp[:,1]-yn)/gr).astype(int), 0, ih-1)
            occ[py, px] = 1
            raw_fill = occ.sum() / max(1, occ.size)

            radius_px = max(0, int(np.ceil(free_splat_radius_m / gr)))
            occ = np.zeros((ih, iw), dtype=np.uint8)
            occ = _splat_pixels(occ, px, py, radius_px)
            fill = occ.sum() / max(1, occ.size)
            print(f"    Free-space evidence splat: fill {100*raw_fill:.2f}% -> "
                  f"{100*fill:.2f}%, radius={free_splat_radius_m:.2f}m ({radius_px}px)")
        else:
            band = samp.get("floor_band", 0.3)
            band_below = float(samp.get("floor_band_below", band))
            band_above = float(samp.get("floor_band_above", band))
            fm = up_mask & (points[:,2] > fz-band_below) & (points[:,2] < fz+band_above)
            fp = points[fm]
            print(f"    Floor points: {len(fp)} "
                  f"(z band: -{band_below:.2f}m/+{band_above:.2f}m)")
            if len(fp) < 50:
                print("    Skip")
                continue

            xn, yn = fp[:,0].min()-margin, fp[:,1].min()-margin
            xx, yx = fp[:,0].max()+margin, fp[:,1].max()+margin
            iw = int(np.ceil((xx-xn)/gr)); ih = int(np.ceil((yx-yn)/gr))
            print(f"    Image: {iw}x{ih}px (res={gr}m/px)")

            occ = np.zeros((ih, iw), dtype=np.uint8)
            px = np.clip(((fp[:,0]-xn)/gr).astype(int), 0, iw-1)
            py = np.clip(((fp[:,1]-yn)/gr).astype(int), 0, ih-1)
            occ[py, px] = 1
            raw_occ_sum = int(occ.sum())
            raw_fill = raw_occ_sum / max(1, occ.size)

            if oversample_sparse_floor:
                med_nn = _estimate_xy_spacing(fp[:, :2])
                should_splat = raw_fill < sparse_fill_ratio or floor_splat_radius_m > 0
                if should_splat and med_nn is not None:
                    if floor_splat_radius_m > 0:
                        radius_m = floor_splat_radius_m
                    else:
                        radius_m = med_nn * floor_splat_radius_scale
                        radius_m = max(radius_m, floor_splat_min_radius_m, gr)
                        radius_m = min(radius_m, floor_splat_max_radius_m)
                    radius_px = int(np.ceil(radius_m / gr))
                    occ = np.zeros((ih, iw), dtype=np.uint8)
                    occ = _splat_pixels(occ, px, py, radius_px)
                    fill = occ.sum() / max(1, occ.size)
                    print(f"    Sparse floor oversample: fill {100*raw_fill:.2f}% -> "
                          f"{100*fill:.2f}%, median_nn={med_nn:.3f}m, "
                          f"radius={radius_m:.3f}m ({radius_px}px)")
                else:
                    print(f"    Sparse floor oversample: skipped "
                          f"(fill={100*raw_fill:.2f}%, median_nn={med_nn})")

        if occupancy_mode in ("height_slice_obstacle", "slice_obstacle", "height_slice"):
            pass
        else:
            obstacle = np.zeros_like(occ)
        if subtract_obstacles and occupancy_mode not in ("height_slice_obstacle", "slice_obstacle", "height_slice"):
            om = ((points[:, 2] > fz + obstacle_min_height_m) &
                  (points[:, 2] < fz + obstacle_max_height_m))
            op = points[om]
            if len(op) > 0:
                ox = np.clip(((op[:, 0] - xn) / gr).astype(int), 0, iw - 1)
                oy = np.clip(((op[:, 1] - yn) / gr).astype(int), 0, ih - 1)
                obstacle[oy, ox] = 1
                dilate_px = max(0, int(np.ceil(obstacle_dilate_m / gr)))
                if dilate_px > 0:
                    ok = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), dtype=np.uint8)
                    obstacle = binary_dilation(obstacle, structure=ok, iterations=1).astype(np.uint8)
                before = int(occ.sum())
                if occupancy_mode not in ("obstacle_free", "free_from_obstacles"):
                    occ[obstacle > 0] = 0
                removed = before - int(occ.sum())
                if occupancy_mode in ("obstacle_free", "free_from_obstacles"):
                    print(f"    Obstacle mask: {len(op)} pts, {int(obstacle.sum())} px "
                          f"(applied after morphology, "
                          f"z +{obstacle_min_height_m:.2f}~+{obstacle_max_height_m:.2f}m)")
                else:
                    print(f"    Obstacle subtraction: {len(op)} pts, "
                          f"{int(obstacle.sum())} px mask, removed {removed} floor px "
                          f"(z +{obstacle_min_height_m:.2f}~+{obstacle_max_height_m:.2f}m)")

        if occupancy_mode not in ("height_slice_obstacle", "slice_obstacle", "height_slice"):
            kern = np.ones((mk, mk), dtype=np.uint8)
            closed = binary_dilation(occ, structure=kern, iterations=mi).astype(np.uint8)
            closed = binary_erosion(closed, structure=kern, iterations=mi).astype(np.uint8)
            if subtract_obstacles:
                closed[obstacle > 0] = 0

        if kl:
            lbl, n_comp = label(closed)
            if n_comp > 1:
                sizes = np.bincount(lbl.ravel())
                sizes[0] = 0  # background
                largest = int(sizes.argmax())
                closed = (lbl == largest).astype(np.uint8)
                kept = sizes[largest]
                total = int(sizes[1:].sum())
                print(f"    Largest component: {kept}/{total}px ({100*kept/total:.1f}%), "
                      f"removed {n_comp-1} small blobs")

        dm = distance_transform_edt(closed)
        dmx = dm.max()
        print(f"    Dist max: {dmx:.1f}px ({dmx*gr:.2f}m)")

        if distance_thresh_m > 0:
            dist_thresh_px = distance_thresh_m / gr
            corr = (dm > dist_thresh_px).astype(np.uint8)
            print(f"    Corridor threshold: {distance_thresh_m:.2f}m "
                  f"({dist_thresh_px:.1f}px, absolute)")
        else:
            dn = dm/dmx if dmx > 0 else dm
            corr = (dn > dr).astype(np.uint8)
            print(f"    Corridor threshold: ratio={dr:.2f} "
                  f"({dmx*dr*gr:.2f}m effective)")

        if corridor_blur_sigma > 0:
            cs = gaussian_filter(corr.astype(float), sigma=corridor_blur_sigma)
            cb = (cs > corridor_blur_threshold).astype(np.uint8)
        else:
            cb = corr
        print(f"    Corridor pixels: raw={int(corr.sum())}, "
              f"post={int(cb.sum())} "
              f"(blur_sigma={corridor_blur_sigma}, thresh={corridor_blur_threshold})")

        skel = skeletonize(cb > 0).astype(np.uint8)

        if sample_mode == "grid":
            # grid 방식: corridor 내부에 격자선을 깔아서 넓은 커버리지
            step_px = max(1, int(round(skel_grid_spacing_m / gr)))
            grid = np.zeros_like(skel)
            for row in range(step_px // 2, ih, step_px):
                grid[row, :] = 1
            for col in range(step_px // 2, iw, step_px):
                grid[:, col] = 1
            cand_px = np.argwhere((grid & cb) > 0)
            print(f"    Grid candidates (spacing={skel_grid_spacing_m}m, step={step_px}px): "
                  f"{len(cand_px)} px")
        else:
            # skeleton 방식: 순수 skeletonize 경로 (가지가 뻗어나가는 형태)
            cand_px = np.argwhere(skel > 0)
            print(f"    Skeleton candidates: {len(cand_px)} px")

        if len(cand_px) == 0:
            cand_px = np.argwhere(dm > (dmx * 0.5))

        if len(cand_px) == 0: print("    No positions"); continue

        swx = cand_px[:,1]*gr + xn
        swy = cand_px[:,0]*gr + yn

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

        base_z = fz + ch
        for si in sel:
            for yi in range(ny):
                yaw = 2*np.pi*yi/ny
                forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
                up = np.array([0.0, 0.0, 1.0])
                right = np.cross(forward, up)
                right = right / np.linalg.norm(right)
                R_cam = np.column_stack([right, -up, forward])  # pitch=0 고정
                for dz in height_offsets:
                    T = np.eye(4)
                    T[:3, :3] = R_cam
                    T[:3, 3]  = [swx[si], swy[si], base_z + dz]
                    all_vp.append({"id": vid, "pose": T, "floor": fi,
                                   "yaw": yaw, "height_offset_m": dz})
                    vid += 1

        debug_imgs[fi] = {
            "occupancy": occ, "closed": closed, "dist_map": dm, "corridor": cb,
            "skeleton": skel, "selected_px": cand_px[sel] if sel else np.array([]),
            "obstacle": obstacle, "xn": xn, "yn": yn, "gr": gr, "fz": fz,
        }

    n_heights = len(height_offsets)
    n_views_per_pos = ny * n_heights
    print(f"\n  Total viewpoints: {len(all_vp)}  "
          f"(yaw×height = {ny}×{n_heights}, height_offsets={height_offsets} m)")

    nf = len(debug_imgs)
    if nf > 0:
        fig, axes = plt.subplots(nf, 6, figsize=(30, 5*nf))
        if nf == 1: axes = axes.reshape(1, -1)
        if occupancy_mode in ("height_slice_obstacle", "slice_obstacle", "height_slice"):
            titles = ["1.Obstacle slice","2.Free space","3.Dist transform",
                      "4.Corridor","5.Skeleton","6.Viewpoints"]
        else:
            titles = ["1.Occupancy","2.Morphology","3.Dist transform",
                      "4.Corridor","5.Skeleton","6.Viewpoints"]
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
        offset_str = "/".join(f"{dz:+.2f}m" for dz in height_offsets)
        fig.suptitle(f"Step 1: Free Path Corridor — {len(all_vp)} viewpoints  "
                     f"(yaw×height = {ny}×{n_heights}, offsets={offset_str})", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "step1_viewpoints.png"), dpi=150); plt.close()

    fig2, ax2 = plt.subplots(1, 2, figsize=(16, 8))
    sub = max(1, len(points)//30000)
    ax2[0].scatter(points[::sub,0],points[::sub,1],c=points[::sub,2],s=0.3,cmap="viridis",alpha=0.3)
    vpp = np.array([v["pose"][:3,3] for v in all_vp])
    if len(vpp)>0:
        # 포지션당 n_views_per_pos 개 뷰 → 첫 번째만 추출해서 위치 표시
        up_vp = vpp[::n_views_per_pos]
        ax2[0].scatter(up_vp[:,0],up_vp[:,1],c="red",s=15,marker="x",label=f"{len(up_vp)} pos")
    ax2[0].set_title("Top-down: map + viewpoints"); ax2[0].set_aspect("equal"); ax2[0].legend()
    ax2[1].scatter(points[::sub,0],points[::sub,2],c="gray",s=0.3,alpha=0.3)
    if len(vpp)>0: ax2[1].scatter(vpp[::n_views_per_pos,0],vpp[::n_views_per_pos,2],c="red",s=15,marker="x")
    for fz in floors:
        ax2[1].axhline(y=fz, color="green", ls="--", alpha=0.5, label="floor")
        for dz in height_offsets:
            ax2[1].axhline(y=fz+ch+dz, color="blue", ls="--", alpha=0.4,
                           label=f"cam {dz:+.2f}m")
    ax2[1].set_title("Side view"); fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "step1_viewpoints_3d.png"), dpi=150); plt.close()
    print(f"  Saved: step1_viewpoints.png, step1_viewpoints_3d.png")

    data = {"viewpoints": all_vp, "floor_levels": floors, "debug_images": debug_imgs}
    pickle.dump(data, open(os.path.join(output_dir, "step1_data.pkl"), "wb"))

    # ── nerfstudio `ns-render camera-path`용 JSON export ───────────────
    # config에서 dataparser transform 경로 옵션을 받음 (없으면 자동 탐지 시도)
    ns_cfg = config.get("nerfstudio", {})
    dp_path = ns_cfg.get("dataparser_transforms_path", None)
    if dp_path is None:
        # output_dir 또는 그 부모 폴더에서 자동 탐지
        for cand in [
            os.path.join(output_dir, "dataparser_transforms.json"),
            os.path.join(os.path.dirname(output_dir), "dataparser_transforms.json"),
        ]:
            if os.path.exists(cand):
                dp_path = cand
                print(f"  [camera-path] dataparser transform 자동 탐지: {cand}")
                break

    # step0의 T_align 가져오기 (없으면 step0_data.pkl에서 fallback 로드)
    T_align = None
    if step0_data is not None and "T_align" in step0_data:
        T_align = step0_data["T_align"]
    else:
        s0_path = os.path.join(output_dir, "step0_data.pkl")
        if os.path.exists(s0_path):
            try:
                s0 = pickle.load(open(s0_path, "rb"))
                T_align = s0.get("T_align")
                if T_align is not None:
                    print(f"  [camera-path] step0_data.pkl에서 T_align 로드")
            except Exception as e:
                print(f"  [warn] step0_data.pkl 로드 실패: {e}")

    try:
        _export_nerfstudio_camera_path(
            all_vp, config, output_dir,
            dataparser_transform_path=dp_path,
            T_align=T_align,
            out_filename=ns_cfg.get("camera_path_filename", "step1_camera_path.json"),
            invert_align=bool(ns_cfg.get("invert_align", True)),
            invert_dataparser=bool(ns_cfg.get("invert_dataparser", False)),
            flip_camera_frame=bool(ns_cfg.get("flip_camera_frame", True)),
            debug_print_first=bool(ns_cfg.get("debug_print_first", True)),
        )
    except Exception as e:
        print(f"  [warn] nerfstudio camera-path JSON export 실패: {e}")

    return all_vp