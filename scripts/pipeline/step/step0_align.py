import os, pickle
import numpy as np
import open3d as o3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


_AXIS_VECTORS = {
    "x":     np.array([ 1.0,  0.0,  0.0]),
    "x_neg": np.array([-1.0,  0.0,  0.0]),
    "y":     np.array([ 0.0,  1.0,  0.0]),
    "y_neg": np.array([ 0.0, -1.0,  0.0]),
    "z":     np.array([ 0.0,  0.0,  1.0]),
    "z_neg": np.array([ 0.0,  0.0, -1.0]),
}


def _axis_to_z_rotation(up_axis_key):
    up = _AXIS_VECTORS[up_axis_key.lower()]
    z  = np.array([0.0, 0.0, 1.0])
    v  = np.cross(up, z)
    s  = np.linalg.norm(v)
    c  = float(np.dot(up, z))
    if s < 1e-6:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _z_yaw_rotation(deg):
    rad = np.deg2rad(float(deg))
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _quat_wxyz_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)


def _write_transformed_ply_like_input(input_path, output_path, points_aligned, R_total,
                                      keep_mask=None):
    """Write aligned PLY while preserving Gaussian/SH/opacity/scale properties.

    keep_mask: optional boolean array over the *original* vertices. When given,
    only the masked-in vertices are written. ``points_aligned`` must already be
    the aligned coordinates of those kept vertices (same length as keep_mask.sum()).
    """
    ply = PlyData.read(input_path)
    vertex = np.array(ply["vertex"].data, copy=True)
    if keep_mask is not None:
        vertex = vertex[keep_mask]
    names = vertex.dtype.names or ()

    for axis, values in zip(("x", "y", "z"), points_aligned.T):
        if axis in names:
            vertex[axis] = values.astype(vertex[axis].dtype, copy=False)

    if all(n in names for n in ("nx", "ny", "nz")):
        normals = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1).astype(np.float64)
        normals = (R_total @ normals.T).T
        for axis, values in zip(("nx", "ny", "nz"), normals.T):
            vertex[axis] = values.astype(vertex[axis].dtype, copy=False)

    if all(f"rot_{i}" in names for i in range(4)):
        quat = np.stack([vertex[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
        rot = _quat_wxyz_to_rotmat(quat)
        rot_aligned = np.einsum("ij,njk->nik", R_total, rot)
        quat_xyzw = Rotation.from_matrix(rot_aligned).as_quat()
        quat_wxyz = np.column_stack([quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]])
        for i in range(4):
            vertex[f"rot_{i}"] = quat_wxyz[:, i].astype(vertex[f"rot_{i}"].dtype, copy=False)

    elements = [PlyElement.describe(vertex, "vertex")]
    elements.extend([el for el in ply.elements if el.name != "vertex"])
    PlyData(
        elements,
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    ).write(output_path)

    preserved = len(names)
    gaussian = all(name in names for name in ("opacity", "scale_0", "rot_0"))
    kind = "Gaussian PLY" if gaussian else "PLY"
    print(f"  Preserved {kind} properties: {preserved} vertex fields")


def _compute_crop_mask(points, crop_cfg):
    """정렬된(Z-up) 좌표에서 불필요한 외곽 포인트를 제거하는 boolean keep-mask.

    지원 옵션 (alignment.crop):
      xy_radius : 이 XY 반경(중심 기준)을 넘는 포인트 제거 (top-down 원형 노이즈)
      center    : [cx, cy] XY 중심. null/미지정이면 origin(0,0) 사용
      z_min/z_max : 이 Z 범위를 벗어나는 포인트 제거
      bbox      : [xmin, xmax, ymin, ymax] XY 박스 밖 포인트 제거
    """
    keep = np.ones(len(points), dtype=bool)

    xy_radius = crop_cfg.get("xy_radius", None)
    if xy_radius is not None:
        center = crop_cfg.get("center", None) or [0.0, 0.0]
        cx, cy = float(center[0]), float(center[1])
        r = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        keep &= r <= float(xy_radius)
        print(f"  Crop: xy_radius={xy_radius} center=({cx:.1f},{cy:.1f})")

    bbox = crop_cfg.get("bbox", None)
    if bbox is not None:
        xmin, xmax, ymin, ymax = [float(v) for v in bbox]
        keep &= (points[:, 0] >= xmin) & (points[:, 0] <= xmax)
        keep &= (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
        print(f"  Crop: bbox x[{xmin},{xmax}] y[{ymin},{ymax}]")

    z_min = crop_cfg.get("z_min", None)
    if z_min is not None:
        keep &= points[:, 2] >= float(z_min)
    z_max = crop_cfg.get("z_max", None)
    if z_max is not None:
        keep &= points[:, 2] <= float(z_max)
    if z_min is not None or z_max is not None:
        print(f"  Crop: z[{z_min},{z_max}]")

    return keep


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

    # 입력 PLY의 "위(중력 반대)" 방향을 +Z로 가져오는 사전 회전.
    # COLMAP/GS 좌표를 그대로 유지하고 싶으면 alignment.apply_pre_rotation=false.
    apply_pre_rotation = bool(align_cfg.get("apply_pre_rotation", False))
    if apply_pre_rotation:
        up_axis = align_cfg.get("up_axis", "z_neg")
        if up_axis.lower() not in _AXIS_VECTORS:
            raise ValueError(f"alignment.up_axis must be one of {list(_AXIS_VECTORS.keys())}, got {up_axis!r}")
        R_flip = _axis_to_z_rotation(up_axis)
        pcd.points = o3d.utility.Vector3dVector((R_flip @ np.asarray(pcd.points).T).T)
        if pcd.has_normals():
            pcd.normals = o3d.utility.Vector3dVector((R_flip @ np.asarray(pcd.normals).T).T)
        print(f"  Pre-rotation: up_axis={up_axis} -> +Z")
    else:
        R_flip = np.eye(3)
        print("  Pre-rotation: disabled")

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
    normal_thr = float(align_cfg.get("normal_threshold", 0.6))
    max_trials = int(align_cfg.get("max_plane_trials", 8))
    print(f"  RANSAC on bottom-40% points ({floor_mask.sum()}) "
          f"(dist={rd}, iter={ri}, normal_thr={normal_thr})...")
    orig_indices = np.where(floor_mask)[0]

    work = pcd_floor
    work_indices = orig_indices.copy()
    plane_model = None
    inliers = np.array([], dtype=int)
    for trial in range(max_trials):
        if len(work_indices) < rn:
            break
        cand_model, cand_inliers_sub = work.segment_plane(
            distance_threshold=rd, ransac_n=rn, num_iterations=ri)
        cand_normal = np.asarray(cand_model[:3], dtype=np.float64)
        cand_normal /= np.linalg.norm(cand_normal)
        z_score = abs(float(cand_normal[2]))
        cand_inliers = work_indices[cand_inliers_sub]
        print(f"    plane#{trial+1}: normal={cand_normal.round(4)}  "
              f"|nz|={z_score:.3f}  inliers={len(cand_inliers)}")
        if z_score >= normal_thr:
            plane_model = cand_model
            inliers = cand_inliers
            break
        keep_mask = np.ones(len(work_indices), dtype=bool)
        keep_mask[cand_inliers_sub] = False
        work_indices = work_indices[keep_mask]
        work = pcd.select_by_index(work_indices)

    if plane_model is None:
        raise RuntimeError(
            "Failed to find a floor-like plane. Try lowering alignment.normal_threshold "
            "or check alignment.up_axis.")

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

    yaw_deg = float(align_cfg.get("yaw_deg", 0.0))
    R_yaw = _z_yaw_rotation(yaw_deg)
    if abs(yaw_deg) > 1e-9:
        print(f"  Post-level yaw: {yaw_deg:.2f}° around +Z")

    points = np.asarray(pcd.points)
    points_rotated = (R @ points.T).T
    points_rotated = (R_yaw @ points_rotated.T).T
    floor_z_after = np.median(points_rotated[inliers, 2])
    points_rotated[:, 2] -= floor_z_after
    print(f"  Floor z shifted: {floor_z_after:.4f} → 0")

    # 정렬된 프레임에서 외곽 노이즈(주차장 외벽 원형 포인트 등) 제거
    crop_cfg = align_cfg.get("crop", {}) or {}
    if crop_cfg.get("enable", False):
        keep_mask = _compute_crop_mask(points_rotated, crop_cfg)
        removed = int((~keep_mask).sum())
        print(f"  Crop: removed {removed}/{len(keep_mask)} points "
              f"({100.0*removed/max(len(keep_mask),1):.1f}%), kept {int(keep_mask.sum())}")
        points_rotated = points_rotated[keep_mask]
    else:
        keep_mask = None

    aligned_path = os.path.join(output_dir, "aligned_map.ply")
    R_total = R_yaw @ R @ R_flip
    _write_transformed_ply_like_input(ply_path, aligned_path, points_rotated, R_total,
                                      keep_mask=keep_mask)
    size_gb = os.path.getsize(aligned_path) / 1e9
    print(f"  Saved: {aligned_path} ({size_gb:.2f} GB)")

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
        "source_ply_path": os.path.abspath(ply_path),
        "source_vertex_count": int(len(points_orig)),
        "aligned_vertex_count": int(len(points_rotated)),
    }
    pickle.dump(data, open(os.path.join(output_dir, "step0_data.pkl"), "wb"))
    return data
