import os, pickle
import numpy as np
import open3d as o3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

    # 입력 PLY의 "위(중력 반대)" 방향을 +Z로 가져오는 사전 회전
    # config: alignment.up_axis ∈ {x, x_neg, y, y_neg, z, z_neg}
    #   - 기본 LiDAR 컨벤션: "z_neg" (원래 동작과 동일)
    #   - COLMAP 표준: "y_neg"
    up_axis = align_cfg.get("up_axis", "z_neg")
    if up_axis.lower() not in _AXIS_VECTORS:
        raise ValueError(f"alignment.up_axis must be one of {list(_AXIS_VECTORS.keys())}, got {up_axis!r}")
    R_flip = _axis_to_z_rotation(up_axis)
    pcd.points = o3d.utility.Vector3dVector((R_flip @ np.asarray(pcd.points).T).T)
    if pcd.has_normals():
        pcd.normals = o3d.utility.Vector3dVector((R_flip @ np.asarray(pcd.normals).T).T)
    print(f"  Pre-rotation: up_axis={up_axis} → +Z")

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

    aligned_path = os.path.join(output_dir, "aligned_map.ply")
    o3d.io.write_point_cloud(aligned_path, pcd_aligned, write_ascii=False, compressed=False)
    size_gb = os.path.getsize(aligned_path) / 1e9
    print(f"  Saved: {aligned_path} ({size_gb:.2f} GB)")

    R_total = R @ R_flip
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
