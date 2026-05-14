"""
STEP 2: Rendering — unified interface for pointcloud (O3D) and Gaussian Splatting modes.

Usage:
    step2_render(..., mode="pointcloud")   # O3D split-render
    step2_render(..., mode="gs", kapture_dir=..., gs_iters=..., ...)  # 3DGS
    step2_render(..., mode="gs_feature", ...)  # Feature Gaussian Splatting
    step2_render(..., mode="gaussian_ply") # render Gaussian PLY directly
"""
import os, pickle, math, sys, site, json
import numpy as np
import open3d as o3d
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Shared helpers
# =============================================================================
_HEAVY_KEYS = ("rgb", "depth", "feature")


def slim_rendered(rendered):
    """rendered 리스트에서 대용량 배열 키를 제거한 경량 복사본을 반환."""
    return [{k: v for k, v in r.items() if k not in _HEAVY_KEYS}
            for r in rendered]


def _fill_holes(rgb, dep, near_dist, near_fill_radius, fill_radius):
    """O3D 렌더 결과의 빈 픽셀(depth==0)을 가장 가까운 채워진 픽셀로 채움."""
    if near_fill_radius <= 0 and fill_radius <= 0:
        return rgb, dep

    from scipy.ndimage import distance_transform_edt
    dep_f = dep.astype(np.float32)
    empty = dep_f == 0
    if not np.any(empty) or np.all(empty):
        return rgb, dep

    dist, (ny, nx) = distance_transform_edt(empty, return_indices=True)
    nearest_depth = dep_f[ny, nx]
    max_fill = np.where(nearest_depth < near_dist,
                        near_fill_radius, fill_radius).astype(np.float32)
    fill = empty & (dist <= max_fill)

    rgb_out = rgb.copy()
    dep_out = dep_f.copy()
    rgb_out[fill] = rgb[ny[fill], nx[fill]]
    dep_out[fill] = nearest_depth[fill]
    return rgb_out, dep_out


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _make_feature_decoder(feature_dim, high_dim, device):
    import torch

    decoder = torch.nn.Conv2d(
        feature_dim, high_dim, kernel_size=3, stride=1, padding=1, bias=False
    ).to(device)
    return decoder


def _load_superpoint_dense(device):
    """Load the local TorchScript SuperPoint model used as dense feature teacher."""
    import torch

    sp_path = os.path.join(_repo_root(), "models", "superpoint_v1.pt")
    if not os.path.exists(sp_path):
        raise FileNotFoundError(
            f"SuperPoint model not found: {sp_path}. Run scripts/setup_models.py first."
        )
    model = torch.jit.load(sp_path, map_location=device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _extract_superpoint_desc(superpoint, image_hwc):
    """Return L2-normalized dense SuperPoint descriptors [1, 256, H/8, W/8]."""
    import torch
    import torch.nn.functional as F

    gray = (
        image_hwc[..., 0] * 0.299
        + image_hwc[..., 1] * 0.587
        + image_hwc[..., 2] * 0.114
    )
    gray = gray.clamp(0, 1).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        _, desc = superpoint(gray)
    return F.normalize(desc, p=2, dim=1)


def _render_feature_map_gsplat(
    rasterization,
    params,
    viewmat,
    K,
    width,
    height,
    near_plane=0.1,
    far_plane=100.0,
    packed=True,
):
    """Splat per-Gaussian loc_features as rasterization colors."""
    import torch
    import torch.nn.functional as F

    loc_features = F.normalize(params["loc_features"], p=2, dim=-1)
    feat_renders, _, _ = rasterization(
        means=params["means"],
        quats=F.normalize(params["quats"], dim=-1),
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=loc_features,
        viewmats=viewmat,
        Ks=K,
        width=int(width),
        height=int(height),
        near_plane=near_plane,
        far_plane=far_plane,
        packed=packed,
    )
    feat = feat_renders[0].permute(2, 0, 1).contiguous()
    return F.normalize(feat, p=2, dim=0)


def _gs_render_pose_from_viewpoint(pose, config):
    """Convert step1 synthetic viewpoint pose to the GS camera frame."""
    pose = np.asarray(pose, dtype=np.float32)
    render_cfg = config.get("rendering", {})
    apply_roll = bool(render_cfg.get("gs_roll_180", True))
    if not apply_roll:
        return pose, False

    # Step1 viewpoints use image-down = world -Z, while the GS training cameras
    # in this pipeline use image-down = world +Z. Rotate the camera frame around
    # its optical axis, preserving position and forward direction.
    roll_180 = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float32)
    return pose @ roll_180, True


# =============================================================================
# GS kapture parsers
# =============================================================================
def _gs_parse_sensors(sensors_txt):
    """sensors.txt → {sensor_id: {fx,fy,cx,cy,w,h}}"""
    cams = {}
    for line in open(sensors_txt):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 10 or parts[2] != "camera":
            continue
        cams[parts[0]] = {
            "w": int(parts[4]), "h": int(parts[5]),
            "fx": float(parts[6]), "fy": float(parts[7]),
            "cx": float(parts[8]), "cy": float(parts[9]),
        }
    return cams


def _gs_parse_trajectories(traj_txt):
    """trajectories.txt → {(ts, dev_id): T_c2w (4×4)}"""
    poses = {}
    for line in open(traj_txt):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = [x.strip() for x in line.split(",")]
        if len(p) < 9:
            continue
        ts, dev = p[0], p[1]
        qw, qx, qy, qz = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        tx, ty, tz      = float(p[6]), float(p[7]), float(p[8])
        R = np.array([
            [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
            [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
        ], dtype=np.float64)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = [tx, ty, tz]
        poses[(ts, dev)] = T
    return poses


def _gs_parse_records(records_txt):
    """records_camera.txt → [(ts, dev_id, filename)]"""
    records = []
    for line in open(records_txt):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3:
            records.append((p[0], p[1], p[2]))
    return records


def _gs_parse_colmap_cameras(cameras_txt):
    """COLMAP cameras.txt → {camera_id: {fx,fy,cx,cy,w,h}}"""
    cams = {}
    for line in open(cameras_txt):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cam_id, model = parts[0], parts[1]
        w, h = int(parts[2]), int(parts[3])
        if model == "SIMPLE_PINHOLE":
            f = float(parts[4])
            fx = fy = f
            cx, cy = float(parts[5]), float(parts[6])
        elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
            fx, fy = float(parts[4]), float(parts[5])
            cx, cy = float(parts[6]), float(parts[7])
        else:
            # fallback: 첫 파라미터를 focal로 사용
            fx = fy = float(parts[4])
            cx, cy = w / 2.0, h / 2.0
        cams[cam_id] = {"w": w, "h": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return cams


def _gs_parse_colmap_images(images_txt):
    """COLMAP images.txt → (records, poses)
    records: [(image_id, camera_id, name)]
    poses:   {(image_id, camera_id): T_c2w (4×4)}
    COLMAP stores w2c; we invert to c2w here.
    """
    records = []
    poses = {}
    lines = [l.rstrip() for l in open(images_txt) if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        i += 2  # 다음 줄은 POINTS2D — 항상 스킵
        if len(parts) < 10:
            continue
        img_id = parts[0]
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = parts[8]
        name = parts[9]

        # COLMAP quaternion → w2c rotation matrix
        R_w2c = np.array([
            [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
            [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
        ], dtype=np.float64)
        t_w2c = np.array([tx, ty, tz], dtype=np.float64)

        # c2w = inv(w2c)
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c
        T_c2w = np.eye(4)
        T_c2w[:3, :3] = R_c2w
        T_c2w[:3,  3] = t_c2w

        key = (img_id, cam_id)
        records.append((img_id, cam_id, name))
        poses[key] = T_c2w
    return records, poses


# =============================================================================
# Pointcloud render (O3D split-render)
# =============================================================================
def _render_pointcloud(ply_path, viewpoints, config, output_dir, step0_data=None):
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

            ren_lo.setup_camera(intr, extrinsic)
            rgb_lo = np.asarray(ren_lo.render_to_image()).copy()
            dep_lo = np.asarray(ren_lo.render_to_depth_image(z_in_view_space=True)).copy()

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

            use_hi = dep_hi_f < dep_lo_f
            rgb = rgb_lo.copy()
            rgb[use_hi] = rgb_hi[use_hi]

            lo_bg = (rgb_lo.max(axis=2) < 5)
            hi_ok = (rgb_hi.max(axis=2) > 5)
            rgb[lo_bg & hi_ok] = rgb_hi[lo_bg & hi_ok]

            dep = np.minimum(dep_lo_f, dep_hi_f)
            dep[dep >= INF] = 0

        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb.astype(np.float32), 0, 1) * 255).astype(np.uint8)
        if bscale != 1.0:
            rgb = np.clip(rgb.astype(np.float32) * bscale, 0, 255).astype(np.uint8)

        rgb, dep = _fill_holes(rgb, dep, near_dist, near_fill_radius, fill_radius)

        if i == 0:
            print(f"  [debug] rgb mean={rgb.mean():.1f}  dep nonzero={np.count_nonzero(dep)}/{dep.size}")

        rp_ = os.path.join(rd, "rgb",   f"{vp['id']:06d}.png")
        dp_ = os.path.join(rd, "depth", f"{vp['id']:06d}.npy")
        cv2.imwrite(rp_, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(dp_, dep.astype(np.float32))
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
    pickle.dump(rendered, open(os.path.join(output_dir, "step2_data.pkl"), "wb"))
    return rendered


# =============================================================================
# Gaussian Splatting render
# =============================================================================
def _resolve_aligned_ply_path(ply_path, output_dir, step0_data=None):
    if step0_data and "aligned_ply_path" in step0_data:
        return step0_data["aligned_ply_path"]
    rp = os.path.join(output_dir, "aligned_map.ply")
    return rp if os.path.exists(rp) else ply_path


def _load_gaussian_ply(ply_path, device, opacity_min=0.0, scale_mul=1.0):
    import torch
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        raise ValueError(f"PLY에 vertex element가 없습니다: {ply_path}")
    v = ply["vertex"].data
    names = set(v.dtype.names or [])

    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(
            "Gaussian PLY 필드가 부족합니다. "
            f"missing={missing}. pointcloud 렌더가 아니라 Gaussian PLY 렌더에는 "
            "x/y/z, opacity, scale_*, rot_* 필드가 필요합니다."
        )

    means_np = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scales_log_np = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    quats_np = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    opacity_raw_np = np.asarray(v["opacity"], dtype=np.float32)

    sh_degree = None
    if {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
        dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
        rest_names = sorted(
            [name for name in names if name.startswith("f_rest_")],
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        num_rest = len(rest_names)
        num_sh = num_rest // 3 + 1
        degree_f = np.sqrt(num_sh) - 1
        if num_rest > 0 and num_rest % 3 == 0 and abs(degree_f - round(degree_f)) < 1e-6:
            sh_degree = int(round(degree_f))
            rest = np.stack([v[name] for name in rest_names], axis=1).astype(np.float32)
            colors_np = np.zeros((len(means_np), num_sh, 3), dtype=np.float32)
            colors_np[:, 0, :] = dc
            colors_np[:, 1:, :] = rest.reshape(len(means_np), 3, num_sh - 1).transpose(0, 2, 1)
        else:
            sh_degree = 0
            colors_np = dc[:, None, :]
    elif {"red", "green", "blue"} <= names:
        colors_np = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32)
        if colors_np.max() > 1.0:
            colors_np /= 255.0
        colors_np = np.clip(colors_np, 0.0, 1.0)
    else:
        colors_np = np.full((len(means_np), 3), 0.5, dtype=np.float32)

    opacities_np = 1.0 / (1.0 + np.exp(-opacity_raw_np))
    valid = np.isfinite(means_np).all(axis=1)
    valid &= np.isfinite(scales_log_np).all(axis=1)
    valid &= np.isfinite(quats_np).all(axis=1)
    valid &= np.isfinite(opacities_np)
    valid &= opacities_np > float(opacity_min)

    if not np.any(valid):
        raise ValueError(f"렌더링할 Gaussian이 없습니다: opacity_min={opacity_min}")

    means_np = means_np[valid]
    scales_log_np = scales_log_np[valid]
    quats_np = quats_np[valid]
    colors_np = colors_np[valid]
    opacity_raw_np = opacity_raw_np[valid]
    opacities_np = opacities_np[valid]

    return {
        "means": torch.from_numpy(means_np).to(device),
        "scales": torch.from_numpy(np.exp(scales_log_np) * float(scale_mul)).to(device),
        "quats": torch.from_numpy(quats_np).to(device),
        "opacities": torch.from_numpy(opacities_np).to(device),
        "colors": torch.from_numpy(colors_np).to(device),
        "sh_degree": sh_degree,
        "num_total": len(v),
        "num_valid": len(means_np),
    }


def _save_render_preview(rendered, config, output_dir, title, filename):
    if not rendered:
        return
    cam_cfg = config["camera"]
    ns = min(8, len(rendered))
    idx = np.linspace(0, len(rendered)-1, ns, dtype=int)
    fig, ax = plt.subplots(2, ns, figsize=(4*ns, 8))
    if ns == 1:
        ax = ax.reshape(2, 1)
    for c, ii in enumerate(idx):
        r = rendered[ii]
        ax[0, c].imshow(cv2.cvtColor(cv2.imread(r["rgb_path"]), cv2.COLOR_BGR2RGB))
        ax[0, c].set_title(f"RGB #{r['id']}", fontsize=9)
        ax[0, c].axis("off")
        dv = np.load(r["depth_path"])
        dv[dv > cam_cfg.get("depth_max", 10)] = 0
        ax[1, c].imshow(dv, cmap="plasma")
        ax[1, c].set_title("Depth", fontsize=9)
        ax[1, c].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def _depth_to_vis(depth, max_depth=20.0):
    dep = np.asarray(depth, dtype=np.float32).copy()
    valid = np.isfinite(dep) & (dep > 0.0) & (dep <= float(max_depth))
    if not np.any(valid):
        return np.zeros((*dep.shape, 3), dtype=np.uint8)

    lo = float(np.percentile(dep[valid], 2))
    hi = float(np.percentile(dep[valid], 98))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.zeros_like(dep, dtype=np.float32)
    norm[valid] = np.clip((dep[valid] - lo) / (hi - lo), 0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    return vis


def _save_depth_vis(depth, path, max_depth=20.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, _depth_to_vis(depth, max_depth=max_depth))


def _focal2fov(focal, pixels):
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def _get_world_view_transform(c2w, device):
    w2c = np.linalg.inv(c2w).astype(np.float32)
    return torch_from_numpy(w2c, device).transpose(0, 1)


def torch_from_numpy(array, device):
    import torch
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device)


def _get_projection_matrix(znear, zfar, fovx, fovy, device):
    import torch

    tan_half_y = math.tan(fovy * 0.5)
    tan_half_x = math.tan(fovx * 0.5)
    top = tan_half_y * znear
    bottom = -top
    right = tan_half_x * znear
    left = -right

    P = torch.zeros(4, 4, device=device)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def _render_gaussian_ply_diff(gs, viewpoints, config, output_dir, rd, device):
    import torch
    import torch.nn.functional as F
    GaussianRasterizationSettings, GaussianRasterizer = _import_diff_gaussian_rasterizer()

    cam_cfg = config["camera"]
    W_out = int(cam_cfg["width"])
    H_out = int(cam_cfg["height"])
    near_plane = float(cam_cfg.get("depth_min", 0.3))
    far_plane = float(cam_cfg.get("depth_max", 100.0))
    fovx = _focal2fov(cam_cfg["fx"], W_out)
    fovy = _focal2fov(cam_cfg["fy"], H_out)
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)
    bg = torch.zeros(3, dtype=torch.float32, device=device)

    # nerfstudio splatfacto-style: pose convention 처리
    # nerfstudio가 export한 가우시안 PLY는 OpenGL convention (Y-up, Z-back)
    # diff_gaussian_rasterization도 OpenCV convention (Y-down, Z-forward)을 기대
    rend_cfg = config.get("rendering", {})
    pose_convention = str(rend_cfg.get("pose_convention", "opengl")).lower()
    depth_alpha_min = float(rend_cfg.get("gaussian_depth_alpha_min", 0.05))
    # OpenGL → OpenCV 변환 (Y, Z축 반전; X축은 동일)
    opengl_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    means = gs["means"].float()
    rotations = F.normalize(gs["quats"].float(), dim=-1)
    scales = gs["scales"].float()
    opacities = gs["opacities"].float().unsqueeze(-1)
    colors = gs["colors"].float()
    sh_degree = gs["sh_degree"]

    rendered = []
    print(f"  Rendering {len(viewpoints)} viewpoints with diff_gaussian_rasterization ...")
    print(f"    pose_convention={pose_convention}")
    with torch.no_grad():
        for i, vp in enumerate(viewpoints):
            pose = vp["pose"].astype(np.float32)  # c2w

            # OpenGL convention인 경우 OpenCV로 변환
            if pose_convention == "opengl":
                pose_cv = pose @ opengl_to_opencv
            else:
                pose_cv = pose

            world_view = _get_world_view_transform(pose_cv, device)
            proj = _get_projection_matrix(near_plane, far_plane, fovx, fovy, device).transpose(0, 1)
            full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
            campos = torch_from_numpy(pose_cv[:3, 3], device)

            settings = GaussianRasterizationSettings(
                image_height=H_out,
                image_width=W_out,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg,
                scale_modifier=1.0,
                viewmatrix=world_view,
                projmatrix=full_proj,
                sh_degree=int(sh_degree or 0),
                campos=campos,
                prefiltered=False,
                debug=False,
            )
            rasterizer = GaussianRasterizer(raster_settings=settings)
            means2d = torch.zeros_like(means, dtype=means.dtype, device=device)

            kwargs = {
                "means3D": means,
                "means2D": means2d,
                "opacities": opacities,
                "scales": scales,
                "rotations": rotations,
                "cov3D_precomp": None,
            }
            if colors.ndim == 3:
                rendered_rgb, _ = rasterizer(shs=colors, colors_precomp=None, **kwargs)
            else:
                rendered_rgb, _ = rasterizer(shs=None, colors_precomp=colors, **kwargs)

            xyz_h = torch.cat([means, torch.ones_like(means[:, :1])], dim=-1)
            cam_xyz = xyz_h @ world_view
            depth_color = cam_xyz[:, 2:3].clamp(min=0.0).repeat(1, 3)
            rendered_depth3, _ = rasterizer(shs=None, colors_precomp=depth_color, **kwargs)
            rendered_alpha3, _ = rasterizer(
                shs=None, colors_precomp=torch.ones_like(means[:, :3]), **kwargs)

            rgb_np = (rendered_rgb.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            depth_sum = rendered_depth3[0].cpu().numpy().astype(np.float32)
            alpha = rendered_alpha3[0].cpu().numpy().astype(np.float32)
            depth_np = np.zeros_like(depth_sum, dtype=np.float32)
            valid = alpha > depth_alpha_min
            depth_np[valid] = depth_sum[valid] / np.maximum(alpha[valid], 1e-6)
            depth_np[(depth_np < near_plane) | (depth_np > far_plane) | ~np.isfinite(depth_np)] = 0.0

            rp_ = os.path.join(rd, "rgb", f"{vp['id']:06d}.png")
            dp_ = os.path.join(rd, "depth", f"{vp['id']:06d}.npy")
            dv_ = os.path.join(rd, "depth_vis", f"{vp['id']:06d}.png")
            cv2.imwrite(rp_, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
            np.save(dp_, depth_np)
            _save_depth_vis(depth_np, dv_, max_depth=cam_cfg.get("depth_max", 20.0))
            rendered.append({
                "id": vp["id"],
                "pose": vp["pose"],
                "rgb_path": rp_,
                "depth_path": dp_,
            })
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(viewpoints)}")
    return rendered


def _import_diff_gaussian_rasterizer():
    import glob

    candidates = []
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    candidates.extend([
        os.path.expanduser("~/.local/lib/python3.10/site-packages"),
        os.path.join(os.path.expanduser("~"), "miniforge3", "envs", "render_loc",
                     "lib", "python3.10", "site-packages"),
        os.path.join(os.getcwd(), "scaffold_gs_result", "2026-04-30_15:53:27",
                     "backup", "submodules", "diff-gaussian-rasterization"),
        os.path.join(os.getcwd(), "third_party", "scaffold_gs", "submodules",
                     "diff-gaussian-rasterization"),
    ])

    compiled_candidates = []
    source_candidates = []
    for path in candidates:
        pkg_dir = os.path.join(path, "diff_gaussian_rasterization") if path else ""
        if not path or not os.path.isdir(pkg_dir):
            continue
        if glob.glob(os.path.join(pkg_dir, "_C*.so")):
            compiled_candidates.append(path)
        else:
            source_candidates.append(path)

    for name in list(sys.modules):
        if name == "diff_gaussian_rasterization" or name.startswith("diff_gaussian_rasterization."):
            sys.modules.pop(name, None)

    for path in reversed(compiled_candidates + source_candidates):
        if path in sys.path:
            sys.path.remove(path)
        if os.path.exists(path):
            sys.path.insert(0, path)

    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    return GaussianRasterizationSettings, GaussianRasterizer


def _render_gaussian_ply_gsplat(gs, viewpoints, config, output_dir, rd, device):
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization

    cam_cfg = config["camera"]
    W_out = int(cam_cfg["width"])
    H_out = int(cam_cfg["height"])
    near_plane = float(cam_cfg.get("depth_min", 0.3))
    far_plane = float(cam_cfg.get("depth_max", 100.0))
    K_render = torch.tensor([
        [cam_cfg["fx"], 0, cam_cfg["cx"]],
        [0, cam_cfg["fy"], cam_cfg["cy"]],
        [0, 0, 1],
    ], device=device).unsqueeze(0).float()

    # nerfstudio splatfacto-style 렌더링 옵션
    rend_cfg = config.get("rendering", {})
    # nerfstudio splatfacto default: "antialiased"
    rasterize_mode = str(rend_cfg.get("rasterize_mode", "antialiased")).lower()
    # nerfstudio가 export한 PLY는 OpenGL pose convention (Y-up, Z-back)
    # gsplat은 OpenCV pose convention (Y-down, Z-forward)을 받음
    pose_convention = str(rend_cfg.get("pose_convention", "opengl")).lower()
    depth_alpha_min = float(rend_cfg.get("gaussian_depth_alpha_min", 0.05))
    # background color (nerfstudio eval default: black)
    bg_color = rend_cfg.get("background_color", [0.0, 0.0, 0.0])

    # OpenGL → OpenCV 변환 행렬 (Y, Z축 반전; X축은 동일)
    opengl_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    bg_tensor = torch.tensor(bg_color, device=device).float().unsqueeze(0)  # (1, 3)

    rendered = []
    print(f"  Rendering {len(viewpoints)} viewpoints with gsplat ...")
    print(f"    rasterize_mode={rasterize_mode}, pose_convention={pose_convention}")
    with torch.no_grad():
        means = gs["means"].float()
        quats = F.normalize(gs["quats"].float(), dim=-1)
        scales = gs["scales"].float()
        opacities = gs["opacities"].float()
        colors = gs["colors"].float()
        sh_degree = gs["sh_degree"]

        for i, vp in enumerate(viewpoints):
            pose = vp["pose"].astype(np.float32)  # c2w

            # OpenGL convention인 경우 OpenCV로 변환
            if pose_convention == "opengl":
                c2w_opencv = pose @ opengl_to_opencv
            else:
                c2w_opencv = pose

            # gsplat은 viewmat (w2c)을 받음
            w2c = np.linalg.inv(c2w_opencv)
            viewmat = torch.from_numpy(w2c).to(device).unsqueeze(0).float()

            renders, alphas, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat,
                Ks=K_render,
                width=W_out,
                height=H_out,
                sh_degree=sh_degree,
                near_plane=near_plane,
                far_plane=far_plane,
                render_mode="RGB+D",
                rasterize_mode=rasterize_mode,   # nerfstudio splatfacto default
                packed=False,                     # nerfstudio splatfacto default
                backgrounds=bg_tensor,            # background color 명시
            )

            rgb_np = (renders[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            depth_np = renders[0, ..., 3].cpu().numpy().astype(np.float32)
            alpha_np = alphas[0].detach().cpu().numpy().astype(np.float32).squeeze()
            depth_np[alpha_np <= depth_alpha_min] = 0.0
            depth_np[(depth_np < near_plane) | (depth_np > far_plane) | ~np.isfinite(depth_np)] = 0.0

            rp_ = os.path.join(rd, "rgb", f"{vp['id']:06d}.png")
            dp_ = os.path.join(rd, "depth", f"{vp['id']:06d}.npy")
            dv_ = os.path.join(rd, "depth_vis", f"{vp['id']:06d}.png")
            cv2.imwrite(rp_, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
            np.save(dp_, depth_np)
            _save_depth_vis(depth_np, dv_, max_depth=cam_cfg.get("depth_max", 20.0))

            rendered.append({
                "id": vp["id"],
                "pose": vp["pose"],
                "rgb_path": rp_,
                "depth_path": dp_,
            })
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(viewpoints)}")
    return rendered


def _render_gaussian_ply(ply_path, viewpoints, config, output_dir, step0_data=None):
    import torch

    print("\n" + "="*60 + "\nSTEP 2: Rendering (Gaussian PLY direct)\n" + "="*60)

    rp = _resolve_aligned_ply_path(ply_path, output_dir, step0_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rend_cfg = config.get("rendering", {})
    opacity_min = float(rend_cfg.get("gaussian_opacity_min", 0.0))
    scale_mul = float(rend_cfg.get("gaussian_scale_mul", 1.0))
    backend = str(rend_cfg.get("gaussian_backend", "diff")).lower()

    print(f"  From   : {rp}")
    print(f"  Device : {device}")
    print(f"  Backend: {backend}")
    print(f"  Params : opacity_min={opacity_min}, scale_mul={scale_mul}")

    gs = _load_gaussian_ply(rp, device, opacity_min=opacity_min, scale_mul=scale_mul)
    print(f"  Gaussians: {gs['num_valid']:,}/{gs['num_total']:,}")
    print(f"  SH degree: {gs['sh_degree'] if gs['sh_degree'] is not None else 'precomputed RGB'}")

    rd = os.path.join(output_dir, "rendered_gaussian_ply")
    os.makedirs(os.path.join(rd, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(rd, "depth"), exist_ok=True)
    os.makedirs(os.path.join(rd, "depth_vis"), exist_ok=True)

    if backend in ("diff", "diff_gaussian", "diff-gaussian"):
        try:
            rendered = _render_gaussian_ply_diff(gs, viewpoints, config, output_dir, rd, device)
        except ImportError as exc:
            raise ImportError(
                "diff_gaussian_rasterization을 import하지 못했습니다. "
                "현재 환경에서는 gsplat도 cuda_runtime.h가 없어 빌드 실패하므로 자동 fallback하지 않습니다. "
                "먼저 다음을 확인하세요: "
                "python -c 'import diff_gaussian_rasterization; print(diff_gaussian_rasterization.__file__)'"
            ) from exc
    elif backend == "gsplat":
        rendered = _render_gaussian_ply_gsplat(gs, viewpoints, config, output_dir, rd, device)
    else:
        raise ValueError(f"알 수 없는 gaussian_backend: {backend}")

    _save_render_preview(
        rendered, config, output_dir,
        f"Step 2: Gaussian PLY Rendered — {len(rendered)} images",
        "step2_gaussian_ply_rendered.png",
    )

    pkl_path = os.path.join(output_dir, "step2_gaussian_ply_data.pkl")
    pickle.dump(rendered, open(pkl_path, "wb"))
    pickle.dump(rendered, open(os.path.join(output_dir, "step2_data.pkl"), "wb"))
    print(f"  Saved: {pkl_path}  ({len(rendered)} entries)")
    return rendered


def _resolve_ns_config(config, output_dir, ns_config_path=None):
    if ns_config_path:
        return os.path.abspath(ns_config_path)

    ns_cfg = config.get("nerfstudio", {})
    for key in ("load_config", "config", "config_path"):
        if ns_cfg.get(key):
            return os.path.abspath(ns_cfg[key])

    import glob
    candidates = sorted(
        glob.glob(os.path.join(os.getcwd(), "nerfstudio", "outputs", "*", "*", "*", "config.yml"))
    )
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        "nerfstudio config.yml을 찾지 못했습니다. "
        "--ns_config 또는 config.nerfstudio.load_config 를 지정하세요."
    )


def _resolve_ns_camera_path(config, output_dir, ns_camera_path=None):
    if ns_camera_path:
        return os.path.abspath(ns_camera_path)
    ns_cfg = config.get("nerfstudio", {})
    filename = ns_cfg.get("camera_path_filename", "step1_camera_path.json")
    return os.path.abspath(os.path.join(output_dir, filename))


def _resolve_ns_train_cwd(ns_config_path, ns_trainer_config):
    """원 학습 시점의 cwd를 추정한다.

    config.yml 위치는 `<train_cwd>/<output_dir>/<exp>/<method>/<timestamp>/config.yml`
    이므로 output_dir 길이만큼 위로 더 올라가면 학습 cwd가 된다.
    """
    from pathlib import Path
    cfg_path = Path(ns_config_path).resolve()
    output_dir = getattr(ns_trainer_config, "output_dir", None)
    extra = len(Path(str(output_dir)).parts) if output_dir is not None else 1
    levels = 4 + extra  # timestamp / method / exp / output_dir(...) → train_cwd
    parents = cfg_path.parents
    if levels < len(parents):
        return parents[levels - 1]
    return cfg_path.parent


def _render_nerfstudio(viewpoints, config, output_dir,
                       ns_config_path=None, ns_camera_path=None):
    import torch
    import yaml
    from pathlib import Path
    from nerfstudio.cameras.camera_paths import get_path_from_json
    from nerfstudio.scripts.render import get_crop_from_json, renderers
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.utils import colormaps

    print("\n" + "="*60 + "\nSTEP 2: Rendering (Nerfstudio ns-render path)\n" + "="*60)

    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    ns_config = _resolve_ns_config(config, output_dir, ns_config_path)
    camera_path_file = _resolve_ns_camera_path(config, output_dir, ns_camera_path)
    if not os.path.exists(camera_path_file):
        raise FileNotFoundError(
            f"ns-render camera path JSON 없음: {camera_path_file}. "
            "먼저 step1_viewpoints를 실행해 step1_camera_path.json을 만드세요."
        )

    rd = os.path.join(output_dir, "rendered_ns")
    rgb_dir = os.path.join(rd, "rgb")
    depth_dir = os.path.join(rd, "depth")
    depth_vis_dir = os.path.join(rd, "depth_vis")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(depth_vis_dir, exist_ok=True)

    rd_abs = os.path.abspath(rd)

    print(f"  ns config   : {ns_config}")
    print(f"  camera path : {camera_path_file}")
    print(f"  output      : {rd_abs}")
    print("  Equivalent RGB command:")
    print("    TORCH_COMPILE_DISABLE=1 ns-render camera-path \\")
    print(f"      --load-config {ns_config} \\")
    print(f"      --camera-path-filename {camera_path_file} \\")
    print(f"      --output-path {rd_abs}/rgb_cli \\")
    print("      --output-format images --image-format png --rendered-output-names rgb")
    print("  Equivalent depth command:")
    print("    TORCH_COMPILE_DISABLE=1 ns-render camera-path \\")
    print(f"      --load-config {ns_config} \\")
    print(f"      --camera-path-filename {camera_path_file} \\")
    print(f"      --output-path {rd_abs}/depth_cli \\")
    print("      --output-format images --image-format png --rendered-output-names depth")

    # ── 데이터파서가 참조하는 상대 경로(예: data: images_metric)는 학습 시점의
    #    cwd 기준이므로, eval_setup 호출 동안 잠시 그 디렉터리로 이동한다.
    trainer_cfg = yaml.load(Path(ns_config).read_text(), Loader=yaml.Loader)
    train_cwd = _resolve_ns_train_cwd(ns_config, trainer_cfg)
    prev_cwd = os.getcwd()
    if os.path.isdir(train_cwd):
        print(f"  train cwd   : {train_cwd}  (chdir for dataparser relative paths)")
        os.chdir(train_cwd)
    try:
        _, pipeline, _, _ = eval_setup(
            Path(ns_config),
            eval_num_rays_per_chunk=config.get("nerfstudio", {}).get("eval_num_rays_per_chunk", None),
            test_mode="inference",
        )
    finally:
        os.chdir(prev_cwd)
    pipeline.eval()

    with open(camera_path_file, "r", encoding="utf-8") as f:
        camera_path_json = json.load(f)
    crop_data = get_crop_from_json(camera_path_json)
    cameras = get_path_from_json(camera_path_json)
    cameras = cameras.to(pipeline.device)

    n_render = int(cameras.size)
    if len(viewpoints) != n_render:
        print(f"  [WARN] viewpoints({len(viewpoints)}) != ns cameras({n_render}); "
              f"앞쪽 {min(len(viewpoints), n_render)}개만 pkl에 저장합니다.")
    n = min(len(viewpoints), n_render)

    cam_cfg = config["camera"]
    dmin = float(cam_cfg.get("depth_min", 0.0))
    dmax = float(cam_cfg.get("depth_max", 100.0))
    ns_cfg = config.get("nerfstudio", {})
    accum_min = float(ns_cfg.get("depth_accumulation_min",
                      config.get("rendering", {}).get("ns_depth_accumulation_min", 0.05)))
    depth_output_name = str(ns_cfg.get("depth_output_name", "depth"))

    rendered = []
    print(f"  Rendering {n} cameras with nerfstudio pipeline ...")
    with torch.no_grad():
        for i in range(n):
            obb_box = crop_data.obb if crop_data is not None else None
            if crop_data is not None:
                ctx = renderers.background_color_override_context(
                    crop_data.background_color.to(pipeline.device)
                )
            else:
                ctx = torch.no_grad()

            with ctx:
                outputs = pipeline.model.get_outputs_for_camera(
                    cameras[i:i + 1], obb_box=obb_box)

            if "rgb" not in outputs:
                raise KeyError(f"nerfstudio output에 rgb가 없습니다: {outputs.keys()}")
            if depth_output_name not in outputs:
                fallback = next((k for k in ("depth", "raw-depth", "expected_depth")
                                 if k in outputs), None)
                if fallback is None:
                    raise KeyError(f"nerfstudio output에 depth가 없습니다: {outputs.keys()}")
                depth_key = fallback
            else:
                depth_key = depth_output_name

            rgb = outputs["rgb"].detach().cpu().numpy()
            rgb = np.asarray(rgb)
            if rgb.ndim == 4:
                rgb = rgb[0]
            rgb_np = (np.clip(rgb[..., :3], 0.0, 1.0) * 255).astype(np.uint8)

            depth_t = outputs[depth_key]
            if depth_t.dim() == 4:
                depth_t = depth_t[0]
            if depth_t.dim() == 2:
                depth_t = depth_t.unsqueeze(-1)
            acc_t = outputs.get("accumulation")
            if acc_t is not None:
                if acc_t.dim() == 4:
                    acc_t = acc_t[0]
                if acc_t.dim() == 2:
                    acc_t = acc_t.unsqueeze(-1)

            # ns-render와 동일한 depth 시각화 (turbo + accumulation 블렌드)
            depth_color = colormaps.apply_depth_colormap(
                depth_t,
                accumulation=acc_t,
                near_plane=None,
                far_plane=None,
                colormap_options=colormaps.ColormapOptions(),
            ).detach().cpu().numpy()
            depth_color_u8 = (np.clip(depth_color, 0.0, 1.0) * 255).astype(np.uint8)

            depth_np = depth_t.detach().cpu().numpy().squeeze().astype(np.float32)
            if depth_np.ndim != 2:
                raise ValueError(f"Unexpected depth shape for {depth_key}: {depth_np.shape}")

            if acc_t is not None:
                acc_np = acc_t.detach().cpu().numpy().squeeze()
                depth_np[acc_np <= accum_min] = 0.0
            depth_np[(depth_np < dmin) | (depth_np > dmax) | ~np.isfinite(depth_np)] = 0.0

            stem = f"{viewpoints[i]['id']:06d}"
            rp_ = os.path.join(rgb_dir, f"{stem}.png")
            dp_ = os.path.join(depth_dir, f"{stem}.npy")
            dv_ = os.path.join(depth_vis_dir, f"{stem}.png")
            cv2.imwrite(rp_, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
            np.save(dp_, depth_np)
            cv2.imwrite(dv_, cv2.cvtColor(depth_color_u8, cv2.COLOR_RGB2BGR))

            rendered.append({
                "id": viewpoints[i]["id"],
                "pose": viewpoints[i]["pose"],
                "rgb_path": rp_,
                "depth_path": dp_,
                "depth_vis_path": dv_,
                "depth_source": f"nerfstudio:{depth_key}",
            })
            if (i + 1) % 50 == 0 or i + 1 == n:
                valid = np.isfinite(depth_np) & (depth_np > 0.0)
                print(f"  {i+1}/{n}  depth_valid={100.0 * valid.mean():.1f}%")

    _save_render_preview(
        rendered, config, output_dir,
        f"Step 2: Nerfstudio Rendered — {len(rendered)} images",
        "step2_ns_rendered.png",
    )

    pkl_path = os.path.join(output_dir, "step2_ns_data.pkl")
    pickle.dump(rendered, open(pkl_path, "wb"))
    pickle.dump(rendered, open(os.path.join(output_dir, "step2_data.pkl"), "wb"))
    print(f"  Saved: {pkl_path}  ({len(rendered)} entries)")
    print(f"  Saved: {os.path.join(output_dir, 'step2_data.pkl')}")
    return rendered


def _render_gs(ply_path, viewpoints, config, output_dir,
               kapture_dir=None, step0_data=None,
               gs_iters=30000, sh_degree=0,
               train_img_size=512, subsample=1,
               voxel_size=None,
               use_ppisp=False,
               colmap_dir=None,
               save_interval=5000,
               log_interval=100,
               wandb_project=None,
               wandb_run_name=None,
               feature_splat=False,
               feature_dim=64,
               feature_weight=1.0,
               feature_stride=8):
    import torch
    import torch.nn.functional as F
    from pytorch_msssim import ssim as _ssim_fn
    try:
        from gsplat import rasterization
        from gsplat.strategy import DefaultStrategy
    except ImportError:
        raise ImportError("gsplat 미설치: pip install gsplat")

    title = "STEP 2-FGS: Feature Gaussian Splatting Rendering" if feature_splat \
            else "STEP 2-GS: 3D Gaussian Splatting Rendering"
    print("\n" + "="*60 + f"\n{title}\n" + "="*60)

    # ── 0. 경로/디바이스 설정
    if step0_data and "aligned_ply_path" in step0_data:
        rp = step0_data["aligned_ply_path"]
    else:
        rp = os.path.join(output_dir, "aligned_map.ply")
        if not os.path.exists(rp):
            rp = ply_path
    T_align = np.array(step0_data["T_align"], dtype=np.float64) \
              if step0_data and "T_align" in step0_data else np.eye(4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  PLY     : {rp}")
    print(f"  Kapture : {kapture_dir}")
    print(f"  Device  : {device}  iters={gs_iters}  sh_degree={sh_degree}")
    if feature_splat:
        print(f"  Feature : SuperPoint teacher, loc_dim={feature_dim}, "
              f"weight={feature_weight}, stride={feature_stride}")
    gs_roll_180 = bool(config.get("rendering", {}).get("gs_roll_180", True))
    print(f"  Camera  : gs_roll_180={gs_roll_180}")

    rd = os.path.join(output_dir, "rendered_gs")
    gs_ckpt = os.path.join(output_dir, "gaussians_feature.pt" if feature_splat else "gaussians.pt")
    os.makedirs(os.path.join(rd, "rgb"),   exist_ok=True)
    os.makedirs(os.path.join(rd, "depth"), exist_ok=True)
    if feature_splat:
        os.makedirs(os.path.join(rd, "feature"), exist_ok=True)

    # ── 1. PLY 로드 → Gaussian 초기화
    pcd = o3d.io.read_point_cloud(rp)
    if voxel_size is not None and voxel_size > 0:
        n_before = len(pcd.points)
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"  Voxel downsample ({voxel_size}m): {n_before:,} → {len(pcd.points):,} pts")
    pts = np.asarray(pcd.points, dtype=np.float32)
    clr = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() \
          else np.full((len(pts), 3), 0.5, dtype=np.float32)
    N = len(pts)
    print(f"  Gaussians: {N:,} pts")

    # scale 초기화: 인접점 거리 기반
    from sklearn.neighbors import KDTree as _KDTree
    sample_pts = pts[np.random.choice(N, min(N, 50000), replace=False)]
    dists, _ = _KDTree(sample_pts).query(sample_pts, k=4)
    init_scale = float(np.log(dists[:, 1:].mean() * 0.5 + 1e-6))

    means     = torch.nn.Parameter(torch.from_numpy(pts).to(device))
    scales    = torch.nn.Parameter(torch.full((N, 3), init_scale, device=device))
    quats     = torch.nn.Parameter(
                    torch.cat([torch.ones(N,1), torch.zeros(N,3)], dim=1).to(device))
    opacities = torch.nn.Parameter(torch.full((N,), -2.0, device=device))

    num_sh = (sh_degree + 1) ** 2
    SH_C0  = 0.28209479177387814
    colors_sh = torch.zeros(N, num_sh, 3, device=device)
    colors_sh[:, 0, :] = torch.from_numpy((clr - 0.5) / SH_C0).to(device)
    colors_sh = torch.nn.Parameter(colors_sh)
    loc_features = None
    feature_decoder = None
    if feature_splat:
        loc_init = torch.randn(N, feature_dim, device=device)
        loc_init = F.normalize(loc_init, p=2, dim=-1)
        loc_features = torch.nn.Parameter(loc_init)
        feature_decoder = _make_feature_decoder(feature_dim, 256, device)

    # ── 2. 학습된 모델이 있으면 로드, 없으면 학습
    if os.path.exists(gs_ckpt):
        print(f"  [GS] 저장된 모델 로드: {gs_ckpt}")
        ckpt = torch.load(gs_ckpt, map_location=device)
        means     = torch.nn.Parameter(ckpt["means"])
        scales    = torch.nn.Parameter(ckpt["scales"])
        quats     = torch.nn.Parameter(ckpt["quats"])
        opacities = torch.nn.Parameter(ckpt["opacities"])
        colors_sh = torch.nn.Parameter(ckpt["colors_sh"])
        if feature_splat:
            if "loc_features" not in ckpt or "feature_decoder" not in ckpt:
                raise RuntimeError(
                    f"{gs_ckpt} is not a feature Gaussian checkpoint. "
                    "Remove it or rerun with --render_mode gs."
                )
            loc_features = torch.nn.Parameter(ckpt["loc_features"])
            feature_decoder = _make_feature_decoder(
                int(loc_features.shape[-1]), 256, device
            )
            feature_decoder.load_state_dict(ckpt["feature_decoder"])
            feature_decoder.to(device)
        N = len(means)
        print(f"  [GS] 로드된 GS: {N:,} pts")
    else:
        # ── 3. 매핑 데이터 로드 (kapture 또는 COLMAP)
        if colmap_dir:
            # ── COLMAP 포맷
            print(f"  [GS] COLMAP 포맷 사용: {colmap_dir}")
            sparse_dir  = os.path.join(colmap_dir, "sparse", "0")
            images_root = os.path.join(colmap_dir, "images")
            cameras_txt = os.path.join(sparse_dir, "cameras.txt")
            images_txt  = os.path.join(sparse_dir, "images.txt")

            sensor_params        = _gs_parse_colmap_cameras(cameras_txt)
            records, traj        = _gs_parse_colmap_images(images_txt)
            records              = records[::subsample]

            # depth: images/cam_X/depths/<stem>.depth
            def _colmap_depth_path(name):
                parts = name.rsplit("/", 1)  # e.g. ["cam_1/rgb", "ts.jpg"]
                if len(parts) != 2:
                    return None
                stem = os.path.splitext(parts[1])[0]
                cam_dir = parts[0].replace("/rgb", "")
                p = os.path.join(images_root, cam_dir, "depths", stem + ".depth")
                return p if os.path.exists(p) else None

            cam_id_map = {}
            train_data = []
            for img_id, cam_id, fname in records:
                key = (img_id, cam_id)
                if key not in traj:
                    continue
                if cam_id not in sensor_params:
                    continue
                img_path = os.path.join(images_root, fname)
                if not os.path.exists(img_path):
                    continue
                if cam_id not in cam_id_map:
                    cam_id_map[cam_id] = len(cam_id_map)
                T_c2w_orig    = traj[key]
                T_c2w_aligned = T_align @ T_c2w_orig
                train_data.append({
                    "path":       img_path,
                    "T_c2w":      T_c2w_aligned.astype(np.float32),
                    "cam":        sensor_params[cam_id],
                    "depth_path": _colmap_depth_path(fname),
                    "camera_idx": cam_id_map[cam_id],
                    "frame_idx":  len(train_data),
                })
        else:
            # ── kapture 포맷
            if not kapture_dir:
                raise ValueError("kapture_dir 또는 colmap_dir 중 하나를 지정하세요.")
            print(f"  [GS] kapture 포맷 사용: {kapture_dir}")
            sensors_txt = os.path.join(kapture_dir, "sensors.txt")
            traj_txt    = os.path.join(kapture_dir, "trajectories.txt")
            records_txt = os.path.join(kapture_dir, "records_camera.txt")
            images_root = os.path.join(kapture_dir, "records_data")

            depth_records_txt = os.path.join(kapture_dir, "records_depth.txt")
            depth_map = {}
            if os.path.exists(depth_records_txt):
                for line in open(depth_records_txt):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    p = [x.strip() for x in line.split(",")]
                    if len(p) >= 3:
                        dpath = os.path.join(images_root, p[2])
                        if os.path.exists(dpath):
                            depth_map[(p[0], p[1])] = dpath
                print(f"  Depth records: {len(depth_map)}")

            sensor_params = _gs_parse_sensors(sensors_txt)
            traj          = _gs_parse_trajectories(traj_txt)
            records       = _gs_parse_records(records_txt)
            records       = records[::subsample]

            cam_id_map = {}
            train_data = []
            for ts, dev, fname in records:
                key = (ts, dev)
                if key not in traj:
                    continue
                if dev not in sensor_params:
                    continue
                img_path = os.path.join(images_root, fname)
                if not os.path.exists(img_path):
                    continue
                if dev not in cam_id_map:
                    cam_id_map[dev] = len(cam_id_map)
                T_c2w_orig    = traj[key]
                T_c2w_aligned = T_align @ T_c2w_orig
                train_data.append({
                    "path":       img_path,
                    "T_c2w":      T_c2w_aligned.astype(np.float32),
                    "cam":        sensor_params[dev],
                    "depth_path": depth_map.get(key),
                    "camera_idx": cam_id_map[dev],
                    "frame_idx":  len(train_data),
                })

        num_cameras = len(cam_id_map)
        num_frames  = len(train_data)
        print(f"  Train frames: {len(train_data)}  "
              f"(with depth: {sum(1 for d in train_data if d['depth_path'])})")
        if not train_data:
            raise RuntimeError("학습 이미지를 찾을 수 없습니다. kapture_dir 또는 colmap_dir 경로를 확인하세요.")

        # ── 4. 학습 루프 (DefaultStrategy: densification + pruning)
        MEANS_LR = 1.6e-4
        params = {
            "means":     means,
            "scales":    scales,
            "quats":     quats,
            "opacities": opacities,
            "colors_sh": colors_sh,
        }
        if feature_splat:
            params["loc_features"] = loc_features
        optimizers = {
            "means":     torch.optim.Adam([means],     lr=MEANS_LR),
            "scales":    torch.optim.Adam([scales],    lr=5e-3),
            "quats":     torch.optim.Adam([quats],     lr=1e-3),
            "opacities": torch.optim.Adam([opacities], lr=5e-2),
            "colors_sh": torch.optim.Adam([colors_sh], lr=2.5e-3),
        }
        if feature_splat:
            optimizers["loc_features"] = torch.optim.Adam([loc_features], lr=1e-3)
            feature_opt = torch.optim.Adam(feature_decoder.parameters(), lr=6.25e-5)
            superpoint = _load_superpoint_dense(device)
        else:
            feature_opt = None
            superpoint = None

        # ── PPISP 초기화 (4대 카메라, 노출 차이 보정)
        ppisp_module = None
        ppisp_opt = None
        ppisp_sched = None
        if use_ppisp:
            try:
                from ppisp import PPISP, PPISPConfig
                ppisp_cfg = PPISPConfig(
                    use_controller=True,
                    controller_distillation=True,
                    controller_activation_ratio=0.8,
                )
                ppisp_module = PPISP(
                    num_cameras=num_cameras,
                    num_frames=num_frames,
                    config=ppisp_cfg,
                ).to(device)
                ppisp_opt, ppisp_sched = ppisp_module.create_optimizers(), None
                try:
                    ppisp_sched = ppisp_module.create_schedulers(ppisp_opt)
                except Exception:
                    pass
                print(f"  [PPISP] 초기화: {num_cameras} cameras, {num_frames} frames")
            except ImportError:
                print("  [PPISP] 미설치 — ppisp 없이 진행")

        # iteration 기반 학습 (표준 gsplat 방식)
        total_iters = gs_iters

        means_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers["means"], T_max=total_iters, eta_min=MEANS_LR * 0.01
        )

        scene_scale = float(np.linalg.norm(
            pts.max(axis=0) - pts.min(axis=0))) / 2.0

        # 표준 3DGS 스케줄: refine 500~50%지점, every 100, reset every 3000
        strategy = DefaultStrategy(
            verbose=False,
            refine_start_iter=500,
            refine_stop_iter=int(total_iters * 0.5),
            refine_every=100,
            reset_every=3000,
            prune_opa=0.01,
            prune_scale3d=0.1,
        )
        state = strategy.initialize_state(scene_scale=scene_scale)

        import random
        from tqdm import tqdm
        train_indices = list(range(len(train_data)))
        random.shuffle(train_indices)
        data_idx = 0

        print(f"  Training {total_iters} iters  "
              f"(save_interval={save_interval}, log_interval={log_interval})")

        # ── wandb 초기화
        _wb = None
        if wandb_project:
            try:
                import wandb as _wandb_mod
                run_name = wandb_run_name or os.path.basename(os.path.abspath(output_dir))
                _wb = _wandb_mod.init(
                    project=wandb_project,
                    name=run_name,
                    config=dict(
                        gs_iters=gs_iters,
                        sh_degree=sh_degree,
                        train_img_size=train_img_size,
                        save_interval=save_interval,
                        log_interval=log_interval,
                        n_train_frames=len(train_data),
                        n_gaussians_init=len(params["means"]),
                    ),
                )
                print(f"  [wandb] run: {_wb.name}  project: {wandb_project}")
            except ImportError:
                print("  [wandb] 미설치 — wandb 없이 진행 (pip install wandb)")

        # ── window 통계 (log_interval iter마다 평균 → wandb)
        win_loss = 0.0
        win_rgb = 0.0
        win_depth = 0.0
        win_depth_count = 0
        win_psnr = 0.0
        win_feature = 0.0
        win_count = 0

        pbar = tqdm(range(total_iters), desc="  Training", ncols=100)
        for it in pbar:
            # epoch처럼 다 돌면 셔플
            if data_idx >= len(train_indices):
                random.shuffle(train_indices)
                data_idx = 0
            entry = train_data[train_indices[data_idx]]
            data_idx += 1

            for opt in optimizers.values():
                opt.zero_grad()
            if ppisp_opt:
                for opt in (ppisp_opt if isinstance(ppisp_opt, list) else [ppisp_opt]):
                    opt.zero_grad()
            if feature_opt is not None:
                feature_opt.zero_grad()

            img_bgr = cv2.imread(entry["path"])
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            cam = entry["cam"]
            orig_h, orig_w = cam["h"], cam["w"]
            rs = train_img_size / max(orig_h, orig_w)
            new_w = int(orig_w * rs)
            new_h = int(orig_h * rs)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h))
            gt = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).to(device)

            sx  = new_w / cam["w"]
            sy  = new_h / cam["h"]
            fx_t = cam["fx"] * sx
            fy_t = cam["fy"] * sy
            cx_t = cam["cx"] * sx
            cy_t = cam["cy"] * sy

            T_c2w = torch.from_numpy(entry["T_c2w"]).to(device)
            viewmat = torch.linalg.inv(T_c2w).unsqueeze(0)
            K = torch.tensor([[fx_t, 0, cx_t],
                               [0, fy_t, cy_t],
                               [0,   0,    1]], device=device).unsqueeze(0).float()

            renders, alphas, info = rasterization(
                means=params["means"],
                quats=F.normalize(params["quats"], dim=-1),
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=params["colors_sh"],
                viewmats=viewmat,
                Ks=K,
                width=new_w,
                height=new_h,
                sh_degree=sh_degree,
                near_plane=0.1,
                far_plane=100.0,
                packed=True,
                render_mode="RGB+D",
            )
            pred = renders[0, ..., :3].clamp(0, 1)
            pred_depth = renders[0, ..., 3]

            if ppisp_module is not None:
                ys = torch.linspace(-1, 1, new_h, device=device)
                xs = torch.linspace(-1, 1, new_w, device=device)
                grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
                pixel_coords = torch.stack([grid_x, grid_y], dim=-1)
                pred = ppisp_module(
                    rgb=pred,
                    pixel_coords=pixel_coords,
                    resolution=(new_w, new_h),
                    camera_idx=entry["camera_idx"],
                    frame_idx=entry["frame_idx"],
                ).clamp(0, 1)

            l1_loss = F.l1_loss(pred, gt)
            pred_bchw = pred.permute(2, 0, 1).unsqueeze(0)
            gt_bchw   = gt.permute(2, 0, 1).unsqueeze(0)
            ssim_val  = _ssim_fn(pred_bchw, gt_bchw, data_range=1.0, size_average=True)
            rgb_loss = 0.8 * l1_loss + 0.2 * (1.0 - ssim_val)

            depth_loss = torch.tensor(0.0, device=device)
            if entry.get("depth_path"):
                gt_depth_raw = np.fromfile(entry["depth_path"], dtype=np.float32)
                if len(gt_depth_raw) == cam["h"] * cam["w"]:
                    gt_depth_full = gt_depth_raw.reshape(cam["h"], cam["w"])
                    gt_depth_resized = cv2.resize(gt_depth_full, (new_w, new_h),
                                                  interpolation=cv2.INTER_NEAREST)
                    gt_depth_t = torch.from_numpy(gt_depth_resized).to(device)
                    valid_mask = gt_depth_t > 0.1
                    if valid_mask.sum() > 10:
                        depth_loss = F.l1_loss(
                            torch.log1p(pred_depth[valid_mask]),
                            torch.log1p(gt_depth_t[valid_mask]))

            scales_exp = torch.exp(params["scales"])
            scale_max_loss = scales_exp.max(dim=-1).values.mean()
            scale_ratio_loss = (scales_exp.max(dim=-1).values /
                                (scales_exp.min(dim=-1).values + 1e-6)).mean()

            ppisp_reg = (ppisp_module.get_regularization_loss()
                         if ppisp_module is not None else 0.0)

            feature_loss = torch.tensor(0.0, device=device)
            if feature_splat:
                target_desc = _extract_superpoint_desc(superpoint, gt)
                feat_h, feat_w = target_desc.shape[-2:]
                sx_f = feat_w / float(new_w)
                sy_f = feat_h / float(new_h)
                K_feat = torch.tensor([[fx_t * sx_f, 0, cx_t * sx_f],
                                       [0, fy_t * sy_f, cy_t * sy_f],
                                       [0, 0, 1]], device=device).unsqueeze(0).float()
                rendered_low_feat = _render_feature_map_gsplat(
                    rasterization, params, viewmat, K_feat, feat_w, feat_h,
                    near_plane=0.1, far_plane=100.0, packed=True
                )
                rendered_high_feat = feature_decoder(rendered_low_feat.unsqueeze(0))
                rendered_high_feat = F.normalize(rendered_high_feat, p=2, dim=1)
                feature_loss = F.l1_loss(rendered_high_feat, target_desc)

            depth_w = max(0.1, 1.0 - 0.9 * (it / total_iters))
            loss = (rgb_loss
                    + depth_w * depth_loss
                    + feature_weight * feature_loss
                    + 1e-4 * scale_max_loss
                    + 1e-3 * scale_ratio_loss
                    + ppisp_reg)

            strategy.step_pre_backward(params, optimizers, state, it, info)
            loss.backward()
            strategy.step_post_backward(params, optimizers, state, it, info, packed=True)

            for opt in optimizers.values():
                opt.step()
            means_scheduler.step()
            if ppisp_opt:
                for opt in (ppisp_opt if isinstance(ppisp_opt, list) else [ppisp_opt]):
                    opt.step()
            if feature_opt is not None:
                feature_opt.step()
            if ppisp_sched:
                for sched in (ppisp_sched if isinstance(ppisp_sched, list) else [ppisp_sched]):
                    sched.step()

            with torch.no_grad():
                params["scales"].clamp_(-6.0, 0.0)

            with torch.no_grad():
                mse = F.mse_loss(pred, gt).item()
                psnr = -10.0 * math.log10(mse + 1e-10)

            # ── window 통계
            win_loss += loss.item()
            win_rgb  += rgb_loss.item()
            if depth_loss.item() > 0:
                win_depth += depth_loss.item()
                win_depth_count += 1
            win_psnr += psnr
            if feature_splat:
                win_feature += feature_loss.item()
            win_count += 1

            postfix = {"loss": f"{loss.item():.4f}",
                       "psnr": f"{psnr:.2f}",
                       "GS": f"{len(params['means']):,}"}
            if feature_splat:
                postfix["feat"] = f"{feature_loss.item():.4f}"
            pbar.set_postfix(postfix)

            # ── wandb 스칼라 로그 (log_interval iter마다)
            if _wb is not None and (it + 1) % log_interval == 0:
                cur_lr = means_scheduler.get_last_lr()[0]
                log_payload = {
                    "loss/total":   win_loss / win_count,
                    "loss/rgb":     win_rgb / win_count,
                    "loss/depth":   win_depth / max(win_depth_count, 1),
                    "psnr":         win_psnr / win_count,
                    "depth_weight": depth_w,
                    "n_gaussians":  len(params["means"]),
                    "lr/means":     cur_lr,
                }
                if feature_splat:
                    log_payload["loss/feature"] = win_feature / win_count
                _wb.log(log_payload, step=it + 1)
                win_loss = win_rgb = win_depth = win_psnr = win_feature = 0.0
                win_depth_count = win_count = 0

            # ── 중간 체크포인트 + 샘플 렌더링 (save_interval iter마다 + 마지막)
            is_save_iter = ((it + 1) % save_interval == 0) or (it + 1 == total_iters)
            if is_save_iter:
                gs_dir = os.path.join(output_dir, "gaussian")
                iter_dir = os.path.join(gs_dir, f"iter_{it+1}")
                os.makedirs(iter_dir, exist_ok=True)

                ckpt_path = os.path.join(iter_dir, "gaussians.pt")
                torch.save({
                    "means":     params["means"].data,
                    "scales":    params["scales"].data,
                    "quats":     params["quats"].data,
                    "opacities": params["opacities"].data,
                    "colors_sh": params["colors_sh"].data,
                    **({"loc_features": params["loc_features"].data,
                        "feature_decoder": feature_decoder.state_dict()}
                       if feature_splat else {}),
                }, ckpt_path)

                n_samples = min(10, len(viewpoints))
                wb_images = {}
                if n_samples > 0:
                    sample_indices = np.linspace(0, len(viewpoints) - 1, n_samples, dtype=int)
                    cam_cfg = config["camera"]
                    W_s, H_s = cam_cfg["width"], cam_cfg["height"]
                    K_s = torch.tensor([
                        [cam_cfg["fx"], 0, cam_cfg["cx"]],
                        [0, cam_cfg["fy"], cam_cfg["cy"]],
                        [0, 0, 1]], device=device).unsqueeze(0).float()

                    for si in sample_indices:
                        sample_vp = viewpoints[si]
                        sample_pose, _ = _gs_render_pose_from_viewpoint(
                            sample_vp["pose"], config
                        )
                        vm_s = torch.from_numpy(np.linalg.inv(sample_pose)).to(device).unsqueeze(0).float()

                        with torch.no_grad():
                            rend_s, _, _ = rasterization(
                                means=params["means"].detach(),
                                quats=F.normalize(params["quats"].detach(), dim=-1),
                                scales=torch.exp(params["scales"].detach()),
                                opacities=torch.sigmoid(params["opacities"].detach()),
                                colors=params["colors_sh"].detach(),
                                viewmats=vm_s, Ks=K_s,
                                width=W_s, height=H_s,
                                sh_degree=sh_degree,
                                near_plane=0.3, far_plane=100.0,
                                render_mode="RGB+D",
                            )
                        rgb_s = (rend_s[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                        sample_path = os.path.join(iter_dir, f"sample_{sample_vp['id']:06d}.png")
                        cv2.imwrite(sample_path, cv2.cvtColor(rgb_s, cv2.COLOR_RGB2BGR))
                        if _wb is not None:
                            import wandb as _wandb_mod
                            wb_images[f"render/vp_{sample_vp['id']:06d}"] = \
                                _wandb_mod.Image(rgb_s, caption=f"iter {it+1}")

                if _wb is not None and wb_images:
                    _wb.log(wb_images, step=it + 1)

                n_gs_now = len(params["means"])
                cur_lr = means_scheduler.get_last_lr()[0]
                tqdm.write(f"  [iter {it+1}/{total_iters}] loss={loss.item():.4f}  "
                           f"GS={n_gs_now:,}  lr={cur_lr:.2e}  → {iter_dir}/")

        pbar.close()

        if _wb is not None:
            _wb.finish()

        # 최종 파라미터 갱신
        means     = params["means"]
        scales    = params["scales"]
        quats     = params["quats"]
        opacities = params["opacities"]
        colors_sh = params["colors_sh"]
        if feature_splat:
            loc_features = params["loc_features"]

        # 최종 모델도 기존 위치에 저장 (호환성)
        ckpt_dict = {
            "means":     means.data,
            "scales":    scales.data,
            "quats":     quats.data,
            "opacities": opacities.data,
            "colors_sh": colors_sh.data,
        }
        if feature_splat:
            ckpt_dict["loc_features"] = loc_features.data
            ckpt_dict["feature_decoder"] = feature_decoder.state_dict()
        if ppisp_module is not None:
            ckpt_dict["ppisp"] = ppisp_module.state_dict()
        torch.save(ckpt_dict, gs_ckpt)
        print(f"  [GS] 최종 모델 저장: {gs_ckpt}  ({len(means):,} GS)"
              + ("  [+PPISP]" if ppisp_module else ""))

        # 최종 가우시안 맵을 PLY로 저장
        SH_C0 = 0.28209479177387814
        gs_pts = means.data.cpu().numpy().astype(np.float32)
        gs_scales_raw = torch.exp(scales.data).cpu().numpy().astype(np.float32)
        gs_quats_raw = F.normalize(quats.data, dim=-1).cpu().numpy().astype(np.float32)
        gs_rgb = (colors_sh.data[:, 0, :].cpu().numpy() * SH_C0 + 0.5).clip(0, 1)
        gs_opa = torch.sigmoid(opacities.data).cpu().numpy().astype(np.float32)

        # opacity 낮은 Gaussian 제거
        valid = gs_opa > 0.05
        gs_dir = os.path.join(output_dir, "gaussian")
        os.makedirs(gs_dir, exist_ok=True)

        # PLY (포인트클라우드 뷰어용)
        gs_pcd = o3d.geometry.PointCloud()
        gs_pcd.points = o3d.utility.Vector3dVector(gs_pts[valid])
        gs_pcd.colors = o3d.utility.Vector3dVector(gs_rgb[valid])
        gs_ply_path = os.path.join(gs_dir, "gaussian_map.ply")
        o3d.io.write_point_cloud(gs_ply_path, gs_pcd, write_ascii=False, compressed=False)
        print(f"  [GS] PLY 저장: {gs_ply_path}  "
              f"({valid.sum():,}/{len(gs_opa):,} pts, opa>0.05)")

        # .splat (WebGL 3D Gaussian Splat Viewer용)
        # 형식: position(3×f32) + scale(3×f32) + rgba(4×u8) + quaternion(4×u8)
        # = 12 + 12 + 4 + 4 = 32 bytes per Gaussian
        v_pts    = gs_pts[valid]
        v_scales = gs_scales_raw[valid]
        v_rgb    = (gs_rgb[valid] * 255).clip(0, 255).astype(np.uint8)
        v_opa    = (gs_opa[valid] * 255).clip(0, 255).astype(np.uint8)
        v_quats  = gs_quats_raw[valid]
        # quaternion을 uint8로: [-1,1] → [0,255]
        v_quats_u8 = ((v_quats * 128) + 128).clip(0, 255).astype(np.uint8)

        n_valid = int(valid.sum())
        splat_buf = np.empty(n_valid, dtype=[
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('sx', '<f4'), ('sy', '<f4'), ('sz', '<f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1'),
            ('qw', 'u1'), ('qx', 'u1'), ('qy', 'u1'), ('qz', 'u1'),
        ])
        splat_buf['x']  = v_pts[:, 0]
        splat_buf['y']  = v_pts[:, 1]
        splat_buf['z']  = v_pts[:, 2]
        splat_buf['sx'] = v_scales[:, 0]
        splat_buf['sy'] = v_scales[:, 1]
        splat_buf['sz'] = v_scales[:, 2]
        splat_buf['r']  = v_rgb[:, 0]
        splat_buf['g']  = v_rgb[:, 1]
        splat_buf['b']  = v_rgb[:, 2]
        splat_buf['a']  = v_opa
        splat_buf['qw'] = v_quats_u8[:, 0]  # gsplat: [w,x,y,z]
        splat_buf['qx'] = v_quats_u8[:, 1]
        splat_buf['qy'] = v_quats_u8[:, 2]
        splat_buf['qz'] = v_quats_u8[:, 3]

        splat_path = os.path.join(gs_dir, "gaussian_map.splat")
        splat_buf.tofile(splat_path)
        splat_mb = os.path.getsize(splat_path) / 1e6
        print(f"  [GS] Splat 저장: {splat_path}  "
              f"({n_valid:,} GS, {splat_mb:.1f} MB)")

    # ── 5. grid viewpoints 렌더링
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization

    cam_cfg = config["camera"]
    W_out   = cam_cfg["width"]
    H_out   = cam_cfg["height"]
    fx_r    = cam_cfg["fx"]
    fy_r    = cam_cfg["fy"]
    cx_r    = cam_cfg["cx"]
    cy_r    = cam_cfg["cy"]

    device = means.device
    params_render = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "colors_sh": colors_sh,
    }
    if feature_splat:
        params_render["loc_features"] = loc_features
        feature_decoder.eval()
    K_render = torch.tensor([[fx_r, 0, cx_r],
                              [0, fy_r, cy_r],
                              [0,   0,    1]], device=device).unsqueeze(0).float()

    rendered = []
    print(f"  Rendering {len(viewpoints)} viewpoints ...")
    with torch.no_grad():
        for i, vp in enumerate(viewpoints):
            raw_pose = vp["pose"].astype(np.float32)
            pose, pose_roll_applied = _gs_render_pose_from_viewpoint(raw_pose, config)
            viewmat = torch.from_numpy(
                          np.linalg.inv(pose)).to(device).unsqueeze(0).float()

            renders, alphas, _ = rasterization(
                means=means.detach(),
                quats=F.normalize(quats.detach(), dim=-1),
                scales=torch.exp(scales.detach()),
                opacities=torch.sigmoid(opacities.detach()),
                colors=colors_sh.detach(),
                viewmats=viewmat,
                Ks=K_render,
                width=W_out,
                height=H_out,
                sh_degree=sh_degree,
                near_plane=0.3,
                far_plane=100.0,
                render_mode="RGB+D",
            )
            rgb_t   = renders[0, ..., :3].clamp(0, 1)
            depth_t = renders[0, ...,  3]

            rgb_np   = (rgb_t.cpu().numpy() * 255).astype(np.uint8)
            depth_np = depth_t.cpu().numpy().astype(np.float32)

            rp_ = os.path.join(rd, "rgb",   f"{vp['id']:06d}.png")
            dp_ = os.path.join(rd, "depth", f"{vp['id']:06d}.npy")
            cv2.imwrite(rp_, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
            np.save(dp_, depth_np)

            fp_ = None
            if feature_splat:
                feat_w = max(1, W_out // feature_stride)
                feat_h = max(1, H_out // feature_stride)
                sx_f = feat_w / float(W_out)
                sy_f = feat_h / float(H_out)
                K_feat = torch.tensor([[fx_r * sx_f, 0, cx_r * sx_f],
                                       [0, fy_r * sy_f, cy_r * sy_f],
                                       [0, 0, 1]], device=device).unsqueeze(0).float()
                low_feat = _render_feature_map_gsplat(
                    rasterization, params_render, viewmat, K_feat, feat_w, feat_h,
                    near_plane=0.3, far_plane=100.0, packed=False
                )
                high_feat = feature_decoder(low_feat.unsqueeze(0))[0]
                high_feat = F.normalize(high_feat, p=2, dim=0)
                feat_np = high_feat.cpu().numpy().astype(np.float16)
                fp_ = os.path.join(rd, "feature", f"{vp['id']:06d}.npy")
                np.save(fp_, feat_np)

            item = {
                "id":         vp["id"],
                "pose":       pose,
                "rgb_path":   rp_,
                "depth_path": dp_,
            }
            if pose_roll_applied:
                item["viewpoint_pose"] = raw_pose
                item["pose_roll_180_applied"] = True
            if fp_ is not None:
                item["feature_path"] = fp_
                item["feature_shape"] = tuple(feat_np.shape)
                item["feature_type"] = "superpoint_fgs"
                item["feature_stride"] = feature_stride
            rendered.append(item)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(viewpoints)}")

    # 시각화
    ns  = min(8, len(rendered))
    idx = np.linspace(0, len(rendered)-1, ns, dtype=int)
    fig, ax = plt.subplots(2, ns, figsize=(4*ns, 8))
    if ns == 1: ax = ax.reshape(2, 1)
    for c, ii in enumerate(idx):
        r = rendered[ii]
        ax[0,c].imshow(cv2.cvtColor(cv2.imread(r["rgb_path"]), cv2.COLOR_BGR2RGB))
        ax[0,c].set_title(f"GS #{r['id']}", fontsize=9); ax[0,c].axis("off")
        dv = np.load(r["depth_path"])
        dv[dv > cam_cfg.get("depth_max", 10)] = 0
        ax[1,c].imshow(dv, cmap="plasma")
        ax[1,c].set_title("Depth", fontsize=9); ax[1,c].axis("off")
    preview_title = "Step 2-FGS: Feature GS Rendered" if feature_splat else "Step 2-GS: 3DGS Rendered"
    fig.suptitle(f"{preview_title} — {len(rendered)} images", fontsize=14)
    fig.tight_layout()
    preview_name = "step2_fgs_rendered.png" if feature_splat else "step2_gs_rendered.png"
    fig.savefig(os.path.join(output_dir, preview_name), dpi=150); plt.close()
    print(f"  Saved: {preview_name}")

    pkl_name = "step2_fgs_data.pkl" if feature_splat else "step2_gs_data.pkl"
    pkl_path = os.path.join(output_dir, pkl_name)
    pickle.dump(rendered, open(pkl_path, "wb"))
    # step2_data.pkl에도 저장 (이후 step3 호환)
    pickle.dump(rendered, open(os.path.join(output_dir, "step2_data.pkl"), "wb"))
    print(f"  Saved: {pkl_path}  ({len(rendered)} entries)")
    return rendered


# =============================================================================
# Unified entry point
# =============================================================================
def step2_render(ply_path, viewpoints, config, output_dir,
                 step0_data=None, mode="pointcloud", **kwargs):
    """
    Rendering step — unified interface.

    Args:
        mode: "pointcloud" (O3D split-render), "gaussian_ply" (direct Gaussian PLY),
              "ns" (nerfstudio/ns-render path), "gs" (train/load 3D Gaussian Splatting),
              or "gs_feature"/"fgs" (Feature Gaussian Splatting)
        **kwargs: GS-specific params (kapture_dir, gs_epochs, voxel_size, accum_steps,
                  use_ppisp, etc.)
    """
    if mode == "gaussian_ply":
        return _render_gaussian_ply(ply_path, viewpoints, config, output_dir,
                                    step0_data=step0_data)
    if mode in ("ns", "nerfstudio", "ns_render"):
        return _render_nerfstudio(
            viewpoints, config, output_dir,
            ns_config_path=kwargs.get("ns_config_path"),
            ns_camera_path=kwargs.get("ns_camera_path"),
        )
    if mode == "gs":
        return _render_gs(ply_path, viewpoints, config, output_dir,
                          step0_data=step0_data, **kwargs)
    if mode in ("gs_feature", "fgs"):
        return _render_gs(ply_path, viewpoints, config, output_dir,
                          step0_data=step0_data, feature_splat=True, **kwargs)
    else:
        return _render_pointcloud(ply_path, viewpoints, config, output_dir, step0_data)
