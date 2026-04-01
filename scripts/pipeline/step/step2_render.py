"""
STEP 2: Rendering — unified interface for pointcloud (O3D) and Gaussian Splatting modes.

Usage:
    step2_render(..., mode="pointcloud")   # O3D split-render
    step2_render(..., mode="gs", kapture_dir=..., gs_iters=..., ...)  # 3DGS
"""
import os, pickle
import numpy as np
import open3d as o3d
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Shared helpers
# =============================================================================
_HEAVY_KEYS = ("rgb", "depth")


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
def _render_gs(ply_path, viewpoints, config, output_dir,
               kapture_dir, step0_data=None,
               gs_epochs=30, sh_degree=0,
               train_img_size=512, subsample=1,
               voxel_size=None, accum_steps=4):
    import torch
    import torch.nn.functional as F
    from pytorch_msssim import ssim as _ssim_fn
    try:
        from gsplat import rasterization
        from gsplat.strategy import DefaultStrategy
    except ImportError:
        raise ImportError("gsplat 미설치: pip install gsplat")

    print("\n" + "="*60 + "\nSTEP 2-GS: 3D Gaussian Splatting Rendering\n" + "="*60)

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
    print(f"  Device  : {device}  epochs={gs_epochs}  sh_degree={sh_degree}  accum={accum_steps}")

    rd = os.path.join(output_dir, "rendered_gs")
    gs_ckpt = os.path.join(output_dir, "gaussians.pt")
    os.makedirs(os.path.join(rd, "rgb"),   exist_ok=True)
    os.makedirs(os.path.join(rd, "depth"), exist_ok=True)

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

    # ── 2. 학습된 모델이 있으면 로드, 없으면 학습
    if os.path.exists(gs_ckpt):
        print(f"  [GS] 저장된 모델 로드: {gs_ckpt}")
        ckpt = torch.load(gs_ckpt, map_location=device)
        means     = torch.nn.Parameter(ckpt["means"])
        scales    = torch.nn.Parameter(ckpt["scales"])
        quats     = torch.nn.Parameter(ckpt["quats"])
        opacities = torch.nn.Parameter(ckpt["opacities"])
        colors_sh = torch.nn.Parameter(ckpt["colors_sh"])
        N = len(means)
        print(f"  [GS] 로드된 GS: {N:,} pts")
    else:
        # ── 3. kapture 매핑 데이터 로드
        sensors_txt = os.path.join(kapture_dir, "sensors.txt")
        traj_txt    = os.path.join(kapture_dir, "trajectories.txt")
        records_txt = os.path.join(kapture_dir, "records_camera.txt")
        images_root = os.path.join(kapture_dir, "records_data")

        sensor_params = _gs_parse_sensors(sensors_txt)
        traj          = _gs_parse_trajectories(traj_txt)
        records       = _gs_parse_records(records_txt)
        records       = records[::subsample]

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
            T_c2w_orig    = traj[key]
            T_c2w_aligned = T_align @ T_c2w_orig
            train_data.append({
                "path": img_path,
                "T_c2w": T_c2w_aligned.astype(np.float32),
                "cam": sensor_params[dev],
            })
        print(f"  Train frames: {len(train_data)}")
        if not train_data:
            raise RuntimeError("학습 이미지를 찾을 수 없습니다. kapture_dir 경로를 확인하세요.")

        # ── 4. 학습 루프 (DefaultStrategy: densification + pruning)
        params = {
            "means":     means,
            "scales":    scales,
            "quats":     quats,
            "opacities": opacities,
            "colors_sh": colors_sh,
        }
        optimizers = {
            "means":     torch.optim.Adam([means],     lr=1.6e-4),
            "scales":    torch.optim.Adam([scales],    lr=5e-3),
            "quats":     torch.optim.Adam([quats],     lr=1e-3),
            "opacities": torch.optim.Adam([opacities], lr=5e-2),
            "colors_sh": torch.optim.Adam([colors_sh], lr=2.5e-3),
        }

        # epoch/iter 계산
        iters_per_epoch = len(train_data) // accum_steps
        total_iters = gs_epochs * iters_per_epoch

        means_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers["means"], T_max=total_iters, eta_min=1.6e-4 * 0.01
        )

        scene_scale = float(np.linalg.norm(
            pts.max(axis=0) - pts.min(axis=0))) / 2.0

        strategy = DefaultStrategy(
            verbose=False,
            refine_start_iter=iters_per_epoch * 2,          # 2 epoch 후 시작
            refine_stop_iter=int(total_iters * 0.75),       # 75% 지점에서 중단
            refine_every=max(1, iters_per_epoch // 5),      # epoch당 ~5회 refine
            reset_every=iters_per_epoch * 3,                # 3 epoch마다 opacity reset
        )
        state = strategy.initialize_state(scene_scale=scene_scale)

        import random
        from tqdm import tqdm
        train_indices = list(range(len(train_data)))

        print(f"  Training {gs_epochs} epochs × {iters_per_epoch} iters/epoch "
              f"= {total_iters} total iters  (accum={accum_steps})")
        global_it = 0
        for epoch in range(gs_epochs):
            random.shuffle(train_indices)
            epoch_loss = 0.0
            epoch_count = 0
            data_idx = 0

            pbar = tqdm(range(iters_per_epoch),
                        desc=f"  Epoch {epoch+1}/{gs_epochs}",
                        leave=True, ncols=100)
            for step in pbar:
                for opt in optimizers.values():
                    opt.zero_grad()

                last_info = None
                step_loss = 0.0
                for acc in range(accum_steps):
                    entry = train_data[train_indices[data_idx]]
                    data_idx += 1

                    img_bgr = cv2.imread(entry["path"])
                    if img_bgr is None:
                        continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    # 비율 유지 리사이즈: 긴 쪽을 train_img_size에 맞춤
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
                    )
                    pred = renders[0, ..., :3].clamp(0, 1)

                    # L1 + SSIM loss
                    l1_loss = F.l1_loss(pred, gt)
                    pred_bchw = pred.permute(2, 0, 1).unsqueeze(0)
                    gt_bchw   = gt.permute(2, 0, 1).unsqueeze(0)
                    ssim_val  = _ssim_fn(pred_bchw, gt_bchw, data_range=1.0, size_average=True)
                    loss = (0.8 * l1_loss + 0.2 * (1.0 - ssim_val)) / accum_steps

                    if acc == accum_steps - 1:
                        strategy.step_pre_backward(params, optimizers, state, global_it, info)

                    loss.backward()
                    last_info = info
                    step_loss += loss.item()

                strategy.step_post_backward(params, optimizers, state, global_it, last_info,
                                            packed=True)

                for opt in optimizers.values():
                    opt.step()
                means_scheduler.step()

                epoch_loss += step_loss * accum_steps
                epoch_count += 1
                global_it += 1

                # tqdm 진행바 업데이트
                pbar.set_postfix(loss=f"{step_loss * accum_steps:.4f}",
                                 GS=f"{len(params['means']):,}")

            pbar.close()
            avg_loss = epoch_loss / max(epoch_count, 1)
            n_gs = len(params["means"])
            print(f"  Epoch {epoch+1}/{gs_epochs}  avg_loss={avg_loss:.4f}  GS={n_gs:,}")

            # ── 중간 체크포인트 + 샘플 렌더링 (20 epoch마다 + 마지막 epoch)
            save_interval = 20
            is_save_epoch = ((epoch + 1) % save_interval == 0) or (epoch + 1 == gs_epochs)
            if is_save_epoch:
                gs_dir = os.path.join(output_dir, "gaussian")
                epoch_dir = os.path.join(gs_dir, f"epoch_{epoch+1}")
                os.makedirs(epoch_dir, exist_ok=True)

                # 체크포인트 저장
                ckpt_path = os.path.join(epoch_dir, "gaussians.pt")
                torch.save({
                    "means":     params["means"].data,
                    "scales":    params["scales"].data,
                    "quats":     params["quats"].data,
                    "opacities": params["opacities"].data,
                    "colors_sh": params["colors_sh"].data,
                }, ckpt_path)

                # 샘플 viewpoint 1장 렌더링
                if len(viewpoints) > 0:
                    sample_vp = viewpoints[len(viewpoints) // 2]
                    cam_cfg = config["camera"]
                    W_s, H_s = cam_cfg["width"], cam_cfg["height"]
                    K_s = torch.tensor([
                        [cam_cfg["fx"], 0, cam_cfg["cx"]],
                        [0, cam_cfg["fy"], cam_cfg["cy"]],
                        [0, 0, 1]], device=device).unsqueeze(0).float()
                    vm_s = torch.from_numpy(
                        np.linalg.inv(sample_vp["pose"].astype(np.float32))
                    ).to(device).unsqueeze(0).float()

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
                            near_plane=0.1, far_plane=100.0,
                            render_mode="RGB+D",
                        )
                    rgb_s = (rend_s[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    sample_path = os.path.join(epoch_dir, "sample_render.png")
                    cv2.imwrite(sample_path, cv2.cvtColor(rgb_s, cv2.COLOR_RGB2BGR))

                print(f"    → Checkpoint saved: {epoch_dir}/")

        # 최종 파라미터 갱신
        means     = params["means"]
        scales    = params["scales"]
        quats     = params["quats"]
        opacities = params["opacities"]
        colors_sh = params["colors_sh"]

        # 최종 모델도 기존 위치에 저장 (호환성)
        torch.save({
            "means":     means.data,
            "scales":    scales.data,
            "quats":     quats.data,
            "opacities": opacities.data,
            "colors_sh": colors_sh.data,
        }, gs_ckpt)
        print(f"  [GS] 최종 모델 저장: {gs_ckpt}  ({len(means):,} GS)")

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
    K_render = torch.tensor([[fx_r, 0, cx_r],
                              [0, fy_r, cy_r],
                              [0,   0,    1]], device=device).unsqueeze(0).float()

    rendered = []
    print(f"  Rendering {len(viewpoints)} viewpoints ...")
    with torch.no_grad():
        for i, vp in enumerate(viewpoints):
            pose    = vp["pose"].astype(np.float32)
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
                near_plane=0.1,
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

            rendered.append({
                "id":         vp["id"],
                "pose":       vp["pose"],
                "rgb_path":   rp_,
                "depth_path": dp_,
            })
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
    fig.suptitle(f"Step 2-GS: 3DGS Rendered — {len(rendered)} images", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "step2_gs_rendered.png"), dpi=150); plt.close()
    print(f"  Saved: step2_gs_rendered.png")

    pkl_path = os.path.join(output_dir, "step2_gs_data.pkl")
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
        mode: "pointcloud" (O3D split-render) or "gs" (3D Gaussian Splatting)
        **kwargs: GS-specific params (kapture_dir, gs_epochs, voxel_size, accum_steps, etc.)
    """
    if mode == "gs":
        return _render_gs(ply_path, viewpoints, config, output_dir,
                          step0_data=step0_data, **kwargs)
    else:
        return _render_pointcloud(ply_path, viewpoints, config, output_dir, step0_data)
