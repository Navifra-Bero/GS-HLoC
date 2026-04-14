"""
STEP 2: Rendering — 2D Gaussian Splatting (2DGS) mode.

2DGS vs 3DGS:
- 3D ellipsoid → 2D disc (surfel): needle artifact 구조적으로 불가능
- depth/normal 정확도 향상 → PnP 정확도 개선
- distortion loss 지원 → geometry 디테일 향상
- gsplat rasterization_2dgs 사용

Usage:
    from pipeline.step.step2_render_2d import step2_render_2dgs
    rendered = step2_render_2dgs(ply_path, viewpoints, config, output_dir,
                                  step0_data=s0, kapture_dir=..., gs_epochs=150, ...)
"""
import os, pickle
import numpy as np
import open3d as o3d
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .step2_render import (
    slim_rendered,
    _gs_parse_sensors,
    _gs_parse_trajectories,
    _gs_parse_records,
)


def _render_2dgs(ply_path, viewpoints, config, output_dir,
                 kapture_dir, step0_data=None,
                 gs_epochs=30, sh_degree=0,
                 train_img_size=512, subsample=1,
                 voxel_size=None, accum_steps=4):
    import torch
    import torch.nn.functional as F
    from pytorch_msssim import ssim as _ssim_fn
    try:
        from gsplat import rasterization_2dgs
        from gsplat.strategy import DefaultStrategy
    except ImportError:
        raise ImportError("gsplat 미설치 또는 2DGS 미지원: pip install gsplat")

    print("\n" + "="*60 + "\nSTEP 2-2DGS: 2D Gaussian Splatting Rendering\n" + "="*60)

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

    rd = os.path.join(output_dir, "rendered_2dgs")
    gs_ckpt = os.path.join(output_dir, "gaussians_2dgs.pt")
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
    # 2DGS: scales (N, 3) — 세 번째 축은 disc 법선 방향 (얇게 초기화)
    scales_xy = torch.full((N, 2), init_scale, device=device)
    scales_z  = torch.full((N, 1), init_scale - 3.0, device=device)  # 법선 방향 매우 얇게
    scales    = torch.nn.Parameter(torch.cat([scales_xy, scales_z], dim=1))
    quats     = torch.nn.Parameter(
                    torch.cat([torch.ones(N, 1), torch.zeros(N, 3)], dim=1).to(device))
    opacities = torch.nn.Parameter(torch.full((N,), -2.0, device=device))

    # 2DGS는 rasterization_2dgs가 SH 미지원 → raw RGB (N, 3)
    colors = torch.nn.Parameter(torch.from_numpy(clr).to(device))

    # ── 2. 학습된 모델이 있으면 로드, 없으면 학습
    if os.path.exists(gs_ckpt):
        print(f"  [2DGS] 저장된 모델 로드: {gs_ckpt}")
        ckpt  = torch.load(gs_ckpt, map_location=device)
        means     = torch.nn.Parameter(ckpt["means"])
        scales    = torch.nn.Parameter(ckpt["scales"])
        quats     = torch.nn.Parameter(ckpt["quats"])
        opacities = torch.nn.Parameter(ckpt["opacities"])
        colors    = torch.nn.Parameter(ckpt["colors"])
        N = len(means)
        print(f"  [2DGS] 로드된 GS: {N:,} pts")
    else:
        # ── 3. kapture 매핑 데이터 로드
        sensors_txt     = os.path.join(kapture_dir, "sensors.txt")
        traj_txt        = os.path.join(kapture_dir, "trajectories.txt")
        records_txt     = os.path.join(kapture_dir, "records_camera.txt")
        images_root     = os.path.join(kapture_dir, "records_data")
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

        train_data = []
        for ts, dev, fname in records:
            key = (ts, dev)
            if key not in traj or dev not in sensor_params:
                continue
            img_path = os.path.join(images_root, fname)
            if not os.path.exists(img_path):
                continue
            T_c2w_aligned = T_align @ traj[key]
            train_data.append({
                "path":       img_path,
                "T_c2w":      T_c2w_aligned.astype(np.float32),
                "cam":        sensor_params[dev],
                "depth_path": depth_map.get(key),
            })
        print(f"  Train frames: {len(train_data)}  "
              f"(with depth: {sum(1 for d in train_data if d['depth_path'])})")
        if not train_data:
            raise RuntimeError("학습 이미지를 찾을 수 없습니다. kapture_dir 경로를 확인하세요.")

        # ── 4. 학습 루프
        MEANS_LR = 1.6e-4
        params = {
            "means":     means,
            "scales":    scales,
            "quats":     quats,
            "opacities": opacities,
            "colors":    colors,
        }
        optimizers = {
            "means":     torch.optim.Adam([means],     lr=MEANS_LR),
            "scales":    torch.optim.Adam([scales],    lr=5e-3),
            "quats":     torch.optim.Adam([quats],     lr=1e-3),
            "opacities": torch.optim.Adam([opacities], lr=5e-2),
            "colors":    torch.optim.Adam([colors],    lr=2.5e-3),
        }

        iters_per_epoch = len(train_data) // accum_steps
        total_iters     = gs_epochs * iters_per_epoch

        means_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers["means"], T_max=total_iters, eta_min=MEANS_LR * 0.01
        )

        scene_scale = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) / 2.0

        strategy = DefaultStrategy(
            verbose=False,
            refine_start_iter=iters_per_epoch * 2,
            refine_stop_iter=int(total_iters * 0.75),
            refine_every=max(1, iters_per_epoch // 5),
            reset_every=iters_per_epoch * 3,
            prune_opa=0.005,
            prune_scale3d=0.1,
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
            epoch_loss = epoch_rgb_loss = 0.0
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

                    cam    = entry["cam"]
                    orig_h, orig_w = cam["h"], cam["w"]
                    rs     = train_img_size / max(orig_h, orig_w)
                    new_w  = int(orig_w * rs)
                    new_h  = int(orig_h * rs)
                    img_rgb = cv2.resize(img_rgb, (new_w, new_h))
                    gt = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).to(device)

                    sx = new_w / cam["w"]; sy = new_h / cam["h"]
                    fx_t = cam["fx"] * sx; fy_t = cam["fy"] * sy
                    cx_t = cam["cx"] * sx; cy_t = cam["cy"] * sy

                    T_c2w   = torch.from_numpy(entry["T_c2w"]).to(device)
                    viewmat = torch.linalg.inv(T_c2w).unsqueeze(0)
                    K = torch.tensor([[fx_t, 0, cx_t],
                                      [0, fy_t, cy_t],
                                      [0,   0,    1]], device=device).unsqueeze(0).float()

                    renders, _, _, _, _, _, info = rasterization_2dgs(
                        means=params["means"],
                        quats=F.normalize(params["quats"], dim=-1),
                        scales=torch.exp(params["scales"]),
                        opacities=torch.sigmoid(params["opacities"]),
                        colors=params["colors"].clamp(0, 1).unsqueeze(0),  # (1, N, 3)
                        viewmats=viewmat,
                        Ks=K,
                        width=new_w,
                        height=new_h,
                        sh_degree=None,
                        near_plane=0.1,
                        far_plane=100.0,
                        packed=False,
                        render_mode="RGB",
                    )
                    pred = renders[0, ..., :3].clamp(0, 1)  # (H, W, 3)

                    # ── RGB loss: L1 + SSIM
                    l1_loss   = F.l1_loss(pred, gt)
                    pred_bchw = pred.permute(2, 0, 1).unsqueeze(0)
                    gt_bchw   = gt.permute(2, 0, 1).unsqueeze(0)
                    ssim_val  = _ssim_fn(pred_bchw, gt_bchw, data_range=1.0, size_average=True)
                    rgb_loss  = 0.8 * l1_loss + 0.2 * (1.0 - ssim_val)

                    # ── Total loss (2DGS: disc 구조가 geometry 제약, depth forward 불필요)
                    loss = rgb_loss / accum_steps

                    if acc == accum_steps - 1:
                        strategy.step_pre_backward(params, optimizers, state, global_it, info)

                    loss.backward()
                    last_info = info
                    step_loss += loss.item()
                    epoch_rgb_loss += rgb_loss.item()

                strategy.step_post_backward(params, optimizers, state, global_it, last_info,
                                            packed=False)

                for opt in optimizers.values():
                    opt.step()
                means_scheduler.step()

                # Hard clamp: log-scale [-6, 0] → exp 범위 [~0.002, 1.0m]
                with torch.no_grad():
                    params["scales"].clamp_(-6.0, 1.0)

                epoch_loss  += step_loss
                epoch_count += 1
                global_it   += 1
                pbar.set_postfix(loss=f"{step_loss:.4f}", GS=f"{len(params['means']):,}")

            pbar.close()
            avg_loss  = epoch_loss  / max(epoch_count, 1)
            n_imgs    = epoch_count * accum_steps
            avg_rgb   = epoch_rgb_loss   / max(n_imgs, 1)
            n_gs      = len(params["means"])
            print(f"  Epoch {epoch+1}/{gs_epochs}  total={avg_loss:.4f}  "
                  f"rgb={avg_rgb:.4f}  GS={n_gs:,}")

            # ── 중간 체크포인트 (10 epoch마다 + 마지막)
            save_interval = 10
            if ((epoch + 1) % save_interval == 0) or (epoch + 1 == gs_epochs):
                gs_dir    = os.path.join(output_dir, "gaussian_2dgs")
                epoch_dir = os.path.join(gs_dir, f"epoch_{epoch+1}")
                os.makedirs(epoch_dir, exist_ok=True)

                torch.save({
                    "means":     params["means"].data,
                    "scales":    params["scales"].data,
                    "quats":     params["quats"].data,
                    "opacities": params["opacities"].data,
                    "colors":    params["colors"].data,
                }, os.path.join(epoch_dir, "gaussians_2dgs.pt"))

                # 샘플 렌더링
                n_samples = min(10, len(viewpoints))
                if n_samples > 0:
                    cam_cfg = config["camera"]
                    W_s, H_s = cam_cfg["width"], cam_cfg["height"]
                    K_s = torch.tensor([[cam_cfg["fx"], 0, cam_cfg["cx"]],
                                        [0, cam_cfg["fy"], cam_cfg["cy"]],
                                        [0, 0, 1]], device=device).unsqueeze(0).float()
                    sample_indices = np.linspace(0, len(viewpoints) - 1, n_samples, dtype=int)
                    for si in sample_indices:
                        sample_vp = viewpoints[si]
                        vm_s = torch.from_numpy(
                            np.linalg.inv(sample_vp["pose"].astype(np.float32))
                        ).to(device).unsqueeze(0).float()
                        with torch.no_grad():
                            rend_s, _, _, _, _, _, _ = rasterization_2dgs(
                                means=params["means"].detach(),
                                quats=F.normalize(params["quats"].detach(), dim=-1),
                                scales=torch.exp(params["scales"].detach()),
                                opacities=torch.sigmoid(params["opacities"].detach()),
                                colors=params["colors"].detach().clamp(0, 1).unsqueeze(0),  # (1, N, 3)
                                viewmats=vm_s, Ks=K_s,
                                width=W_s, height=H_s,
                                sh_degree=None,
                                near_plane=0.3, far_plane=100.0,
                                packed=False, render_mode="RGB",
                            )
                        rgb_s = (rend_s[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                        cv2.imwrite(
                            os.path.join(epoch_dir, f"sample_{sample_vp['id']:06d}.png"),
                            cv2.cvtColor(rgb_s, cv2.COLOR_RGB2BGR))
                print(f"    → Checkpoint saved: {epoch_dir}/")

        # 최종 파라미터 저장
        means     = params["means"]
        scales    = params["scales"]
        quats     = params["quats"]
        opacities = params["opacities"]
        colors    = params["colors"]

        torch.save({
            "means":     means.data,
            "scales":    scales.data,
            "quats":     quats.data,
            "opacities": opacities.data,
            "colors":    colors.data,
        }, gs_ckpt)
        print(f"  [2DGS] 최종 모델 저장: {gs_ckpt}  ({len(means):,} GS)")

        # PLY 저장
        gs_pts   = means.data.cpu().numpy().astype(np.float32)
        gs_rgb   = colors.data.clamp(0, 1).cpu().numpy().astype(np.float32)
        gs_opa   = torch.sigmoid(opacities.data).cpu().numpy().astype(np.float32)
        valid    = gs_opa > 0.05
        gs_dir   = os.path.join(output_dir, "gaussian_2dgs")
        os.makedirs(gs_dir, exist_ok=True)
        gs_pcd   = o3d.geometry.PointCloud()
        gs_pcd.points = o3d.utility.Vector3dVector(gs_pts[valid])
        gs_pcd.colors = o3d.utility.Vector3dVector(gs_rgb[valid])
        gs_ply_path = os.path.join(gs_dir, "gaussian_2dgs_map.ply")
        o3d.io.write_point_cloud(gs_ply_path, gs_pcd)
        print(f"  [2DGS] PLY 저장: {gs_ply_path}  ({valid.sum():,}/{len(gs_opa):,} pts)")

    # ── 5. grid viewpoints 렌더링
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization_2dgs

    cam_cfg  = config["camera"]
    W_out, H_out = cam_cfg["width"], cam_cfg["height"]
    K_render = torch.tensor([
        [[cam_cfg["fx"], 0, cam_cfg["cx"]],
         [0, cam_cfg["fy"], cam_cfg["cy"]],
         [0, 0, 1]]], device=device).float()

    rendered = []
    print(f"  Rendering {len(viewpoints)} viewpoints ...")
    with torch.no_grad():
        for i, vp in enumerate(viewpoints):
            pose    = vp["pose"].astype(np.float32)
            viewmat = torch.from_numpy(np.linalg.inv(pose)).to(device).unsqueeze(0).float()
            renders, _, _, _, _, _, _ = rasterization_2dgs(
                means=means.detach(),
                quats=F.normalize(quats.detach(), dim=-1),
                scales=torch.exp(scales.detach()),
                opacities=torch.sigmoid(opacities.detach()),
                colors=colors.detach().clamp(0, 1).unsqueeze(0),  # (1, N, 3)
                viewmats=viewmat,
                Ks=K_render,
                width=W_out,
                height=H_out,
                sh_degree=None,
                near_plane=0.3,
                far_plane=100.0,
                packed=False,
                render_mode="RGB+D",
            )
            rgb_np   = (renders[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            depth_np = renders[0, ..., 3].cpu().numpy().astype(np.float32)

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
    idx = np.linspace(0, len(rendered) - 1, ns, dtype=int)
    fig, ax = plt.subplots(2, ns, figsize=(4 * ns, 8))
    if ns == 1: ax = ax.reshape(2, 1)
    for c, ii in enumerate(idx):
        r = rendered[ii]
        ax[0, c].imshow(cv2.cvtColor(cv2.imread(r["rgb_path"]), cv2.COLOR_BGR2RGB))
        ax[0, c].set_title(f"2DGS #{r['id']}", fontsize=9); ax[0, c].axis("off")
        dv = np.load(r["depth_path"])
        dv[dv > cam_cfg.get("depth_max", 10)] = 0
        ax[1, c].imshow(dv, cmap="plasma")
        ax[1, c].set_title("Depth", fontsize=9); ax[1, c].axis("off")
    fig.suptitle(f"Step 2-2DGS: Rendered — {len(rendered)} images", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "step2_2dgs_rendered.png"), dpi=150); plt.close()
    print(f"  Saved: step2_2dgs_rendered.png")

    pkl_path = os.path.join(output_dir, "step2_2dgs_data.pkl")
    pickle.dump(rendered, open(pkl_path, "wb"))
    pickle.dump(rendered, open(os.path.join(output_dir, "step2_data.pkl"), "wb"))
    print(f"  Saved: {pkl_path}  ({len(rendered)} entries)")
    return rendered


def step2_render_2dgs(ply_path, viewpoints, config, output_dir,
                      step0_data=None, **kwargs):
    """2DGS 렌더링 entry point."""
    return _render_2dgs(ply_path, viewpoints, config, output_dir,
                        step0_data=step0_data, **kwargs)
