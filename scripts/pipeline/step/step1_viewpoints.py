import os, pickle
import numpy as np
import open3d as o3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import (binary_dilation, binary_erosion, distance_transform_edt,
                           gaussian_filter, maximum_filter, label, binary_fill_holes)
from skimage.morphology import skeletonize


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
    if len(points) == 0:
        raise ValueError(f"No points loaded from {ap}")

    max_floors = samp.get("max_floors", 1)
    min_gap = samp.get("min_floor_gap", 2.5)
    max_fh = samp.get("max_floor_height", 5.0)
    min_fp = samp.get("min_floor_points", 100)

    normals_need_estimate = not pcd.has_normals()
    if not normals_need_estimate:
        normals0 = np.asarray(pcd.normals)
        normal_norm = np.linalg.norm(normals0, axis=1)
        normals_need_estimate = (
            len(normals0) != len(points)
            or not np.isfinite(normals0).all()
            or np.percentile(normal_norm, 95) < 1e-6
        )
    if normals_need_estimate:
        print("  Normals missing/invalid; estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    normals = np.asarray(pcd.normals)

    nth = align_cfg.get("normal_threshold", 0.8)
    dots = normals @ np.array([0.0, 0.0, 1.0])
    up_mask = dots > nth
    down_mask = dots < -nth
    horizontal_mask = np.abs(dots) > nth

    if up_mask.sum() >= min_fp:
        floor_normal_mask = up_mask
        normal_mode = "upward"
    elif down_mask.sum() >= min_fp:
        floor_normal_mask = down_mask
        normal_mode = "downward (flipped normal orientation)"
    elif horizontal_mask.sum() >= min_fp:
        floor_normal_mask = horizontal_mask
        normal_mode = "horizontal abs(normal_z)"
    else:
        floor_normal_mask = np.ones(len(points), dtype=bool)
        normal_mode = "all-points fallback (no reliable normals)"

    up_pts = points[floor_normal_mask]
    print(f"  Upward-facing: {int(up_mask.sum())} ({100*up_mask.sum()/len(points):.1f}%)")
    print(f"  Downward-facing: {int(down_mask.sum())} ({100*down_mask.sum()/len(points):.1f}%)")
    print(f"  Floor normal mode: {normal_mode}  ({len(up_pts)} pts)")

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
    ch = samp.get("height_above_floor", 1.2)
    ny = samp.get("num_yaw_angles", 4)
    occupancy_mode = str(samp.get("occupancy_mode", "floor_normals")).lower()
    # height_offsets_m: 기준 높이(ch)에서의 오프셋 목록 (미터)
    # e.g. [0.0, -0.2, 0.2] → 기준 / 20cm 아래 / 20cm 위
    height_offsets = list(samp.get("height_offsets_m", [0.0, -0.2, 0.2]))
    print(f"  Height offsets: {height_offsets} m")
    mk = samp.get("morph_kernel_size", 3)
    mi = samp.get("morph_iterations", 4)
    dr = samp.get("distance_thresh_ratio", 0.2)
    sample_mode = samp.get("sample_mode", "skeleton")   # "skeleton" or "grid"
    skel_grid_spacing_m = samp.get("skel_grid_spacing_m", 0.8)
    keep_largest_component = bool(samp.get("keep_largest_component", True))
    all_vp = []; debug_imgs = {}; vid = 0

    for fi, fz in enumerate(floors):
        print(f"\n  --- Floor {fi}: z={fz:.2f}m ---")
        band = samp.get("floor_band", 0.3)
        if occupancy_mode in ("height_slice_obstacle", "height_slice", "bev_slice"):
            fm = (points[:,2] > fz-band) & (points[:,2] < fz+band)
        else:
            fm = floor_normal_mask & (points[:,2] > fz-band) & (points[:,2] < fz+band)
        fp = points[fm]
        print(f"    Floor points: {len(fp)}")
        if len(fp) < 50: print("    Skip"); continue

        if occupancy_mode in ("height_slice_obstacle", "height_slice", "bev_slice"):
            slice_min_h = float(samp.get("slice_min_height_m", 0.10))
            slice_max_h = float(samp.get("slice_max_height_m", 1.50))
            floor_radius_m = float(samp.get("slice_floor_splat_radius_m",
                                           samp.get("slice_splat_radius_m", 0.25)))
            obstacle_radius_m = float(samp.get("slice_obstacle_inflate_m", 0.35))
            obstacle_close_m = float(samp.get("slice_obstacle_close_m",
                                             max(obstacle_radius_m, 0.8)))
            obstacle_min_count = int(samp.get("slice_obstacle_min_points_per_cell", 5))
            clearance_m = float(samp.get("slice_wall_close_m", 0.30))

            obstacle_mask = (
                (points[:, 2] > fz + slice_min_h) &
                (points[:, 2] < fz + slice_max_h)
            )
            obstacle_pts = points[obstacle_mask]
            support_pts = np.vstack([fp, obstacle_pts]) if len(obstacle_pts) else fp
            print(f"    BEV slice: z=[{fz + slice_min_h:.2f}, {fz + slice_max_h:.2f}] "
                  f"obstacles={len(obstacle_pts)}")
        else:
            support_pts = fp
            obstacle_pts = None
            floor_radius_m = 0.0
            obstacle_radius_m = 0.0
            clearance_m = 0.0

        margin = float(samp.get("slice_margin_m", 1.0))
        xn, yn = support_pts[:,0].min()-margin, support_pts[:,1].min()-margin
        xx, yx = support_pts[:,0].max()+margin, support_pts[:,1].max()+margin
        iw = int(np.ceil((xx-xn)/gr)); ih = int(np.ceil((yx-yn)/gr))
        print(f"    Image: {iw}x{ih}px (res={gr}m/px)")

        def rasterize_xy(xy_pts, radius_m=0.0, min_count=1):
            mask = np.zeros((ih, iw), dtype=np.uint8)
            if xy_pts is None or len(xy_pts) == 0:
                return mask
            px = np.clip(((xy_pts[:,0]-xn)/gr).astype(int), 0, iw-1)
            py = np.clip(((xy_pts[:,1]-yn)/gr).astype(int), 0, ih-1)
            if min_count <= 1:
                mask[py, px] = 1
            else:
                counts = np.zeros((ih, iw), dtype=np.int32)
                np.add.at(counts, (py, px), 1)
                mask = (counts >= min_count).astype(np.uint8)
            radius_px = int(np.ceil(float(radius_m) / gr))
            if radius_px > 0:
                st = np.ones((2 * radius_px + 1, 2 * radius_px + 1), dtype=np.uint8)
                mask = binary_dilation(mask, structure=st).astype(np.uint8)
            return mask

        kern = np.ones((mk, mk), dtype=np.uint8)
        if occupancy_mode in ("height_slice_obstacle", "height_slice", "bev_slice"):
            # Build free-space from detected floor support, then remove the
            # height-slice obstacles.  Obstacle-only hole filling is brittle
            # when sparse wall loops have gaps; floor support keeps us inside
            # the observed traversable area instead of selecting exterior rings.
            occ = rasterize_xy(obstacle_pts, radius_m=0.0, min_count=obstacle_min_count)
            obstacle_occ = rasterize_xy(
                obstacle_pts, radius_m=obstacle_radius_m, min_count=obstacle_min_count)
            floor_occ = rasterize_xy(fp, radius_m=floor_radius_m)
            close_iter = int(samp.get("slice_floor_close_iterations", mi))
            floor_support = binary_dilation(
                floor_occ, structure=kern, iterations=max(close_iter, 1))
            floor_support = binary_erosion(
                floor_support, structure=kern, iterations=max(close_iter // 2, 1))
            floor_support = binary_fill_holes(floor_support > 0)
            closed = floor_support & (obstacle_occ == 0)

            if closed.sum() < 50 and occ.sum() > 0:
                print("    [WARN] floor support too sparse; trying obstacle-loop fallback")
                obstacle_closed = rasterize_xy(obstacle_pts, radius_m=obstacle_close_m)
                filled = binary_fill_holes(obstacle_closed > 0)
                closed = filled & (obstacle_occ == 0)

            if keep_largest_component and np.any(closed):
                cc, ncc = label(closed)
                if ncc > 1:
                    sizes = np.bincount(cc.ravel())
                    sizes[0] = 0
                    closed = cc == int(np.argmax(sizes))
            closed = closed.astype(np.uint8)
        else:
            occ = rasterize_xy(fp)
            closed = binary_dilation(occ, structure=kern, iterations=mi).astype(np.uint8)
            closed = binary_erosion(closed, structure=kern, iterations=mi).astype(np.uint8)

        dm = distance_transform_edt(closed)
        dmx = dm.max()
        print(f"    Dist max: {dmx:.1f}px ({dmx*gr:.2f}m)")

        dn = dm/dmx if dmx > 0 else dm
        if occupancy_mode in ("height_slice_obstacle", "height_slice", "bev_slice"):
            min_clear_px = float(clearance_m) / gr
            corr = ((dm > min_clear_px) & (dn > dr)).astype(np.uint8)
            if corr.sum() == 0 and np.any(closed):
                print("    [WARN] clearance threshold too strict; falling back to distance-ratio corridor")
                corr = (dn > dr).astype(np.uint8)
        else:
            corr = (dn > dr).astype(np.uint8)
        cs = gaussian_filter(corr.astype(float), sigma=3)
        cb = (cs > 0.3).astype(np.uint8)

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
            "xn": xn, "yn": yn, "gr": gr, "fz": fz,
        }

    n_heights = len(height_offsets)
    n_views_per_pos = ny * n_heights
    print(f"\n  Total viewpoints: {len(all_vp)}  "
          f"(yaw×height = {ny}×{n_heights}, height_offsets={height_offsets} m)")

    nf = len(debug_imgs)
    if nf > 0:
        fig, axes = plt.subplots(nf, 6, figsize=(30, 5*nf))
        if nf == 1: axes = axes.reshape(1, -1)
        if occupancy_mode in ("height_slice_obstacle", "height_slice", "bev_slice"):
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
    return all_vp
