import os, pickle
import numpy as np
import open3d as o3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import (binary_dilation, binary_erosion, distance_transform_edt,
                           gaussian_filter, maximum_filter)
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
    sample_mode = samp.get("sample_mode", "skeleton")   # "skeleton" or "grid"
    skel_grid_spacing_m = samp.get("skel_grid_spacing_m", 0.5)
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

        cz = fz + ch
        for si in sel:
            for yi in range(ny):
                yaw = 2*np.pi*yi/ny
                forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
                up = np.array([0.0, 0.0, 1.0])
                right = np.cross(forward, up)
                right = right / np.linalg.norm(right)
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
