import os, pickle
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ── PnP solvers ────────────────────────────────────────────────────────────

def _solve_pnp_opencv(pts2d, pts3d, K, config):
    """OpenCV EPnP + RANSAC."""
    pnp    = config.get("pnp", {})
    reproj = float(pnp.get("reproj_threshold",
                   pnp.get("ransac_reproj_threshold", 8.0)))
    iters  = int(pnp.get("ransac_iterations", 1000))
    conf   = float(pnp.get("pnp_confidence", 0.99))
    dist_c = np.zeros(4)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, dist_c,
        iterationsCount=iters,
        reprojectionError=reproj,
        confidence=conf,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None:
        return None, None, np.array([])

    inlier_idx = inliers.flatten()
    R_mat, _   = cv2.Rodrigues(rvec)
    T_QR       = np.eye(4)
    T_QR[:3, :3] = R_mat
    T_QR[:3, 3]  = tvec.flatten()
    return T_QR, rvec, inlier_idx


def _solve_pnp_superansac(pts2d, pts3d_world, K, config):
    """
    SuperANSAC (PROSAC + MAGSAC + GCRANSAC) PnP.
    pts3d_world: world frame 3D points (T_WR @ pts3d_cam 변환 후 전달)
    Returns T_WQ (4x4), inlier_idx
    """
    import pysuperansac

    pnp    = config.get("pnp", {})
    reproj = float(pnp.get("reproj_threshold", 4.0))
    iters  = int(pnp.get("superansac_max_iterations", 1000))
    conf   = float(pnp.get("superansac_confidence", 0.99))

    # SimplePinhole: [focal, cx, cy]
    f_avg = float((K[0, 0] + K[1, 1]) / 2.0)
    cam_params = np.array([f_avg, float(K[0, 2]), float(K[1, 2])], dtype=np.float64)

    # 3D를 양수 영역으로 shift (neighborhood graph 구성 안정화)
    min_coords   = pts3d_world.min(axis=0)
    pts3d_shifted = pts3d_world - min_coords

    # Nx5: [x, y, X, Y, Z]  — bounding_box = 각 열의 최댓값 (5 elements)
    corr         = np.hstack([pts2d, pts3d_shifted]).astype(np.float64)
    bounding_box = np.max(corr, axis=0)  # shape (5,)

    cfg = pysuperansac.RANSACSettings()
    cfg.inlier_threshold   = reproj
    cfg.max_iterations     = iters
    cfg.confidence         = conf
    cfg.sampler            = pysuperansac.SamplerType.PROSAC
    cfg.scoring            = pysuperansac.ScoringType.MAGSAC
    cfg.local_optimization = pysuperansac.LocalOptimizationType.GCRANSAC

    R, t, inlier_idx, score, used_iters = pysuperansac.estimateAbsolutePose(
        np.ascontiguousarray(corr),
        pysuperansac.CameraType.SimplePinhole,
        cam_params,
        bounding_box,
        [],
        cfg,
    )

    # shift 역변환: t_orig = t_shifted - R @ min_coords
    t = t.flatten() - R @ min_coords
    print(f"  SuperANSAC: {len(inlier_idx)} inliers  score={score:.4f}  iters={used_iters}")

    if len(inlier_idx) == 0:
        return None, np.array([])

    # SuperANSAC 출력 R, t 는 world→camera (T_CW) convention
    # T_WQ (camera→world) = inv(T_CW)
    T_CW = np.eye(4)
    T_CW[:3, :3] = R
    T_CW[:3, 3]  = t
    T_WQ = np.linalg.inv(T_CW)
    return T_WQ, np.array(inlier_idx)


def _refine_pnp(pts2d_in, pts3d_in, K, rvec_init, tvec_init):
    """inlier만으로 iterative refine (SOLVEPNP_ITERATIVE)."""
    dist_c = np.zeros(4)
    ok, rvec_r, tvec_r = cv2.solvePnP(
        pts3d_in, pts2d_in, K, dist_c,
        rvec=rvec_init, tvec=tvec_init,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return rvec_init, tvec_init
    return rvec_r, tvec_r


# ── Main step ──────────────────────────────────────────────────────────────

def step7_pnp(step6_data, step5_data, config, output_dir, save_images=True):
    """
    2D-3D correspondence → PnP → T_WQ

    solver 선택 (config.pnp.solver):
      "opencv"     : cv2.solvePnPRansac (EPnP + vanilla RANSAC)
      "superansac" : PROSAC + MAGSAC + GCRANSAC (SuperANSAC)

    refine (config.pnp.refine = true):
      RANSAC 후 inlier만으로 cv2.solvePnP ITERATIVE refine
    """
    print("\n" + "="*60 + "\nSTEP 7: 2D-3D correspondence + PnP\n" + "="*60)

    cam = config["camera"]
    pnp = config.get("pnp", {})
    fx, fy = cam["fx"], cam["fy"]
    cx, cy = cam["cx"], cam["cy"]
    K      = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dmin   = cam.get("depth_min", 0.3)
    dmax   = cam.get("depth_max", 20.0)

    solver    = pnp.get("solver", "opencv").strip().lower()
    do_refine = bool(pnp.get("refine", True))
    min_in    = int(pnp.get("min_inliers", 6))

    matched_q = step6_data["matched_q_kps"]
    matched_r = step6_data["matched_r_kps"]
    ref_entry = step6_data["best_cand"]
    query_rgb = step6_data["query_rgb"]
    ref_rgb   = step6_data["ref_rgb"]
    gt_entry  = step5_data.get("gt_entry")

    dep  = np.load(ref_entry["depth_path"])
    T_WR = np.array(ref_entry["pose"], dtype=np.float64)

    # ── 2D-3D lifting (ref camera frame) ──────────────────────────────
    pts2d = []; pts3d_cam = []; invalid = 0
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
        pts3d_cam.append([(ru - cx) * pz / fx,
                           (rv - cy) * pz / fy,
                           pz])

    pts2d     = np.array(pts2d,     dtype=np.float64)
    pts3d_cam = np.array(pts3d_cam, dtype=np.float64)

    print(f"  Matched pairs      : {len(matched_q)}")
    print(f"  Valid (depth OK)   : {len(pts2d)}  (depth invalid: {invalid})")
    print(f"  Ref pose           : {T_WR[:3, 3].round(3)}")
    print(f"  Solver             : {solver}  refine={do_refine}")

    estimated_pose = None
    inlier_count   = 0
    T_QR           = None

    if len(pts2d) < min_in:
        print(f"  PnP SKIPPED: {len(pts2d)} < {min_in} correspondences")

    elif solver == "superansac":
        # ref cam → world frame
        pts3d_h     = np.hstack([pts3d_cam, np.ones((len(pts3d_cam), 1))])
        pts3d_world = (T_WR @ pts3d_h.T).T[:, :3]

        T_WQ_raw, inlier_idx = _solve_pnp_superansac(
            pts2d, pts3d_world, K, config)

        if T_WQ_raw is not None and len(inlier_idx) >= min_in:
            inlier_count = len(inlier_idx)

            if do_refine:
                # world → ref cam frame으로 변환 후 ITERATIVE refine
                T_RW = np.linalg.inv(T_WR)
                pts3d_cam_in = (T_RW @ np.hstack([
                    pts3d_world[inlier_idx],
                    np.ones((inlier_count, 1))]).T).T[:, :3]
                pts2d_in = pts2d[inlier_idx]
                # T_WQ = T_WR @ inv(T_QR)  →  T_QR = inv(T_WQ) @ T_WR
                T_QR_init = np.linalg.inv(T_WQ_raw) @ T_WR
                rvec_init, _ = cv2.Rodrigues(T_QR_init[:3, :3])
                tvec_init    = T_QR_init[:3, 3].reshape(3, 1)
                rvec_r, tvec_r = _refine_pnp(
                    pts2d_in, pts3d_cam_in, K, rvec_init, tvec_init)
                R_r, _ = cv2.Rodrigues(rvec_r)
                T_QR_r = np.eye(4)
                T_QR_r[:3, :3] = R_r
                T_QR_r[:3, 3]  = tvec_r.flatten()
                estimated_pose = T_WR @ np.linalg.inv(T_QR_r)
                print(f"  Refine applied (SuperANSAC → ITERATIVE)")
            else:
                estimated_pose = T_WQ_raw

            print(f"  SuperANSAC SUCCESS: {inlier_count}/{len(pts2d)} inliers")
            print(f"  T_WQ position: {estimated_pose[:3, 3].round(4)}")
        else:
            print("  SuperANSAC FAILED: insufficient inliers")

    else:  # opencv
        T_QR_raw, rvec, inlier_idx = _solve_pnp_opencv(pts2d, pts3d_cam, K, config)

        if T_QR_raw is not None and len(inlier_idx) >= min_in:
            inlier_count = len(inlier_idx)
            T_QR = T_QR_raw

            if do_refine:
                pts2d_in     = pts2d[inlier_idx]
                pts3d_cam_in = pts3d_cam[inlier_idx]
                tvec_init    = T_QR_raw[:3, 3].reshape(3, 1)
                rvec_r, tvec_r = _refine_pnp(
                    pts2d_in, pts3d_cam_in, K, rvec, tvec_init)
                R_r, _ = cv2.Rodrigues(rvec_r)
                T_QR = np.eye(4)
                T_QR[:3, :3] = R_r
                T_QR[:3, 3]  = tvec_r.flatten()
                print(f"  Refine applied (EPnP → ITERATIVE)")

            estimated_pose = T_WR @ np.linalg.inv(T_QR)
            print(f"  OpenCV PnP SUCCESS: {inlier_count}/{len(pts2d)} inliers")
            print(f"  T_WQ position: {estimated_pose[:3, 3].round(4)}")
        else:
            print("  OpenCV PnP FAILED: insufficient inliers")

    if gt_entry is not None and estimated_pose is not None:
        err = np.linalg.norm(
            estimated_pose[:3, 3] - np.array(gt_entry["pose"])[:3, 3])
        print(f"  GT  position  : {np.array(gt_entry['pose'])[:3, 3].round(4)}")
        print(f"  Position error: {err:.4f} m")

    if save_images:
        # ── 시각화 ───────────────────────────────────────────────────────
        fig = plt.figure(figsize=(18, 10))
        gs_ = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

        ax = fig.add_subplot(gs_[0, 0])
        ax.imshow(query_rgb)
        if len(pts2d) > 0:
            ax.scatter(pts2d[:, 0], pts2d[:, 1], c="red", s=10, marker="x", alpha=0.7)
        ax.set_title(f"Query + {len(pts2d)} 2D corr", fontsize=9); ax.axis("off")

        ax = fig.add_subplot(gs_[0, 1])
        ax.imshow(ref_rgb)
        ax.set_title(f"Reference #{ref_entry['id']}", fontsize=9); ax.axis("off")

        ax = fig.add_subplot(gs_[0, 2])
        if len(pts3d_cam) > 0:
            sc = ax.scatter(pts3d_cam[:, 0], pts3d_cam[:, 2],
                            c=pts3d_cam[:, 1], cmap="RdYlBu", s=20)
            plt.colorbar(sc, ax=ax, fraction=0.046, label="Y (m)")
        ax.set_title("3D pts: X vs Z (ref cam frame)")
        ax.set_xlabel("X"); ax.set_ylabel("Z")

        ax = fig.add_subplot(gs_[1, :])
        cands = step5_data["candidates"]
        all_pos = np.array([e["pose"][:3, 3] for e in cands])
        ax.scatter(all_pos[:, 0], all_pos[:, 1],
                   c="lightgray", s=5, alpha=0.4, label="Candidates")
        ref_pos = np.array(ref_entry["pose"])[:3, 3]
        ax.scatter(ref_pos[0], ref_pos[1], c="orange", s=200, marker="D",
                   zorder=5, label=f"Best Ref #{ref_entry['id']}")
        if gt_entry is not None:
            gp = np.array(gt_entry["pose"])[:3, 3]
            ax.scatter(gp[0], gp[1], c="blue", s=300, marker="*",
                       zorder=6, label="GT query")
        if estimated_pose is not None:
            ep = estimated_pose[:3, 3]
            ax.scatter(ep[0], ep[1], c="red", s=300, marker="^",
                       zorder=7, label="Estimated")
            ax.plot([ref_pos[0], ep[0]], [ref_pos[1], ep[1]], "r--", alpha=0.5)
            if gt_entry is not None:
                gp = np.array(gt_entry["pose"])[:3, 3]
                err = np.linalg.norm(ep - gp)
                ax.plot([ep[0], gp[0]], [ep[1], gp[1]], "b:", alpha=0.7,
                        label=f"err={err:.3f}m")
        ax.set_aspect("equal"); ax.legend(fontsize=8, loc="best")
        ax.set_title("Top-down: Ref (orange), GT (blue), Estimated (red)")

        status  = "SUCCESS" if estimated_pose is not None else "FAILED"
        err_str = ""
        if gt_entry is not None and estimated_pose is not None:
            err_str = (f" | err="
                       f"{np.linalg.norm(estimated_pose[:3,3]-np.array(gt_entry['pose'])[:3,3]):.3f}m")
        fig.suptitle(
            f"Step 7: PnP [{solver}{'→refine' if do_refine else ''}]"
            f" — {status} ({inlier_count} inliers, {len(pts2d)} corr){err_str}",
            fontsize=12)
        fig.savefig(os.path.join(output_dir, "step7_pnp.png"), dpi=150); plt.close()
        print(f"  Saved: step7_pnp.png")

    result = {
        "estimated_pose":    estimated_pose,
        "T_WR":              T_WR,
        "T_QR":              T_QR,
        "inlier_count":      inlier_count,
        "n_correspondences": len(pts2d),
        "best_ref_id":       ref_entry["id"],
        "gt_entry":          gt_entry,
        "solver":            solver,
    }
    if save_images:
        pickle.dump(result, open(os.path.join(output_dir, "step7_data.pkl"), "wb"))
    return result
