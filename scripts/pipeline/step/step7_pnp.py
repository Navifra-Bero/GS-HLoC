import os, pickle
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def step7_pnp(step6_data, step5_data, config, output_dir):
    """
    논문 Algorithm 1:
    2D-3D correspondence → PnP → T_WQ = T_WR × T_QR⁻¹
    """
    print("\n" + "="*60 + "\nSTEP 7: 2D-3D correspondence + PnP\n" + "="*60)
    cam      = config["camera"]
    onl      = config.get("online", {})
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    K        = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    dist_c   = np.zeros(4)
    dmin     = cam.get("depth_min", 0.3)
    dmax     = cam.get("depth_max", 20.0)

    matched_q = step6_data["matched_q_kps"]
    matched_r = step6_data["matched_r_kps"]
    ref_entry = step6_data["best_cand"]
    query_rgb = step6_data["query_rgb"]
    ref_rgb   = step6_data["ref_rgb"]
    gt_entry  = step5_data.get("gt_entry")

    dep  = np.load(ref_entry["depth_path"])
    T_WR = np.array(ref_entry["pose"], dtype=np.float64)

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
        pts3d.append([(ru-cx)*pz/fx, (rv-cy)*pz/fy, pz])

    pts2d = np.array(pts2d, dtype=np.float64)
    pts3d = np.array(pts3d, dtype=np.float64)

    print(f"  Matched pairs  : {len(matched_q)}")
    print(f"  Valid (depth OK): {len(pts2d)}  (depth invalid: {invalid})")
    print(f"  Ref pose        : {T_WR[:3,3].round(3)}")

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

    # 시각화
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
    ax.set_title("Top-down: Ref (orange), GT (blue), Estimated (red)")

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
