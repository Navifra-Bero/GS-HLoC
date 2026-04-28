import os, re, pickle, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .step.step5_retrieval import step5_retrieval
from .step.step5_retrieval_type2 import step5_retrieval_type2
from .step.step6_match import step6_match
from .step.step6_match_type2 import step6_match_type2
from .step.step7_pnp import step7_pnp
from .step.multi_cam import load_multi_cam_config, parse_kapture_records, find_sister_images


def localize_single(query_image_path, db, config, work_dir, save_images=True,
                    query_images=None):
    """단일 쿼리 이미지에 대해 step5→step7 파이프라인 실행.

    Args:
        query_images: multi-cam 모드 시 {cam_id: path} 딕셔너리. None = 단일 캠.

    config.multi_cam.retrieval_type == "type2"이면 step5_retrieval_type2 사용.
    """
    os.makedirs(work_dir, exist_ok=True)
    retrieval_type = config.get("multi_cam", {}).get("retrieval_type", "type1")
    step5_fn = step5_retrieval_type2 if retrieval_type == "type2" else step5_retrieval
    step6_fn = step6_match_type2 if retrieval_type == "type2" else step6_match
    try:
        s5 = step5_fn(query_image_path, db, config, work_dir,
                      save_images=save_images,
                      query_images=query_images)
        s6 = step6_fn(s5, config, work_dir, save_images=save_images)
        result = step7_pnp(s6, s5, config, work_dir, save_images=save_images)
        return result.get("estimated_pose") if result else None
    except Exception as e:
        print(f"    localize_single error: {e}")
        return None


def run_test_batch(test_dir, db, config, output_dir, gt_poses_path=None,
                   save_images=True):
    """test_dir 안의 모든 이미지에 대해 온라인 파이프라인 실행."""
    from scipy.spatial.transform import Rotation

    print("\n" + "="*60)
    print("BATCH TEST: Online localization on test set")
    print("="*60)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if os.path.isfile(test_dir):
        img_paths = [test_dir]
    else:
        img_paths = sorted([
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])
    if not img_paths:
        print(f"  ERROR: {test_dir} 에 이미지가 없습니다.")
        return

    print(f"  Test images: {len(img_paths)} ({test_dir})")

    # ── multi-cam 설정 로드 ───────────────────────────────────────────────
    mc_enabled, mc_cam_ids, mc_kapture_dir, mc_primary = load_multi_cam_config(config)
    mc_records = None
    if mc_enabled:
        if not os.path.isabs(mc_kapture_dir):
            mc_kapture_dir = os.path.join(os.getcwd(), mc_kapture_dir)
        if os.path.exists(mc_kapture_dir):
            mc_records = parse_kapture_records(mc_kapture_dir)
            print(f"  Multi-cam enabled: cams={mc_cam_ids}  primary={mc_primary}  "
                  f"records={len(mc_records)} timestamps")
        else:
            print(f"  WARNING: multi_cam.enabled=true but kapture_dir not found: {mc_kapture_dir}")

    gt_map = {}
    if gt_poses_path and os.path.exists(gt_poses_path):
        raw = json.load(open(gt_poses_path))
        for fname, val in raw.items():
            arr = np.array(val)
            if arr.shape == (4, 4):
                gt_map[fname] = arr[:3, 3]
            elif arr.ndim == 1 and len(arr) >= 3:
                gt_map[fname] = arr[:3]
        print(f"  GT poses loaded: {len(gt_map)} entries")
    else:
        print("  GT poses: 없음 (추정 경로만 표시)")

    cam_match = re.search(r'cam_\d+', os.path.abspath(test_dir))
    cam_subdir = cam_match.group(0) if cam_match else "cam_unknown"
    test_out = os.path.join(output_dir, "test_results", cam_subdir)
    print(f"  Output dir: {test_out}")
    os.makedirs(test_out, exist_ok=True)

    results = []
    for i, img_path in enumerate(img_paths):
        fname    = os.path.basename(img_path)
        work_dir = os.path.join(test_out, os.path.splitext(fname)[0])
        print(f"\n  [{i+1}/{len(img_paths)}] {fname}")

        # multi-cam: 자매 이미지 탐색
        query_images = None
        if mc_enabled and mc_records is not None:
            sisters = find_sister_images(img_path, mc_records, mc_cam_ids)
            if len(sisters) > 1:
                query_images = sisters
            elif sisters:
                print(f"    multi-cam: only 1 cam found {list(sisters.keys())}, "
                      "falling back to single-cam")

        est_pose = localize_single(img_path, db, config, work_dir,
                                   save_images=save_images,
                                   query_images=query_images)
        est_xyz  = est_pose[:3, 3] if est_pose is not None else None
        gt_xyz   = gt_map.get(fname)
        err      = float(np.linalg.norm(est_xyz - gt_xyz)) \
                   if (est_xyz is not None and gt_xyz is not None) else None

        results.append({
            "fname":    fname,
            "success":  est_pose is not None,
            "est_pose": est_pose,
            "est_xyz":  est_xyz,
            "gt_xyz":   gt_xyz,
            "error_m":  err,
        })
        status  = "OK" if est_pose is not None else "FAIL"
        err_str = f"  err={err:.3f}m" if err is not None else ""
        print(f"    → {status}{err_str}")

    n_ok    = sum(r["success"] for r in results)
    n_total = len(results)
    errors  = [r["error_m"] for r in results if r["error_m"] is not None]
    print(f"\n  Success rate: {n_ok}/{n_total} ({100*n_ok/n_total:.1f}%)")
    if errors:
        print(f"  Error  mean={np.mean(errors):.3f}m  "
              f"median={np.median(errors):.3f}m  "
              f"max={np.max(errors):.3f}m")

    # Plot
    has_gt  = any(r["gt_xyz"]  is not None for r in results)
    has_est = any(r["est_xyz"] is not None for r in results)
    ncols   = 3 if (has_gt and has_est and errors) else 2

    fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 7))

    ax = axes[0]
    db_poses = np.array([e["pose"][:3, 3] for e in db["entries"]])
    ax.scatter(db_poses[:,0], db_poses[:,1], c="lightgray", s=2, alpha=0.4, label="DB viewpoints")

    if has_gt:
        gt_pts = np.array([r["gt_xyz"] for r in results if r["gt_xyz"] is not None])
        ax.plot(gt_pts[:,0], gt_pts[:,1], "b-o", lw=1.5, ms=4, label="GT path", zorder=3)

    if has_est:
        est_ok = [(r["est_xyz"], r["gt_xyz"]) for r in results if r["est_xyz"] is not None]
        ep = np.array([x[0] for x in est_ok])
        ax.plot(ep[:,0], ep[:,1], "r-^", lw=1.5, ms=5, label="Estimated", zorder=4)
        if has_gt:
            for r in results:
                if r["est_xyz"] is not None and r["gt_xyz"] is not None:
                    ax.plot([r["gt_xyz"][0], r["est_xyz"][0]],
                            [r["gt_xyz"][1], r["est_xyz"][1]],
                            "k-", lw=0.5, alpha=0.4)
        fail_gt = [r["gt_xyz"] for r in results if not r["success"] and r["gt_xyz"] is not None]
        if fail_gt:
            fp = np.array(fail_gt)
            ax.scatter(fp[:,0], fp[:,1], c="red", marker="x", s=60, zorder=5, label="Failed")

    ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.set_title(f"Top-down: GT vs Estimated\n({n_ok}/{n_total} success)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    ax2 = axes[1]
    colors_b = []
    vals     = []
    xlabels  = []
    for r in results:
        xlabels.append(os.path.splitext(r["fname"])[0][-12:])
        if not r["success"]:
            vals.append(0); colors_b.append("red")
        elif r["error_m"] is not None:
            vals.append(r["error_m"]); colors_b.append("steelblue")
        else:
            vals.append(0); colors_b.append("orange")

    xpos = np.arange(len(vals))
    ax2.bar(xpos, vals, color=colors_b)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(xlabels, rotation=60, fontsize=6, ha="right")
    ax2.set_ylabel("Position error (m)")
    ax2.set_title("Per-image error\n(red=failed, orange=no GT, blue=error)")
    if errors:
        ax2.axhline(np.mean(errors), color="black", ls="--", lw=1,
                    label=f"mean={np.mean(errors):.3f}m")
        ax2.axhline(np.median(errors), color="gray", ls=":", lw=1,
                    label=f"median={np.median(errors):.3f}m")
        ax2.legend(fontsize=7)

    if ncols == 3:
        ax3 = axes[2]
        errs_sorted = np.sort(errors)
        cdf = np.arange(1, len(errs_sorted)+1) / len(errs_sorted)
        ax3.plot(errs_sorted, cdf*100, "b-", lw=2)
        for thresh in [0.25, 0.5, 1.0, 2.0]:
            pct = 100 * np.mean(np.array(errors) <= thresh)
            ax3.axvline(thresh, color="gray", ls="--", lw=0.8, alpha=0.7)
            ax3.text(thresh, pct+2, f"{pct:.0f}%\n@{thresh}m", fontsize=7, ha="center")
        ax3.set_xlabel("Error threshold (m)"); ax3.set_ylabel("Recall (%)")
        ax3.set_title("Error CDF"); ax3.set_ylim(0, 105); ax3.grid(True, alpha=0.3)

    title = (f"Batch Test — {n_ok}/{n_total} success"
             + (f" | mean err={np.mean(errors):.3f}m" if errors else ""))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    out_png = os.path.join(test_out, "test_trajectory.png")
    fig.savefig(out_png, dpi=150); plt.close()
    print(f"\n  Saved: {out_png}")

    pickle.dump(results, open(os.path.join(test_out, "batch_results.pkl"), "wb"))
    print(f"  Saved: {os.path.join(test_out, 'batch_results.pkl')}")

    # Trajectory 파일 저장
    tum_lines = []
    csv_rows  = [["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "success"]]
    traj_json = {}

    for r in results:
        stem = os.path.splitext(r["fname"])[0]
        try:
            ts = float(stem) / 1e6
        except ValueError:
            ts = results.index(r)

        if r["est_pose"] is not None:
            T = r["est_pose"]
            tx, ty, tz = T[:3, 3]
            quat = Rotation.from_matrix(T[:3, :3]).as_quat()
            qx, qy, qz, qw = quat
            tum_lines.append(f"{ts:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                             f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")
            csv_rows.append([f"{ts:.6f}", f"{tx:.6f}", f"{ty:.6f}", f"{tz:.6f}",
                             f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}", f"{qw:.6f}", "1"])
            traj_json[r["fname"]] = T.tolist()
        else:
            csv_rows.append([f"{ts:.6f}", "", "", "", "", "", "", "", "0"])

    tum_path = os.path.join(test_out, "trajectory_tum.txt")
    with open(tum_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        f.write("\n".join(tum_lines))
    print(f"  Saved: {tum_path}  ({len(tum_lines)} poses, TUM format)")

    csv_path = os.path.join(test_out, "trajectory.csv")
    with open(csv_path, "w") as f:
        for row in csv_rows:
            f.write(",".join(row) + "\n")
    print(f"  Saved: {csv_path}")

    json_path = os.path.join(test_out, "trajectory_poses.json")
    with open(json_path, "w") as f:
        json.dump(traj_json, f, indent=2)
    print(f"  Saved: {json_path}  (4x4 matrices)")

    return results
