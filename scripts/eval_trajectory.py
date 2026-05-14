#!/usr/bin/env python3
"""
Trajectory evaluation: predicted (aligned PLY frame) vs GT (kapture frame).

step0_align에서 PLY를 T_align으로 변환했으므로,
GT(kapture world frame)를 T_align으로 변환한 뒤 비교한다.

Multi-cam type2의 step7 결과는 rig frame pose로 저장된다. 이 경우 GT camera
pose도 rigs.txt의 T_rig_to_cam을 사용해 rig pose로 변환한 뒤 비교한다.

Usage:
  python3 scripts/eval_trajectory.py
  python3 scripts/eval_trajectory.py --outlier_thresh 3.0 --cam_id cam_3
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


# ── Default paths ──────────────────────────────────────────────────────────
STEP0_PKL      = "output/colmap_sgs_test/step0_data.pkl"
GT_TRAJ_TXT    = "kapture_1_3/sensors/trajectories.txt"
RIGS_TXT       = "kapture_1_3/sensors/rigs.txt"
PRED_TUM_TXT   = "output/colmap_sgs_test/test_results/cam_3/trajectory_tum.txt"
OUTPUT_DIR     = "output/colmap_sgs_test/test_results/cam_3"
CAM_ID         = "cam_3"   # GT를 읽을 카메라 ID
OUTLIER_THRESH = 5.0   # m — 이 이상의 오차는 outlier로 분류


# ── Loaders ────────────────────────────────────────────────────────────────

def load_T_align(pkl_path: str) -> np.ndarray:
    """step0_data.pkl에서 T_align (4x4) 로드."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    T = data["T_align"]
    print(f"  T_align loaded from {pkl_path}")
    print(f"  R_total:\n{T[:3,:3]}")
    print(f"  t_align: {T[:3,3]}")
    return T


def load_rig_extrinsic(rigs_path: str, cam_id: str) -> np.ndarray | None:
    """rigs.txt에서 해당 cam의 T_rig_to_cam (4×4) 로드."""
    if not os.path.exists(rigs_path):
        return None
    with open(rigs_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9 or parts[1] != cam_id:
                continue
            from scipy.spatial.transform import Rotation as _Rot
            qw, qx, qy, qz = map(float, parts[2:6])
            tx, ty, tz      = map(float, parts[6:9])
            R = _Rot.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3]  = [tx, ty, tz]
            return T   # T_rig_to_cam
    return None


def to_rig_pose(T_cam: np.ndarray, T_rig_to_cam: np.ndarray) -> np.ndarray:
    """camera c2w pose를 rig c2w pose로 변환.

    kapture rigs.txt는 현재 코드베이스에서 T_rig_to_cam으로 사용한다.
    따라서 T_W_rig = T_W_cam @ T_rig_to_cam.
    """
    return T_cam @ T_rig_to_cam


def rig_to_cam_pose(T_rig: np.ndarray, T_rig_to_cam: np.ndarray) -> np.ndarray:
    """rig c2w pose를 camera c2w pose로 변환."""
    return T_rig @ np.linalg.inv(T_rig_to_cam)


def load_gt_trajectory(traj_path: str, cam_id: str) -> dict:
    """
    kapture trajectories.txt 파싱.
    format: # timestamp, device_id, qw, qx, qy, qz, tx, ty, tz
    camera-to-world pose.
    Returns: {timestamp_us (int) -> 4x4 np.ndarray (camera-to-world)}
    """
    poses = {}
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue
            if parts[1] != cam_id:
                continue
            ts_us = int(parts[0])
            qw, qx, qy, qz = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            tx, ty, tz      = float(parts[6]), float(parts[7]), float(parts[8])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # scipy: [x,y,z,w]
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3]  = [tx, ty, tz]
            poses[ts_us] = T
    return poses


def load_pred_trajectory(tum_path: str) -> dict:
    """
    TUM format: # timestamp tx ty tz qx qy qz qw
    camera-to-world pose (aligned PLY frame).
    Returns: {timestamp_us (int) -> 4x4 np.ndarray}
    """
    poses = {}
    with open(tum_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            if len(vals) < 8:
                continue
            ts_sec = float(vals[0])
            tx, ty, tz      = float(vals[1]), float(vals[2]), float(vals[3])
            qx, qy, qz, qw  = float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3]  = [tx, ty, tz]
            ts_us = int(round(ts_sec * 1e6))
            poses[ts_us] = T
    return poses


# ── Metrics ────────────────────────────────────────────────────────────────

def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """두 회전행렬 간 각도 오차 (degrees)."""
    dR    = R1.T @ R2
    trace = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step0_pkl",      default=STEP0_PKL)
    parser.add_argument("--gt_traj",        default=GT_TRAJ_TXT)
    parser.add_argument("--rigs_txt",       default=RIGS_TXT,
                        help="kapture rigs.txt 경로. 있으면 GT와 pred 모두 rig frame으로 비교.")
    parser.add_argument("--pred_tum",       default=PRED_TUM_TXT)
    parser.add_argument("--output_dir",     default=OUTPUT_DIR)
    parser.add_argument("--cam_id",         default=CAM_ID,
                        help="GT trajectories.txt에서 읽을 카메라 ID")
    parser.add_argument("--gt_frame",       default="rig",
                        choices=["cam", "rig"],
                        help="GT를 비교할 frame. type2 step7 평가는 rig 권장.")
    parser.add_argument("--pred_frame",     default="rig",
                        choices=["cam", "rig"],
                        help="pred_tum pose frame. type2 step7 출력은 rig.")
    parser.add_argument("--pred_cam_id",    default=None,
                        help="pred_frame=cam일 때 pred를 rig로 올릴 카메라 ID. 기본값은 cam_id.")
    parser.add_argument("--outlier_thresh", type=float, default=OUTLIER_THRESH,
                        help="이 거리(m) 이상의 오차는 outlier로 제외")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    T_align       = load_T_align(args.step0_pkl)
    gt_raw        = load_gt_trajectory(args.gt_traj, args.cam_id)
    pred_poses    = load_pred_trajectory(args.pred_tum)
    print(f"  GT poses ({args.cam_id}): {len(gt_raw)}")
    print(f"  Pred poses: {len(pred_poses)}")

    # ── Rig extrinsic 로드 ─────────────────────────────────────────────────
    pred_cam_id = args.pred_cam_id or args.cam_id
    T_gt_rig_to_cam = load_rig_extrinsic(args.rigs_txt, args.cam_id)
    T_pred_rig_to_cam = load_rig_extrinsic(args.rigs_txt, pred_cam_id)

    if args.gt_frame == "rig":
        if T_gt_rig_to_cam is None:
            raise FileNotFoundError(
                f"GT frame=rig requires {args.cam_id} in rigs.txt: {args.rigs_txt}")
        print(f"  GT rig extrinsic: T_rig_to_{args.cam_id} from {args.rigs_txt}")
        print(f"    t_rig_to_cam: {T_gt_rig_to_cam[:3,3].round(4)}")
    else:
        print(f"  GT frame: camera ({args.cam_id}); no rig conversion")

    if args.pred_frame == "cam":
        if T_pred_rig_to_cam is None:
            raise FileNotFoundError(
                f"pred_frame=cam requires {pred_cam_id} in rigs.txt: {args.rigs_txt}")
        print(f"  Pred frame: camera ({pred_cam_id}) → converted to rig for comparison")
    else:
        print("  Pred frame: rig (type2 step7 default)")

    # ── Transform GT to aligned frame (+ rig frame if requested) ──────────
    # T_align transforms points: p_aligned = T_align @ p_world
    # camera-to-aligned = T_align @ camera-to-world
    eval_frame = "rig" if args.gt_frame == "rig" else args.cam_id
    if args.gt_frame != args.pred_frame:
        print(f"  Frame normalization: pred {args.pred_frame} → GT eval frame {eval_frame}")
    print(f"\n[2] Transforming trajectories to {eval_frame} + aligned frame...")
    gt_aligned = {}
    for ts, T_cam in gt_raw.items():
        T_cam_aligned = T_align @ T_cam          # cam c2w in aligned world
        if args.gt_frame == "rig":
            gt_aligned[ts] = to_rig_pose(T_cam_aligned, T_gt_rig_to_cam)
        else:
            gt_aligned[ts] = T_cam_aligned

    if args.pred_frame == args.gt_frame:
        pred_norm = pred_poses
    elif args.pred_frame == "cam" and args.gt_frame == "rig":
        pred_norm = {
            ts: to_rig_pose(T_pred, T_pred_rig_to_cam)
            for ts, T_pred in pred_poses.items()
        }
    elif args.pred_frame == "rig" and args.gt_frame == "cam":
        if T_pred_rig_to_cam is None:
            raise FileNotFoundError(
                f"pred_frame=rig → gt_frame=cam requires {pred_cam_id} in rigs.txt")
        pred_norm = {
            ts: rig_to_cam_pose(T_pred, T_pred_rig_to_cam)
            for ts, T_pred in pred_poses.items()
        }

    # ── Match timestamps ──────────────────────────────────────────────────
    common_ts = sorted(set(gt_aligned.keys()) & set(pred_norm.keys()))
    print(f"\n[3] Matched timestamps: {len(common_ts)}")
    if len(common_ts) == 0:
        print("  ERROR: No matching timestamps. 타임스탬프 단위를 확인하세요.")
        return

    # ── Compute errors ────────────────────────────────────────────────────
    print("\n[4] Computing errors...")
    gt_xyz_list   = []
    pred_xyz_list = []
    trans_errors  = []
    rot_errors    = []

    for ts in common_ts:
        T_gt   = gt_aligned[ts]
        T_pred = pred_norm[ts]

        p_gt   = T_gt[:3, 3]
        p_pred = T_pred[:3, 3]
        t_err  = float(np.linalg.norm(p_gt - p_pred))
        r_err  = rotation_error_deg(T_gt[:3, :3], T_pred[:3, :3])

        gt_xyz_list.append(p_gt)
        pred_xyz_list.append(p_pred)
        trans_errors.append(t_err)
        rot_errors.append(r_err)

    gt_xyz   = np.array(gt_xyz_list)    # (N, 3)
    pred_xyz = np.array(pred_xyz_list)  # (N, 3)
    t_errs   = np.array(trans_errors)   # (N,)
    r_errs   = np.array(rot_errors)     # (N,)

    # ── Outlier separation ────────────────────────────────────────────────
    thresh        = args.outlier_thresh
    inlier_mask   = t_errs <= thresh
    outlier_mask  = ~inlier_mask
    n_total       = len(t_errs)
    n_inliers     = int(inlier_mask.sum())
    n_outliers    = int(outlier_mask.sum())

    t_in = t_errs[inlier_mask]
    r_in = r_errs[inlier_mask]

    # ── Print results & save txt ──────────────────────────────────────────
    lines = []
    lines.append(f"{'='*55}")
    lines.append(f"  GT trajectory      : {args.gt_traj}")
    lines.append(f"  Pred trajectory    : {args.pred_tum}")
    lines.append(f"  GT cam/frame       : {args.cam_id} / {args.gt_frame}")
    lines.append(f"  Pred frame         : {args.pred_frame}"
                 + (f" ({pred_cam_id})" if args.pred_frame == "cam" else ""))
    lines.append(f"  Eval frame         : {eval_frame} + aligned map")
    lines.append(f"  Outlier threshold  : {thresh} m")
    lines.append(f"  Total matched      : {n_total}")
    lines.append(f"  Inliers            : {n_inliers}")
    lines.append(f"  Outliers (excluded): {n_outliers}")
    lines.append(f"{'='*55}")

    if n_inliers > 0:
        lines.append(f"\n  Translation Error (N={n_inliers})")
        lines.append(f"    Mean   : {np.mean(t_in):.4f} m")
        lines.append(f"    Median : {np.median(t_in):.4f} m")
        lines.append(f"    RMSE   : {np.sqrt(np.mean(t_in**2)):.4f} m")
        lines.append(f"    Std    : {np.std(t_in):.4f} m")
        lines.append(f"    Min    : {np.min(t_in):.4f} m")
        lines.append(f"    Max    : {np.max(t_in):.4f} m")
        lines.append(f"    @0.10m : {100*np.mean(t_in<=0.10):.1f}%")
        lines.append(f"    @0.25m : {100*np.mean(t_in<=0.25):.1f}%")
        lines.append(f"    @0.50m : {100*np.mean(t_in<=0.50):.1f}%")
        lines.append(f"    @1.00m : {100*np.mean(t_in<=1.00):.1f}%")
        lines.append(f"    @2.00m : {100*np.mean(t_in<=2.00):.1f}%")

        lines.append(f"\n  Rotation Error (N={n_inliers})")
        lines.append(f"    Mean   : {np.mean(r_in):.2f} deg")
        lines.append(f"    Median : {np.median(r_in):.2f} deg")
        lines.append(f"    RMSE   : {np.sqrt(np.mean(r_in**2)):.2f} deg")
        lines.append(f"    Std    : {np.std(r_in):.2f} deg")
    lines.append(f"{'='*55}")

    report = "\n".join(lines)
    print(report)

    out_txt = os.path.join(args.output_dir, "eval_results.txt")
    with open(out_txt, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved: {out_txt}")

    # ── Full GT trajectory for reference (all timestamps) ─────────────────
    all_gt_ts  = sorted(gt_aligned.keys())
    all_gt_xyz = np.array([gt_aligned[t][:3, 3] for t in all_gt_ts])

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # --- Plot 1: Top-down XY trajectory ---
    ax = axes[0]
    ax.plot(all_gt_xyz[:, 0], all_gt_xyz[:, 1],
            color="lightblue", lw=1.5, alpha=0.6, label="GT full")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1],
            "b-o", lw=2, ms=3, label=f"GT (matched, N={n_total})")
    ax.plot(pred_xyz[inlier_mask, 0], pred_xyz[inlier_mask, 1],
            "r-^", lw=2, ms=4, label=f"Predicted (inlier, N={n_inliers})")
    if outlier_mask.any():
        ax.scatter(pred_xyz[outlier_mask, 0], pred_xyz[outlier_mask, 1],
                   c="orange", marker="x", s=100, zorder=6,
                   label=f"Outlier (>{thresh}m, N={n_outliers})")
    # Error lines between matched pairs (inliers only)
    for i in range(n_total):
        if inlier_mask[i]:
            ax.plot([gt_xyz[i, 0], pred_xyz[i, 0]],
                    [gt_xyz[i, 1], pred_xyz[i, 1]],
                    "k-", lw=0.3, alpha=0.25)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="best")
    ax.set_title("Trajectory: GT vs Predicted (top-down XY)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Per-frame translation error ---
    ax2 = axes[1]
    colors = ["steelblue" if m else "orange" for m in inlier_mask]
    ax2.bar(np.arange(n_total), t_errs, color=colors, width=1.0, edgecolor="none")
    ax2.axhline(thresh, color="red", ls="--", lw=1.5,
                label=f"Outlier thresh ({thresh}m)")
    if n_inliers > 0:
        ax2.axhline(np.mean(t_in), color="black", ls="--", lw=1.5,
                    label=f"Mean={np.mean(t_in):.3f}m")
        ax2.axhline(np.median(t_in), color="gray", ls=":", lw=1.5,
                    label=f"Median={np.median(t_in):.3f}m")
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("Translation error (m)")
    ax2.set_title("Per-frame translation error\n(blue=inlier, orange=outlier)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Error CDF (inliers only) ---
    ax3 = axes[2]
    if n_inliers > 0:
        es = np.sort(t_in)
        cdf = np.arange(1, len(es) + 1) / len(es)
        ax3.plot(es, cdf * 100, "b-", lw=2, label=f"CDF (N={n_inliers})")
        thresh_colors = [(0.25, "green"), (0.5, "orange"), (1.0, "red"), (2.0, "purple")]
        for thr, col in thresh_colors:
            pct = 100.0 * np.mean(t_in <= thr)
            ax3.axvline(thr, color=col, ls="--", lw=1, alpha=0.8)
            ax3.text(thr, pct + 2, f"{pct:.0f}%\n@{thr}m",
                     fontsize=8, ha="center", color=col)
        ax3.legend(fontsize=9)
    ax3.set_xlabel("Error threshold (m)")
    ax3.set_ylabel("Recall (%)")
    ax3.set_title("Error CDF (inliers only)")
    ax3.set_ylim(0, 108)
    ax3.grid(True, alpha=0.3)

    if n_inliers > 0:
        suptitle = (f"{args.cam_id} Trajectory Evaluation [{eval_frame} frame]  "
                    f"(matched={n_total}, inliers={n_inliers}, outliers={n_outliers})\n"
                    f"Trans — mean={np.mean(t_in):.3f}m  "
                    f"median={np.median(t_in):.3f}m  "
                    f"RMSE={np.sqrt(np.mean(t_in**2)):.3f}m    "
                    f"Rot — mean={np.mean(r_in):.1f}°  "
                    f"median={np.median(r_in):.1f}°")
    else:
        suptitle = f"{args.cam_id} Trajectory Evaluation [{eval_frame} frame]"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()

    out_png = os.path.join(args.output_dir, "eval_trajectory.png")
    fig.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
