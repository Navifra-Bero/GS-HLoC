#!/usr/bin/env python3
"""
Evaluate predicted trajectory against COLMAP images.bin poses.

COLMAP images.bin stores world-to-camera poses. This script converts them to
camera-to-world, applies step0 T_align, then compares against the localization
trajectory saved by scripts/pipeline/batch_test.py.

Usage:
  python3 scripts/eval_trajectory_colmap.py
  python3 scripts/eval_trajectory_colmap.py --cam_id cam_3 --outlier_thresh 3.0
"""

import argparse
import csv
import json
import os
import pickle
import re
import struct
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation


STEP0_PKL = "output/sgs_splat_proxy/step0_data.pkl"
COLMAP_IMAGES_BIN = "nerfstudio/scene0/sparse/0/images.bin"
TRANSFORMS_JSON = "nerfstudio/images_metric/transforms.json"
RIGS_TXT = "kapture_data/kapture/sensors/rigs.txt"
PRED_JSON = "output/sgs_splat_proxy/test_results/cam_3/trajectory_poses.json"
PRED_TUM = "output/sgs_splat_proxy/test_results/cam_3/trajectory_tum.txt"
OUTPUT_DIR = "output/sgs_splat_proxy/test_results/cam_3"
MAP_PLY = "output/sgs_splat_proxy/aligned_map.ply"
CAM_ID = "cam_3"
OUTLIER_THRESH = 5.0


def load_T_align(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    T = np.asarray(data["T_align"], dtype=np.float64)
    print(f"  T_align loaded: {pkl_path}")
    print(f"  R_align:\n{T[:3, :3]}")
    print(f"  t_align: {T[:3, 3]}")
    return T


def load_applied_transform(json_path, disabled=False):
    if disabled or not json_path:
        print("  COLMAP applied_transform: disabled (identity)")
        return np.eye(4)
    if not os.path.exists(json_path):
        print(f"  COLMAP applied_transform: not found ({json_path}); using identity")
        return np.eye(4)

    with open(json_path) as f:
        data = json.load(f)
    arr = data.get("applied_transform")
    if arr is None:
        print(f"  COLMAP applied_transform: missing in {json_path}; using identity")
        return np.eye(4)

    arr = np.asarray(arr, dtype=np.float64)
    T = np.eye(4)
    if arr.shape == (3, 4):
        T[:3, :4] = arr
    elif arr.shape == (4, 4):
        T = arr
    else:
        raise ValueError(f"Unexpected applied_transform shape: {arr.shape}")
    print(f"  COLMAP applied_transform loaded: {json_path}")
    print(f"  applied_transform:\n{T[:3, :4]}")
    return T


def load_rig_extrinsic(rigs_path, cam_id):
    if not rigs_path or not os.path.exists(rigs_path):
        return None
    with open(rigs_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9 or parts[1] != cam_id:
                continue
            qw, qx, qy, qz = map(float, parts[2:6])
            tx, ty, tz = map(float, parts[6:9])
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            print(f"  Rig extrinsic loaded: {rigs_path} ({cam_id})")
            print(f"  T_rig_to_{cam_id} t: {T[:3, 3]}")
            return T
    return None


def qvec_wxyz_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def read_c_string(fid):
    name = bytearray()
    while True:
        ch = fid.read(1)
        if ch == b"":
            raise EOFError("Unexpected EOF while reading COLMAP image name")
        if ch == b"\x00":
            return name.decode("utf-8")
        name.extend(ch)


def timestamp_key_from_name(name):
    stem = Path(name).stem
    match = re.search(r"(\d{12,})", stem)
    return match.group(1) if match else stem


def load_colmap_images(images_bin, cam_id=None):
    """
    Read COLMAP images.bin.

    Returns:
      dict timestamp_key -> {
        "name": image name,
        "T_c2w": 4x4 camera-to-world in COLMAP world frame,
      }
    """
    poses = {}
    total = 0
    used = 0

    with open(images_bin, "rb") as fid:
        num_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_images):
            total += 1
            image_id = struct.unpack("<i", fid.read(4))[0]
            qvec = struct.unpack("<dddd", fid.read(32))
            tvec = np.array(struct.unpack("<ddd", fid.read(24)), dtype=np.float64)
            camera_id = struct.unpack("<i", fid.read(4))[0]
            name = read_c_string(fid)
            num_points2d = struct.unpack("<Q", fid.read(8))[0]
            fid.seek(num_points2d * 24, os.SEEK_CUR)

            if cam_id and cam_id not in Path(name).parts and cam_id not in name:
                continue

            R_w2c = qvec_wxyz_to_rotmat(qvec)
            T_c2w = np.eye(4)
            T_c2w[:3, :3] = R_w2c.T
            T_c2w[:3, 3] = -R_w2c.T @ tvec

            key = timestamp_key_from_name(name)
            poses[key] = {
                "image_id": image_id,
                "camera_id": camera_id,
                "name": name,
                "T_c2w": T_c2w,
            }
            used += 1

    if cam_id and used == 0:
        raise RuntimeError(f"No COLMAP images matched cam_id={cam_id!r} in {images_bin}")

    print(f"  COLMAP images loaded: {used}/{total}"
          + (f" for {cam_id}" if cam_id else ""))
    return poses


def load_pred_json(json_path):
    with open(json_path) as f:
        raw = json.load(f)
    poses = {}
    for name, mat in raw.items():
        arr = np.asarray(mat, dtype=np.float64)
        if arr.shape != (4, 4):
            continue
        poses[timestamp_key_from_name(name)] = {
            "name": name,
            "T_c2w": arr,
        }
    print(f"  Pred JSON poses loaded: {len(poses)} from {json_path}")
    return poses


def load_pred_tum(tum_path):
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
            tx, ty, tz = map(float, vals[1:4])
            qx, qy, qz, qw = map(float, vals[4:8])
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            key = str(int(round(ts_sec * 1e6)))
            poses[key] = {
                "name": key,
                "T_c2w": T,
            }
    print(f"  Pred TUM poses loaded: {len(poses)} from {tum_path}")
    return poses


def load_pred_poses(pred_json, pred_tum, prefer_json=True):
    if prefer_json and pred_json and os.path.exists(pred_json):
        return load_pred_json(pred_json), pred_json
    if pred_tum and os.path.exists(pred_tum):
        return load_pred_tum(pred_tum), pred_tum
    if pred_json and os.path.exists(pred_json):
        return load_pred_json(pred_json), pred_json
    raise FileNotFoundError(f"No prediction file found: {pred_json} or {pred_tum}")


def rotation_error_deg(R_gt, R_pred):
    dR = R_gt.T @ R_pred
    cos_angle = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def load_map_xy(map_ply, max_points=100000):
    if not map_ply or not os.path.exists(map_ply):
        return None
    try:
        vertex = PlyData.read(map_ply)["vertex"].data
        xy = np.column_stack([
            np.asarray(vertex["x"], dtype=np.float64),
            np.asarray(vertex["y"], dtype=np.float64),
        ])
        if len(xy) > max_points:
            rng = np.random.default_rng(7)
            xy = xy[rng.choice(len(xy), size=max_points, replace=False)]
        print(f"  Map background loaded: {len(xy)} sampled points from {map_ply}")
        return xy
    except Exception as e:
        print(f"  [warn] failed to load map background {map_ply}: {e}")
        return None


def write_matches_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "key", "gt_name", "pred_name",
            "gt_x", "gt_y", "gt_z",
            "pred_x", "pred_y", "pred_z",
            "trans_err_m", "rot_err_deg",
        ])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step0_pkl", default=STEP0_PKL)
    parser.add_argument("--colmap_images", default=COLMAP_IMAGES_BIN,
                        help="COLMAP sparse/0/images.bin")
    parser.add_argument("--applied_transform_json", default=TRANSFORMS_JSON,
                        help="JSON containing nerfstudio/COLMAP applied_transform. "
                             "Use --disable_applied_transform for identity.")
    parser.add_argument("--disable_applied_transform", action="store_true",
                        help="Do not apply the nerfstudio/COLMAP applied_transform.")
    parser.add_argument("--pred_json", default=PRED_JSON,
                        help="trajectory_poses.json. Used first by default.")
    parser.add_argument("--pred_tum", default=PRED_TUM,
                        help="trajectory_tum.txt fallback")
    parser.add_argument("--prefer_tum", action="store_true",
                        help="Use TUM first instead of trajectory_poses.json")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--cam_id", default=CAM_ID,
                        help="Filter COLMAP image names by cam id, e.g. cam_3. Empty string = all.")
    parser.add_argument("--rigs_txt", default=RIGS_TXT,
                        help="kapture rigs.txt. GT is converted cam->rig by default.")
    parser.add_argument("--gt_frame", default="rig", choices=["camera", "rig"],
                        help="Frame to compare GT in. step7 predictions are rig by default.")
    parser.add_argument("--map_ply", default=MAP_PLY,
                        help="Aligned map PLY to draw as XY background. Empty string disables it.")
    parser.add_argument("--map_max_points", type=int, default=100000)
    parser.add_argument("--outlier_thresh", type=float, default=OUTLIER_THRESH)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cam_id = args.cam_id or None
    map_ply = args.map_ply or None

    print("\n[1] Loading data...")
    T_align = load_T_align(args.step0_pkl)
    T_applied = load_applied_transform(
        args.applied_transform_json,
        disabled=args.disable_applied_transform,
    )
    T_rig_to_cam = None
    if args.gt_frame == "rig":
        if not cam_id:
            raise ValueError("--gt_frame rig requires --cam_id")
        T_rig_to_cam = load_rig_extrinsic(args.rigs_txt, cam_id)
        if T_rig_to_cam is None:
            raise FileNotFoundError(
                f"Could not find {cam_id} in rigs file: {args.rigs_txt}")
    gt_colmap = load_colmap_images(args.colmap_images, cam_id=cam_id)
    pred, pred_source = load_pred_poses(
        args.pred_json,
        args.pred_tum,
        prefer_json=not args.prefer_tum,
    )
    map_xy = load_map_xy(map_ply, args.map_max_points)

    print("\n[2] Transforming COLMAP GT to eval frame...")
    gt_aligned = {}
    for key, item in gt_colmap.items():
        T_gt = T_align @ T_applied @ item["T_c2w"]
        if args.gt_frame == "rig":
            # Keep this consistent with step7_pnp.py:
            # c2w_rig = c2w_cam @ T_rig_to_cam
            T_gt = T_gt @ T_rig_to_cam
        gt_aligned[key] = {
            "name": item["name"],
            "T_c2w": T_gt,
        }

    common_keys = sorted(set(gt_aligned.keys()) & set(pred.keys()))
    print(f"\n[3] Matched poses: {len(common_keys)}")
    print(f"  GT keys: {len(gt_aligned)}")
    print(f"  Pred keys: {len(pred)}")
    if not common_keys:
        raise RuntimeError("No matching timestamps / image stems between COLMAP GT and predictions")

    gt_xyz = []
    pred_xyz = []
    t_errs = []
    r_errs = []
    match_rows = []

    for key in common_keys:
        T_gt = gt_aligned[key]["T_c2w"]
        T_pred = pred[key]["T_c2w"]
        p_gt = T_gt[:3, 3]
        p_pred = T_pred[:3, 3]
        t_err = float(np.linalg.norm(p_gt - p_pred))
        r_err = rotation_error_deg(T_gt[:3, :3], T_pred[:3, :3])

        gt_xyz.append(p_gt)
        pred_xyz.append(p_pred)
        t_errs.append(t_err)
        r_errs.append(r_err)
        match_rows.append([
            key, gt_aligned[key]["name"], pred[key]["name"],
            *p_gt.tolist(), *p_pred.tolist(), t_err, r_err,
        ])

    gt_xyz = np.asarray(gt_xyz)
    pred_xyz = np.asarray(pred_xyz)
    t_errs = np.asarray(t_errs)
    r_errs = np.asarray(r_errs)

    thresh = args.outlier_thresh
    inlier_mask = t_errs <= thresh
    outlier_mask = ~inlier_mask
    n_total = len(t_errs)
    n_in = int(inlier_mask.sum())
    n_out = int(outlier_mask.sum())
    t_in = t_errs[inlier_mask]
    r_in = r_errs[inlier_mask]

    print("\n[4] Results...")
    lines = [
        "=" * 62,
        f"  COLMAP images      : {args.colmap_images}",
        f"  Applied transform  : {'disabled' if args.disable_applied_transform else args.applied_transform_json}",
        f"  Prediction source  : {pred_source}",
        f"  Step0 align        : {args.step0_pkl}",
        f"  Camera filter      : {cam_id or 'all'}",
        f"  Rig file           : {args.rigs_txt if args.gt_frame == 'rig' else 'unused'}",
        f"  Eval frame         : {args.gt_frame} + aligned map frame",
        f"  Outlier threshold  : {thresh:.3f} m",
        f"  Total matched      : {n_total}",
        f"  Inliers            : {n_in}",
        f"  Outliers           : {n_out}",
        "=" * 62,
    ]

    if n_in > 0:
        lines += [
            f"\n  Translation Error (N={n_in})",
            f"    Mean   : {np.mean(t_in):.4f} m",
            f"    Median : {np.median(t_in):.4f} m",
            f"    RMSE   : {np.sqrt(np.mean(t_in ** 2)):.4f} m",
            f"    Std    : {np.std(t_in):.4f} m",
            f"    Min    : {np.min(t_in):.4f} m",
            f"    Max    : {np.max(t_in):.4f} m",
            f"    @0.10m : {100 * np.mean(t_in <= 0.10):.1f}%",
            f"    @0.25m : {100 * np.mean(t_in <= 0.25):.1f}%",
            f"    @0.50m : {100 * np.mean(t_in <= 0.50):.1f}%",
            f"    @1.00m : {100 * np.mean(t_in <= 1.00):.1f}%",
            f"    @2.00m : {100 * np.mean(t_in <= 2.00):.1f}%",
            f"\n  Rotation Error (N={n_in})",
            f"    Mean   : {np.mean(r_in):.2f} deg",
            f"    Median : {np.median(r_in):.2f} deg",
            f"    RMSE   : {np.sqrt(np.mean(r_in ** 2)):.2f} deg",
            f"    Std    : {np.std(r_in):.2f} deg",
        ]
    lines.append("=" * 62)

    report = "\n".join(lines)
    print(report)

    out_txt = os.path.join(args.output_dir, "eval_colmap_results.txt")
    with open(out_txt, "w") as f:
        f.write(report + "\n")
    print(f"  Saved: {out_txt}")

    out_csv = os.path.join(args.output_dir, "eval_colmap_matches.csv")
    write_matches_csv(out_csv, match_rows)
    print(f"  Saved: {out_csv}")

    all_gt_keys = sorted(gt_aligned.keys())
    all_gt_xyz = np.asarray([gt_aligned[k]["T_c2w"][:3, 3] for k in all_gt_keys])

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    ax = axes[0]
    if map_xy is not None:
        ax.scatter(map_xy[:, 0], map_xy[:, 1], c="lightgray", s=0.2,
                   alpha=0.25, label="aligned map")
    ax.plot(all_gt_xyz[:, 0], all_gt_xyz[:, 1],
            color="lightblue", lw=1.4, alpha=0.7, label="COLMAP GT full")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1],
            "b-o", lw=1.8, ms=3, label=f"COLMAP GT matched (N={n_total})")
    ax.plot(pred_xyz[inlier_mask, 0], pred_xyz[inlier_mask, 1],
            "r-^", lw=1.7, ms=4, label=f"Pred inlier (N={n_in})")
    if outlier_mask.any():
        ax.scatter(pred_xyz[outlier_mask, 0], pred_xyz[outlier_mask, 1],
                   c="orange", marker="x", s=90, zorder=6,
                   label=f"Pred outlier (N={n_out})")
    for i in range(n_total):
        if inlier_mask[i]:
            ax.plot([gt_xyz[i, 0], pred_xyz[i, 0]],
                    [gt_xyz[i, 1], pred_xyz[i, 1]],
                    "k-", lw=0.25, alpha=0.25)
    ax.set_aspect("equal")
    ax.set_title("COLMAP GT vs Predicted (XY)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    ax2 = axes[1]
    colors = ["steelblue" if ok else "orange" for ok in inlier_mask]
    ax2.bar(np.arange(n_total), t_errs, color=colors, width=1.0, edgecolor="none")
    ax2.axhline(thresh, color="red", ls="--", lw=1.4,
                label=f"outlier thresh={thresh:.1f}m")
    if n_in > 0:
        ax2.axhline(np.mean(t_in), color="black", ls="--", lw=1.3,
                    label=f"mean={np.mean(t_in):.3f}m")
        ax2.axhline(np.median(t_in), color="gray", ls=":", lw=1.5,
                    label=f"median={np.median(t_in):.3f}m")
    ax2.set_xlabel("Matched frame index")
    ax2.set_ylabel("Translation error (m)")
    ax2.set_title("Per-frame Translation Error")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = axes[2]
    if n_in > 0:
        es = np.sort(t_in)
        cdf = np.arange(1, len(es) + 1) / len(es)
        ax3.plot(es, cdf * 100, "b-", lw=2, label=f"CDF (N={n_in})")
        for thr, color in [(0.1, "green"), (0.25, "orange"), (0.5, "red"),
                           (1.0, "purple"), (2.0, "brown")]:
            pct = 100.0 * np.mean(t_in <= thr)
            ax3.axvline(thr, color=color, ls="--", lw=1, alpha=0.8)
            ax3.text(thr, min(103, pct + 2), f"{pct:.0f}%\n@{thr:g}m",
                     fontsize=8, ha="center", color=color)
        ax3.legend(fontsize=8)
    ax3.set_xlabel("Error threshold (m)")
    ax3.set_ylabel("Recall (%)")
    ax3.set_title("Translation Error CDF (inliers)")
    ax3.set_ylim(0, 108)
    ax3.grid(True, alpha=0.3)

    if n_in > 0:
        title = (f"COLMAP Trajectory Evaluation [{cam_id or 'all'}]  "
                 f"matched={n_total}, inliers={n_in}, outliers={n_out}\n"
                 f"Trans mean={np.mean(t_in):.3f}m median={np.median(t_in):.3f}m "
                 f"RMSE={np.sqrt(np.mean(t_in ** 2)):.3f}m | "
                 f"Rot mean={np.mean(r_in):.1f}deg median={np.median(r_in):.1f}deg")
    else:
        title = f"COLMAP Trajectory Evaluation [{cam_id or 'all'}]"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    out_png = os.path.join(args.output_dir, "eval_trajectory_colmap.png")
    fig.savefig(out_png, dpi=150)
    plt.close()
    print(f"  Saved: {out_png}")


if __name__ == "__main__":
    main()
