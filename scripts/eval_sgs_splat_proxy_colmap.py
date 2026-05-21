#!/usr/bin/env python3
"""
Evaluate output/sgs_splat_proxy test trajectory against COLMAP poses.

This is a focused variant of eval_trajectory_colmap.py for the current
SplatHLoc + Scaffold-GS pipeline:
  - predictions: output/sgs_splat_proxy/test_results/<cam>/trajectory_*.*
  - GT: COLMAP images.bin in the same source map frame as step0 input
  - output: translation/rotation report, CSV, and trajectory figure

COLMAP stores world-to-camera poses. This script converts them to c2w, applies
step0 T_align, optionally applies an extra COLMAP/nerfstudio transform, and then
normalizes camera/rig frames to compare with step7 predictions.
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
from scipy.spatial.transform import Rotation


DEFAULT_OUTPUT_ROOT = "output/sgs_splat_proxy"
DEFAULT_CAM_ID = "cam_3"
DEFAULT_QUERY_DIR = "kapture_1_3/sensors/records_data/cam_3/images"


def _timestamp_key(name: str) -> str:
    stem = Path(name).stem
    m = re.search(r"(\d{12,})", stem)
    return m.group(1) if m else stem


def _read_c_string(fid) -> str:
    buf = bytearray()
    while True:
        ch = fid.read(1)
        if ch == b"":
            raise EOFError("Unexpected EOF while reading COLMAP image name")
        if ch == b"\x00":
            return buf.decode("utf-8")
        buf.extend(ch)


def _qvec_wxyz_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def load_step0_align(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    T = np.asarray(data["T_align"], dtype=np.float64)
    print(f"  T_align loaded: {path}")
    print(f"  R_align:\n{T[:3, :3]}")
    print(f"  t_align: {T[:3, 3]}")
    return T


def load_optional_transform(path: str, disabled: bool) -> np.ndarray:
    if disabled or not path:
        print("  Extra COLMAP transform: identity")
        return np.eye(4, dtype=np.float64)
    if not os.path.exists(path):
        raise FileNotFoundError(f"extra transform JSON not found: {path}")

    with open(path) as f:
        data = json.load(f)
    raw = (
        data.get("applied_transform")
        or data.get("transform")
        or data.get("model_to_map_transform")
    )
    if raw is None:
        print(f"  Extra COLMAP transform: no transform key in {path}; identity")
        return np.eye(4, dtype=np.float64)

    arr = np.asarray(raw, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    if arr.shape == (3, 4):
        T[:3, :4] = arr
    elif arr.shape == (4, 4):
        T = arr
    else:
        raise ValueError(f"Unexpected transform shape {arr.shape}: {path}")
    print(f"  Extra COLMAP transform loaded: {path}")
    print(f"  extra_transform:\n{T[:3, :4]}")
    return T


def load_rig_extrinsic(path: str, cam_id: str):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9 or parts[1] != cam_id:
                continue
            qw, qx, qy, qz = map(float, parts[2:6])
            tx, ty, tz = map(float, parts[6:9])
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            print(f"  Rig extrinsic loaded: {path} ({cam_id})")
            print(f"  T_rig_to_{cam_id} t: {T[:3, 3]}")
            return T
    return None


def load_colmap_images(images_bin: str, cam_id: str | None):
    poses = {}
    total = 0
    used = 0
    with open(images_bin, "rb") as fid:
        n_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(n_images):
            total += 1
            image_id = struct.unpack("<i", fid.read(4))[0]
            qvec = struct.unpack("<dddd", fid.read(32))
            tvec = np.asarray(struct.unpack("<ddd", fid.read(24)), dtype=np.float64)
            camera_id = struct.unpack("<i", fid.read(4))[0]
            name = _read_c_string(fid)
            n_pts = struct.unpack("<Q", fid.read(8))[0]
            fid.seek(n_pts * 24, os.SEEK_CUR)

            if cam_id and cam_id not in Path(name).parts and cam_id not in name:
                continue

            R_w2c = _qvec_wxyz_to_rotmat(qvec)
            T_c2w = np.eye(4, dtype=np.float64)
            T_c2w[:3, :3] = R_w2c.T
            T_c2w[:3, 3] = -R_w2c.T @ tvec

            poses[_timestamp_key(name)] = {
                "name": name,
                "image_id": image_id,
                "camera_id": camera_id,
                "T_c2w": T_c2w,
            }
            used += 1

    print(f"  COLMAP images loaded: {used}/{total}"
          + (f" for {cam_id}" if cam_id else ""))
    if cam_id and used == 0:
        raise RuntimeError(f"No COLMAP image matched cam_id={cam_id!r}: {images_bin}")
    return poses


def load_pred_json(path: str):
    with open(path) as f:
        raw = json.load(f)
    poses = {}
    for name, mat in raw.items():
        T = np.asarray(mat, dtype=np.float64)
        if T.shape != (4, 4):
            continue
        poses[_timestamp_key(name)] = {"name": name, "T_c2w": T}
    print(f"  Pred JSON poses loaded: {len(poses)} from {path}")
    return poses


def load_pred_tum(path: str):
    poses = {}
    with open(path) as f:
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
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            key = str(int(round(ts_sec * 1e6)))
            poses[key] = {"name": key, "T_c2w": T}
    print(f"  Pred TUM poses loaded: {len(poses)} from {path}")
    return poses


def load_predictions(pred_json: str, pred_tum: str, prefer_tum: bool):
    if prefer_tum and pred_tum and os.path.exists(pred_tum):
        return load_pred_tum(pred_tum), pred_tum
    if pred_json and os.path.exists(pred_json):
        return load_pred_json(pred_json), pred_json
    if pred_tum and os.path.exists(pred_tum):
        return load_pred_tum(pred_tum), pred_tum
    raise FileNotFoundError(f"No prediction file found: {pred_json} or {pred_tum}")


def load_query_keys(query_dir: str):
    if not query_dir:
        return None
    if os.path.isfile(query_dir):
        keys = {_timestamp_key(query_dir)}
    elif os.path.isdir(query_dir):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        keys = {
            _timestamp_key(p)
            for p in os.listdir(query_dir)
            if os.path.splitext(p)[1].lower() in exts
        }
    else:
        raise FileNotFoundError(f"query_dir not found: {query_dir}")
    print(f"  Query key filter loaded: {len(keys)} from {query_dir}")
    return keys


def rotation_error_deg(R_gt, R_pred) -> float:
    dR = R_gt.T @ R_pred
    c = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _rotation_offset_matrix(gt_eval, pred_eval, keys, mode):
    """Return constant R_off where R_pred_corrected = R_pred @ R_off."""
    if mode == "none":
        return np.eye(3, dtype=np.float64)
    if mode != "auto":
        raise ValueError(f"Unknown rotation offset mode: {mode}")
    if not keys:
        return np.eye(3, dtype=np.float64)

    rel_rots = []
    for key in keys:
        R_gt = gt_eval[key]["T_c2w"][:3, :3]
        R_pr = pred_eval[key]["T_c2w"][:3, :3]
        rel_rots.append(R_pr.T @ R_gt)
    rotvecs = np.asarray([Rotation.from_matrix(R).as_rotvec() for R in rel_rots])
    mean_rot = Rotation.from_rotvec(np.median(rotvecs, axis=0)).as_matrix()
    print("  Rotation offset: auto median R_pred.T @ R_gt")
    print(f"  R_offset:\n{mean_rot}")
    return mean_rot


def _with_rotation_offset(pred_eval, R_offset):
    if np.linalg.norm(R_offset - np.eye(3)) < 1e-12:
        return pred_eval
    out = {}
    for key, item in pred_eval.items():
        T = np.asarray(item["T_c2w"], dtype=np.float64).copy()
        T[:3, :3] = T[:3, :3] @ R_offset
        out[key] = {"name": item["name"], "T_c2w": T}
    return out


def load_map_xy(path: str, max_points: int):
    if not path or not os.path.exists(path):
        return None
    try:
        from plyfile import PlyData
        vertex = PlyData.read(path)["vertex"].data
        xy = np.column_stack([
            np.asarray(vertex["x"], dtype=np.float64),
            np.asarray(vertex["y"], dtype=np.float64),
        ])
        if len(xy) > max_points:
            rng = np.random.default_rng(7)
            xy = xy[rng.choice(len(xy), size=max_points, replace=False)]
        print(f"  Map background loaded: {len(xy)} sampled points from {path}")
        return xy
    except Exception as e:
        print(f"  [warn] failed to load map PLY: {e}")
        return None


def write_matches_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "key", "gt_name", "pred_name",
            "gt_x", "gt_y", "gt_z",
            "pred_x", "pred_y", "pred_z",
            "trans_err_m", "rot_err_deg",
        ])
        w.writerows(rows)


def default_paths(output_root: str, cam_id: str):
    eval_dir = os.path.join(output_root, "test_results", cam_id)
    return {
        "step0_pkl": os.path.join(output_root, "step0_data.pkl"),
        "pred_json": os.path.join(eval_dir, "trajectory_poses.json"),
        "pred_tum": os.path.join(eval_dir, "trajectory_tum.txt"),
        "output_dir": eval_dir,
        "map_ply": os.path.join(output_root, "aligned_map.ply"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cam_id", default=DEFAULT_CAM_ID)
    parser.add_argument("--query_dir", default=DEFAULT_QUERY_DIR,
                        help="Kapture records image dir/file used in test. "
                             "Only these timestamps are evaluated. Empty disables.")
    parser.add_argument("--colmap_images", default="nerfstudio/scene0/sparse/0/images.bin",
                        help="COLMAP sparse images.bin used as GT")
    parser.add_argument("--step0_pkl", default=None)
    parser.add_argument("--extra_transform_json", default="",
                        help="Optional transform applied before T_align, e.g. nerfstudio transforms.json")
    parser.add_argument("--disable_extra_transform", action="store_true")
    parser.add_argument("--rigs_txt", default="kapture_1_3/sensors/rigs.txt")
    parser.add_argument("--gt_frame", choices=["camera", "rig"], default="rig")
    parser.add_argument("--pred_frame", choices=["camera", "rig"], default="rig")
    parser.add_argument("--pred_cam_id", default=None,
                        help="Camera id for pred_frame conversion; default = cam_id")
    parser.add_argument("--pred_json", default=None)
    parser.add_argument("--pred_tum", default=None)
    parser.add_argument("--prefer_tum", action="store_true")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--map_ply", default=None)
    parser.add_argument("--map_max_points", type=int, default=100000)
    parser.add_argument("--outlier_thresh", type=float, default=5.0)
    parser.add_argument("--rotation_offset", choices=["none", "auto"], default="none",
                        help="Optional constant rotation-convention correction for reporting "
                             "rotation error only. Translation is unchanged.")
    args = parser.parse_args()

    paths = default_paths(args.output_root, args.cam_id)
    step0_pkl = args.step0_pkl or paths["step0_pkl"]
    pred_json = args.pred_json or paths["pred_json"]
    pred_tum = args.pred_tum or paths["pred_tum"]
    output_dir = args.output_dir or paths["output_dir"]
    map_ply = args.map_ply if args.map_ply is not None else paths["map_ply"]
    pred_cam_id = args.pred_cam_id or args.cam_id

    os.makedirs(output_dir, exist_ok=True)

    print("\n[1] Loading data...")
    T_align = load_step0_align(step0_pkl)
    T_extra = load_optional_transform(
        args.extra_transform_json,
        args.disable_extra_transform or not args.extra_transform_json,
    )
    gt_colmap = load_colmap_images(args.colmap_images, args.cam_id or None)
    pred, pred_source = load_predictions(pred_json, pred_tum, args.prefer_tum)
    query_keys = load_query_keys(args.query_dir)
    map_xy = load_map_xy(map_ply, args.map_max_points)

    T_gt_rig_to_cam = load_rig_extrinsic(args.rigs_txt, args.cam_id)
    T_pred_rig_to_cam = load_rig_extrinsic(args.rigs_txt, pred_cam_id)
    if args.gt_frame == "rig" and T_gt_rig_to_cam is None:
        raise FileNotFoundError(f"GT frame=rig needs {args.cam_id} in {args.rigs_txt}")
    if args.pred_frame != args.gt_frame and T_pred_rig_to_cam is None:
        raise FileNotFoundError(
            f"pred/gt frame conversion needs {pred_cam_id} in {args.rigs_txt}")

    print("\n[2] Transforming COLMAP GT to aligned eval frame...")
    gt_eval = {}
    for key, item in gt_colmap.items():
        T = T_align @ T_extra @ item["T_c2w"]
        if args.gt_frame == "rig":
            T = T @ T_gt_rig_to_cam
        gt_eval[key] = {"name": item["name"], "T_c2w": T}

    if args.pred_frame == args.gt_frame:
        pred_eval = pred
    elif args.pred_frame == "camera" and args.gt_frame == "rig":
        pred_eval = {
            k: {"name": v["name"], "T_c2w": v["T_c2w"] @ T_pred_rig_to_cam}
            for k, v in pred.items()
        }
    elif args.pred_frame == "rig" and args.gt_frame == "camera":
        pred_eval = {
            k: {"name": v["name"], "T_c2w": v["T_c2w"] @ np.linalg.inv(T_pred_rig_to_cam)}
            for k, v in pred.items()
        }
    else:
        raise RuntimeError("unreachable frame conversion")

    common_set = set(gt_eval) & set(pred_eval)
    if query_keys is not None:
        common_set &= query_keys
    common = sorted(common_set)
    print(f"\n[3] Matched poses: {len(common)}")
    print(f"  GT keys: {len(gt_eval)}")
    print(f"  Pred keys: {len(pred_eval)}")
    if query_keys is not None:
        print(f"  Query filter keys: {len(query_keys)}")
    if not common:
        print("  Example GT keys:", list(sorted(gt_eval))[:5])
        print("  Example Pred keys:", list(sorted(pred_eval))[:5])
        raise RuntimeError("No matching timestamps / stems between COLMAP GT and predictions")

    R_offset = _rotation_offset_matrix(gt_eval, pred_eval, common, args.rotation_offset)
    pred_eval_for_rot = _with_rotation_offset(pred_eval, R_offset)

    gt_xyz, pred_xyz, t_errs, r_errs, rows = [], [], [], [], []
    for key in common:
        T_gt = gt_eval[key]["T_c2w"]
        T_pr = pred_eval[key]["T_c2w"]
        T_pr_rot = pred_eval_for_rot[key]["T_c2w"]
        p_gt = T_gt[:3, 3]
        p_pr = T_pr[:3, 3]
        t_err = float(np.linalg.norm(p_gt - p_pr))
        r_err = rotation_error_deg(T_gt[:3, :3], T_pr_rot[:3, :3])
        gt_xyz.append(p_gt)
        pred_xyz.append(p_pr)
        t_errs.append(t_err)
        r_errs.append(r_err)
        rows.append([
            key, gt_eval[key]["name"], pred_eval[key]["name"],
            *p_gt.tolist(), *p_pr.tolist(), t_err, r_err,
        ])

    gt_xyz = np.asarray(gt_xyz)
    pred_xyz = np.asarray(pred_xyz)
    t_errs = np.asarray(t_errs)
    r_errs = np.asarray(r_errs)
    inlier = t_errs <= float(args.outlier_thresh)
    outlier = ~inlier
    t_in = t_errs[inlier]
    r_in = r_errs[inlier]

    print("\n[4] Results...")
    report_lines = [
        "=" * 66,
        f"  COLMAP images      : {args.colmap_images}",
        f"  Prediction source  : {pred_source}",
        f"  Step0 align        : {step0_pkl}",
        f"  Extra transform    : {args.extra_transform_json or 'identity'}",
        f"  Camera filter      : {args.cam_id or 'all'}",
        f"  Rig file           : {args.rigs_txt}",
        f"  Eval frame         : {args.gt_frame} + aligned map frame",
        f"  Pred frame         : {args.pred_frame}",
        f"  Query dir          : {args.query_dir or 'disabled'}",
        f"  Rotation offset    : {args.rotation_offset}",
        f"  Outlier threshold  : {args.outlier_thresh:.3f} m",
        f"  Total matched      : {len(common)}",
        f"  Inliers            : {int(inlier.sum())}",
        f"  Outliers           : {int(outlier.sum())}",
        "=" * 66,
    ]
    if len(t_in):
        report_lines += [
            f"\n  Translation Error (N={len(t_in)})",
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
            f"\n  Rotation Error (N={len(r_in)})",
            f"    Mean   : {np.mean(r_in):.2f} deg",
            f"    Median : {np.median(r_in):.2f} deg",
            f"    RMSE   : {np.sqrt(np.mean(r_in ** 2)):.2f} deg",
            f"    Std    : {np.std(r_in):.2f} deg",
        ]
    report_lines.append("=" * 66)
    report = "\n".join(report_lines)
    print(report)

    txt_path = os.path.join(output_dir, "eval_sgs_colmap_results.txt")
    with open(txt_path, "w") as f:
        f.write(report + "\n")
    print(f"  Saved: {txt_path}")

    csv_path = os.path.join(output_dir, "eval_sgs_colmap_matches.csv")
    write_matches_csv(csv_path, rows)
    print(f"  Saved: {csv_path}")

    all_gt_keys = sorted(gt_eval)
    all_gt_xyz = np.asarray([gt_eval[k]["T_c2w"][:3, 3] for k in all_gt_keys])

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    ax = axes[0]
    if map_xy is not None:
        ax.scatter(map_xy[:, 0], map_xy[:, 1], c="lightgray", s=0.2,
                   alpha=0.25, label="aligned map")
    ax.plot(all_gt_xyz[:, 0], all_gt_xyz[:, 1], color="lightblue",
            lw=1.3, alpha=0.7, label="COLMAP GT full")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], "b-o", lw=1.8, ms=3,
            label=f"COLMAP GT matched (N={len(common)})")
    ax.plot(pred_xyz[inlier, 0], pred_xyz[inlier, 1], "r-^",
            lw=1.7, ms=4, label=f"Pred inlier (N={int(inlier.sum())})")
    if outlier.any():
        ax.scatter(pred_xyz[outlier, 0], pred_xyz[outlier, 1],
                   c="orange", marker="x", s=90, zorder=6,
                   label=f"Pred outlier (N={int(outlier.sum())})")
    for i in range(len(common)):
        if inlier[i]:
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
    colors = ["steelblue" if ok else "orange" for ok in inlier]
    ax2.bar(np.arange(len(common)), t_errs, color=colors, width=1.0,
            edgecolor="none")
    ax2.axhline(args.outlier_thresh, color="red", ls="--", lw=1.4,
                label=f"outlier thresh={args.outlier_thresh:g}m")
    if len(t_in):
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
    if len(t_in):
        es = np.sort(t_in)
        cdf = np.arange(1, len(es) + 1) / len(es)
        ax3.plot(es, cdf * 100, "b-", lw=2, label=f"CDF (N={len(es)})")
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

    if len(t_in):
        title = (f"SGS Splat Proxy COLMAP Eval [{args.cam_id}]  "
                 f"matched={len(common)}, inliers={int(inlier.sum())}, "
                 f"outliers={int(outlier.sum())}\n"
                 f"Trans mean={np.mean(t_in):.3f}m median={np.median(t_in):.3f}m "
                 f"RMSE={np.sqrt(np.mean(t_in ** 2)):.3f}m | "
                 f"Rot mean={np.mean(r_in):.1f}deg median={np.median(r_in):.1f}deg")
    else:
        title = f"SGS Splat Proxy COLMAP Eval [{args.cam_id}]"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    png_path = os.path.join(output_dir, "eval_sgs_colmap_trajectory.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {png_path}")


if __name__ == "__main__":
    main()
