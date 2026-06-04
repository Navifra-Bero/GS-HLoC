#!/usr/bin/env python3
"""
Trajectory evaluation: predicted (aligned PLY frame) vs GT.

step0_align에서 PLY를 T_align으로 변환했으므로,
GT 원본 world frame를 T_align으로 변환한 뒤 비교한다.

GT 입력은 kapture trajectories.txt 또는 COLMAP images.txt를 지원한다.
COLMAP images.txt의 qvec/tvec는 world-to-camera이므로 inverse해서
camera-to-world pose로 바꾼 뒤 사용한다.

Multi-cam type2의 step7 결과는 rig frame pose로 저장된다. 이 경우 GT camera
pose도 rigs.txt의 T_rig_to_cam을 사용해 rig pose로 변환한 뒤 비교한다.

Usage:
  python3 scripts/eval_trajectory.py
  python3 scripts/eval_trajectory.py --outlier_thresh 3.0 --cam_id cam_3
  python3 scripts/eval_trajectory.py --align_pred_to_gt rigid
  python3 scripts/eval_trajectory.py \
    --gt_format colmap \
    --gt_traj test_data_rectified/images.txt \
    --pred_tum output/gs_sdf_omni/test_results/cam_0/trajectory_tum.txt \
    --output_dir output/gs_sdf_omni/test_results/cam_0/eval_colmap \
    --cam_id cam_0 --gt_frame rig --pred_frame rig
"""

import os
import re
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


# ── Default paths ──────────────────────────────────────────────────────────
STEP0_PKL      = "output/gs_sdf_omni/step0_data.pkl"
GT_TRAJ_TXT    = "kapture_1_3/sensors/trajectories.txt"
RIGS_TXT       = "kapture_1_3/sensors/rigs.txt"
PRED_TUM_TXT   = "output/gs_sdf_omni/test_results/cam_3/trajectory_tum.txt"
OUTPUT_DIR     = "output/gs_sdf_omni/test_results/cam_3"
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


def _infer_cam_id_from_path(path: str) -> str | None:
    norm = os.path.normpath(str(path))
    parts = norm.split(os.sep)
    for part in reversed(parts):
        if re.fullmatch(r"cam_\d+", part):
            return part
    match = re.findall(r"cam_\d+", norm)
    return match[-1] if match else None


def colmap_qt_to_c2w(qvec, tvec) -> np.ndarray:
    """COLMAP qvec/tvec(world-to-camera) -> camera-to-world pose."""
    qw, qx, qy, qz = [float(v) for v in qvec]
    tx, ty, tz = [float(v) for v in tvec]
    R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T_w2c = np.eye(4, dtype=np.float64)
    T_w2c[:3, :3] = R_w2c
    T_w2c[:3, 3] = [tx, ty, tz]
    return np.linalg.inv(T_w2c)


def parse_colmap_image_rows(images_txt: str) -> list[dict]:
    """COLMAP images.txt 파싱. 각 image block의 첫 줄만 pose로 사용."""
    rows = []
    with open(images_txt) as f:
        lines = [line.strip() for line in f
                 if line.strip() and not line.startswith("#")]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        i += 2
        if len(parts) < 10:
            continue
        name = parts[9]
        if not re.search(r"\.(jpg|jpeg|png|bmp)$", name, re.IGNORECASE):
            continue
        try:
            image_id = int(parts[0])
            qvec = np.array([float(v) for v in parts[1:5]], dtype=np.float64)
            tvec = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
            camera_id = int(parts[8])
        except ValueError:
            continue
        stem = os.path.splitext(os.path.basename(name))[0]
        try:
            ts_us = int(stem)
        except ValueError:
            continue
        rows.append({
            "image_id": image_id,
            "camera_id": camera_id,
            "name": name,
            "cam_id": _infer_cam_id_from_path(name),
            "timestamp_us": ts_us,
            "pose": colmap_qt_to_c2w(qvec, tvec),
        })
    return rows


def load_gt_colmap_images(images_txt: str, cam_id: str) -> dict:
    """COLMAP images.txt에서 특정 camera path(cam_X)의 c2w pose를 읽는다."""
    poses = {}
    for row in parse_colmap_image_rows(images_txt):
        if cam_id and row["cam_id"] != cam_id:
            continue
        poses[row["timestamp_us"]] = row["pose"]
    return poses


def parse_colmap_rigs_from_images(images_txt: str, primary_cam: str = "cam_0",
                                  tolerance_ns: int = 200_000_000) -> dict:
    """COLMAP 동시 프레임 pose로 T_rig_to_cam을 추정한다.

    parse_colmap_records와 동일하게 primary_cam을 rig frame으로 둔다.
    """
    rows = [r for r in parse_colmap_image_rows(images_txt)
            if r["cam_id"] and r["timestamp_us"] is not None]
    by_cam = {}
    for row in rows:
        by_cam.setdefault(row["cam_id"], []).append(row)
    for cid in by_cam:
        by_cam[cid].sort(key=lambda r: r["timestamp_us"])
    if primary_cam not in by_cam and by_cam:
        primary_cam = sorted(by_cam)[0]

    rigs = {primary_cam: np.eye(4, dtype=np.float64)}
    primary_rows = by_cam.get(primary_cam, [])
    if not primary_rows:
        return rigs

    for cid, seq in by_cam.items():
        if cid == primary_cam:
            continue
        rels = []
        for row in seq:
            ref = min(primary_rows,
                      key=lambda r: abs(r["timestamp_us"] - row["timestamp_us"]))
            if abs(ref["timestamp_us"] - row["timestamp_us"]) > tolerance_ns:
                continue
            T_w_cam = row["pose"]
            T_w_rig = ref["pose"]
            rels.append(np.linalg.inv(T_w_cam) @ T_w_rig)
        if not rels:
            continue
        Ts = np.stack(rels, axis=0)
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = np.median(Ts[:, :3, 3], axis=0)
        try:
            T[:3, :3] = Rotation.from_matrix(Ts[:, :3, :3]).mean().as_matrix()
        except Exception:
            T[:3, :3] = Ts[0, :3, :3]
        rigs[cid] = T
    return rigs


def detect_gt_format(gt_format: str, gt_path: str) -> str:
    if gt_format != "auto":
        return gt_format
    base = os.path.basename(gt_path)
    if base == "images.txt":
        return "colmap"
    return "kapture"


def load_gt_poses_auto(gt_path: str, cam_id: str, fmt: str) -> dict:
    if fmt == "colmap":
        return load_gt_colmap_images(gt_path, cam_id)
    return load_gt_trajectory(gt_path, cam_id)


def load_rig_extrinsic_auto(rigs_path: str, cam_id: str, fmt: str,
                            gt_path: str, primary_cam: str) -> np.ndarray | None:
    if rigs_path and os.path.isfile(rigs_path):
        return load_rig_extrinsic(rigs_path, cam_id)
    if fmt == "colmap":
        rigs = parse_colmap_rigs_from_images(gt_path, primary_cam=primary_cam)
        return rigs.get(cam_id)
    return None


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


def estimate_similarity(src: np.ndarray, dst: np.ndarray,
                        estimate_scale: bool = False):
    """Umeyama/Kabsch alignment: dst ~= scale * R @ src + t."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 3:
        raise ValueError("trajectory alignment requires at least 3 matched poses")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    X = src - mu_src
    Y = dst - mu_dst

    H = (X.T @ Y) / len(src)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    if estimate_scale:
        var_src = np.mean(np.sum(X * X, axis=1))
        scale = float(np.sum(S) / max(var_src, 1e-12))
    else:
        scale = 1.0
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def estimate_translation(src: np.ndarray, dst: np.ndarray):
    """Translation-only alignment: dst ~= src + t."""
    t = np.asarray(dst, dtype=np.float64).mean(axis=0) - \
        np.asarray(src, dtype=np.float64).mean(axis=0)
    return 1.0, np.eye(3), t


def estimate_pred_to_gt_alignment(pred_xyz: np.ndarray, gt_xyz: np.ndarray,
                                  mode: str, robust: bool = True,
                                  keep_ratio: float = 0.7, iters: int = 3):
    """matched positions로 pred frame → GT eval frame 변환을 추정."""
    mode = str(mode).lower()
    if mode == "none":
        return 1.0, np.eye(3), np.zeros(3), np.ones(len(pred_xyz), dtype=bool)

    pred_xyz = np.asarray(pred_xyz, dtype=np.float64)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64)
    mask = np.ones(len(pred_xyz), dtype=bool)
    n_keep = max(3, int(round(len(pred_xyz) * float(keep_ratio))))

    for _ in range(max(1, int(iters))):
        if mode == "translation":
            scale, R, t = estimate_translation(pred_xyz[mask], gt_xyz[mask])
        elif mode in ("rigid", "se3"):
            scale, R, t = estimate_similarity(pred_xyz[mask], gt_xyz[mask],
                                              estimate_scale=False)
        elif mode in ("similarity", "sim3"):
            scale, R, t = estimate_similarity(pred_xyz[mask], gt_xyz[mask],
                                              estimate_scale=True)
        else:
            raise ValueError(f"Unknown --align_pred_to_gt mode: {mode}")

        if not robust:
            break
        aligned = scale * (pred_xyz @ R.T) + t
        residual = np.linalg.norm(aligned - gt_xyz, axis=1)
        keep_idx = np.argsort(residual)[:n_keep]
        new_mask = np.zeros(len(pred_xyz), dtype=bool)
        new_mask[keep_idx] = True
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask

    return scale, R, t, mask


def apply_similarity_to_pose(T: np.ndarray, scale: float,
                             R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """월드 프레임 similarity를 c2w pose에 적용."""
    out = np.array(T, dtype=np.float64, copy=True)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = scale * (R @ T[:3, 3]) + t
    return out


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step0_pkl",      default=STEP0_PKL)
    parser.add_argument("--gt_traj",        default=GT_TRAJ_TXT)
    parser.add_argument("--gt_format",      default="auto",
                        choices=["auto", "kapture", "colmap"],
                        help="GT pose file format. auto는 images.txt면 COLMAP으로 판단.")
    parser.add_argument("--rigs_txt",       default=RIGS_TXT,
                        help="kapture rigs.txt 경로. COLMAP GT에서는 파일이 없으면 "
                             "images.txt 동시 프레임에서 cam_0 기준 rig extrinsic을 추정.")
    parser.add_argument("--rig_primary_cam", default="cam_0",
                        help="COLMAP images.txt에서 rig frame으로 둘 기준 카메라.")
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
    parser.add_argument("--align_pred_to_gt", default="none",
                        choices=["none", "translation", "rigid", "se3",
                                 "similarity", "sim3"],
                        help="GT와 pred가 다른 map frame일 때 matched trajectory로 "
                             "pred→GT 정렬을 추정해 적용. scale이 맞으면 rigid 권장.")
    parser.add_argument("--align_no_robust", action="store_true", default=False,
                        help="pred→GT 정렬 추정 시 residual 기반 반복 trimming을 끔.")
    parser.add_argument("--align_keep_ratio", type=float, default=0.7,
                        help="robust alignment에서 residual이 작은 비율만 유지 "
                             "(기본 0.7).")
    parser.add_argument("--align_iters", type=int, default=3,
                        help="robust alignment 반복 횟수.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    T_align       = load_T_align(args.step0_pkl)
    gt_format     = detect_gt_format(args.gt_format, args.gt_traj)
    gt_raw        = load_gt_poses_auto(args.gt_traj, args.cam_id, gt_format)
    pred_poses    = load_pred_trajectory(args.pred_tum)
    print(f"  GT format: {gt_format}")
    print(f"  GT poses ({args.cam_id}): {len(gt_raw)}")
    print(f"  Pred poses: {len(pred_poses)}")

    # ── Rig extrinsic 로드 ─────────────────────────────────────────────────
    pred_cam_id = args.pred_cam_id or args.cam_id
    T_gt_rig_to_cam = load_rig_extrinsic_auto(
        args.rigs_txt, args.cam_id, gt_format, args.gt_traj, args.rig_primary_cam)
    T_pred_rig_to_cam = load_rig_extrinsic_auto(
        args.rigs_txt, pred_cam_id, gt_format, args.gt_traj, args.rig_primary_cam)

    if args.gt_frame == "rig":
        if T_gt_rig_to_cam is None:
            raise FileNotFoundError(
                f"GT frame=rig requires rig extrinsic for {args.cam_id}. "
                f"For COLMAP, check --gt_traj images.txt and --rig_primary_cam.")
        rig_src = args.rigs_txt if os.path.isfile(args.rigs_txt) else args.gt_traj
        print(f"  GT rig extrinsic: T_rig_to_{args.cam_id} from {rig_src}")
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

    alignment_summary = None
    if args.align_pred_to_gt != "none":
        gt_for_align = np.array([gt_aligned[ts][:3, 3] for ts in common_ts])
        pred_for_align = np.array([pred_norm[ts][:3, 3] for ts in common_ts])
        scale_a, R_a, t_a, align_mask = estimate_pred_to_gt_alignment(
            pred_for_align, gt_for_align,
            args.align_pred_to_gt,
            robust=not args.align_no_robust,
            keep_ratio=args.align_keep_ratio,
            iters=args.align_iters,
        )
        pred_norm = {
            ts: apply_similarity_to_pose(T, scale_a, R_a, t_a)
            for ts, T in pred_norm.items()
        }
        rot_a = Rotation.from_matrix(R_a).as_euler("xyz", degrees=True)
        alignment_summary = {
            "mode": args.align_pred_to_gt,
            "scale": scale_a,
            "R": R_a,
            "t": t_a,
            "euler_xyz_deg": rot_a,
            "n_used": int(align_mask.sum()),
            "n_total": int(len(align_mask)),
            "robust": not args.align_no_robust,
        }
        print("\n[3b] Pred trajectory frame alignment")
        print(f"  mode={args.align_pred_to_gt}  robust={not args.align_no_robust}  "
              f"used={align_mask.sum()}/{len(align_mask)}")
        print(f"  scale={scale_a:.8f}")
        print(f"  R_pred_to_gt:\n{R_a}")
        print(f"  euler_xyz_deg={rot_a.round(3)}")
        print(f"  t_pred_to_gt={t_a.round(4)}")

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
    lines.append(f"  GT format          : {gt_format}")
    lines.append(f"  Pred trajectory    : {args.pred_tum}")
    lines.append(f"  GT cam/frame       : {args.cam_id} / {args.gt_frame}")
    lines.append(f"  Pred frame         : {args.pred_frame}"
                 + (f" ({pred_cam_id})" if args.pred_frame == "cam" else ""))
    lines.append(f"  Eval frame         : {eval_frame} + aligned map")
    if alignment_summary is not None:
        lines.append(f"  Pred→GT alignment  : {alignment_summary['mode']} "
                     f"(robust={alignment_summary['robust']}, "
                     f"used={alignment_summary['n_used']}/"
                     f"{alignment_summary['n_total']})")
        lines.append(f"    scale            : {alignment_summary['scale']:.8f}")
        lines.append(f"    t                : "
                     f"{np.array2string(alignment_summary['t'], precision=4)}")
        lines.append(f"    euler xyz deg    : "
                     f"{np.array2string(alignment_summary['euler_xyz_deg'], precision=3)}")
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
