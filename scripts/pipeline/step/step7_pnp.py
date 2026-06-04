import os, pickle
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .multi_cam import parse_kapture_rigs, parse_kapture_sensors, load_multi_cam_config


def _load_ref_depth(ref_entry):
    path = ref_entry.get("depth_path", "")
    if not path or not os.path.exists(path):
        return None
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".depth"):
        cam = ref_entry.get("camera", {}) or {}
        h = int(ref_entry.get("height") or cam.get("height") or 0)
        w = int(ref_entry.get("width") or cam.get("width") or 0)
        arr = np.fromfile(path, dtype=np.float32)
        if h <= 0 or w <= 0:
            if arr.size == 1200 * 1920:
                h, w = 1200, 1920
            elif arr.size == 1:
                return None
            else:
                return None
        if arr.size == h * w:
            return arr.reshape(h, w)
    return None


def _K_from_ref_entry(ref_entry, fallback_K):
    cam = ref_entry.get("camera", {}) or {}
    if all(k in cam for k in ("fx", "fy", "cx", "cy")):
        return np.array([[cam["fx"], 0, cam["cx"]],
                         [0, cam["fy"], cam["cy"]],
                         [0, 0, 1]], dtype=np.float64)
    return fallback_K


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


# ── Point + line pose refinement ───────────────────────────────────────────

def _project_world_points_pose(T_WQ, pts3d_world, K):
    """World 3D points를 query image pixel로 투영."""
    if len(pts3d_world) == 0:
        return np.zeros((0, 2)), np.zeros((0,), dtype=bool)
    R_WQ = T_WQ[:3, :3]
    t_WQ = T_WQ[:3, 3]
    pts_q = (R_WQ.T @ (pts3d_world - t_WQ).T).T
    z = pts_q[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    uv = np.zeros((len(pts_q), 2), dtype=np.float64)
    uv[valid, 0] = K[0, 0] * pts_q[valid, 0] / z[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * pts_q[valid, 1] / z[valid] + K[1, 2]
    valid &= np.isfinite(uv).all(axis=1)
    return uv, valid


def _line_coeff_from_xy(line_xy):
    """2D line segment endpoints(xy) → normalized infinite-line coeff [a,b,c]."""
    p0 = np.asarray(line_xy[0], dtype=np.float64)
    p1 = np.asarray(line_xy[1], dtype=np.float64)
    a = p0[1] - p1[1]
    b = p1[0] - p0[0]
    c = p0[0] * p1[1] - p1[0] * p0[1]
    n = np.hypot(a, b)
    if n < 1e-9:
        return None
    return np.array([a / n, b / n, c / n], dtype=np.float64)


def _depth_at_nearest(depth, xy, dmin, dmax, radius_px=0):
    x, y = float(xy[0]), float(xy[1])
    i, j = int(round(y)), int(round(x))
    if not (0 <= i < depth.shape[0] and 0 <= j < depth.shape[1]):
        return None
    r = int(max(0, radius_px))
    y0, y1 = max(0, i - r), min(depth.shape[0], i + r + 1)
    x0, x1 = max(0, j - r), min(depth.shape[1], j + r + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch >= dmin) & (patch <= dmax)]
    if valid.size == 0:
        return None
    return float(valid.min())


def _lift_line_matches_to_world(line_matches, ref_entry, K_render, dmin, dmax, lookup_radius_px=0):
    """SOLD2 2D line matches의 ref endpoints를 depth로 3D world line segment로 lift."""
    q_lines = np.asarray(line_matches.get("query_lines", []), dtype=np.float64)
    r_lines = np.asarray(line_matches.get("ref_lines", []), dtype=np.float64)
    if q_lines.ndim != 3 or r_lines.ndim != 3 or len(q_lines) == 0:
        return np.zeros((0, 2, 2)), np.zeros((0, 2, 3))

    depth = _load_ref_depth(ref_entry)
    if depth is None:
        return np.zeros((0, 2, 2)), np.zeros((0, 2, 3))
    T_WR = np.asarray(ref_entry["pose"], dtype=np.float64)
    fx, fy = K_render[0, 0], K_render[1, 1]
    cx, cy = K_render[0, 2], K_render[1, 2]

    q_keep = []
    world_keep = []
    for q_line, r_line in zip(q_lines, r_lines):
        coeff = _line_coeff_from_xy(q_line)
        if coeff is None:
            continue
        pts_ref = []
        ok = True
        for xy in r_line:
            z = _depth_at_nearest(depth, xy, dmin, dmax, lookup_radius_px)
            if z is None:
                ok = False
                break
            x = (xy[0] - cx) * z / fx
            y = (xy[1] - cy) * z / fy
            pts_ref.append([x, y, z])
        if not ok:
            continue
        pts_ref_h = np.hstack([np.asarray(pts_ref), np.ones((2, 1))])
        pts_world = (T_WR @ pts_ref_h.T).T[:, :3]
        q_keep.append(q_line)
        world_keep.append(pts_world)

    if not q_keep:
        return np.zeros((0, 2, 2)), np.zeros((0, 2, 3))
    return np.asarray(q_keep, dtype=np.float64), np.asarray(world_keep, dtype=np.float64)


def _line_distance_residuals(T_WQ, q_lines, line3d_world, K):
    """Projected 3D line endpoints와 query 2D infinite-line 사이 거리 residual(px)."""
    if len(q_lines) == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=bool)
    all_pts = line3d_world.reshape(-1, 3)
    uv, valid_pt = _project_world_points_pose(T_WQ, all_pts, K)
    uv = uv.reshape(-1, 2, 2)
    valid_pair = valid_pt.reshape(-1, 2).all(axis=1)

    residuals = []
    valid_lines = []
    for i, q_line in enumerate(q_lines):
        coeff = _line_coeff_from_xy(q_line)
        if coeff is None or not valid_pair[i]:
            residuals.extend([1e3, 1e3])
            valid_lines.append(False)
            continue
        d = uv[i] @ coeff[:2] + coeff[2]
        residuals.extend(d.tolist())
        valid_lines.append(True)
    return np.asarray(residuals, dtype=np.float64), np.asarray(valid_lines, dtype=bool)


def _pose_to_params(T_WQ):
    from scipy.spatial.transform import Rotation

    rvec = Rotation.from_matrix(T_WQ[:3, :3]).as_rotvec()
    return np.hstack([rvec, T_WQ[:3, 3]])


def _params_to_pose(params):
    from scipy.spatial.transform import Rotation

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_rotvec(params[:3]).as_matrix()
    T[:3, 3] = params[3:6]
    return T


def _rmse(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if len(x) else float("inf")


def _refine_pose_points_lines(T_init, pts2d, pts3d_world, q_lines, line3d_world,
                              K, config):
    """PnP pose를 초기값으로 point reprojection + line endpoint-to-line residual을 LM refine."""
    from scipy.optimize import least_squares

    pnp = config.get("pnp", {})
    min_lines = int(pnp.get("line_refine_min_lines", 3))
    if len(q_lines) < min_lines:
        return None, {"success": False, "reason": "not_enough_lines",
                      "n_lines": int(len(q_lines))}

    line_thr = float(pnp.get("line_refine_inlier_threshold", 12.0))
    line_res0, line_valid0 = _line_distance_residuals(T_init, q_lines, line3d_world, K)
    if len(line_res0):
        pair_res0 = np.abs(line_res0.reshape(-1, 2))
        keep = line_valid0 & (np.max(pair_res0, axis=1) <= line_thr)
        if int(keep.sum()) >= min_lines:
            q_lines = q_lines[keep]
            line3d_world = line3d_world[keep]

    if len(q_lines) < min_lines:
        return None, {"success": False, "reason": "not_enough_line_inliers",
                      "n_lines": int(len(q_lines))}

    point_weight = float(pnp.get("line_refine_point_weight", 1.0))
    line_weight = float(pnp.get("line_refine_line_weight", 0.7))

    def residual(params):
        T = _params_to_pose(params)
        chunks = []
        if len(pts2d) > 0:
            uv, valid = _project_world_points_pose(T, pts3d_world, K)
            point_res = uv - pts2d
            point_res[~valid] = 1e3
            chunks.append(point_weight * point_res.reshape(-1))
        line_res, _ = _line_distance_residuals(T, q_lines, line3d_world, K)
        chunks.append(line_weight * line_res)
        return np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float64)

    x0 = _pose_to_params(T_init)
    before = residual(x0)
    opt = least_squares(
        residual, x0,
        loss=pnp.get("line_refine_loss", "huber"),
        f_scale=float(pnp.get("line_refine_fscale", 5.0)),
        max_nfev=int(pnp.get("line_refine_max_nfev", 40)),
        xtol=1e-6, ftol=1e-6, gtol=1e-6,
    )
    T_refined = _params_to_pose(opt.x)
    after = residual(opt.x)

    stats = {
        "success": bool(opt.success),
        "cost_before": float(0.5 * np.sum(before * before)),
        "cost_after": float(0.5 * np.sum(after * after)),
        "rmse_before": _rmse(before),
        "rmse_after": _rmse(after),
        "n_points": int(len(pts2d)),
        "n_lines": int(len(q_lines)),
        "nfev": int(opt.nfev),
    }
    return T_refined, stats


def _line_distance_residuals_rig(T_W_rig, T_rig_to_cam, q_lines, line3d_world, K):
    """Rig pose 기준 projected 3D line endpoints와 query line 사이 거리 residual(px)."""
    if len(q_lines) == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=bool)
    all_pts = line3d_world.reshape(-1, 3)
    uv, valid_pt = _project_world_points(T_W_rig, T_rig_to_cam, all_pts, K)
    uv = uv.reshape(-1, 2, 2)
    valid_pair = valid_pt.reshape(-1, 2).all(axis=1)

    residuals = []
    valid_lines = []
    for i, q_line in enumerate(q_lines):
        coeff = _line_coeff_from_xy(q_line)
        if coeff is None or not valid_pair[i]:
            residuals.extend([1e3, 1e3])
            valid_lines.append(False)
            continue
        d = uv[i] @ coeff[:2] + coeff[2]
        residuals.extend(d.tolist())
        valid_lines.append(True)
    return np.asarray(residuals, dtype=np.float64), np.asarray(valid_lines, dtype=bool)


def _refine_rig_pose_points_lines(T_init, cam_result, line_matches, K_render, config):
    """Multi-cam rig pose를 best cam의 point inliers + SOLD2 lines로 추가 refine."""
    from scipy.optimize import least_squares

    pnp = config.get("pnp", {})
    dmin = float(config["camera"].get("depth_min", 0.3))
    dmax = float(config["camera"].get("depth_max", 20.0))
    min_lines = int(pnp.get("line_refine_min_lines", 3))

    entry = cam_result.get("entry")
    T_rig_to_cam = cam_result.get("T_rig_to_cam")
    if entry is None:
        return None, {"success": False, "reason": "missing_entry"}
    if T_rig_to_cam is None:
        T_rig_to_cam = np.eye(4, dtype=np.float64)

    q_lines, line3d_world = _lift_line_matches_to_world(
        line_matches, entry, K_render, dmin, dmax)
    if len(q_lines) < min_lines:
        return None, {"success": False, "reason": "not_enough_lines",
                      "n_lines": int(len(q_lines))}

    K_query = cam_result.get("K_query", K_render)
    pts2d = np.asarray(cam_result.get("pts2d", np.zeros((0, 2))), dtype=np.float64)
    pts3d_cam = np.asarray(cam_result.get("pts3d_cam", np.zeros((0, 3))), dtype=np.float64)
    inlier_idx = np.asarray(cam_result.get("pnp_inlier_idx", []), dtype=int)
    if len(inlier_idx) > 0:
        pts2d = pts2d[inlier_idx]
        pts3d_cam = pts3d_cam[inlier_idx]
    if len(pts3d_cam) > 0:
        T_WR = np.asarray(entry["pose"], dtype=np.float64)
        pts3d_h = np.hstack([pts3d_cam, np.ones((len(pts3d_cam), 1))])
        pts3d_world = (T_WR @ pts3d_h.T).T[:, :3]
    else:
        pts3d_world = np.zeros((0, 3), dtype=np.float64)

    line_thr = float(pnp.get("line_refine_inlier_threshold", 12.0))
    line_res0, line_valid0 = _line_distance_residuals_rig(
        T_init, T_rig_to_cam, q_lines, line3d_world, K_query)
    if len(line_res0):
        pair_res0 = np.abs(line_res0.reshape(-1, 2))
        keep = line_valid0 & (np.max(pair_res0, axis=1) <= line_thr)
        if int(keep.sum()) >= min_lines:
            q_lines = q_lines[keep]
            line3d_world = line3d_world[keep]
    if len(q_lines) < min_lines:
        return None, {"success": False, "reason": "not_enough_line_inliers",
                      "n_lines": int(len(q_lines))}

    point_weight = float(pnp.get("line_refine_point_weight", 1.0))
    line_weight = float(pnp.get("line_refine_line_weight", 0.7))

    def residual(params):
        T = _params_to_pose(params)
        chunks = []
        if len(pts2d) > 0:
            uv, valid = _project_world_points(T, T_rig_to_cam, pts3d_world, K_query)
            point_res = uv - pts2d
            point_res[~valid] = 1e3
            chunks.append(point_weight * point_res.reshape(-1))
        line_res, _ = _line_distance_residuals_rig(
            T, T_rig_to_cam, q_lines, line3d_world, K_query)
        chunks.append(line_weight * line_res)
        return np.concatenate(chunks)

    x0 = _pose_to_params(T_init)
    before = residual(x0)
    opt = least_squares(
        residual, x0,
        loss=pnp.get("line_refine_loss", "huber"),
        f_scale=float(pnp.get("line_refine_fscale", 5.0)),
        max_nfev=int(pnp.get("line_refine_max_nfev", 40)),
        xtol=1e-6, ftol=1e-6, gtol=1e-6,
    )
    after = residual(opt.x)
    stats = {
        "success": bool(opt.success),
        "cost_before": float(0.5 * np.sum(before * before)),
        "cost_after": float(0.5 * np.sum(after * after)),
        "rmse_before": _rmse(before),
        "rmse_after": _rmse(after),
        "n_points": int(len(pts2d)),
        "n_lines": int(len(q_lines)),
        "nfev": int(opt.nfev),
    }
    return _params_to_pose(opt.x), stats


# ── Multi-cam PnP helper ───────────────────────────────────────────────────

def _lift_2d_to_3d(mkpts_r, ref_entry, K_render, dmin, dmax):
    """렌더 이미지의 2D 매칭점 → ref cam frame 3D점 (depth lifting)."""
    dep = _load_ref_depth(ref_entry)
    if dep is None:
        return np.array(pts3d_cam, dtype=np.float64), np.array(valid_idx, dtype=int)
    fx_r, fy_r = K_render[0, 0], K_render[1, 1]
    cx_r, cy_r = K_render[0, 2], K_render[1, 2]
    lookup_radius = ref_entry.get("depth_lookup_radius_px", None)
    if lookup_radius is None:
        lookup_radius = 2 if str(ref_entry.get("depth_path", "")).endswith(".depth") else 0
    lookup_radius = int(lookup_radius)
    pts3d_cam = []
    valid_idx = []
    for i, (ru, rv) in enumerate(mkpts_r):
        pz = _depth_at_nearest(dep, (ru, rv), dmin, dmax, lookup_radius)
        if pz is None:
            continue
        pts3d_cam.append([(ru - cx_r) * pz / fx_r,
                           (rv - cy_r) * pz / fy_r,
                           pz])
        valid_idx.append(i)
    return np.array(pts3d_cam, dtype=np.float64), np.array(valid_idx, dtype=int)


def _pnp_cam(mkpts_q, mkpts_r, ref_entry, K_query, K_render, config, cam_id=""):
    """단일 카메라 PnP 실행.

    Parameters
    ----------
    mkpts_q   : (N,2) query 이미지 2D 점 (cam의 실제 intrinsics K_query 기준)
    mkpts_r   : (N,2) render 이미지 2D 점 (렌더 intrinsics K_render 기준)
    ref_entry : render DB entry (pose c2w, depth_path)
    K_query   : 쿼리 카메라 intrinsic matrix (3×3)
    K_render  : 렌더 intrinsic matrix (3×3) — 3D lifting에 사용
    config    : pipeline config

    Returns
    -------
    c2w_cam   : (4,4) 쿼리 카메라의 c2w (world frame), None if failed
    inlier_n  : int
    pts2d     : valid 2D pts (after depth filter)
    pts3d_cam : valid 3D pts in render cam frame
    inlier_idx: RANSAC inlier indices into pts2d/pts3d_cam
    """
    pnp = config.get("pnp", {})
    dmin, dmax = float(config["camera"].get("depth_min", 0.3)), \
                 float(config["camera"].get("depth_max", 20.0))
    solver    = pnp.get("solver", "opencv").strip().lower()
    do_refine = bool(pnp.get("refine", True))
    min_in    = int(pnp.get("min_inliers", 6))

    pts3d_cam, valid_idx = _lift_2d_to_3d(mkpts_r, ref_entry, K_render, dmin, dmax)
    if len(pts3d_cam) < min_in:
        return None, 0, np.zeros((0, 2)), pts3d_cam, np.array([], dtype=int)

    pts2d = mkpts_q[valid_idx]
    T_WR  = np.array(ref_entry["pose"], dtype=np.float64)

    if solver == "superansac":
        pts3d_h     = np.hstack([pts3d_cam, np.ones((len(pts3d_cam), 1))])
        pts3d_world = (T_WR @ pts3d_h.T).T[:, :3]
        T_WQ_raw, inlier_idx = _solve_pnp_superansac(pts2d, pts3d_world, K_query, config)
        if T_WQ_raw is None or len(inlier_idx) < min_in:
            return None, 0, pts2d, pts3d_cam, np.array([], dtype=int)
        inlier_n = len(inlier_idx)
        if do_refine:
            T_RW = np.linalg.inv(T_WR)
            pts3d_cam_in = (T_RW @ np.hstack([
                pts3d_world[inlier_idx], np.ones((inlier_n, 1))]).T).T[:, :3]
            T_QR_init = np.linalg.inv(T_WQ_raw) @ T_WR
            rvec_i, _  = cv2.Rodrigues(T_QR_init[:3, :3])
            tvec_i     = T_QR_init[:3, 3].reshape(3, 1)
            rvec_r, tvec_r = _refine_pnp(pts2d[inlier_idx], pts3d_cam_in, K_query, rvec_i, tvec_i)
            R_r, _ = cv2.Rodrigues(rvec_r)
            T_QR_r = np.eye(4); T_QR_r[:3, :3] = R_r; T_QR_r[:3, 3] = tvec_r.flatten()
            c2w_cam = T_WR @ np.linalg.inv(T_QR_r)
        else:
            c2w_cam = T_WQ_raw
    else:  # opencv
        T_QR_raw, rvec, inlier_idx = _solve_pnp_opencv(pts2d, pts3d_cam, K_query, config)
        if T_QR_raw is None or len(inlier_idx) < min_in:
            return None, 0, pts2d, pts3d_cam, np.array([], dtype=int)
        inlier_n = len(inlier_idx)
        T_QR = T_QR_raw
        if do_refine:
            rvec_r, tvec_r = _refine_pnp(
                pts2d[inlier_idx], pts3d_cam[inlier_idx], K_query, rvec,
                T_QR_raw[:3, 3].reshape(3, 1))
            R_r, _ = cv2.Rodrigues(rvec_r)
            T_QR = np.eye(4); T_QR[:3, :3] = R_r; T_QR[:3, 3] = tvec_r.flatten()
        c2w_cam = T_WR @ np.linalg.inv(T_QR)

    label = f"[{cam_id}] " if cam_id else ""
    print(f"  {label}PnP {solver}: {inlier_n}/{len(pts2d)} inliers  "
          f"pos={c2w_cam[:3,3].round(3)}")
    return c2w_cam, inlier_n, pts2d, pts3d_cam, np.asarray(inlier_idx, dtype=int)


def _rotation_diff_deg(T_a, T_b):
    """두 c2w pose의 rotation 차이를 degree로 반환."""
    from scipy.spatial.transform import Rotation
    R_rel = T_a[:3, :3].T @ T_b[:3, :3]
    return float(np.degrees(Rotation.from_matrix(R_rel).magnitude()))


def _weighted_pose_mean(poses, weights):
    """SE(3) pose들의 가중 평균. translation은 선형, rotation은 scipy mean."""
    from scipy.spatial.transform import Rotation

    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 1e-6)
    weights = weights / weights.sum()

    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = np.sum([w * T[:3, 3] for T, w in zip(poses, weights)], axis=0)
    rots = Rotation.from_matrix([T[:3, :3] for T in poses])
    out[:3, :3] = rots.mean(weights=weights).as_matrix()
    return out


def _pose_to_vec(T):
    """c2w pose → 6D [rotvec, t]."""
    from scipy.spatial.transform import Rotation
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def _vec_to_pose(x):
    """6D [rotvec, t] → c2w pose."""
    from scipy.spatial.transform import Rotation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = x[3:6]
    return T


def _project_world_points(T_W_rig, T_rig_to_cam, pts3d_world, K):
    """World 3D points를 query camera image plane으로 project."""
    T_cam_world = T_rig_to_cam @ np.linalg.inv(T_W_rig)
    pts_h = np.hstack([pts3d_world, np.ones((len(pts3d_world), 1))])
    pts_cam = (T_cam_world @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    valid = z > 1e-6
    proj = np.empty((len(pts_cam), 2), dtype=np.float64)
    proj[:, 0] = K[0, 0] * pts_cam[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
    proj[:, 1] = K[1, 1] * pts_cam[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    return proj, valid


def _joint_refine_rig_pose(T_init, cam_results, config):
    """고정 rig extrinsic을 사용해 rig pose 하나를 multi-cam reprojection refine."""
    try:
        from scipy.optimize import least_squares
    except Exception as e:
        print(f"  Joint refine unavailable: {e}")
        return None, {}

    pnp = config.get("pnp", {})
    reproj = float(pnp.get("joint_reproj_fscale",
                   pnp.get("reproj_threshold",
                   pnp.get("ransac_reproj_threshold", 8.0))))
    max_nfev = int(pnp.get("joint_max_nfev", 50))
    max_pts_per_cam = int(pnp.get("joint_max_points_per_cam", 1200))

    blocks = []
    for cam_id, r in cam_results.items():
        pts2d = r["pts2d"]
        pts3d_ref = r["pts3d_cam"]
        if len(pts2d) == 0 or r.get("T_rig_to_cam") is None:
            continue

        inlier_idx = np.asarray(r.get("pnp_inlier_idx", []), dtype=int)
        if len(inlier_idx) > 0:
            pts2d = pts2d[inlier_idx]
            pts3d_ref = pts3d_ref[inlier_idx]

        if len(pts2d) > max_pts_per_cam:
            idx = np.linspace(0, len(pts2d) - 1, max_pts_per_cam).astype(int)
            pts2d = pts2d[idx]
            pts3d_ref = pts3d_ref[idx]

        T_W_ref = np.asarray(r["entry"]["pose"], dtype=np.float64)
        pts3d_world = (T_W_ref @ np.hstack([
            pts3d_ref, np.ones((len(pts3d_ref), 1))]).T).T[:, :3]
        blocks.append({
            "cam_id": cam_id,
            "pts2d": pts2d,
            "pts3d_world": pts3d_world,
            "K": r["K_query"],
            "T_rig_to_cam": r["T_rig_to_cam"],
        })

    n_corr = sum(len(b["pts2d"]) for b in blocks)
    if not blocks or n_corr < int(config.get("pnp", {}).get("min_inliers", 6)):
        return None, {"n_corr": n_corr}

    def residual(x):
        T = _vec_to_pose(x)
        res = []
        for b in blocks:
            proj, valid = _project_world_points(
                T, b["T_rig_to_cam"], b["pts3d_world"], b["K"])
            r = proj - b["pts2d"]
            if not np.all(valid):
                r[~valid] = 1e3
            res.append(r.reshape(-1))
        return np.concatenate(res)

    x0 = _pose_to_vec(T_init)
    err0 = residual(x0)
    opt = least_squares(
        residual, x0,
        loss="soft_l1",
        f_scale=reproj,
        max_nfev=max_nfev,
    )
    err1 = residual(opt.x)

    stats = {
        "n_corr": n_corr,
        "rmse_before": float(np.sqrt(np.mean(err0 ** 2))),
        "rmse_after": float(np.sqrt(np.mean(err1 ** 2))),
        "cost": float(opt.cost),
        "success": bool(opt.success),
        "message": opt.message,
    }
    return _vec_to_pose(opt.x), stats


def _rig_pose_reprojection_stats(T_W_rig, cam_results, config):
    """rig pose 후보를 PnP inlier correspondences 기준으로 평가."""
    max_pts_per_cam = int(config.get("pnp", {}).get("joint_max_points_per_cam", 1200))
    per_cam = {}
    all_errs = []

    for cam_id, r in cam_results.items():
        pts2d = r["pts2d"]
        pts3d_ref = r["pts3d_cam"]
        if len(pts2d) == 0 or r.get("T_rig_to_cam") is None:
            continue

        inlier_idx = np.asarray(r.get("pnp_inlier_idx", []), dtype=int)
        if len(inlier_idx) > 0:
            pts2d = pts2d[inlier_idx]
            pts3d_ref = pts3d_ref[inlier_idx]

        if len(pts2d) > max_pts_per_cam:
            idx = np.linspace(0, len(pts2d) - 1, max_pts_per_cam).astype(int)
            pts2d = pts2d[idx]
            pts3d_ref = pts3d_ref[idx]

        if len(pts2d) == 0:
            continue

        T_W_ref = np.asarray(r["entry"]["pose"], dtype=np.float64)
        pts3d_world = (T_W_ref @ np.hstack([
            pts3d_ref, np.ones((len(pts3d_ref), 1))]).T).T[:, :3]
        proj, valid = _project_world_points(
            T_W_rig, r["T_rig_to_cam"], pts3d_world, r["K_query"])
        err = np.linalg.norm(proj - pts2d, axis=1)
        err = err[valid]
        if len(err) == 0:
            continue

        per_cam[cam_id] = {
            "n": int(len(err)),
            "median": float(np.median(err)),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mean": float(np.mean(err)),
        }
        all_errs.append(err)

    if not all_errs:
        return {
            "n": 0,
            "median": float("inf"),
            "rmse": float("inf"),
            "mean": float("inf"),
            "per_cam": per_cam,
        }

    errs = np.concatenate(all_errs)
    return {
        "n": int(len(errs)),
        "median": float(np.median(errs)),
        "rmse": float(np.sqrt(np.mean(errs ** 2))),
        "mean": float(np.mean(errs)),
        "per_cam": per_cam,
    }


def _select_multicam_pose(candidates, baseline_name, best_cam_id, config):
    """best-cam baseline을 기준으로 fusion/joint 후보를 보수적으로 채택."""
    pnp = config.get("pnp", {})
    policy = pnp.get("multi_cam_pose_selection", "adaptive").strip().lower()
    score_key = pnp.get("multi_cam_pose_score", "median").strip().lower()
    improve_ratio = float(pnp.get("multi_cam_accept_improve_ratio", 0.90))
    best_cam_degrade = float(pnp.get("multi_cam_best_cam_degrade_ratio", 1.10))

    if policy in candidates:
        return policy, "forced"
    if policy == "best":
        return baseline_name, "forced_best"
    if policy in ("fusion", "fused") and "fusion" in candidates:
        return "fusion", "forced_fusion"
    if policy == "joint" and "joint" in candidates:
        return "joint", "forced_joint"

    baseline = candidates[baseline_name]["stats"]
    baseline_score = baseline.get(score_key, float("inf"))
    baseline_best_cam = baseline.get("per_cam", {}).get(best_cam_id, {})
    baseline_best_score = baseline_best_cam.get(score_key, baseline_score)

    chosen = baseline_name
    reason = "baseline"
    best_score = baseline_score

    for name in ("joint", "fusion"):
        if name not in candidates:
            continue
        st = candidates[name]["stats"]
        score = st.get(score_key, float("inf"))
        cam_score = st.get("per_cam", {}).get(best_cam_id, {}).get(score_key, score)
        improves = score <= baseline_score * improve_ratio
        preserves_best_cam = cam_score <= baseline_best_score * best_cam_degrade
        if improves and preserves_best_cam and score < best_score:
            chosen = name
            best_score = score
            reason = (f"{name} improves {score_key} "
                      f"{baseline_score:.2f}->{score:.2f}px")

    return chosen, reason


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
    cam = config["camera"]
    pnp = config.get("pnp", {})
    line_label = " + point-line refinement" if bool(pnp.get("line_refine", False)) else ""
    print("\n" + "="*60 +
          f"\nSTEP 7: 2D-3D pose estimation (PnP{line_label})"
          "\n" + "="*60)

    fx, fy = cam["fx"], cam["fy"]
    cx, cy = cam["cx"], cam["cy"]
    K      = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    solver    = pnp.get("solver", "opencv").strip().lower()
    do_refine = bool(pnp.get("refine", True))
    min_in    = int(pnp.get("min_inliers", 6))

    ref_entry = step6_data["best_cand"]
    query_rgb = step6_data["query_rgb"]
    ref_rgb   = step6_data["ref_rgb"]
    gt_entry  = step5_data.get("gt_entry")
    T_WR      = np.array(ref_entry["pose"], dtype=np.float64)

    # K_render: 렌더는 항상 config camera 파라미터로 생성
    K_render = K

    # ── Multi-cam PnP ────────────────────────────────────────────────────
    active_cams = step6_data.get("active_cams", [])
    cam_mkpts_q = step6_data.get("cam_mkpts_q", {})
    cam_mkpts_r = step6_data.get("cam_mkpts_r", {})
    cam_entries = step6_data.get("cam_entries", {})
    mc_primary  = (step6_data.get("mc_primary")
                   or config.get("multi_cam", {}).get("primary_cam", ""))
    is_multi_pnp = len(active_cams) > 1 and bool(cam_entries)

    estimated_pose = None
    inlier_count   = 0
    T_QR           = None   # single-cam only
    pts2d          = np.zeros((0, 2))
    pts3d_cam      = np.zeros((0, 3))
    pnp_method     = "single"
    pnp_cams_used  = []
    best_pnp_cam   = None
    joint_stats    = {}
    line_refine_stats = {}
    pnp_results    = {}
    pose_selection = {}

    if is_multi_pnp:
        # ── 각 카메라 독립 PnP → rig frame 통일 → 최고 inlier 선택 ──────
        _, _, kapture_dir, _ = load_multi_cam_config(config)
        if not os.path.isabs(kapture_dir):
            kapture_dir = os.path.join(os.getcwd(), kapture_dir)

        rigs    = parse_kapture_rigs(kapture_dir)     # {cam_id: T_rig_to_cam}
        sensors = parse_kapture_sensors(kapture_dir)  # {cam_id: {fx,fy,cx,cy,...}}

        print(f"  Mode: multi-cam PnP  active={active_cams}  primary={mc_primary}")
        print(f"  Solver: {solver}  refine={do_refine}")

        best_inliers = 0
        best_cam_id  = None

        for cam_id in active_cams:
            if cam_id not in cam_mkpts_q or cam_id not in cam_entries:
                continue
            mq = cam_mkpts_q[cam_id]
            mr = cam_mkpts_r[cam_id]
            if len(mq) == 0:
                continue

            # 카메라별 intrinsics
            s = sensors.get(cam_id)
            if s:
                K_q = np.array([[s["fx"], 0, s["cx"]],
                                 [0, s["fy"], s["cy"]],
                                 [0, 0, 1]], dtype=np.float64)
                print(f"    [{cam_id}] query K from sensors.txt: "
                      f"fx={s['fx']:.1f} fy={s['fy']:.1f} "
                      f"cx={s['cx']:.1f} cy={s['cy']:.1f}")
            else:
                K_q = K_render
                print(f"    [{cam_id}] sensors.txt 미존재 → config K 사용")

            K_ref = _K_from_ref_entry(cam_entries[cam_id], K_render)
            c2w_cam, n_in, p2d, p3d, inlier_idx = _pnp_cam(
                mq, mr, cam_entries[cam_id], K_q, K_ref, config, cam_id)

            if cam_id == mc_primary:
                pts2d, pts3d_cam = p2d, p3d  # 시각화용

            if c2w_cam is None:
                print(f"    [{cam_id}] PnP FAILED")
                pnp_results[cam_id] = {
                    "T_W_rig": None,
                    "n_inliers": 0,
                }
                continue

            # cam c2w → rig c2w:  T_WQ_rig = c2w_cam @ T_rig_to_cam
            T_rig_to_cam = rigs.get(cam_id)
            if T_rig_to_cam is not None:
                c2w_rig = c2w_cam @ T_rig_to_cam
            else:
                c2w_rig = c2w_cam
                print(f"    [{cam_id}] rigs.txt 미존재 → cam pose 그대로 사용")

            pnp_results[cam_id] = {
                "T_W_rig":      c2w_rig,
                "T_W_cam":      c2w_cam,
                "n_inliers":    int(n_in),
                "pts2d":        p2d,
                "pts3d_cam":    p3d,
                "pnp_inlier_idx": inlier_idx,
                "entry":        cam_entries[cam_id],
                "K_query":      K_q,
                "T_rig_to_cam": T_rig_to_cam,
            }
            print(f"    [{cam_id}] rig pos={c2w_rig[:3,3].round(3)}  inliers={n_in}")

            if n_in > best_inliers:
                best_inliers = n_in
                estimated_pose = c2w_rig
                best_cam_id    = cam_id

        if estimated_pose is not None:
            trans_thr = float(pnp.get(
                "multi_cam_consistency_trans_thresh",
                config.get("multi_cam", {}).get("consistency_trans_thresh", 0.1)))
            rot_thr = float(pnp.get(
                "multi_cam_consistency_rot_thresh_deg",
                config.get("multi_cam", {}).get("consistency_rot_thresh_deg", 1.0)))
            do_joint = bool(pnp.get(
                "multi_cam_joint_refine",
                config.get("multi_cam", {}).get("joint_refine", True)))

            print(f"  Multi-cam PnP seed: best_cam={best_cam_id}  "
                  f"{best_inliers} inliers")

            ok_poses = [(cid, r) for cid, r in pnp_results.items()
                        if r.get("T_W_rig") is not None]
            consistent = []
            if len(ok_poses) > 1:
                print("  Consistency check vs best rig pose:")
                for cid, r in ok_poses:
                    c2w_rig = r["T_W_rig"]
                    d = float(np.linalg.norm(c2w_rig[:3, 3] - estimated_pose[:3, 3]))
                    a = _rotation_diff_deg(estimated_pose, c2w_rig)
                    keep = (d <= trans_thr and a <= rot_thr) or cid == best_cam_id
                    flag = "✓ use" if keep else "✗ reject"
                    print(f"    [{cid}] Δpos={d:.3f}m  Δrot={a:.2f}deg  "
                          f"{flag}  ({r['n_inliers']} inliers)")
                    if keep:
                        consistent.append((cid, r))
            else:
                consistent = ok_poses

            if not consistent and best_cam_id in pnp_results:
                consistent = [(best_cam_id, pnp_results[best_cam_id])]

            if len(consistent) >= 2:
                consistent_dict = {c: r for c, r in consistent}
                pnp_cams_used = [c for c, _ in consistent]
                poses = [r["T_W_rig"] for _, r in consistent]
                weights = [max(r["n_inliers"], 1) for _, r in consistent]
                best_pose = pnp_results[best_cam_id]["T_W_rig"]
                fused_pose = _weighted_pose_mean(poses, weights)
                print(f"  Weighted rig fusion: cams={[c for c, _ in consistent]}  "
                      f"weights={weights}")
                print(f"  Fused rig position: {fused_pose[:3,3].round(4)}")

                pose_candidates = {
                    "best": {
                        "pose": best_pose,
                        "cams_used": [best_cam_id],
                        "method": "multi_best_cam",
                    },
                    "fusion": {
                        "pose": fused_pose,
                        "cams_used": pnp_cams_used,
                        "method": "multi_fusion",
                    },
                }

                if do_joint:
                    refined_pose, stats = _joint_refine_rig_pose(
                        fused_pose, consistent_dict, config)
                    if refined_pose is not None:
                        joint_stats = stats
                        print(f"  Joint rig refine: corr={stats.get('n_corr', 0)}  "
                              f"rmse {stats.get('rmse_before', 0):.2f}"
                              f"→{stats.get('rmse_after', 0):.2f}px  "
                              f"success={stats.get('success')}")
                        pose_candidates["joint"] = {
                            "pose": refined_pose,
                            "cams_used": pnp_cams_used,
                            "method": "multi_joint_refine",
                        }
                    else:
                        print("  Joint rig refine skipped/failed → using fused pose")

                for name, cand in pose_candidates.items():
                    cand["stats"] = _rig_pose_reprojection_stats(
                        cand["pose"], consistent_dict, config)

                print("  Pose candidate reprojection scores (PnP inliers):")
                for name, cand in pose_candidates.items():
                    st = cand["stats"]
                    per = "  ".join(
                        f"{cid}:med={v['median']:.1f}px"
                        for cid, v in st.get("per_cam", {}).items())
                    print(f"    {name:<6} median={st['median']:.2f}px  "
                          f"rmse={st['rmse']:.2f}px  n={st['n']}  {per}")

                chosen_name, selection_reason = _select_multicam_pose(
                    pose_candidates, "best", best_cam_id, config)
                chosen = pose_candidates[chosen_name]
                estimated_pose = chosen["pose"]
                pnp_method = chosen["method"]
                pnp_cams_used = chosen["cams_used"]
                pose_selection = {
                    "chosen": chosen_name,
                    "reason": selection_reason,
                    "candidates": {
                        name: cand["stats"] for name, cand in pose_candidates.items()
                    },
                }
                print(f"  Adaptive pose selection: {chosen_name} "
                      f"({selection_reason})")

                inlier_count = int(sum(r["n_inliers"] for _, r in consistent))
                pnp_cam_for_viz = best_cam_id
            else:
                cid, r = consistent[0]
                estimated_pose = r["T_W_rig"]
                inlier_count = r["n_inliers"]
                pnp_cam_for_viz = cid
                pnp_cams_used = [cid]
                pnp_method = "multi_single_cam"
                print(f"  Single consistent cam → using [{cid}] rig pose")
            best_pnp_cam = pnp_cam_for_viz

            if pnp_cam_for_viz in pnp_results:
                viz = pnp_results[pnp_cam_for_viz]
                pts2d, pts3d_cam = viz["pts2d"], viz["pts3d_cam"]
                ref_entry = viz["entry"]
                T_WR = np.array(ref_entry["pose"], dtype=np.float64)
                if pnp_cam_for_viz in step6_data.get("cam_entries", {}):
                    ref_rgb = cv2.cvtColor(
                        cv2.imread(ref_entry["rgb_path"]), cv2.COLOR_BGR2RGB)

            if bool(pnp.get("line_refine", False)):
                line_matches = step6_data.get("line_matches") or {}
                line_n = int(line_matches.get("n_matches", 0))
                line_cam = line_matches.get("cam_id")
                if line_n > 0 and line_cam in (None, pnp_cam_for_viz):
                    refined_pose, stats = _refine_rig_pose_points_lines(
                        estimated_pose, pnp_results[pnp_cam_for_viz],
                        line_matches, K_render, config)
                    line_refine_stats = stats
                    if (refined_pose is not None and stats.get("success", False)
                            and stats.get("cost_after", float("inf"))
                            <= stats.get("cost_before", float("inf"))):
                        estimated_pose = refined_pose
                        pnp_method = f"{pnp_method}+line_refine"
                        print("  Multi-cam point+line refine: "
                              f"cam={pnp_cam_for_viz}  "
                              f"points={stats['n_points']}  lines={stats['n_lines']}  "
                              f"rmse {stats['rmse_before']:.2f}"
                              f"→{stats['rmse_after']:.2f}px")
                    else:
                        print(f"  Multi-cam point+line refine skipped: "
                              f"{stats.get('reason', 'optimization_failed')}")
                elif line_n > 0:
                    print(f"  Multi-cam point+line refine skipped: "
                          f"SOLD2 cam={line_cam}, PnP cam={pnp_cam_for_viz}")
                else:
                    print("  Multi-cam point+line refine skipped: no SOLD2 line matches")

            print(f"  Multi-cam rig position: {estimated_pose[:3,3].round(4)}")
        else:
            print("  Multi-cam PnP 전 cam 실패 → single-cam fallback")
            is_multi_pnp = False

    if not is_multi_pnp:
        # ── 단일 캠 경로 ─────────────────────────────────────────────────
        matched_q   = step6_data["matched_q_kps"]
        matched_r   = step6_data["matched_r_kps"]
        _pnp_cam_id = step6_data.get("best_cam_id") or step6_data.get("mc_primary")
        best_pnp_cam = _pnp_cam_id
        pnp_cams_used = [_pnp_cam_id] if _pnp_cam_id else []
        dmin = cam.get("depth_min", 0.3)
        dmax = cam.get("depth_max", 20.0)

        # ref_entry/T_WR: pnp_cam의 실제 렌더 entry 사용
        # sub cam fallback 시 best_cand(primary 렌더)와 matched_r의 렌더가 다를 수 있음
        _cam_entries = step6_data.get("cam_entries", {})
        if _pnp_cam_id and _pnp_cam_id in _cam_entries:
            ref_entry = _cam_entries[_pnp_cam_id]
            T_WR      = np.array(ref_entry["pose"], dtype=np.float64)
            print(f"  Ref entry: [{_pnp_cam_id}] render #{ref_entry['id']}")

        # K_render: 렌더 intrinsic (항상 config 파라미터)
        # K_query : 실제 쿼리 cam intrinsic (sensors.txt 우선, fallback = config K)
        K_render = K
        K_query  = K
        if _pnp_cam_id:
            _, _, _kap_dir, _ = load_multi_cam_config(config)
            if not os.path.isabs(_kap_dir):
                _kap_dir = os.path.join(os.getcwd(), _kap_dir)
            if os.path.isdir(_kap_dir):
                _sensors = parse_kapture_sensors(_kap_dir)
                s = _sensors.get(_pnp_cam_id)
                if s:
                    K_query = np.array([[s["fx"], 0, s["cx"]],
                                        [0, s["fy"], s["cy"]],
                                        [0, 0, 1]], dtype=np.float64)
                    print(f"  Query cam [{_pnp_cam_id}] K from sensors.txt: "
                          f"fx={s['fx']:.1f}  fy={s['fy']:.1f}  "
                          f"cx={s['cx']:.1f}  cy={s['cy']:.1f}")

        K_render = _K_from_ref_entry(ref_entry, K_render)
        dep = _load_ref_depth(ref_entry)
        if dep is None:
            dep = np.zeros((0, 0), dtype=np.float32)
        # 3D lifting은 K_render 사용 (렌더는 항상 config 파라미터로 생성)
        pts2d_list = []; pts3d_list = []; invalid = 0
        for i in range(len(matched_q)):
            qu, qv = matched_q[i]
            ru, rv = matched_r[i]
            ri, rj = int(round(rv)), int(round(ru))
            if not (0 <= ri < dep.shape[0] and 0 <= rj < dep.shape[1]):
                invalid += 1; continue
            pz = float(dep[ri, rj])
            if not np.isfinite(pz) or pz < dmin or pz > dmax:
                invalid += 1; continue
            pts2d_list.append([qu, qv])
            pts3d_list.append([(ru - K_render[0,2]) * pz / K_render[0,0],
                                (rv - K_render[1,2]) * pz / K_render[1,1],
                                pz])

        pts2d     = np.array(pts2d_list,  dtype=np.float64)
        pts3d_cam = np.array(pts3d_list,  dtype=np.float64)

        print(f"  Matched pairs      : {len(matched_q)}")
        print(f"  Valid (depth OK)   : {len(pts2d)}  (invalid: {invalid})")
        print(f"  Ref pose           : {T_WR[:3,3].round(3)}")
        print(f"  Solver             : {solver}  refine={do_refine}")

        if len(pts2d) < min_in:
            print(f"  PnP SKIPPED: {len(pts2d)} < {min_in} correspondences")

        elif solver == "superansac":
            pts3d_h     = np.hstack([pts3d_cam, np.ones((len(pts3d_cam), 1))])
            pts3d_world = (T_WR @ pts3d_h.T).T[:, :3]
            T_WQ_raw, inlier_idx = _solve_pnp_superansac(
                pts2d, pts3d_world, K_query, config)

            if T_WQ_raw is not None and len(inlier_idx) >= min_in:
                inlier_count = len(inlier_idx)
                if do_refine:
                    T_RW = np.linalg.inv(T_WR)
                    pts3d_cam_in = (T_RW @ np.hstack([
                        pts3d_world[inlier_idx], np.ones((inlier_count, 1))]).T).T[:, :3]
                    T_QR_init = np.linalg.inv(T_WQ_raw) @ T_WR
                    rvec_i, _ = cv2.Rodrigues(T_QR_init[:3, :3])
                    tvec_i    = T_QR_init[:3, 3].reshape(3, 1)
                    rvec_r, tvec_r = _refine_pnp(
                        pts2d[inlier_idx], pts3d_cam_in, K_query, rvec_i, tvec_i)
                    R_r, _ = cv2.Rodrigues(rvec_r)
                    T_QR_r = np.eye(4); T_QR_r[:3, :3] = R_r; T_QR_r[:3, 3] = tvec_r.flatten()
                    estimated_pose = T_WR @ np.linalg.inv(T_QR_r)
                    print(f"  Refine applied (SuperANSAC → ITERATIVE)")
                else:
                    estimated_pose = T_WQ_raw
                print(f"  SuperANSAC SUCCESS: {inlier_count}/{len(pts2d)} inliers")
                print(f"  T_WQ position: {estimated_pose[:3,3].round(4)}")
            else:
                print("  SuperANSAC FAILED: insufficient inliers")

        else:  # opencv
            T_QR_raw, rvec, inlier_idx = _solve_pnp_opencv(
                pts2d, pts3d_cam, K_query, config)
            if T_QR_raw is not None and len(inlier_idx) >= min_in:
                inlier_count = len(inlier_idx)
                T_QR = T_QR_raw
                if do_refine:
                    rvec_r, tvec_r = _refine_pnp(
                        pts2d[inlier_idx], pts3d_cam[inlier_idx], K_query, rvec,
                        T_QR_raw[:3, 3].reshape(3, 1))
                    R_r, _ = cv2.Rodrigues(rvec_r)
                    T_QR = np.eye(4); T_QR[:3, :3] = R_r; T_QR[:3, 3] = tvec_r.flatten()
                    print(f"  Refine applied (EPnP → ITERATIVE)")
                estimated_pose = T_WR @ np.linalg.inv(T_QR)
                print(f"  OpenCV PnP SUCCESS: {inlier_count}/{len(pts2d)} inliers")
                print(f"  T_WQ position: {estimated_pose[:3,3].round(4)}")
            else:
                print("  OpenCV PnP FAILED: insufficient inliers")

        if estimated_pose is not None and bool(pnp.get("line_refine", False)):
            line_matches = step6_data.get("line_matches") or {}
            line_n = int(line_matches.get("n_matches", 0))
            if line_n > 0:
                # PnP inliers만 point residual에 사용하고, SOLD2 lines는 별도 line residual로 사용.
                if len(inlier_idx) > 0:
                    p2d_ref = pts2d[inlier_idx]
                    p3d_ref_cam = pts3d_cam[inlier_idx]
                else:
                    p2d_ref = pts2d
                    p3d_ref_cam = pts3d_cam
                if len(p3d_ref_cam) > 0:
                    p3d_ref_h = np.hstack([
                        p3d_ref_cam, np.ones((len(p3d_ref_cam), 1))])
                    p3d_ref_world = (T_WR @ p3d_ref_h.T).T[:, :3]
                else:
                    p3d_ref_world = np.zeros((0, 3), dtype=np.float64)

                q_lines, line3d_world = _lift_line_matches_to_world(
                    line_matches, ref_entry, K_render, dmin, dmax)
                refined_pose, stats = _refine_pose_points_lines(
                    estimated_pose, p2d_ref, p3d_ref_world,
                    q_lines, line3d_world, K_query, config)
                line_refine_stats = stats
                if refined_pose is not None and stats.get("success", False):
                    estimated_pose = refined_pose
                    pnp_method = f"{pnp_method}+line_refine"
                    print("  Point+line refine: "
                          f"points={stats['n_points']}  lines={stats['n_lines']}  "
                          f"rmse {stats['rmse_before']:.2f}"
                          f"→{stats['rmse_after']:.2f}px  "
                          f"pos={estimated_pose[:3,3].round(4)}")
                else:
                    print(f"  Point+line refine skipped: "
                          f"{stats.get('reason', 'optimization_failed')}")
            else:
                print("  Point+line refine skipped: no SOLD2 line matches")

    # ── Rig frame 정규화 (single-cam 경로) ──────────────────────────────────
    # multi-cam PnP 경로는 이미 rig frame으로 변환됨.
    # single-cam 경로는 해당 cam의 c2w이므로 rig frame으로 변환해야
    # eval에서 카메라 간 frame 불일치 없이 비교 가능.
    if estimated_pose is not None and not is_multi_pnp:
        # best_cam_id = step6에서 실제 PnP에 사용된 cam (매칭 수 최대)
        # mc_primary  = avg_sim 기준 primary (실제 사용 cam과 다를 수 있음)
        _used_cam = step6_data.get("best_cam_id") or step6_data.get("mc_primary")
        if _used_cam:
            _, _, _kap_dir, _ = load_multi_cam_config(config)
            if not os.path.isabs(_kap_dir):
                _kap_dir = os.path.join(os.getcwd(), _kap_dir)
            if os.path.isdir(_kap_dir):
                _rigs = parse_kapture_rigs(_kap_dir)
                _T_rc = _rigs.get(_used_cam)
                if _T_rc is not None:
                    # c2w_rig = c2w_cam @ T_rig_to_cam
                    estimated_pose = estimated_pose @ _T_rc
                    print(f"  → rig frame (cam={_used_cam})")

    if gt_entry is not None and estimated_pose is not None:
        err = np.linalg.norm(
            estimated_pose[:3, 3] - np.array(gt_entry["pose"])[:3, 3])
        print(f"  GT  position  : {np.array(gt_entry['pose'])[:3, 3].round(4)}")
        print(f"  Position error: {err:.4f} m")

    if save_images:
        # ── 시각화 ───────────────────────────────────────────────────────
        if is_multi_pnp and pnp_results:
            query_images = step5_data.get("query_images") or {}
            cam_rows = [c for c in active_cams if c in pnp_results]
            if not cam_rows:
                cam_rows = list(pnp_results.keys())

            fig = plt.figure(figsize=(18, 4.1 * len(cam_rows) + 5.0))
            gs_ = GridSpec(len(cam_rows) + 1, 3, figure=fig,
                           height_ratios=([1.0] * len(cam_rows)) + [1.25],
                           hspace=0.45, wspace=0.22)

            for row, cam_id in enumerate(cam_rows):
                r = pnp_results.get(cam_id, {})
                q_path = query_images.get(cam_id)
                if q_path and os.path.isfile(q_path):
                    q_img = cv2.cvtColor(cv2.imread(q_path), cv2.COLOR_BGR2RGB)
                else:
                    q_img = query_rgb

                entry = r.get("entry") or cam_entries.get(cam_id)
                if entry is not None and os.path.isfile(entry["rgb_path"]):
                    r_img = cv2.cvtColor(cv2.imread(entry["rgb_path"]), cv2.COLOR_BGR2RGB)
                    ref_title = f"{cam_id} render #{entry['id']}"
                else:
                    r_img = np.zeros_like(q_img)
                    ref_title = f"{cam_id} render unavailable"

                p2d = r.get("pts2d", np.zeros((0, 2)))
                p3d = r.get("pts3d_cam", np.zeros((0, 3)))
                n_in = r.get("n_inliers", 0)
                status = "used" if cam_id in pnp_cams_used else "rejected"
                if r.get("T_W_rig") is None:
                    status = "PnP failed"
                star = " ★" if cam_id == best_pnp_cam else ""

                ax = fig.add_subplot(gs_[row, 0])
                ax.imshow(q_img)
                if len(p2d) > 0:
                    ax.scatter(p2d[:, 0], p2d[:, 1], c="red", s=6,
                               marker="x", alpha=0.55)
                ax.set_title(f"[{cam_id}{star}] query  corr={len(p2d)}  "
                             f"inliers={n_in}  {status}", fontsize=9)
                ax.axis("off")

                ax = fig.add_subplot(gs_[row, 1])
                ax.imshow(r_img)
                ax.set_title(ref_title, fontsize=9)
                ax.axis("off")

                ax = fig.add_subplot(gs_[row, 2])
                if len(p3d) > 0:
                    ax.scatter(p3d[:, 0], p3d[:, 2],
                               c=p3d[:, 1], cmap="RdYlBu", s=6, alpha=0.8)
                if r.get("T_W_rig") is not None:
                    rp = r["T_W_rig"][:3, 3]
                    ax.text(0.02, 0.95,
                            f"rig pos=({rp[0]:.2f}, {rp[1]:.2f}, {rp[2]:.2f})",
                            transform=ax.transAxes, va="top", fontsize=8)
                ax.set_title(f"[{cam_id}] lifted 3D: X vs Z", fontsize=9)
                ax.set_xlabel("X"); ax.set_ylabel("Z")

            ax = fig.add_subplot(gs_[len(cam_rows), :])
            cands = step5_data["candidates"]
            all_pos = np.array([e["pose"][:3, 3] for e in cands])
            ax.scatter(all_pos[:, 0], all_pos[:, 1],
                       c="lightgray", s=8, alpha=0.4, label="Retrieval candidates")

            for cam_id in cam_rows:
                r = pnp_results.get(cam_id, {})
                entry = r.get("entry") or cam_entries.get(cam_id)
                if entry is not None:
                    ref_pos = np.array(entry["pose"])[:3, 3]
                    ax.scatter(ref_pos[0], ref_pos[1], s=150, marker="D",
                               alpha=0.8, label=f"{cam_id} ref #{entry['id']}")
                if r.get("T_W_rig") is not None:
                    rp = r["T_W_rig"][:3, 3]
                    color = "tab:green" if cam_id in pnp_cams_used else "tab:gray"
                    ax.scatter(rp[0], rp[1], c=color, s=170, marker="x",
                               linewidths=2.5, label=f"{cam_id} rig PnP")
                    if estimated_pose is not None:
                        ep = estimated_pose[:3, 3]
                        ax.plot([rp[0], ep[0]], [rp[1], ep[1]],
                                color=color, alpha=0.45, linestyle=":")

            if gt_entry is not None:
                gp = np.array(gt_entry["pose"])[:3, 3]
                ax.scatter(gp[0], gp[1], c="blue", s=260, marker="*",
                           zorder=6, label="GT query")
            if estimated_pose is not None:
                ep = estimated_pose[:3, 3]
                ax.scatter(ep[0], ep[1], c="red", s=260, marker="^",
                           zorder=7, label=f"Final rig ({pnp_method})")
                if gt_entry is not None:
                    gp = np.array(gt_entry["pose"])[:3, 3]
                    err = np.linalg.norm(ep - gp)
                    ax.plot([ep[0], gp[0]], [ep[1], gp[1]], "b:", alpha=0.7,
                            label=f"err={err:.3f}m")

            ax.set_aspect("equal")
            ax.legend(fontsize=7, loc="best", ncols=2)
            ax.set_title("Top-down multi-cam rig pose: refs, per-cam PnP, final fused/joint")

        else:
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
            f"Step 7: PnP [{solver}{'→refine' if do_refine else ''}] "
            f"{pnp_method} — {status} ({inlier_count} inliers, "
            f"{len(pts2d)} corr){err_str}",
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
        "pnp_method":        pnp_method,
        "pnp_cams_used":     pnp_cams_used,
        "best_pnp_cam":      best_pnp_cam,
        "joint_refine_stats": joint_stats,
        "line_refine_stats": line_refine_stats,
        "pose_selection":    pose_selection,
    }
    if save_images:
        pickle.dump(result, open(os.path.join(output_dir, "step7_data.pkl"), "wb"))
    return result
