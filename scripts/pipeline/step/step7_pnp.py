import os, pickle
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .multi_cam import parse_kapture_rigs, parse_kapture_sensors, load_multi_cam_config


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


# ── Multi-cam PnP helper ───────────────────────────────────────────────────

def _lift_2d_to_3d(mkpts_r, ref_entry, K_render, dmin, dmax):
    """렌더 이미지의 2D 매칭점 → ref cam frame 3D점 (depth lifting)."""
    dep = np.load(ref_entry["depth_path"])
    fx_r, fy_r = K_render[0, 0], K_render[1, 1]
    cx_r, cy_r = K_render[0, 2], K_render[1, 2]
    pts3d_cam = []
    valid_idx = []
    for i, (ru, rv) in enumerate(mkpts_r):
        ri, rj = int(round(rv)), int(round(ru))
        if not (0 <= ri < dep.shape[0] and 0 <= rj < dep.shape[1]):
            continue
        pz = float(dep[ri, rj])
        if not np.isfinite(pz) or pz < dmin or pz > dmax:
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
    """
    pnp = config.get("pnp", {})
    dmin, dmax = float(config["camera"].get("depth_min", 0.3)), \
                 float(config["camera"].get("depth_max", 20.0))
    solver    = pnp.get("solver", "opencv").strip().lower()
    do_refine = bool(pnp.get("refine", True))
    min_in    = int(pnp.get("min_inliers", 6))

    pts3d_cam, valid_idx = _lift_2d_to_3d(mkpts_r, ref_entry, K_render, dmin, dmax)
    if len(pts3d_cam) < min_in:
        return None, 0, np.zeros((0, 2)), pts3d_cam

    pts2d = mkpts_q[valid_idx]
    T_WR  = np.array(ref_entry["pose"], dtype=np.float64)

    if solver == "superansac":
        pts3d_h     = np.hstack([pts3d_cam, np.ones((len(pts3d_cam), 1))])
        pts3d_world = (T_WR @ pts3d_h.T).T[:, :3]
        T_WQ_raw, inlier_idx = _solve_pnp_superansac(pts2d, pts3d_world, K_query, config)
        if T_WQ_raw is None or len(inlier_idx) < min_in:
            return None, 0, pts2d, pts3d_cam
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
            return None, 0, pts2d, pts3d_cam
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
    return c2w_cam, inlier_n, pts2d, pts3d_cam


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

    if is_multi_pnp:
        # ── 각 카메라 독립 PnP → rig frame 통일 → 최고 inlier 선택 ──────
        _, _, kapture_dir, _ = load_multi_cam_config(config)
        if not os.path.isabs(kapture_dir):
            kapture_dir = os.path.join(os.getcwd(), kapture_dir)

        rigs    = parse_kapture_rigs(kapture_dir)     # {cam_id: T_rig_to_cam}
        sensors = parse_kapture_sensors(kapture_dir)  # {cam_id: {fx,fy,cx,cy,...}}

        print(f"  Mode: multi-cam PnP  active={active_cams}  primary={mc_primary}")
        print(f"  Solver: {solver}  refine={do_refine}")

        pnp_results = {}  # {cam_id: (c2w_rig, n_inliers)}
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
            else:
                K_q = K_render
                print(f"    [{cam_id}] sensors.txt 미존재 → config K 사용")

            c2w_cam, n_in, p2d, p3d = _pnp_cam(
                mq, mr, cam_entries[cam_id], K_q, K_render, config, cam_id)

            if cam_id == mc_primary:
                pts2d, pts3d_cam = p2d, p3d  # 시각화용

            if c2w_cam is None:
                print(f"    [{cam_id}] PnP FAILED")
                pnp_results[cam_id] = (None, 0)
                continue

            # cam c2w → rig c2w:  T_WQ_rig = c2w_cam @ T_rig_to_cam
            T_rig_to_cam = rigs.get(cam_id)
            if T_rig_to_cam is not None:
                c2w_rig = c2w_cam @ T_rig_to_cam
            else:
                c2w_rig = c2w_cam
                print(f"    [{cam_id}] rigs.txt 미존재 → cam pose 그대로 사용")

            pnp_results[cam_id] = (c2w_rig, n_in)
            print(f"    [{cam_id}] rig pos={c2w_rig[:3,3].round(3)}  inliers={n_in}")

            if n_in > best_inliers:
                best_inliers = n_in
                estimated_pose = c2w_rig
                best_cam_id    = cam_id

        if estimated_pose is not None:
            inlier_count = best_inliers
            print(f"  Multi-cam PnP BEST: cam={best_cam_id}  {inlier_count} inliers")
            print(f"  Rig position: {estimated_pose[:3,3].round(4)}")

            # 카메라 간 일관성 체크
            ok_poses = [(cid, r) for cid, r in pnp_results.items() if r[0] is not None]
            if len(ok_poses) > 1:
                print("  Consistency check:")
                for cid, (c2w_r, n) in ok_poses:
                    d = np.linalg.norm(c2w_r[:3,3] - estimated_pose[:3,3])
                    flag = "✓" if d < 0.5 else "✗ inconsistent"
                    print(f"    [{cid}] dist_to_best={d:.3f}m  {flag}  ({n} inliers)")
        else:
            print("  Multi-cam PnP 전 cam 실패 → single-cam fallback")
            is_multi_pnp = False

    if not is_multi_pnp:
        # ── 단일 캠 경로 ─────────────────────────────────────────────────
        matched_q   = step6_data["matched_q_kps"]
        matched_r   = step6_data["matched_r_kps"]
        _pnp_cam_id = step6_data.get("best_cam_id") or step6_data.get("mc_primary")
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

        dep = np.load(ref_entry["depth_path"])
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
