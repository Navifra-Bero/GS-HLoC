#!/usr/bin/env python3
"""
kapture_localize.py
===================
kapture 포맷으로 저장된 실제 카메라 데이터(mapping)를 DB로,
test_data의 실제 쿼리 이미지를 query로 사용하는 localization 파이프라인.

real DB vs real Query → domain gap 없음.

단계:
  step2  : kapture_mapping 이미지 샘플 시각화 (2×6 grid)
  step3  : SuperPoint 특징점 추출
  step4  : Global descriptor (NetVLAD/EigenPlaces)
  step5  : Depth → 3D keypoints (backproject)
  step6  : KDTree 데이터베이스 구축
  step7a : Global retrieval
  step7b : Local feature extraction (query + ref)
  step7c : SuperGlue matching
  step7d : 2D-3D correspondence
  step7e : PnP pose estimation

사용법:
  python3 scripts/kapture_localize.py [--step STEP] [--query PATH] [--subsample N]
"""

import os, sys, argparse, pickle, json, math
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── 경로 기본값 ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)
MAPPING_DIR  = os.path.join(ROOT_DIR, "kapture_mapping", "sensors")
QUERY_DIR    = os.path.join(ROOT_DIR, "test_data")
OUTPUT_DIR   = os.path.join(ROOT_DIR, "output", "kapture_localize")
MODELS_DIR   = os.path.join(ROOT_DIR, "models")

# ─── 기본 설정 ────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "camera": {
        "depth_min": 0.3,
        "depth_max": 50.0,
    },
    "features": {
        "max_keypoints":        1024,
        "keypoint_threshold":   0.005,
        "global_desc_dim":      512,
        "superpoint_model":     os.path.join(MODELS_DIR, "superpoint_v1.pt"),
        "global_model":         os.path.join(MODELS_DIR, "netvlad.pt"),
        "superglue_model":      os.path.join(MODELS_DIR, "superglue_indoor.pth"),
    },
    "online": {
        "top_k":                5,
        "reprojection_error":   8.0,
        "pnp_iterations":       1000,
        "pnp_confidence":       0.99,
    },
}

CAM_LABELS = {
    "mapping_cam_0": "cam_0",
    "mapping_cam_1": "cam_1",
    "mapping_cam_2": "cam_2",
    "mapping_cam_3": "cam_3",
}

# =============================================================================
# kapture 파싱 유틸
# =============================================================================

def quat_to_rot(qw, qx, qy, qz):
    """쿼터니언 → 3×3 회전행렬 (world←camera)"""
    r = np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])
    return r

def pose_to_T(qw, qx, qy, qz, tx, ty, tz):
    """kapture pose (T_world_camera) → 4×4 행렬"""
    R = quat_to_rot(qw, qx, qy, qz)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [tx, ty, tz]
    return T

def parse_sensors(sensors_txt):
    """sensors.txt → {sensor_id: {fx,fy,cx,cy,w,h}}"""
    cams = {}
    for line in open(sensors_txt):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 10 or parts[2] != "camera":
            continue
        sid = parts[0]
        cams[sid] = {
            "w":  int(parts[4]),
            "h":  int(parts[5]),
            "fx": float(parts[6]),
            "fy": float(parts[7]),
            "cx": float(parts[8]),
            "cy": float(parts[9]),
        }
    return cams

def parse_trajectories(traj_txt):
    """trajectories.txt → {(timestamp_str, device_id): T_wc (4×4)}"""
    poses = {}
    for line in open(traj_txt):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        ts, dev = parts[0], parts[1]
        qw, qx, qy, qz = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        tx, ty, tz      = float(parts[6]), float(parts[7]), float(parts[8])
        poses[(ts, dev)] = pose_to_T(qw, qx, qy, qz, tx, ty, tz)
    return poses

def parse_records(records_txt):
    """records_camera.txt → [(timestamp, device_id, filename)]"""
    records = []
    for line in open(records_txt):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        records.append((parts[0], parts[1], parts[2]))
    return records

def load_depth(depth_path, h, w):
    """float32 binary depth → (h, w) ndarray"""
    raw = np.fromfile(depth_path, dtype=np.float32)
    if raw.size == h * w:
        return raw.reshape(h, w)
    # fallback: try PNG visualization
    vis = depth_path.replace(".depth", "_vis.png")
    if os.path.exists(vis):
        return cv2.imread(vis, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    return np.zeros((h, w), dtype=np.float32)

def load_mapping_entries(mapping_dir, subsample=1):
    """
    kapture_mapping 전체 파싱 → list of dicts
    {id, timestamp, cam_id, rgb_path, depth_path, pose (T_wc), cam_params}
    subsample: N번째마다 하나씩 선택
    """
    sensors_txt  = os.path.join(mapping_dir, "sensors.txt")
    traj_txt     = os.path.join(mapping_dir, "trajectories.txt")
    records_txt  = os.path.join(mapping_dir, "records_camera.txt")
    data_dir     = os.path.join(mapping_dir, "records_data")

    cams    = parse_sensors(sensors_txt)
    poses   = parse_trajectories(traj_txt)
    records = parse_records(records_txt)

    # timestamp별로 그룹화 (4 cams per timestamp)
    from collections import defaultdict
    ts_groups = defaultdict(list)
    for ts, dev, fname in records:
        ts_groups[ts].append((dev, fname))

    timestamps = sorted(ts_groups.keys())
    entries = []
    idx = 0
    for i, ts in enumerate(timestamps):
        if i % subsample != 0:
            continue
        for dev, fname in sorted(ts_groups[ts]):
            if (ts, dev) not in poses:
                continue
            cam_id = dev  # e.g. "mapping_cam_0"
            if cam_id not in cams:
                continue
            rgb_path   = os.path.join(data_dir, fname)
            # depth: 같은 timestamp, depth 폴더
            depth_fname = fname.replace("/images/", "/depths/").replace(".jpg", ".depth")
            depth_path  = os.path.join(data_dir, depth_fname)
            if not os.path.exists(rgb_path):
                continue
            entries.append({
                "id":          idx,
                "timestamp":   ts,
                "cam_id":      cam_id,
                "rgb_path":    rgb_path,
                "depth_path":  depth_path if os.path.exists(depth_path) else None,
                "pose":        poses[(ts, dev)],
                "cam_params":  cams[cam_id],
            })
            idx += 1

    return entries

def load_query_entries(query_dir):
    """
    test_data kapture 파싱 → list of query dicts
    (trajectories.txt가 없으면 pose=None)
    """
    records_txt = os.path.join(query_dir, "records_camera.txt")
    traj_txt    = os.path.join(query_dir, "trajectories.txt")
    sensors_txt = os.path.join(query_dir, "sensors.txt")
    data_dir    = os.path.join(query_dir, "records_data")

    cams    = parse_sensors(sensors_txt) if os.path.exists(sensors_txt) else {}
    poses   = parse_trajectories(traj_txt) if os.path.exists(traj_txt) else {}
    records = parse_records(records_txt) if os.path.exists(records_txt) else []

    entries = []
    for i, (ts, dev, fname) in enumerate(records):
        rgb_path = os.path.join(data_dir, fname)
        if not os.path.exists(rgb_path):
            continue
        # depth
        depth_fname = fname.replace("/images/", "/depths/").replace(".jpg", ".depth")
        depth_path  = os.path.join(data_dir, depth_fname)
        cam_params  = cams.get(dev)
        pose        = poses.get((ts, dev))
        entries.append({
            "id":         i,
            "timestamp":  ts,
            "cam_id":     dev,
            "rgb_path":   rgb_path,
            "depth_path": depth_path if os.path.exists(depth_path) else None,
            "pose":       pose,
            "cam_params": cam_params,
        })
    return entries

# =============================================================================
# STEP 2: 이미지 샘플 시각화
# =============================================================================

def step2_visualize(entries, output_dir):
    print("\n" + "="*60 + "\nSTEP 2: Mapping 데이터 시각화\n" + "="*60)
    os.makedirs(output_dir, exist_ok=True)

    # 6개 timestamp 샘플 (균등 간격)
    # entries는 timestamp×4cam 순서로 들어있음
    timestamps = []
    seen = set()
    for e in entries:
        if e["timestamp"] not in seen:
            timestamps.append(e["timestamp"])
            seen.add(e["timestamp"])

    n_ts = min(6, len(timestamps))
    idxs = np.linspace(0, len(timestamps)-1, n_ts, dtype=int)
    sel_ts = [timestamps[i] for i in idxs]

    # 2행 × 6열: row0=RGB, row1=Depth(vis)
    fig, axes = plt.subplots(2, n_ts, figsize=(4*n_ts, 8))
    if n_ts == 1:
        axes = axes.reshape(2, 1)

    for col, ts in enumerate(sel_ts):
        # 해당 timestamp의 cam_0 (또는 첫 번째 cam) 찾기
        candidates = [e for e in entries if e["timestamp"] == ts]
        e = candidates[0]  # cam_0

        rgb = cv2.cvtColor(cv2.imread(e["rgb_path"]), cv2.COLOR_BGR2RGB)
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"{e['cam_id']}\n{ts[:10]}", fontsize=7)
        axes[0, col].axis("off")

        # depth
        cam = e["cam_params"] or {"h": 1200, "w": 1920}
        h, w = cam.get("h", 1200), cam.get("w", 1920)
        if e["depth_path"]:
            dep = load_depth(e["depth_path"], h, w)
            im = axes[1, col].imshow(dep, cmap="plasma",
                                     vmin=0, vmax=20)
            plt.colorbar(im, ax=axes[1, col], fraction=0.046)
        else:
            axes[1, col].text(0.5, 0.5, "No depth", ha="center", va="center")
        axes[1, col].axis("off")

    fig.suptitle(f"Step 2: Mapping 데이터 — {len(entries)} entries "
                 f"({len(timestamps)} timestamps × 4 cams)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, "step2_mapping.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Entries  : {len(entries)}")
    print(f"  Timestamps: {len(timestamps)}")
    print(f"  Saved: {out}")
    return entries

# =============================================================================
# STEP 3: SuperPoint
# =============================================================================

def _extract_superpoint(sp_model, rgb, mk, kt, dev):
    """SuperPoint 추출 헬퍼. 출력: kp_xy (N,2), descriptors (256,N), scores (N,)"""
    import torch
    from scipy.ndimage import maximum_filter as max_filter
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    inp  = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        sc, dm = sp_model(inp)          # sc:(1,H,W)  dm:(1,256,H/8,W/8)
    s  = sc[0].cpu().numpy()           # (H, W)
    lm = max_filter(s, size=5)
    m  = (s == lm) & (s > kt)
    ys, xs = np.where(m)
    ks = s[m]
    if len(xs) > mk:
        top = np.argsort(ks)[::-1][:mk]
        xs, ys, ks = xs[top], ys[top], ks[top]
    d  = dm[0].cpu().numpy()           # (256, H/8, W/8)
    dh, dw = d.shape[1], d.shape[2]
    if len(xs) > 0:
        descs = np.array([d[:, min(int(y/8), dh-1), min(int(x/8), dw-1)]
                          for x, y in zip(xs, ys)])   # (N, 256)
        descs = descs.T                                # (256, N)
    else:
        descs = np.zeros((256, 0), dtype=np.float32)
    kps = np.stack([xs, ys], axis=1).astype(np.float32) if len(xs) > 0 else np.zeros((0, 2))
    return kps, descs, ks


def step3_superpoint(entries, config, output_dir):
    print("\n" + "="*60 + "\nSTEP 3: SuperPoint\n" + "="*60)
    import torch
    fc  = config["features"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mk  = fc.get("max_keypoints", 1024)
    kt  = fc.get("keypoint_threshold", 0.005)

    sp_model = torch.jit.load(fc["superpoint_model"], map_location=dev)
    sp_model.eval()

    total_kp = 0
    for i, e in enumerate(entries):
        rgb = cv2.cvtColor(cv2.imread(e["rgb_path"]), cv2.COLOR_BGR2RGB)
        kps, descs, scores = _extract_superpoint(sp_model, rgb, mk, kt, dev)

        e["kp_xy"]     = kps      # (N, 2)
        e["kp_desc"]   = descs    # (256, N)
        e["kp_scores"] = scores   # (N,)
        total_kp += len(kps)

        if (i+1) % 100 == 0:
            print(f"  [{i+1}/{len(entries)}] avg kp so far: {total_kp/(i+1):.0f}")

    avg = total_kp / max(len(entries), 1)
    print(f"  Total entries: {len(entries)}, avg keypoints: {avg:.0f}")

    # 시각화: 6개 샘플
    n = min(6, len(entries))
    idxs = np.linspace(0, len(entries)-1, n, dtype=int)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: axes = [axes]
    for col, idx in enumerate(idxs):
        e = entries[idx]
        rgb = cv2.cvtColor(cv2.imread(e["rgb_path"]), cv2.COLOR_BGR2RGB)
        axes[col].imshow(rgb)
        kps = e["kp_xy"]
        if len(kps) > 0:
            axes[col].scatter(kps[:, 0], kps[:, 1], c="lime", s=3, alpha=0.7)
        axes[col].set_title(f"#{e['id']} kp={len(kps)}", fontsize=8)
        axes[col].axis("off")
    fig.suptitle(f"Step 3: SuperPoint — avg {avg:.0f} kp/img", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, "step3_superpoint.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    pickle.dump(entries, open(os.path.join(output_dir, "step3_data.pkl"), "wb"))
    return entries

# =============================================================================
# STEP 4: Global Descriptors
# =============================================================================

def step4_global_desc(entries, config, output_dir):
    print("\n" + "="*60 + "\nSTEP 4: Global Descriptors\n" + "="*60)
    import torch
    fc  = config["features"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gd  = fc.get("global_desc_dim", 512)

    try:
        gm = torch.jit.load(fc["global_model"], map_location=dev)
        gm.eval()
        use_model = True
        print(f"  Model: {fc['global_model']}")
    except Exception as ex:
        print(f"  Model load failed ({ex}) → fallback: HSV histogram")
        use_model = False

    for i, e in enumerate(entries):
        rgb = cv2.cvtColor(cv2.imread(e["rgb_path"]), cv2.COLOR_BGR2RGB)
        if use_model:
            t = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
            t = torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0).to(dev)
            with torch.no_grad():
                d = gm(t).cpu().numpy().flatten()
        else:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            d = np.concatenate([
                cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten(),
                cv2.calcHist([hsv], [1], None, [64], [0, 256]).flatten(),
            ])
            d = d / (np.linalg.norm(d) + 1e-8)
        d = d[:gd]
        if len(d) < gd:
            d = np.pad(d, (0, gd - len(d)))
        e["global_descriptor"] = d
        if (i+1) % 200 == 0:
            print(f"  [{i+1}/{len(entries)}]")

    # 시각화
    nv  = min(200, len(entries))
    idx = np.linspace(0, len(entries)-1, nv, dtype=int)
    dm  = np.array([entries[i]["global_descriptor"] for i in idx])
    dm  = dm / (np.linalg.norm(dm, axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    im = ax[0].imshow(dm @ dm.T, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax[0].set_title(f"Similarity {nv}×{nv}")
    plt.colorbar(im, ax=ax[0], fraction=0.046)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit_transform(dm)
    ps  = np.array([entries[i]["pose"][:3, 3] for i in idx])
    ax[1].scatter(pca[:, 0], pca[:, 1], c=ps[:, 0], cmap="viridis", s=10)
    ax[1].set_title("PCA (color=X)")
    fig.suptitle("Step 4: Global Descriptors", fontsize=14)
    fig.tight_layout()
    out = os.path.join(output_dir, "step4_global_desc.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    pickle.dump(entries, open(os.path.join(output_dir, "step4_data.pkl"), "wb"))
    return entries

# =============================================================================
# STEP 5: Backproject
# =============================================================================

def step5_backproject(entries, config, output_dir):
    print("\n" + "="*60 + "\nSTEP 5: Backproject\n" + "="*60)
    cfg   = config["camera"]
    dmin  = cfg.get("depth_min", 0.3)
    dmax  = cfg.get("depth_max", 50.0)
    tk = vk = 0
    all3d = []

    for e in entries:
        cam = e.get("cam_params") or {}
        fx, fy = cam.get("fx", 1027.0), cam.get("fy", 1030.0)
        cx, cy = cam.get("cx", 960.0),  cam.get("cy", 600.0)
        h,  w  = cam.get("h", 1200),    cam.get("w", 1920)

        dep = None
        if e["depth_path"] and os.path.exists(e["depth_path"]):
            dep = load_depth(e["depth_path"], h, w)

        k3, vl = [], []
        for kp in e["kp_xy"]:
            u, v = kp
            ui, vi = int(round(u)), int(round(v))
            tk += 1
            if dep is None:
                k3.append([0, 0, 0]); vl.append(False); continue
            if not (0 <= vi < dep.shape[0] and 0 <= ui < dep.shape[1]):
                k3.append([0, 0, 0]); vl.append(False); continue
            d = float(dep[vi, ui])
            if d < dmin or d > dmax or not math.isfinite(d):
                k3.append([0, 0, 0]); vl.append(False); continue
            pc = np.array([(u-cx)*d/fx, (v-cy)*d/fy, d, 1.0])
            pw = e["pose"] @ pc
            k3.append(pw[:3]); vl.append(True); vk += 1
            all3d.append(pw[:3])

        e["kps_3d"]   = np.array(k3)
        e["kps_valid"] = np.array(vl)

    print(f"  Valid: {vk}/{tk} ({100*vk/(tk+1e-8):.1f}%)")

    all3d = np.array(all3d) if all3d else np.zeros((0, 3))
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    if len(all3d) > 0:
        s = max(1, len(all3d) // 20000)
        ax[0].scatter(all3d[::s, 0], all3d[::s, 1],
                      c=all3d[::s, 2], s=1, cmap="viridis", alpha=0.5)
    ax[0].set_title(f"3D keypoints ({vk})")
    ax[0].set_aspect("equal")

    # 중간 이미지 예시
    mid = entries[len(entries)//2]
    rgb = cv2.cvtColor(cv2.imread(mid["rgb_path"]), cv2.COLOR_BGR2RGB)
    ax[1].imshow(rgb)
    kps, vld = mid["kp_xy"], mid["kps_valid"]
    if len(kps) > 0:
        ax[1].scatter(kps[vld,  0], kps[vld,  1], c="lime", s=5, label=f"Valid({vld.sum()})")
        ax[1].scatter(kps[~vld, 0], kps[~vld, 1], c="red",  s=5, label=f"Invalid({(~vld).sum()})")
    ax[1].legend(fontsize=8); ax[1].axis("off")

    fig.suptitle("Step 5: Backproject", fontsize=14)
    fig.tight_layout()
    out = os.path.join(output_dir, "step5_backproject.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    pickle.dump(entries, open(os.path.join(output_dir, "step5_data.pkl"), "wb"))
    return entries

# =============================================================================
# STEP 6: Build Database (KDTree)
# =============================================================================

def step6_build_database(entries, output_dir):
    print("\n" + "="*60 + "\nSTEP 6: Build Database (KDTree)\n" + "="*60)
    from scipy.spatial import KDTree

    global_descs = []
    db_entries   = []

    for e in entries:
        gd = e.get("global_descriptor")
        if gd is None:
            continue
        global_descs.append(gd)
        db_entries.append({
            "id":         e["id"],
            "timestamp":  e["timestamp"],
            "cam_id":     e["cam_id"],
            "pose":       e["pose"],
            "rgb_path":   e["rgb_path"],
            "depth_path": e["depth_path"],
            "cam_params": e["cam_params"],
        })

    global_descs = np.array(global_descs, dtype=np.float32)
    norms = np.linalg.norm(global_descs, axis=1, keepdims=True) + 1e-8
    global_descs_normed = (global_descs / norms).astype(np.float32)

    kdtree = KDTree(global_descs_normed)
    db = {
        "global_descs": global_descs_normed,
        "kdtree":        kdtree,
        "entries":       db_entries,
    }

    pkl_path = os.path.join(output_dir, "step6_database.pkl")
    pickle.dump(db, open(pkl_path, "wb"))
    print(f"  DB entries : {len(db_entries)}")
    print(f"  Descriptor : {global_descs_normed.shape}")
    print(f"  KDTree     : {kdtree.n} nodes")
    print(f"  Saved: {pkl_path}")

    # 시각화
    nv  = min(200, len(db_entries))
    idx = np.linspace(0, len(db_entries)-1, nv, dtype=int)
    sub = global_descs_normed[idx]
    sim = sub @ sub.T
    pos = np.array([db_entries[i]["pose"][:3, 3] for i in idx])

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    im = ax[0].imshow(sim, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax[0].set_title(f"Cosine similarity ({nv}×{nv})")
    plt.colorbar(im, ax=ax[0], fraction=0.046)
    sc = ax[1].scatter(pos[:, 0], pos[:, 1], c=np.arange(nv), cmap="viridis", s=5)
    ax[1].set_title("DB positions (top-down)")
    ax[1].set_aspect("equal")
    plt.colorbar(sc, ax=ax[1], fraction=0.046)
    fig.suptitle(f"Step 6: Database — {len(db_entries)} entries, KDTree ready", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, "step6_database.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    return db

# =============================================================================
# STEP 7a: Global Retrieval
# =============================================================================

def step7a_retrieval(query_entry, db, config, output_dir):
    print("\n" + "="*60 + "\nSTEP 7a: Global Retrieval\n" + "="*60)
    import torch
    fc    = config["features"]
    onl   = config.get("online", {})
    top_k = onl.get("top_k", 5)
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gd_dim = fc.get("global_desc_dim", 512)

    # query_entry가 dict이면 rgb_path 사용, 문자열이면 파일 경로
    if isinstance(query_entry, str):
        q_rgb_path = query_entry
        q_pose     = None
        q_cam      = None
    else:
        q_rgb_path = query_entry["rgb_path"]
        q_pose     = query_entry.get("pose")
        q_cam      = query_entry.get("cam_params")

    query_rgb = cv2.cvtColor(cv2.imread(q_rgb_path), cv2.COLOR_BGR2RGB)
    print(f"  Query: {q_rgb_path}")

    # global descriptor
    try:
        gm = torch.jit.load(fc["global_model"], map_location=dev)
        gm.eval()
        t = cv2.resize(query_rgb, (224, 224)).astype(np.float32) / 255.0
        t = torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0).to(dev)
        with torch.no_grad():
            q_gd = gm(t).cpu().numpy().flatten()
    except Exception as ex:
        print(f"  Model failed ({ex}), HSV fallback")
        hsv  = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2HSV)
        q_gd = np.concatenate([
            cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten(),
            cv2.calcHist([hsv], [1], None, [64], [0, 256]).flatten(),
        ])
        q_gd = q_gd / (np.linalg.norm(q_gd) + 1e-8)
    q_gd = q_gd[:gd_dim]
    if len(q_gd) < gd_dim:
        q_gd = np.pad(q_gd, (0, gd_dim - len(q_gd)))
    q_gd_norm = (q_gd / (np.linalg.norm(q_gd) + 1e-8)).astype(np.float32)

    # KDTree retrieval
    dists, idxs = db["kdtree"].query(q_gd_norm, k=top_k)
    cos_sims    = 1.0 - dists**2 / 2.0
    candidates  = [db["entries"][i] for i in idxs]

    print(f"  Top-{top_k}:")
    for rank, (cand, sim) in enumerate(zip(candidates, cos_sims)):
        gt_str = ""
        if q_pose is not None:
            d = np.linalg.norm(np.array(cand["pose"])[:3, 3] - q_pose[:3, 3])
            gt_str = f"  GT_dist={d:.2f}m"
        print(f"    Rank{rank+1}: #{cand['id']} {cand['cam_id']}  sim={sim:.4f}{gt_str}")

    # 시각화
    n_show = min(top_k, 5) + 1
    fig, axes = plt.subplots(2, n_show, figsize=(4*n_show, 8))
    if n_show == 1: axes = axes.reshape(2, 1)
    axes[0, 0].imshow(query_rgb)
    axes[0, 0].set_title("QUERY", color="blue")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, os.path.basename(q_rgb_path), ha="center",
                    va="center", fontsize=7, wrap=True)
    for rank, (cand, sim) in enumerate(zip(candidates[:n_show-1], cos_sims[:n_show-1])):
        ref = cv2.cvtColor(cv2.imread(cand["rgb_path"]), cv2.COLOR_BGR2RGB)
        axes[0, rank+1].imshow(ref)
        col = "green" if rank == 0 else "orange"
        axes[0, rank+1].set_title(f"Rank{rank+1} #{cand['id']}\nsim={sim:.3f}", color=col, fontsize=9)
        axes[0, rank+1].axis("off")
        p = np.array(cand["pose"])[:3, 3]
        axes[1, rank+1].axis("off")
        axes[1, rank+1].text(0.5, 0.5, f"{cand['cam_id']}\n({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})",
                             ha="center", va="center", fontsize=7)
    fig.suptitle(f"Step 7a: KDTree Retrieval — Top-{top_k}", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, "step7a_retrieval.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    data = {
        "query_rgb":     query_rgb,
        "query_rgb_path": q_rgb_path,
        "query_gd_norm": q_gd_norm,
        "query_pose":    q_pose,
        "query_cam":     q_cam,
        "candidates":    candidates,
        "cos_sims":      cos_sims.tolist(),
        "best_match":    candidates[0],
    }
    pickle.dump(data, open(os.path.join(output_dir, "step7a_data.pkl"), "wb"))
    return data

# =============================================================================
# STEP 7b: Local Feature Extraction
# =============================================================================

def step7b_features(step7a_data, config, output_dir):
    print("\n" + "="*60 + "\nSTEP 7b: Local Feature Extraction\n" + "="*60)
    import torch
    fc  = config["features"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mk  = fc.get("max_keypoints", 1024)
    kt  = fc.get("keypoint_threshold", 0.005)

    sp = torch.jit.load(fc["superpoint_model"], map_location=dev)
    sp.eval()

    def extract(rgb, label):
        kps, descs, scores = _extract_superpoint(sp, rgb, mk, kt, dev)
        print(f"    {label}: {len(kps)} keypoints")
        return kps, descs, scores

    q_rgb = step7a_data["query_rgb"]
    r_rgb = cv2.cvtColor(cv2.imread(step7a_data["best_match"]["rgb_path"]), cv2.COLOR_BGR2RGB)

    q_kps, q_descs, q_scores = extract(q_rgb, "Query")
    r_kps, r_descs, r_scores = extract(r_rgb, f"Ref #{step7a_data['best_match']['id']}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(q_rgb)
    if len(q_kps): axes[0].scatter(q_kps[:, 0], q_kps[:, 1], c="lime", s=3)
    axes[0].set_title(f"Query — {len(q_kps)} kp")
    axes[0].axis("off")
    axes[1].imshow(r_rgb)
    if len(r_kps): axes[1].scatter(r_kps[:, 0], r_kps[:, 1], c="lime", s=3)
    axes[1].set_title(f"Ref #{step7a_data['best_match']['id']} — {len(r_kps)} kp")
    axes[1].axis("off")
    fig.suptitle("Step 7b: SuperPoint Features", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, "step7b_features.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    data = {**step7a_data,
            "q_kps": q_kps, "q_descs": q_descs, "q_scores": q_scores,
            "r_kps": r_kps, "r_descs": r_descs, "r_scores": r_scores,
            "r_rgb": r_rgb}
    pickle.dump(data, open(os.path.join(output_dir, "step7b_data.pkl"), "wb"))
    return data

# =============================================================================
# STEP 7c: SuperGlue Matching
# =============================================================================

def step7c_matching(step7b_data, config, output_dir):
    print("\n" + "="*60 + "\nSTEP 7c: SuperGlue Matching\n" + "="*60)
    import torch
    fc  = config["features"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_kps   = step7b_data["q_kps"]
    r_kps   = step7b_data["r_kps"]
    q_descs = step7b_data["q_descs"]
    r_descs = step7b_data["r_descs"]
    q_scores = step7b_data["q_scores"]
    r_scores = step7b_data["r_scores"]
    q_rgb   = step7b_data["query_rgb"]
    r_rgb   = step7b_data["r_rgb"]

    H, W = q_rgb.shape[:2]

    matched_q = matched_r = confs = np.array([])

    try:
        sg = torch.jit.load(fc["superglue_model"], map_location=dev)
        sg.eval()

        def to_t(arr): return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).to(dev)

        kp_q_n = (q_kps - np.array([[W/2, H/2]])) / (max(W, H) / 2)
        kp_r_n = (r_kps - np.array([[W/2, H/2]])) / (max(W, H) / 2)

        data = {
            "keypoints0":   to_t(q_kps),
            "keypoints1":   to_t(r_kps),
            "descriptors0": torch.from_numpy(q_descs.astype(np.float32)).unsqueeze(0).to(dev),
            "descriptors1": torch.from_numpy(r_descs.astype(np.float32)).unsqueeze(0).to(dev),
            "scores0":      to_t(q_scores),
            "scores1":      to_t(r_scores),
            "image_shape":  (H, W),
        }
        with torch.no_grad():
            pred = sg(data)
        m0   = pred["matches0"][0].cpu().numpy()
        conf = pred["matching_scores0"][0].cpu().numpy()
        valid = (m0 >= 0) & (conf > 0)
        matched_q = q_kps[valid]
        matched_r = r_kps[m0[valid]]
        confs     = conf[valid]
        print(f"  SuperGlue: {valid.sum()} matches (conf>0)")

    except Exception as ex:
        print(f"  SuperGlue failed ({ex}), MNN fallback")
        if len(q_kps) > 0 and len(r_kps) > 0:
            sim  = q_descs.T @ r_descs
            nn_q = np.argmax(sim, axis=1)
            nn_r = np.argmax(sim, axis=0)
            ids  = np.arange(len(q_kps))
            mutual = nn_r[nn_q[ids]] == ids
            if mutual.any():
                matched_q = q_kps[mutual]
                matched_r = r_kps[nn_q[ids[mutual]]]
                confs     = np.ones(matched_q.shape[0])
                print(f"  MNN: {len(matched_q)} matches")

    print(f"  Matched pairs: {len(matched_q)}")

    # 시각화
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    h1, w1 = q_rgb.shape[:2]
    h2, w2 = r_rgb.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = q_rgb
    canvas[:h2, w1:w1+w2] = r_rgb
    ax.imshow(canvas)
    for i in range(min(len(matched_q), 200)):
        x0, y0 = matched_q[i]
        x1, y1 = matched_r[i]
        c = plt.cm.RdYlGn(float(i) / max(len(matched_q), 1))
        ax.plot([x0, x1 + w1], [y0, y1], "-", color=c, linewidth=0.5, alpha=0.7)
        ax.scatter([x0, x1+w1], [y0, y1], c=[c], s=5)
    ax.set_title(f"Step 7c: {len(matched_q)} matches")
    ax.axis("off")
    fig.tight_layout()
    out = os.path.join(output_dir, "step7c_matching.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    data = {**step7b_data,
            "matched_q": matched_q, "matched_r": matched_r, "confs": confs}
    pickle.dump(data, open(os.path.join(output_dir, "step7c_data.pkl"), "wb"))
    return data

# =============================================================================
# STEP 7d: 2D-3D Correspondence
# =============================================================================

def step7d_correspondence(step7c_data, config, output_dir):
    print("\n" + "="*60 + "\nSTEP 7d: 2D-3D Correspondence\n" + "="*60)
    matched_q = step7c_data["matched_q"]
    matched_r = step7c_data["matched_r"]
    ref_entry = step7c_data["best_match"]
    cfg       = config["camera"]
    dmin      = cfg.get("depth_min", 0.3)
    dmax      = cfg.get("depth_max", 50.0)

    cam = ref_entry.get("cam_params") or {}
    fx, fy = cam.get("fx", 1027.0), cam.get("fy", 1030.0)
    cx, cy = cam.get("cx", 960.0),  cam.get("cy", 600.0)
    h,  w  = cam.get("h", 1200),    cam.get("w", 1920)

    dep = None
    if ref_entry.get("depth_path") and os.path.exists(ref_entry["depth_path"]):
        dep = load_depth(ref_entry["depth_path"], h, w)

    pts2d, pts3d = [], []
    for (qu, qv), (ru, rv) in zip(matched_q, matched_r):
        ri, rj = int(round(rv)), int(round(ru))
        if dep is None:
            continue
        if not (0 <= ri < dep.shape[0] and 0 <= rj < dep.shape[1]):
            continue
        pz = float(dep[ri, rj])
        if pz < dmin or pz > dmax or not math.isfinite(pz):
            continue
        px = (ru - cx) * pz / fx
        py = (rv - cy) * pz / fy
        pts2d.append([qu, qv])
        pts3d.append([px, py, pz])

    pts2d = np.array(pts2d, dtype=np.float32)
    pts3d = np.array(pts3d, dtype=np.float32)
    print(f"  Valid correspondences: {len(pts2d)} / {len(matched_q)}")

    # 시각화
    q_rgb = step7c_data["query_rgb"]
    r_rgb = step7c_data["r_rgb"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(q_rgb)
    if len(pts2d): axes[0].scatter(pts2d[:, 0], pts2d[:, 1], c="lime", s=8)
    axes[0].set_title(f"Query 2D — {len(pts2d)} pts")
    axes[0].axis("off")
    axes[1].imshow(r_rgb)
    if len(matched_r): axes[1].scatter(matched_r[:, 0], matched_r[:, 1], c="orange", s=8)
    axes[1].set_title(f"Ref (depth valid={len(pts3d)})")
    axes[1].axis("off")
    fig.suptitle("Step 7d: 2D-3D Correspondence", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, "step7d_correspondence.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    data = {**step7c_data, "pts2d": pts2d, "pts3d": pts3d}
    pickle.dump(data, open(os.path.join(output_dir, "step7d_data.pkl"), "wb"))
    return data

# =============================================================================
# STEP 7e: PnP Pose Estimation
# =============================================================================

def step7e_pnp(step7d_data, config, output_dir):
    print("\n" + "="*60 + "\nSTEP 7e: PnP Pose Estimation\n" + "="*60)
    pts2d     = step7d_data["pts2d"]
    pts3d     = step7d_data["pts3d"]
    ref_entry = step7d_data["best_match"]
    onl       = config.get("online", {})

    if len(pts2d) < 6:
        print(f"  Too few correspondences ({len(pts2d)} < 6), skip")
        return None

    # ref camera intrinsics (query camera intrinsics for PnP)
    q_cam = step7d_data.get("query_cam") or ref_entry.get("cam_params") or {}
    fx, fy = q_cam.get("fx", 1027.0), q_cam.get("fy", 1030.0)
    cx, cy = q_cam.get("cx", 960.0),  q_cam.get("cy", 600.0)

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(4)

    ret, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d.reshape(-1, 1, 3).astype(np.float64),
        pts2d.reshape(-1, 1, 2).astype(np.float64),
        K, dist,
        iterationsCount = onl.get("pnp_iterations", 1000),
        reprojectionError = onl.get("reprojection_error", 8.0),
        confidence = onl.get("pnp_confidence", 0.99),
        flags = cv2.SOLVEPNP_EPNP,
    )

    if not ret or inliers is None or len(inliers) < 4:
        print(f"  PnP failed (inliers={len(inliers) if inliers is not None else 0})")
        return None

    R_qr, _ = cv2.Rodrigues(rvec)
    T_QR = np.eye(4)
    T_QR[:3, :3] = R_qr
    T_QR[:3,  3] = tvec.flatten()

    T_WR = np.array(ref_entry["pose"])
    T_WQ = T_WR @ np.linalg.inv(T_QR)

    pos = T_WQ[:3, 3]
    print(f"  PnP success! inliers={len(inliers)}/{len(pts2d)}")
    print(f"  Estimated position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    q_pose = step7d_data.get("query_pose")
    if q_pose is not None:
        gt_pos = np.array(q_pose)[:3, 3]
        err = np.linalg.norm(pos - gt_pos)
        print(f"  GT position     : ({gt_pos[0]:.3f}, {gt_pos[1]:.3f}, {gt_pos[2]:.3f})")
        print(f"  Position error  : {err:.3f} m")

    # 시각화
    q_rgb = step7d_data["query_rgb"]
    r_rgb = step7d_data["r_rgb"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(q_rgb)
    pts2d_all = step7d_data["pts2d"]
    if len(pts2d_all):
        axes[0].scatter(pts2d_all[:, 0], pts2d_all[:, 1], c="orange", s=5, label="all")
    if inliers is not None and len(inliers):
        in_pts = pts2d_all[inliers.flatten()]
        axes[0].scatter(in_pts[:, 0], in_pts[:, 1], c="lime", s=8, label=f"inliers({len(inliers)})")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Query — PnP inliers")
    axes[0].axis("off")
    axes[1].imshow(r_rgb)
    axes[1].set_title(f"Best Ref #{ref_entry['id']}")
    axes[1].axis("off")
    ax = axes[2]
    ax.set_title("Top-down: estimated vs ref")
    ax.set_aspect("equal")
    ax.scatter(*np.array(ref_entry["pose"])[:2, 3], c="blue",  s=80, zorder=5, label="Ref")
    ax.scatter(*pos[:2],                             c="green", s=80, zorder=5, label="Est. Query")
    if q_pose is not None:
        ax.scatter(*np.array(q_pose)[:2, 3],         c="red",   s=80, zorder=5, label="GT Query")
    ax.legend(fontsize=8)
    fig.suptitle(f"Step 7e: PnP — pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, "step7e_pnp.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    result = {
        "T_WQ":     T_WQ,
        "position": pos,
        "inliers":  len(inliers),
        "total_pts": len(pts2d),
    }
    if q_pose is not None:
        result["gt_pos"]       = np.array(q_pose)[:3, 3]
        result["position_err"] = float(np.linalg.norm(pos - result["gt_pos"]))
    pickle.dump(result, open(os.path.join(output_dir, "step7e_result.pkl"), "wb"))
    return result

# =============================================================================
# STEP EVAL: 전체 query 평가 + GT 비교
# =============================================================================

def step_eval(query_dir, db, config, output_dir, query_cam_filter=None):
    """
    test_data의 모든 query 이미지에 대해 7a~7e를 실행하고
    추정 trajectory를 GT(trajectories.txt)와 비교.

    출력:
      eval_results.pkl  - 전체 결과
      eval_trajectory.png - top-down 궤적 비교
      eval_error.png      - 포즈 오차 그래프
    """
    print("\n" + "="*60 + "\nEVAL: Full trajectory evaluation\n" + "="*60)

    q_entries = load_query_entries(query_dir)
    if query_cam_filter:
        filtered = [e for e in q_entries if e["cam_id"] == query_cam_filter]
        if filtered:
            q_entries = filtered
    print(f"  Query entries: {len(q_entries)} (cam={query_cam_filter or 'all'})")

    results = []   # {ts, cam_id, gt_pos, est_pos, pos_err, success}

    for i, qe in enumerate(q_entries):
        print(f"\n  [{i+1}/{len(q_entries)}] {qe['cam_id']} ts={qe['timestamp']}")
        try:
            # 7a: retrieval (임시 출력 억제)
            d = step7a_retrieval(qe, db, config, output_dir)
            # 7b: features
            d = step7b_features(d, config, output_dir)
            # 7c: matching
            d = step7c_matching(d, config, output_dir)
            # 7d: correspondence
            d = step7d_correspondence(d, config, output_dir)
            # 7e: pnp
            res = step7e_pnp(d, config, output_dir)

            gt_pos  = np.array(qe["pose"])[:3, 3] if qe["pose"] is not None else None
            est_pos = res["position"] if res else None
            pos_err = float(np.linalg.norm(est_pos - gt_pos)) if (est_pos is not None and gt_pos is not None) else None

            results.append({
                "timestamp": qe["timestamp"],
                "cam_id":    qe["cam_id"],
                "gt_pos":    gt_pos,
                "est_pos":   est_pos,
                "pos_err":   pos_err,
                "success":   res is not None,
                "cos_sim":   d["cos_sims"][0] if d else None,
                "inliers":   res["inliers"] if res else 0,
            })
        except Exception as ex:
            print(f"    FAILED: {ex}")
            gt_pos = np.array(qe["pose"])[:3, 3] if qe["pose"] is not None else None
            results.append({
                "timestamp": qe["timestamp"],
                "cam_id":    qe["cam_id"],
                "gt_pos":    gt_pos,
                "est_pos":   None,
                "pos_err":   None,
                "success":   False,
                "cos_sim":   None,
                "inliers":   0,
            })

    # ── 통계 ────────────────────────────────────────────────────────────
    n_total   = len(results)
    n_success = sum(r["success"] for r in results)
    errs      = [r["pos_err"] for r in results if r["pos_err"] is not None]
    print(f"\n  Success: {n_success}/{n_total} ({100*n_success/max(n_total,1):.1f}%)")
    if errs:
        print(f"  Position error — mean={np.mean(errs):.3f}m  median={np.median(errs):.3f}m  max={np.max(errs):.3f}m")

    pickle.dump(results, open(os.path.join(output_dir, "eval_results.pkl"), "wb"))

    # ── Plot 1: top-down 궤적 비교 ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    gt_pts  = np.array([r["gt_pos"]  for r in results if r["gt_pos"]  is not None])
    est_ok  = [r for r in results if r["est_pos"] is not None]
    est_fail= [r for r in results if r["gt_pos"] is not None and r["est_pos"] is None]

    if len(gt_pts):
        ax.plot(gt_pts[:, 0], gt_pts[:, 1], "b-", linewidth=1.5, alpha=0.6, label="GT trajectory")
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], c="blue", s=8, zorder=3)

    if est_ok:
        est_pts = np.array([r["est_pos"] for r in est_ok])
        gt_ok   = np.array([r["gt_pos"]  for r in est_ok])
        errs_ok = np.array([r["pos_err"] for r in est_ok])
        sc = ax.scatter(est_pts[:, 0], est_pts[:, 1],
                        c=errs_ok, cmap="RdYlGn_r", s=30, zorder=5,
                        vmin=0, vmax=min(5.0, errs_ok.max()) if len(errs_ok) else 1,
                        label=f"Estimated ({len(est_ok)})")
        plt.colorbar(sc, ax=ax, label="Position error (m)")
        # GT→Est 연결선
        for r in est_ok:
            ax.plot([r["gt_pos"][0], r["est_pos"][0]],
                    [r["gt_pos"][1], r["est_pos"][1]],
                    "gray", linewidth=0.4, alpha=0.4)

    if est_fail:
        fail_pts = np.array([r["gt_pos"] for r in est_fail])
        ax.scatter(fail_pts[:, 0], fail_pts[:, 1], c="red", marker="x",
                   s=40, zorder=6, label=f"Failed ({len(est_fail)})")

    ax.set_aspect("equal")
    ax.set_title("Top-down: GT vs Estimated trajectory")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(fontsize=8)

    # ── Plot 2: 오차 그래프 ──────────────────────────────────────────────
    ax2 = axes[1]
    idx_list  = list(range(len(results)))
    err_list  = [r["pos_err"] if r["pos_err"] is not None else np.nan for r in results]
    suc_list  = [r["success"] for r in results]

    ax2.plot(idx_list, err_list, "b-o", markersize=3, linewidth=1, label="Position error")
    ax2.axhline(1.0, color="g", linestyle="--", linewidth=1, label="1m threshold")
    ax2.axhline(2.0, color="r", linestyle="--", linewidth=1, label="2m threshold")
    for j, (ok, err) in enumerate(zip(suc_list, err_list)):
        if not ok:
            ax2.axvspan(j-0.4, j+0.4, color="red", alpha=0.2)

    if errs:
        ax2.axhline(np.mean(errs), color="orange", linestyle="-.",
                    linewidth=1.5, label=f"Mean {np.mean(errs):.2f}m")
    ax2.set_xlabel("Query index")
    ax2.set_ylabel("Position error (m)")
    ax2.set_title(f"Position Error — {n_success}/{n_total} success "
                  f"({'%.1f'%(100*n_success/max(n_total,1))}%)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(bottom=0)

    fig.suptitle(f"Trajectory Evaluation — cam={query_cam_filter or 'all'}", fontsize=14)
    fig.tight_layout()
    out = os.path.join(output_dir, "eval_trajectory.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    # ── Plot 3: 오차 분포 히스토그램 ────────────────────────────────────
    if errs:
        fig2, ax3 = plt.subplots(figsize=(8, 5))
        ax3.hist(errs, bins=20, color="steelblue", edgecolor="white")
        ax3.axvline(np.mean(errs),   color="orange", linestyle="-.", label=f"Mean {np.mean(errs):.2f}m")
        ax3.axvline(np.median(errs), color="green",  linestyle="--", label=f"Median {np.median(errs):.2f}m")
        ax3.set_xlabel("Position error (m)")
        ax3.set_ylabel("Count")
        ax3.set_title("Error distribution")
        ax3.legend()
        fig2.tight_layout()
        out2 = os.path.join(output_dir, "eval_error_hist.png")
        fig2.savefig(out2, dpi=150)
        plt.close()
        print(f"  Saved: {out2}")

    return results


# =============================================================================
# MAIN
# =============================================================================

OFFLINE_STEPS = ["2_visualize", "3_superpoint", "4_global_desc", "5_backproject", "6_build_db"]
ONLINE_STEPS  = ["7a_retrieval", "7b_features", "7c_matching", "7d_correspondence", "7e_pnp"]
EVAL_STEPS    = ["eval"]
ALL_STEPS     = OFFLINE_STEPS + ONLINE_STEPS + EVAL_STEPS

def main():
    parser = argparse.ArgumentParser(description="Kapture real-data localization pipeline")
    parser.add_argument("--step",      default="all",
                        choices=["all", "offline", "online", "eval"] + ALL_STEPS)
    parser.add_argument("--mapping",   default=MAPPING_DIR,   help="kapture_mapping/sensors 경로")
    parser.add_argument("--query_dir", default=QUERY_DIR,     help="test_data 경로")
    parser.add_argument("--query",     default=None,          help="단일 query 이미지 경로 (7a~)")
    parser.add_argument("--output",    default=OUTPUT_DIR)
    parser.add_argument("--subsample", type=int, default=3,
                        help="Mapping timestamps 중 N번째마다 선택 (기본=3)")
    parser.add_argument("--query_cam", default="cam_0",
                        choices=["cam_0","cam_1","cam_2","cam_3"],
                        help="query 카메라 선택 (7a~, query 미지정 시)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    config = DEFAULT_CONFIG

    run_offline = args.step in ("all", "offline") or args.step in OFFLINE_STEPS
    run_online  = args.step in ("all", "online")  or args.step in ONLINE_STEPS

    def should(step_name):
        if args.step == "all":     return True
        if args.step == "offline": return step_name in OFFLINE_STEPS
        if args.step == "online":  return step_name in ONLINE_STEPS
        if args.step == "eval":    return step_name == "eval"
        return args.step == step_name

    # ── Load / resume entries ─────────────────────────────────────────────
    entries = None
    db      = None

    if should("2_visualize"):
        print(f"\nLoading kapture_mapping (subsample={args.subsample})...")
        entries = load_mapping_entries(args.mapping, subsample=args.subsample)
        print(f"  Loaded {len(entries)} entries")
        entries = step2_visualize(entries, args.output)

    if should("3_superpoint"):
        if entries is None:
            pkl = os.path.join(args.output, "step3_data.pkl")
            if os.path.exists(pkl):
                entries = pickle.load(open(pkl, "rb"))
            else:
                print("Loading mapping entries...")
                entries = load_mapping_entries(args.mapping, subsample=args.subsample)
                step2_visualize(entries, args.output)
        entries = step3_superpoint(entries, config, args.output)

    if should("4_global_desc"):
        if entries is None:
            pkl = os.path.join(args.output, "step3_data.pkl")
            entries = pickle.load(open(pkl, "rb")) if os.path.exists(pkl) else None
            if entries is None:
                raise RuntimeError("step3_data.pkl 없음. 먼저 --step 3_superpoint 실행")
        entries = step4_global_desc(entries, config, args.output)

    if should("5_backproject"):
        if entries is None:
            pkl = os.path.join(args.output, "step4_data.pkl")
            entries = pickle.load(open(pkl, "rb")) if os.path.exists(pkl) else None
            if entries is None:
                raise RuntimeError("step4_data.pkl 없음. 먼저 --step 4_global_desc 실행")
        entries = step5_backproject(entries, config, args.output)

    if should("6_build_db"):
        if entries is None:
            pkl = os.path.join(args.output, "step5_data.pkl")
            entries = pickle.load(open(pkl, "rb")) if os.path.exists(pkl) else None
            if entries is None:
                raise RuntimeError("step5_data.pkl 없음. 먼저 --step 5_backproject 실행")
        db = step6_build_database(entries, args.output)

    # ── Online ────────────────────────────────────────────────────────────
    if any(should(s) for s in ONLINE_STEPS):
        if db is None:
            pkl = os.path.join(args.output, "step6_database.pkl")
            if not os.path.exists(pkl):
                raise RuntimeError("step6_database.pkl 없음. 먼저 offline 실행")
            print("  Loading DB...")
            db = pickle.load(open(pkl, "rb"))

    # query 결정
    query_entry = None
    if any(should(s) for s in ONLINE_STEPS):
        if args.query and os.path.exists(args.query):
            query_entry = args.query
        else:
            # test_data에서 query_cam의 첫 번째 이미지 선택
            q_entries = load_query_entries(args.query_dir)
            cam_entries = [e for e in q_entries if e["cam_id"] == args.query_cam]
            if not cam_entries:
                cam_entries = q_entries
            query_entry = cam_entries[0] if cam_entries else None
            if query_entry:
                print(f"  Auto-query: {query_entry['rgb_path']}")

    data7 = None
    if should("7a_retrieval"):
        data7 = step7a_retrieval(query_entry, db, config, args.output)
    if should("7b_features"):
        if data7 is None:
            data7 = pickle.load(open(os.path.join(args.output, "step7a_data.pkl"), "rb"))
        data7 = step7b_features(data7, config, args.output)
    if should("7c_matching"):
        if data7 is None or "q_kps" not in data7:
            data7 = pickle.load(open(os.path.join(args.output, "step7b_data.pkl"), "rb"))
        data7 = step7c_matching(data7, config, args.output)
    if should("7d_correspondence"):
        if data7 is None or "matched_q" not in data7:
            data7 = pickle.load(open(os.path.join(args.output, "step7c_data.pkl"), "rb"))
        data7 = step7d_correspondence(data7, config, args.output)
    if should("7e_pnp"):
        if data7 is None or "pts2d" not in data7:
            data7 = pickle.load(open(os.path.join(args.output, "step7d_data.pkl"), "rb"))
        step7e_pnp(data7, config, args.output)

    if should("eval"):
        if db is None:
            pkl = os.path.join(args.output, "step6_database.pkl")
            if not os.path.exists(pkl):
                raise RuntimeError("step6_database.pkl 없음. 먼저 offline 실행")
            db = pickle.load(open(pkl, "rb"))
        step_eval(args.query_dir, db, config, args.output, args.query_cam)

    print("\n✓ Done.")

if __name__ == "__main__":
    main()
