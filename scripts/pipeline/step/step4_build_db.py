import os, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter


def _feature_meta_from_path(path):
    if not path or not os.path.exists(path):
        return None
    feat = np.load(path, mmap_mode="r")
    return tuple(int(x) for x in feat.shape)


def step4_build_db(rendered, output_dir):
    """오프라인 마지막 단계: global descriptors를 KDTree로 인덱싱."""
    from scipy.spatial import KDTree

    print("\n" + "="*60 + "\nSTEP 4: Build database (KDTree)\n" + "="*60)

    global_descs = []
    entries = []

    depth_descs_list = []
    feature_info = {"type": None, "stride": None, "shape": None, "n_with_feature": 0}
    for r in rendered:
        gd = r.get("global_descriptor")
        if gd is None:
            print(f"  WARNING: #{r['id']} global_descriptor 없음, 건너뜀")
            continue
        global_descs.append(gd)
        depth_descs_list.append(r.get("depth_descriptor"))   # None 가능
        entry = {
            "id":         r["id"],
            "pose":       r["pose"],
            "rgb_path":   r["rgb_path"],
            "depth_path": r["depth_path"],
        }
        for key in ("source", "cam_id", "camera_id", "timestamp", "view_group_id",
                    "width", "height", "camera", "depth_lookup_radius_px"):
            if r.get(key) is not None:
                entry[key] = r.get(key)
        # ── SplatHLoc coarse matching용 rendered FGS feature 경로 전달 ─────
        fpath = r.get("feature_path")
        if fpath:
            if not os.path.isabs(fpath) and not os.path.exists(fpath):
                candidate = os.path.join(output_dir, fpath)
                if os.path.exists(candidate):
                    fpath = candidate
            entry["feature_path"]   = fpath
            entry["feature_shape"]  = r.get("feature_shape") or _feature_meta_from_path(fpath)
            entry["feature_stride"] = r.get("feature_stride")
            entry["feature_type"]   = r.get("feature_type") or "rendered_feature"
            feature_info["n_with_feature"] += 1
            feature_info["type"]   = feature_info["type"]   or entry["feature_type"]
            feature_info["stride"] = feature_info["stride"] or entry["feature_stride"]
            feature_info["shape"]  = feature_info["shape"]  or entry["feature_shape"]
        entries.append(entry)

    global_descs = np.array(global_descs, dtype=np.float32)
    norms = np.linalg.norm(global_descs, axis=1, keepdims=True) + 1e-8
    global_descs_normed = (global_descs / norms).astype(np.float32)

    # depth descriptor 행렬 구성 (late fusion re-ranking용)
    has_depth = any(d is not None for d in depth_descs_list)
    if has_depth:
        depth_dim = next(d for d in depth_descs_list if d is not None).shape[0]
        depth_descs_normed = np.zeros((len(depth_descs_list), depth_dim), dtype=np.float32)
        for i, dd in enumerate(depth_descs_list):
            if dd is not None:
                depth_descs_normed[i] = dd   # 이미 step3에서 L2 정규화됨
    else:
        depth_descs_normed = None

    kdtree = KDTree(global_descs_normed)

    vlad_centers = None
    pca_model    = None
    global_desc_method = "anyloc"
    for r in rendered:
        if r.get("global_desc_method"):
            global_desc_method = r["global_desc_method"]
        if r.get("vlad_vocab") is not None:
            vlad_centers = r["vlad_vocab"]
            pca_model    = r.get("pca_model")
            break

    has_feature = feature_info["n_with_feature"] > 0
    db = {"global_descs": global_descs_normed, "kdtree": kdtree,
          "depth_descs": depth_descs_normed, "has_depth": has_depth,
          "entries": entries, "vlad_centers": vlad_centers, "pca_model": pca_model,
          "global_desc_method": global_desc_method,
          "has_feature": has_feature,
          "feature_type":   feature_info["type"],
          "feature_stride": feature_info["stride"],
          "feature_shape":  feature_info["shape"]}
    print(f"  Depth descs  : {'stored (' + str(depth_descs_normed.shape) + ')' if has_depth else 'none'}")
    if has_feature:
        print(f"  FGS features : {feature_info['n_with_feature']}/{len(entries)} "
              f"type={feature_info['type']}  stride={feature_info['stride']}  "
              f"shape={feature_info['shape']}")
    else:
        print(f"  FGS features : none (step2 feature_splat 결과 없음)")
    db_pkl = os.path.join(output_dir, "step4_database.pkl")
    db_npz = os.path.join(output_dir, "step4_database.npz")
    pickle.dump(db, open(db_pkl, "wb"))
    np.savez(db_npz,
             global_descs=global_descs_normed,
             poses=np.array([e["pose"] for e in entries]),
             sources=np.array([e.get("source", "rendered") for e in entries], dtype=object),
             cam_ids=np.array([e.get("cam_id", "") for e in entries], dtype=object),
             timestamps=np.array([e.get("timestamp", "") for e in entries], dtype=object),
             feature_paths=np.array([e.get("feature_path", "") for e in entries], dtype=object),
             feature_shapes=np.array([e.get("feature_shape", None) for e in entries], dtype=object),
             feature_strides=np.array([e.get("feature_stride", -1) or -1 for e in entries], dtype=np.int32))

    print(f"  DB entries : {len(entries)}")
    print(f"  Descriptor : shape={global_descs_normed.shape}")
    print(f"  KDTree     : {kdtree.n} nodes built")
    print(f"  Saved: {db_pkl}")

    # ── 전체 디스크립터 통계 및 outlier 감지 ──────────────────────────────
    mean_sim_per_entry = (global_descs_normed @ global_descs_normed.T).mean(axis=1)
    outlier_thresh = np.percentile(mean_sim_per_entry, 5)   # 하위 5%
    outlier_mask = mean_sim_per_entry < outlier_thresh
    n_outliers = outlier_mask.sum()
    print(f"  Sim stats (all): mean={mean_sim_per_entry.mean():.3f}  "
          f"min={mean_sim_per_entry.min():.3f}  max={mean_sim_per_entry.max():.3f}")
    print(f"  Outlier entries (mean sim < {outlier_thresh:.3f}): {n_outliers}")

    if n_outliers > 0:
        import cv2, shutil
        out_bad = os.path.join(output_dir, "step4_outlier_renders")
        os.makedirs(out_bad, exist_ok=True)
        for oi in np.where(outlier_mask)[0][:20]:   # 최대 20개만 저장
            src = entries[oi]["rgb_path"]
            if os.path.exists(src):
                dst = os.path.join(out_bad, f"{oi:04d}_{os.path.basename(src)}")
                shutil.copy2(src, dst)
        print(f"  Outlier renders saved → {out_bad}/  (최대 20개)")

    # ── 시각화: DB 구성/coverage/outlier를 source별로 표시 ────────────────
    poses_all = np.array([e["pose"][:3, 3] for e in entries], dtype=np.float64)
    sources = np.array([e.get("source", "rendered") for e in entries], dtype=object)
    cam_ids = np.array([e.get("cam_id", "") for e in entries], dtype=object)
    src_counts = Counter(sources.tolist())
    real_cam_counts = Counter(cam_ids[sources == "real_train"].tolist())
    depth_counts = Counter(
        sources[i] for i, e in enumerate(entries)
        if e.get("depth_path") and os.path.exists(e.get("depth_path", ""))
    )
    print(f"  Source counts: {dict(src_counts)}")
    if real_cam_counts:
        print(f"  Real train cams: {dict(real_cam_counts)}")

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.15, 1.0],
                          hspace=0.30, wspace=0.22)

    ax_map = fig.add_subplot(gs[:, 0])
    aligned_map = os.path.join(output_dir, "aligned_map.ply")
    if os.path.exists(aligned_map):
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(aligned_map)
            pts = np.asarray(pcd.points)
            sub = max(1, len(pts) // 80000)
            ax_map.scatter(pts[::sub, 0], pts[::sub, 1], c="0.78",
                           s=0.25, alpha=0.35, linewidths=0, label="aligned map")
        except Exception as exc:
            print(f"  WARNING: failed to draw aligned map background: {exc}")

    render_mask = sources != "real_train"
    real_mask = sources == "real_train"
    if np.any(render_mask):
        ax_map.scatter(poses_all[render_mask, 0], poses_all[render_mask, 1],
                       c="#e24a33", s=11, marker="x", linewidths=0.7,
                       alpha=0.80, label=f"rendered ({int(render_mask.sum())})")
    if np.any(real_mask):
        cam_colors = {
            "cam_0": "#1f77b4",
            "cam_1": "#17becf",
            "cam_2": "#2ca02c",
            "cam_3": "#9467bd",
        }
        for cam_id in sorted(set(cam_ids[real_mask])):
            cm = real_mask & (cam_ids == cam_id)
            ax_map.scatter(poses_all[cm, 0], poses_all[cm, 1],
                           c=cam_colors.get(cam_id, "#4c72b0"),
                           s=5, marker=".", alpha=0.45,
                           label=f"{cam_id} real ({int(cm.sum())})")
    if n_outliers > 0:
        oi = np.where(outlier_mask)[0]
        ax_map.scatter(poses_all[oi, 0], poses_all[oi, 1],
                       facecolors="none", edgecolors="yellow", s=70,
                       linewidths=1.2, label=f"outlier ({len(oi)})")
    ax_map.set_title("DB coverage in aligned map frame")
    ax_map.set_xlabel("X [m]")
    ax_map.set_ylabel("Y [m]")
    ax_map.set_aspect("equal")
    ax_map.legend(loc="best", fontsize=8, markerscale=1.8)

    ax_counts = fig.add_subplot(gs[0, 1])
    labels = []
    values = []
    colors = []
    for src in ("rendered", "real_train"):
        if src_counts.get(src, 0):
            labels.append(src)
            values.append(src_counts[src])
            colors.append("#e24a33" if src == "rendered" else "#1f77b4")
    for cam_id in sorted(real_cam_counts):
        if cam_id:
            labels.append(f"real {cam_id}")
            values.append(real_cam_counts[cam_id])
            colors.append("#7aa6c2")
    y = np.arange(len(labels))
    ax_counts.barh(y, values, color=colors)
    ax_counts.set_yticks(y)
    ax_counts.set_yticklabels(labels)
    ax_counts.invert_yaxis()
    ax_counts.set_xlabel("entries")
    ax_counts.set_title("DB entry counts")
    for yi, val in zip(y, values):
        ax_counts.text(val, yi, f" {val}", va="center", fontsize=9)
    summary = (
        f"total={len(entries)}\n"
        f"desc={global_descs_normed.shape[1]}D\n"
        f"depth stored={sum(depth_counts.values())}\n"
        + "\n".join(f"{k} depth={v}" for k, v in sorted(depth_counts.items()))
    )
    ax_counts.text(0.98, 0.04, summary, transform=ax_counts.transAxes,
                   ha="right", va="bottom", fontsize=9,
                   bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.75"))

    ax_hist = fig.add_subplot(gs[1, 1])
    bins = np.linspace(float(mean_sim_per_entry.min()), float(mean_sim_per_entry.max()), 50)
    for src, color in (("rendered", "#e24a33"), ("real_train", "#1f77b4")):
        sm = mean_sim_per_entry[sources == src]
        if sm.size:
            ax_hist.hist(sm, bins=bins, alpha=0.55, color=color, label=f"{src} n={len(sm)}")
    ax_hist.axvline(outlier_thresh, color="black", linestyle="--",
                    linewidth=1.2, label=f"5% outlier thresh={outlier_thresh:.3f}")
    ax_hist.set_title("Descriptor mean similarity distribution")
    ax_hist.set_xlabel("mean cosine similarity to all DB entries")
    ax_hist.set_ylabel("entry count")
    ax_hist.legend(fontsize=8)

    fig.suptitle(f"Step 4: Database overview — {len(entries)} entries, KDTree ready",
                 fontsize=15)
    fig.savefig(os.path.join(output_dir, "step4_database.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: step4_database.png")
    return db
