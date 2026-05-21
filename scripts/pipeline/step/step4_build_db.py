import os, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

    # ── 시각화 ────────────────────────────────────────────────────────────
    nv = min(200, len(entries))
    idx = np.linspace(0, len(entries)-1, nv, dtype=int)
    sub_d = global_descs_normed[idx]
    sim_mat = sub_d @ sub_d.T
    poses_sub = np.array([entries[i]["pose"][:3,3] for i in idx])

    off_diag = sim_mat[~np.eye(nv, dtype=bool)]
    print(f"  Sim stats (sampled {nv}, off-diag): "
          f"min={off_diag.min():.3f}  mean={off_diag.mean():.3f}  max={off_diag.max():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im = axes[0].imshow(sim_mat, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[0].set_title(f"Global desc cosine similarity ({nv}×{nv})\n"
                      f"off-diag: mean={off_diag.mean():.3f}  max={off_diag.max():.3f}")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # outlier 위치를 빨간 X로 표시
    outlier_in_sub = np.isin(idx, np.where(outlier_mask)[0])
    axes[1].scatter(poses_sub[:,0], poses_sub[:,1],
                    c=np.arange(nv), cmap="viridis", s=10)
    if outlier_in_sub.any():
        axes[1].scatter(poses_sub[outlier_in_sub, 0], poses_sub[outlier_in_sub, 1],
                        c="red", s=60, marker="x", linewidths=1.5, label="outlier")
        axes[1].legend(fontsize=8)
    axes[1].set_title("DB viewpoint positions (top-down)"); axes[1].set_aspect("equal")
    fig.suptitle(f"Step 4: Database — {len(entries)} entries, KDTree ready", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "step4_database.png"), dpi=150); plt.close()
    print(f"  Saved: step4_database.png")
    return db
