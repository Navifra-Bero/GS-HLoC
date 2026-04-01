import os, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def step4_build_db(rendered, output_dir):
    """오프라인 마지막 단계: global descriptors를 KDTree로 인덱싱."""
    from scipy.spatial import KDTree

    print("\n" + "="*60 + "\nSTEP 4: Build database (KDTree)\n" + "="*60)

    global_descs = []
    entries = []

    for r in rendered:
        gd = r.get("global_descriptor")
        if gd is None:
            print(f"  WARNING: #{r['id']} global_descriptor 없음, 건너뜀")
            continue
        global_descs.append(gd)
        entries.append({
            "id":         r["id"],
            "pose":       r["pose"],
            "rgb_path":   r["rgb_path"],
            "depth_path": r["depth_path"],
        })

    global_descs = np.array(global_descs, dtype=np.float32)
    norms = np.linalg.norm(global_descs, axis=1, keepdims=True) + 1e-8
    global_descs_normed = (global_descs / norms).astype(np.float32)

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

    db = {"global_descs": global_descs_normed, "kdtree": kdtree,
          "entries": entries, "vlad_centers": vlad_centers, "pca_model": pca_model,
          "global_desc_method": global_desc_method}
    db_pkl = os.path.join(output_dir, "step4_database.pkl")
    db_npz = os.path.join(output_dir, "step4_database.npz")
    pickle.dump(db, open(db_pkl, "wb"))
    np.savez(db_npz, global_descs=global_descs_normed,
             poses=np.array([e["pose"] for e in entries]))

    print(f"  DB entries : {len(entries)}")
    print(f"  Descriptor : shape={global_descs_normed.shape}")
    print(f"  KDTree     : {kdtree.n} nodes built")
    print(f"  Saved: {db_pkl}")

    nv = min(200, len(entries))
    idx = np.linspace(0, len(entries)-1, nv, dtype=int)
    sub_d = global_descs_normed[idx]
    sim_mat = sub_d @ sub_d.T
    poses_sub = np.array([entries[i]["pose"][:3,3] for i in idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im = axes[0].imshow(sim_mat, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[0].set_title(f"Global desc cosine similarity ({nv}×{nv})")
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].scatter(poses_sub[:,0], poses_sub[:,1],
                    c=np.arange(nv), cmap="viridis", s=10)
    axes[1].set_title("DB viewpoint positions (top-down)"); axes[1].set_aspect("equal")
    fig.suptitle(f"Step 4: Database — {len(entries)} entries, KDTree ready", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "step4_database.png"), dpi=150); plt.close()
    print(f"  Saved: step4_database.png")
    return db
