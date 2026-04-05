import os, pickle
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .step3_global_desc import _extract_megaloc_desc, _extract_dino_patches, _compute_vlad


def step5_retrieval(query_image_path, db, config, output_dir):
    """Query 이미지 → global descriptor → KDTree top-K retrieval."""
    import torch
    print("\n" + "="*60 + "\nSTEP 5: Global retrieval\n" + "="*60)
    fc    = config["features"]
    onl   = config.get("online", {})
    top_k = onl.get("top_k", 5)
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if query_image_path and os.path.exists(query_image_path):
        query_rgb = cv2.cvtColor(cv2.imread(query_image_path), cv2.COLOR_BGR2RGB)
        gt_entry  = None
        print(f"  Query: {query_image_path} ({query_rgb.shape[1]}×{query_rgb.shape[0]})")
    else:
        qi = len(db["entries"]) // 3
        gt_entry  = db["entries"][qi]
        query_rgb = cv2.cvtColor(cv2.imread(gt_entry["rgb_path"]), cv2.COLOR_BGR2RGB)
        print(f"  Query: DB entry #{gt_entry['id']} (self-test)")

    _cache = step5_retrieval.__dict__
    global_desc_method = db.get("global_desc_method", "anyloc")

    if global_desc_method == "megaloc":
        print(f"  Method: MegaLoc  (dim={fc.get('megaloc_dim', 8448)})")
        if "_megaloc_model" not in _cache:
            print("  Loading MegaLoc …")
            model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
            model.eval().to(dev)
            _cache["_megaloc_model"] = model
        model = _cache["_megaloc_model"]
        q_gd_norm = _extract_megaloc_desc(query_rgb, model, dev)
        q_gd_norm = q_gd_norm / (np.linalg.norm(q_gd_norm) + 1e-8)
    else:
        vlad_centers = db.get("vlad_centers")
        if vlad_centers is None:
            raise RuntimeError("DB에 vlad_centers가 없습니다. step3→step4를 재실행하세요.")
        dino_name  = fc.get("dino_model", "dinov2_vitb14")
        img_size   = int(fc.get("dino_img_size", 322))
        n_clusters = vlad_centers.shape[0]
        feat_dim   = vlad_centers.shape[1]
        print(f"  Method: AnyLoc  DINOv2={dino_name}  VLAD K={n_clusters}  dim={n_clusters*feat_dim}")
        if "_dino_model" not in _cache or _cache.get("_dino_name") != dino_name:
            model = torch.hub.load("facebookresearch/dinov2", dino_name, pretrained=True)
            model.eval().to(dev)
            _cache["_dino_model"] = model
            _cache["_dino_name"]  = dino_name
            print(f"  Loaded DINOv2: {dino_name}")
        model = _cache["_dino_model"]
        q_patches = _extract_dino_patches(query_rgb, model, dev, img_size)
        q_gd_norm = _compute_vlad(q_patches, vlad_centers)

    dists, idxs = db["kdtree"].query(q_gd_norm, k=top_k)
    cos_sims    = 1.0 - dists**2 / 2.0
    candidates  = [db["entries"][i] for i in idxs]

    print(f"  Top-{top_k} results:")
    for rank, (cand, sim) in enumerate(zip(candidates, cos_sims)):
        gt_str = ""
        if gt_entry:
            d = np.linalg.norm(np.array(cand["pose"])[:3,3]
                               - np.array(gt_entry["pose"])[:3,3])
            gt_str = f"  GT_dist={d:.2f}m"
        print(f"    Rank{rank+1}: #{cand['id']}  sim={sim:.4f}{gt_str}")

    n_show  = min(top_k, 5) + 1
    fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 4))
    if n_show == 1: axes = [axes]
    axes[0].imshow(query_rgb)
    axes[0].set_title("QUERY", color="blue", fontsize=10)
    axes[0].axis("off")
    for rank, (cand, sim) in enumerate(zip(candidates[:n_show-1], cos_sims[:n_show-1])):
        col = "green" if rank == 0 else "orange"
        ref_rgb = cv2.cvtColor(cv2.imread(cand["rgb_path"]), cv2.COLOR_BGR2RGB)
        p = np.array(cand["pose"])[:3, 3]
        axes[rank+1].imshow(ref_rgb)
        axes[rank+1].set_title(
            f"Rank{rank+1} #{cand['id']}  sim={sim:.3f}\n"
            f"({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})",
            color=col, fontsize=8)
        axes[rank+1].axis("off")
    fig.suptitle(f"Step 5: KDTree Retrieval — Top-{top_k}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,"step5_retrieval.png"), dpi=150); plt.close()
    print(f"  Saved: step5_retrieval.png")

    data = {
        "query_rgb":        query_rgb,
        "query_gd_norm":    q_gd_norm,
        "candidates":       candidates,
        "cos_sims":         cos_sims.tolist(),
        "gt_entry":         gt_entry,
        "query_image_path": query_image_path,
    }
    pickle.dump(data, open(os.path.join(output_dir,"step5_data.pkl"),"wb"))
    return data
