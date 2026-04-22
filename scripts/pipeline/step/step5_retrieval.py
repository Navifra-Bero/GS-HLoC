import os, pickle
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .step3_global_desc import (_extract_megaloc_desc, _extract_megaloc_spatial,
                                _extract_mixvpr_desc, _extract_mixvpr_spatial,
                                _extract_depth_spatial, _load_query_depth,
                                _load_mixvpr_model)


def step5_retrieval(query_image_path, db, config, output_dir, save_images=True):
    """Query 이미지 → global descriptor → KDTree top-K retrieval."""
    import torch
    print("\n" + "="*60 + "\nSTEP 5: Global retrieval\n" + "="*60)
    fc    = config["features"]
    top_k = config.get("matching", {}).get("top_k_retrieval", 10)
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
    global_desc_method = db.get("global_desc_method", "megaloc")

    # ── 공통 파라미터 ──────────────────────────────────────────────────────
    use_depth        = bool(fc.get("use_depth_desc", False))
    n_bins           = int(fc.get("depth_bins", 32))
    w_rgb            = float(fc.get("depth_rgb_weight", 0.7))
    w_depth          = 1.0 - w_rgb
    cam_h            = int(config.get("camera", {}).get("height", 1200))
    cam_w            = int(config.get("camera", {}).get("width", 1920))
    retrieval_factor = int(config.get("matching", {}).get("retrieval_factor", 3))
    expand_k         = top_k * retrieval_factor
    q_depth_desc     = None

    # ── MegaLoc 분기 ──────────────────────────────────────────────────────
    if global_desc_method == "megaloc":
        grid_n      = int(fc.get("dino_grid_n", db["entries"][0].get("dino_grid_n", 1)
                                 if db["entries"] else 1))
        use_spatial = grid_n > 1
        depth_grid  = grid_n if use_spatial else 1

        parts = []
        if use_spatial: parts.append(f"{grid_n}×{grid_n} grid")
        if use_depth:   parts.append(f"depth late-fusion w={w_rgb:.1f}/{w_depth:.1f}")
        print(f"  Method: MegaLoc" + (f" ({', '.join(parts)})" if parts else ""))

        if "_megaloc_model" not in _cache:
            print("  Loading MegaLoc …")
            model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
            model.eval().to(dev)
            _cache["_megaloc_model"] = model
        model = _cache["_megaloc_model"]

        q_gd_norm = (_extract_megaloc_spatial(query_rgb, model, dev, grid_n)
                     if use_spatial else _extract_megaloc_desc(query_rgb, model, dev))

        if use_depth and query_image_path:
            depth_map, depth_path = _load_query_depth(query_image_path, cam_h, cam_w)
            if depth_map is not None:
                q_depth_desc = _extract_depth_spatial(depth_map, depth_grid, n_bins)
                print(f"  Query depth loaded: {depth_path}")
            else:
                print(f"  WARNING: depth not found ({depth_path}), RGB-only")

    # ── MixVPR 분기 ───────────────────────────────────────────────────────
    elif global_desc_method == "mixvpr":
        ckpt_path   = fc.get("mixvpr_ckpt", "")
        out_dim     = int(fc.get("mixvpr_out_dim", 512))
        grid_n      = int(fc.get("grid_n", db["entries"][0].get("grid_n", 1)
                                  if db["entries"] else 1))
        use_spatial = grid_n > 1
        depth_grid  = grid_n if use_spatial else 1

        parts = []
        if use_spatial: parts.append(f"{grid_n}×{grid_n} grid")
        if use_depth:   parts.append(f"depth late-fusion w={w_rgb:.1f}/{w_depth:.1f}")
        print(f"  Method: MixVPR (out_dim={out_dim})" +
              (f" ({', '.join(parts)})" if parts else ""))

        if ("_mixvpr_model" not in _cache
                or _cache.get("_mixvpr_ckpt") != ckpt_path
                or _cache.get("_mixvpr_out_dim") != out_dim):
            model = _load_mixvpr_model(ckpt_path, out_dim, dev)
            _cache["_mixvpr_model"]   = model
            _cache["_mixvpr_ckpt"]    = ckpt_path
            _cache["_mixvpr_out_dim"] = out_dim
        model = _cache["_mixvpr_model"]

        q_gd_norm = (_extract_mixvpr_spatial(query_rgb, model, dev, grid_n)
                     if use_spatial else _extract_mixvpr_desc(query_rgb, model, dev))

        if use_depth and query_image_path:
            depth_map, depth_path = _load_query_depth(query_image_path, cam_h, cam_w)
            if depth_map is not None:
                q_depth_desc = _extract_depth_spatial(depth_map, depth_grid, n_bins)
                print(f"  Query depth loaded: {depth_path}")
            else:
                print(f"  WARNING: depth not found ({depth_path}), RGB-only")

    else:
        raise ValueError(f"Unknown global_desc_method in DB: '{global_desc_method}'. "
                         "Re-run step3 with a supported method (megaloc, mixvpr).")

    # ── Step 1: RGB KDTree로 expand_k개 후보 추출 ────────────────────────
    actual_expand_k = min(expand_k, len(db["entries"]))
    dists, idxs = db["kdtree"].query(q_gd_norm, k=actual_expand_k)
    rgb_sims    = 1.0 - dists ** 2 / 2.0   # L2 dist → cosine sim

    # ── Step 2: depth late fusion re-ranking ─────────────────────────────
    depth_descs_db = db.get("depth_descs")
    if (q_depth_desc is not None and depth_descs_db is not None
            and db.get("has_depth", True)):
        cand_depth = depth_descs_db[idxs]
        depth_sims = cand_depth @ q_depth_desc
        final_sims = w_rgb * rgb_sims + w_depth * depth_sims
        print(f"  Late fusion: {actual_expand_k} candidates → "
              f"re-rank (w_rgb={w_rgb:.1f}, w_depth={w_depth:.1f})")
    else:
        final_sims = rgb_sims
        if use_depth:
            print("  Late fusion: depth 없음, RGB-only 사용")

    # ── Step 3: 최종 top_k 선택 ──────────────────────────────────────────
    order      = np.argsort(-final_sims)[:top_k]
    top_idxs   = idxs[order]
    cos_sims   = final_sims[order]
    candidates = [db["entries"][i] for i in top_idxs]

    print(f"  Top-{top_k} results:")
    for rank, (cand, sim) in enumerate(zip(candidates, cos_sims)):
        gt_str = ""
        if gt_entry:
            d = np.linalg.norm(np.array(cand["pose"])[:3,3]
                               - np.array(gt_entry["pose"])[:3,3])
            gt_str = f"  GT_dist={d:.2f}m"
        print(f"    Rank{rank+1}: #{cand['id']}  sim={sim:.4f}{gt_str}")

    if save_images:
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
        print("  Saved: step5_retrieval.png")

    data = {
        "query_rgb":        query_rgb,
        "query_gd_norm":    q_gd_norm,
        "candidates":       candidates,
        "cos_sims":         cos_sims.tolist(),
        "gt_entry":         gt_entry,
        "query_image_path": query_image_path,
    }
    if save_images:
        pickle.dump(data, open(os.path.join(output_dir,"step5_data.pkl"),"wb"))
    return data
