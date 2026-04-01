import os, sys, pickle, glob
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .step3_global_desc import _extract_megaloc_desc, _extract_dino_patches, _compute_vlad


def _load_eloftr(config, dev):
    """EfficientLoFTR 로드 헬퍼."""
    import torch

    eloftr_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "EfficientLoFTR"))
    if eloftr_root not in sys.path:
        sys.path.insert(0, eloftr_root)

    from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

    fc = config["features"]
    ckpt_path = fc.get("eloftr_ckpt",
        os.path.join(eloftr_root, "weights", "ELoFTR", "weights", "eloftr_outdoor.ckpt"))
    use_opt = fc.get("eloftr_opt", False)

    _cfg = (opt_default_cfg if use_opt else full_default_cfg).copy()
    matcher = LoFTR(config=_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    matcher.load_state_dict(state["state_dict"])
    matcher = reparameter(matcher).eval().to(dev)
    return matcher, ckpt_path, use_opt


def _make_gray_tensor_fn(dev, max_dim):
    """to_gray_tensor 클로저 생성."""
    import torch

    def to_gray_tensor(rgb_img):
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        orig_h, orig_w = gray.shape
        h, w = orig_h, orig_w
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            nw = int(w * s); nh = int(h * s)
            gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
            h, w = gray.shape
        scale_x = orig_w / w
        scale_y = orig_h / h
        ph = ((h + 31) // 32) * 32
        pw = ((w + 31) // 32) * 32
        if ph != h or pw != w:
            padded = np.zeros((ph, pw), dtype=np.float32)
            padded[:h, :w] = gray
            gray = padded
        return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(dev), scale_x, scale_y

    return to_gray_tensor


def step6_match(step5_data, config, output_dir):
    """EfficientLoFTR matching: query vs top-K candidates → best match."""
    import torch

    print("\n" + "="*60 + "\nSTEP 6: EfficientLoFTR matching\n" + "="*60)

    fc          = config["features"]
    dev         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_thresh = float(fc.get("match_conf_thresh", 0.2))
    max_dim     = int(fc.get("eloftr_max_dim", 840))

    query_rgb  = step5_data["query_rgb"]
    candidates = step5_data["candidates"]

    matcher, ckpt_path, use_opt = _load_eloftr(config, dev)
    print(f"  EfficientLoFTR: ckpt={os.path.basename(ckpt_path)}, "
          f"mode={'opt' if use_opt else 'full'}, device={dev}")

    if dev.type == "cuda":
        torch.cuda.empty_cache()

    to_gray_tensor = _make_gray_tensor_fn(dev, max_dim)
    q_tensor, q_scale_x, q_scale_y = to_gray_tensor(query_rgb)

    best_mkpts_q = np.zeros((0, 2))
    best_mkpts_r = np.zeros((0, 2))
    best_confs   = np.array([])
    best_cand    = candidates[0]
    best_ref_rgb = cv2.cvtColor(cv2.imread(candidates[0]["rgb_path"]), cv2.COLOR_BGR2RGB)
    best_n       = -1
    all_match_counts = []

    for rank, cand in enumerate(candidates):
        ref_rgb  = cv2.cvtColor(cv2.imread(cand["rgb_path"]), cv2.COLOR_BGR2RGB)
        r_tensor, r_scale_x, r_scale_y = to_gray_tensor(ref_rgb)

        try:
            batch = {"image0": q_tensor, "image1": r_tensor}
            with torch.no_grad():
                matcher(batch)
            mkpts0 = batch["mkpts0_f"].cpu().numpy()
            mkpts1 = batch["mkpts1_f"].cpu().numpy()
            confs  = batch["mconf"].cpu().numpy()
        except Exception as e:
            print(f"  Rank{rank+1} ELoFTR failed: {e}")
            all_match_counts.append(0)
            continue

        mkpts0 = mkpts0 * np.array([q_scale_x, q_scale_y], dtype=np.float32)
        mkpts1 = mkpts1 * np.array([r_scale_x, r_scale_y], dtype=np.float32)

        mask   = confs >= conf_thresh
        n_good = int(mask.sum())
        all_match_counts.append(n_good)
        print(f"  Rank{rank+1} #{cand['id']}: {len(mkpts0)} raw → {n_good} conf≥{conf_thresh}")

        if n_good > best_n:
            best_n       = n_good
            best_mkpts_q = mkpts0[mask]
            best_mkpts_r = mkpts1[mask]
            best_confs   = confs[mask]
            best_cand    = cand
            best_ref_rgb = ref_rgb

    print(f"\n  Best: #{best_cand['id']}  matches={best_n}")

    # 시각화
    h1, w1 = query_rgb.shape[:2]; h2, w2 = best_ref_rgb.shape[:2]
    th = max(h1, h2)
    sq, sr = 1.0, 1.0
    if h1 != th:
        sq = th / h1
        query_rgb    = cv2.resize(query_rgb,    (int(w1*sq), th))
    if h2 != th:
        sr = th / h2
        best_ref_rgb = cv2.resize(best_ref_rgb, (int(w2*sr), th))
    h1, w1 = query_rgb.shape[:2]; h2, w2 = best_ref_rgb.shape[:2]
    canvas = np.concatenate([query_rgb, best_ref_rgb], axis=1)
    mkpts_q_v = best_mkpts_q * sq
    mkpts_r_v = best_mkpts_r * sr

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.imshow(canvas)
    if len(mkpts_q_v) > 0:
        cmap_v = plt.cm.RdYlGn(best_confs / (best_confs.max() + 1e-8))
        step_v = max(1, len(mkpts_q_v) // 200)
        for i in range(0, len(mkpts_q_v), step_v):
            ax.plot([mkpts_q_v[i,0], mkpts_r_v[i,0]+w1],
                    [mkpts_q_v[i,1], mkpts_r_v[i,1]],
                    c=cmap_v[i], alpha=0.5, linewidth=0.8)
        ax.scatter(mkpts_q_v[:,0], mkpts_q_v[:,1], c="cyan",   s=8, zorder=3)
        ax.scatter(mkpts_r_v[:,0]+w1, mkpts_r_v[:,1], c="yellow", s=8, zorder=3)

    match_summary = "  |  ".join(
        f"R{r+1}:{n}" for r, n in enumerate(all_match_counts))
    ax.set_title(f"Step 6: EfficientLoFTR — best=#{best_cand['id']} ({best_n} matches)\n"
                 f"{match_summary}")
    ax.axis("off"); fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "step6_matching.png"), dpi=150); plt.close()
    print(f"  Saved: step6_matching.png")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors_bar = ["green" if c["id"]==best_cand["id"] else "steelblue"
                  for c in candidates[:len(all_match_counts)]]
    ax2.bar([f"R{r+1}\n#{c['id']}" for r, c in enumerate(candidates[:len(all_match_counts)])],
            all_match_counts, color=colors_bar)
    ax2.set_ylabel("Matches (conf≥{:.2f})".format(conf_thresh))
    ax2.set_title("EfficientLoFTR match counts per candidate")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "step6_match_counts.png"), dpi=150); plt.close()

    data = {
        "matched_q_kps":    best_mkpts_q,
        "matched_r_kps":    best_mkpts_r,
        "confidences":      best_confs,
        "best_cand":        best_cand,
        "query_rgb":        query_rgb,
        "ref_rgb":          best_ref_rgb,
        "all_match_counts": all_match_counts,
        "candidates":       candidates,
    }
    pickle.dump(data, open(os.path.join(output_dir,"step6_data.pkl"),"wb"))
    return data


def step6a_match_viz(query_dir, db, config, output_dir):
    """배치 retrieval + matching 시각화."""
    import torch

    print("\n" + "="*60 + "\nSTEP 6a: Batch retrieval + match viz\n" + "="*60)
    print(f"  Query dir : {query_dir}")

    fc          = config["features"]
    dev         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_thresh = float(fc.get("match_conf_thresh", 0.2))
    max_dim     = int(fc.get("eloftr_max_dim", 840))
    onl         = config.get("online", {})
    top_k       = onl.get("top_k", 5)

    query_files = sorted(
        glob.glob(os.path.join(query_dir, "*.jpg")) +
        glob.glob(os.path.join(query_dir, "*.png")) +
        glob.glob(os.path.join(query_dir, "*.jpeg"))
    )
    if not query_files:
        print(f"  ERROR: {query_dir} 에 이미지 없음"); return
    print(f"  {len(query_files)} query images found")

    global_desc_method = db.get("global_desc_method", "anyloc")
    retr_model = None
    if global_desc_method == "megaloc":
        print("  Loading MegaLoc …")
        retr_model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        retr_model.eval().to(dev)

    matcher, _, _ = _load_eloftr(config, dev)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  EfficientLoFTR loaded  (device={dev})")

    to_gray_tensor = _make_gray_tensor_fn(dev, max_dim)

    save_dir = os.path.join(output_dir, "step6a_results")
    os.makedirs(save_dir, exist_ok=True)

    for qi, qpath in enumerate(query_files):
        stem = os.path.splitext(os.path.basename(qpath))[0]
        print(f"\n  [{qi+1}/{len(query_files)}] {stem}")

        query_rgb = cv2.cvtColor(cv2.imread(qpath), cv2.COLOR_BGR2RGB)

        # Retrieval
        if global_desc_method == "megaloc":
            q_gd = _extract_megaloc_desc(query_rgb, retr_model, dev)
        else:
            vlad_centers = db.get("vlad_centers")
            img_size = int(fc.get("dino_img_size", 322))
            dino_name = fc.get("dino_model", "dinov2_vitb14")
            from .step5_retrieval import step5_retrieval
            if "_dino_model" not in step5_retrieval.__dict__:
                step5_retrieval.__dict__["_dino_model"] = torch.hub.load(
                    "facebookresearch/dinov2", dino_name).eval().to(dev)
            dino = step5_retrieval.__dict__["_dino_model"]
            q_patches = _extract_dino_patches(query_rgb, dino, dev, img_size)
            q_gd = _compute_vlad(q_patches, vlad_centers)

        q_gd = q_gd / (np.linalg.norm(q_gd) + 1e-8)
        dists, idxs = db["kdtree"].query(q_gd, k=top_k)
        cos_sims  = 1.0 - dists**2 / 2.0
        candidates = [db["entries"][i] for i in idxs]
        print(f"    Retrieval top1: #{candidates[0]['id']}  sim={cos_sims[0]:.4f}")

        # LoFTR matching
        q_tensor, q_sx, q_sy = to_gray_tensor(query_rgb)
        best_n = -1; best_mkpts_q = best_mkpts_r = best_confs = None
        best_cand = candidates[0]; best_ref_rgb = None

        for rank, cand in enumerate(candidates):
            ref_rgb = cv2.cvtColor(cv2.imread(cand["rgb_path"]), cv2.COLOR_BGR2RGB)
            r_tensor, r_sx, r_sy = to_gray_tensor(ref_rgb)
            try:
                batch = {"image0": q_tensor, "image1": r_tensor}
                with torch.no_grad():
                    matcher(batch)
                mkpts0 = batch["mkpts0_f"].cpu().numpy() * np.array([q_sx, q_sy])
                mkpts1 = batch["mkpts1_f"].cpu().numpy() * np.array([r_sx, r_sy])
                confs  = batch["mconf"].cpu().numpy()
            except Exception as e:
                print(f"    Rank{rank+1} LoFTR failed: {e}"); continue

            mask   = confs >= conf_thresh
            n_good = int(mask.sum())
            print(f"    Rank{rank+1} #{cand['id']}: {n_good} matches")
            if n_good > best_n:
                best_n = n_good
                best_mkpts_q = mkpts0[mask]; best_mkpts_r = mkpts1[mask]
                best_confs = confs[mask]; best_cand = cand; best_ref_rgb = ref_rgb

        if best_ref_rgb is None:
            best_ref_rgb = cv2.cvtColor(cv2.imread(candidates[0]["rgb_path"]), cv2.COLOR_BGR2RGB)

        h1, w1 = query_rgb.shape[:2]; h2, w2 = best_ref_rgb.shape[:2]
        th = max(h1, h2)
        sq, sr = 1.0, 1.0
        if h1 != th:
            sq = th / h1
            query_rgb    = cv2.resize(query_rgb,    (int(w1*sq), th))
        if h2 != th:
            sr = th / h2
            best_ref_rgb = cv2.resize(best_ref_rgb, (int(w2*sr), th))
        h1, w1 = query_rgb.shape[:2]; h2, w2 = best_ref_rgb.shape[:2]
        canvas = np.concatenate([query_rgb, best_ref_rgb], axis=1)
        mkpts_q_v = best_mkpts_q * sq if best_mkpts_q is not None else None
        mkpts_r_v = best_mkpts_r * sr if best_mkpts_r is not None else None

        fig, ax = plt.subplots(1, 1, figsize=(18, 6))
        ax.imshow(canvas)
        if best_n > 0 and mkpts_q_v is not None:
            cmap_v = plt.cm.RdYlGn(best_confs / (best_confs.max() + 1e-8))
            step_v = max(1, best_n // 300)
            for i in range(0, best_n, step_v):
                ax.plot([mkpts_q_v[i,0], mkpts_r_v[i,0]+w1],
                        [mkpts_q_v[i,1], mkpts_r_v[i,1]],
                        c=cmap_v[i], alpha=0.5, linewidth=0.8)
            ax.scatter(mkpts_q_v[:,0], mkpts_q_v[:,1], c="cyan",   s=6, zorder=3)
            ax.scatter(mkpts_r_v[:,0]+w1, mkpts_r_v[:,1], c="yellow", s=6, zorder=3)

        ax.set_title(f"{stem}  →  best=#{best_cand['id']}  ({best_n} matches, conf≥{conf_thresh})")
        ax.axis("off"); fig.tight_layout()
        out_png = os.path.join(save_dir, f"{stem}.png")
        fig.savefig(out_png, dpi=120); plt.close()
        print(f"    Saved: step6a_results/{stem}.png")

    print(f"\n  Done. {len(query_files)} results in {save_dir}/")
