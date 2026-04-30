"""
STEP 6 (DeDoDe): Feature matching

DeDoDe-D  — 3D-consistent keypoint detector  (step 1)
DeDoDe-G  — keypoint descriptor              (step 1)
DualSoftMaxMatcher — learned matcher         (step 2)

LightGlue 공식 패키지는 DeDoDe를 지원하지 않음.
DeDoDe 저자의 DualSoftMaxMatcher 가 DeDoDe descriptor 에 맞게 설계된 matcher.

설치:
    pip install git+https://github.com/Parskatt/DeDoDe

Usage (config features 블록):
    matcher_name: dedode_lightglue
    dedode_num_keypoints: 10000   # 기본값
    dedode_match_threshold: 0.01  # 기본값
"""

import os, sys, pickle, glob
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .step3_global_desc import (_extract_megaloc_desc,
                                _extract_mixvpr_desc, _load_mixvpr_model)


# ── DeDoDe loader ──────────────────────────────────────────────────────────

def _load_dedode(config, dev):
    """DeDoDe detector + descriptor + matcher 로드."""
    try:
        from DeDoDe import dedode_detector_L, dedode_descriptor_G
        from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
    except ImportError:
        raise ImportError(
            "DeDoDe 가 없습니다:\n"
            "  pip install git+https://github.com/Parskatt/DeDoDe\n"
        )

    import torch
    fc       = config["features"]
    num_kp   = int(fc.get("dedode_num_keypoints", 10000))

    detector   = dedode_detector_L(weights=None).to(dev).eval()
    descriptor = dedode_descriptor_G(weights=None).to(dev).eval()
    matcher    = DualSoftMaxMatcher()

    info = f"DeDoDe-L + DeDoDe-G + DualSoftMaxMatcher  num_kp={num_kp}  device={dev}"
    return detector, descriptor, matcher, num_kp, info


# ── 이미지 변환 헬퍼 ────────────────────────────────────────────────────────

def _rgb_to_dedode_tensor(rgb_arr: np.ndarray, max_dim, dev):
    """
    numpy RGB (H,W,3) uint8 → float tensor (1,3,H,W) [0,1] on device.
    Returns tensor, scale_x, scale_y (resized→original).
    """
    import torch

    orig_h, orig_w = rgb_arr.shape[:2]
    img = rgb_arr

    if max_dim is not None and max(orig_h, orig_w) > max_dim:
        s     = max_dim / max(orig_h, orig_w)
        new_w = int(orig_w * s)
        new_h = int(orig_h * s)
        img   = cv2.resize(rgb_arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    new_h, new_w = img.shape[:2]
    scale_x = orig_w / new_w
    scale_y = orig_h / new_h

    t = torch.from_numpy(img).float() / 255.0  # (H, W, 3)
    t = t.permute(2, 0, 1).unsqueeze(0).to(dev)  # (1, 3, H, W)
    return t, scale_x, scale_y, new_h, new_w


# ── 단일 쌍 매칭 ───────────────────────────────────────────────────────────

def _run_dedode(detector, descriptor, matcher, query_rgb, ref_rgb, config, dev):
    """
    DeDoDe detect → describe → DualSoftMaxMatcher 로 단일 쌍 매칭.

    Returns:
        mkpts0  : (M, 2) query keypoints (original image coords)
        mkpts1  : (M, 2) ref   keypoints (original image coords)
        confs   : (M,) match confidence scores
        n_good  : 유효 매칭 수
        score   : re-ranking 점수 (confidence 합)
    """
    import torch

    fc            = config["features"]
    num_kp        = int(fc.get("dedode_num_keypoints", 10000))
    thresh        = float(fc.get("dedode_match_threshold", 0.01))
    max_dim       = fc.get("dedode_resize", None)  # None = 원본 해상도

    q_t, q_sx, q_sy, q_h, q_w = _rgb_to_dedode_tensor(query_rgb, max_dim, dev)
    r_t, r_sx, r_sy, r_h, r_w = _rgb_to_dedode_tensor(ref_rgb,   max_dim, dev)

    with torch.no_grad():
        # ── Detection ──────────────────────────────────────────────────
        det_q = detector.detect({"image": q_t}, num_keypoints=num_kp)
        det_r = detector.detect({"image": r_t}, num_keypoints=num_kp)

        kpts_q = det_q["keypoints"]     # (1, N, 2)  normalized [-1,1]
        kpts_r = det_r["keypoints"]     # (1, M, 2)
        conf_q = det_q["confidence"]    # (1, N)
        conf_r = det_r["confidence"]    # (1, M)

        # ── Description ────────────────────────────────────────────────
        desc_q = descriptor.describe_keypoints({"image": q_t, "keypoints": kpts_q})
        desc_r = descriptor.describe_keypoints({"image": r_t, "keypoints": kpts_r})

        descs_q = desc_q["descriptions"]  # (1, N, D)
        descs_r = desc_r["descriptions"]  # (1, M, D)

    # ── Matching ────────────────────────────────────────────────────────
    matches_q, matches_r, _ = matcher.match(
        kpts_q, descs_q,
        kpts_r, descs_r,
        P_A=conf_q, P_B=conf_r,
        normalize=True, inv_temp=20, threshold=thresh,
    )

    if len(matches_q) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.array([]), 0, 0.0

    # normalized [-1,1] → pixel coords
    matches_q_px, matches_r_px = matcher.to_pixel_coords(
        matches_q, matches_r, q_h, q_w, r_h, r_w
    )

    mkpts0 = matches_q_px.cpu().numpy() * np.array([q_sx, q_sy], dtype=np.float32)
    mkpts1 = matches_r_px.cpu().numpy() * np.array([r_sx, r_sy], dtype=np.float32)

    # confidence: cosine similarity (desc dot product) 를 score 로 사용
    with torch.no_grad():
        d0 = descs_q[0][matches_q[:, 1].long()] if matches_q.dim() == 2 else descs_q[0]
        d1 = descs_r[0][matches_r[:, 1].long()] if matches_r.dim() == 2 else descs_r[0]
        cos_sim = (d0 * d1).sum(dim=-1).clamp(0, 1).cpu().numpy()

    n_good = len(mkpts0)
    score  = float(cos_sim.sum())
    return mkpts0, mkpts1, cos_sim, n_good, score


# ── step6_match_dedode ─────────────────────────────────────────────────────

def step6_match_dedode(step5_data, config, output_dir, save_images=True):
    """DeDoDe + DualSoftMaxMatcher: query vs top-K candidates → best match."""
    import torch

    print("\n" + "="*60 + "\nSTEP 6: Feature matching  [DeDoDe + DualSoftMaxMatcher]\n" + "="*60)

    fc         = config["features"]
    dev        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query_rgb  = step5_data["query_rgb"]
    candidates = step5_data["candidates"]
    cos_sims   = step5_data.get("cos_sims", [1.0] * len(candidates))

    detector, descriptor, matcher, num_kp, info_str = _load_dedode(config, dev)
    print(f"  Matcher: {info_str}")

    if dev.type == "cuda":
        torch.cuda.empty_cache()

    best_mkpts_q = np.zeros((0, 2))
    best_mkpts_r = np.zeros((0, 2))
    best_confs   = np.array([])
    best_cand    = candidates[0]
    best_ref_rgb = cv2.cvtColor(cv2.imread(candidates[0]["rgb_path"]), cv2.COLOR_BGR2RGB)
    best_score   = -1.0
    best_n       = 0
    all_match_counts = []
    all_scores       = []

    for rank, cand in enumerate(candidates):
        ref_rgb = cv2.cvtColor(cv2.imread(cand["rgb_path"]), cv2.COLOR_BGR2RGB)

        try:
            mkpts0, mkpts1, confs, n_good, score = _run_dedode(
                detector, descriptor, matcher, query_rgb, ref_rgb, config, dev
            )
        except Exception as e:
            print(f"  Rank{rank+1} matching failed: {e}")
            all_match_counts.append(0)
            all_scores.append(0.0)
            continue

        mean_conf = float(confs.mean()) if len(confs) > 0 else 0.0
        ret_sim   = cos_sims[rank] if rank < len(cos_sims) else 0.0
        all_match_counts.append(n_good)
        all_scores.append(score)

        print(f"  Rank{rank+1} #{cand['id']:>4}: {n_good:>4} matches  "
              f"score={score:>7.2f}  mean_conf={mean_conf:.3f}  ret_sim={ret_sim:.4f}")

        if score > best_score:
            best_score   = score
            best_n       = n_good
            best_mkpts_q = mkpts0
            best_mkpts_r = mkpts1
            best_confs   = confs
            best_cand    = cand
            best_ref_rgb = ref_rgb

    print(f"\n  Best: #{best_cand['id']}  matches={best_n}  score={best_score:.2f}")

    if save_images:
        _save_viz(query_rgb, best_ref_rgb, best_mkpts_q, best_mkpts_r, best_confs,
                  best_cand, best_n, best_score,
                  candidates, all_match_counts, all_scores, output_dir)

    data = {
        "matched_q_kps":    best_mkpts_q,
        "matched_r_kps":    best_mkpts_r,
        "confidences":      best_confs,
        "best_cand":        best_cand,
        "query_rgb":        query_rgb,
        "ref_rgb":          best_ref_rgb,
        "all_match_counts": all_match_counts,
        "all_scores":       all_scores,
        "candidates":       candidates,
        "matcher_name":     "dedode",
    }
    if save_images:
        pickle.dump(data, open(os.path.join(output_dir, "step6_data.pkl"), "wb"))
    return data


# ── step6a_match_viz_dedode ────────────────────────────────────────────────

def step6a_match_viz_dedode(query_dir, db, config, output_dir):
    """배치 retrieval + DeDoDe matching 시각화."""
    import torch

    print("\n" + "="*60 + "\nSTEP 6a: Batch retrieval + match viz  [DeDoDe]\n" + "="*60)
    print(f"  Query dir : {query_dir}")

    fc      = config["features"]
    dev     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    onl     = config.get("online", {})
    top_k   = onl.get("top_k", 5)

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

    detector, descriptor, matcher, num_kp, info_str = _load_dedode(config, dev)
    print(f"  Matcher: {info_str}")

    if dev.type == "cuda":
        torch.cuda.empty_cache()

    save_dir = os.path.join(output_dir, "step6a_results")
    os.makedirs(save_dir, exist_ok=True)

    for qi, qpath in enumerate(query_files):
        stem = os.path.splitext(os.path.basename(qpath))[0]
        print(f"\n  [{qi+1}/{len(query_files)}] {stem}")

        query_rgb = cv2.cvtColor(cv2.imread(qpath), cv2.COLOR_BGR2RGB)

        # ── Retrieval ──────────────────────────────────────────────────
        if global_desc_method == "megaloc":
            q_gd = _extract_megaloc_desc(query_rgb, retr_model, dev)
        elif global_desc_method == "mixvpr":
            from .step5_retrieval import step5_retrieval
            _sc = step5_retrieval.__dict__
            if "_mixvpr_model" not in _sc:
                ckpt_path = fc.get("mixvpr_ckpt", "")
                out_dim   = int(fc.get("mixvpr_out_dim", 512))
                _sc["_mixvpr_model"] = _load_mixvpr_model(ckpt_path, out_dim, dev)
            q_gd = _extract_mixvpr_desc(query_rgb, _sc["_mixvpr_model"], dev)
        else:
            raise ValueError(f"Unknown global_desc_method: '{global_desc_method}'")

        q_gd = q_gd / (np.linalg.norm(q_gd) + 1e-8)
        dists, idxs = db["kdtree"].query(q_gd, k=top_k)
        cos_sims    = 1.0 - dists**2 / 2.0
        candidates  = [db["entries"][i] for i in idxs]
        print(f"    Retrieval top1: #{candidates[0]['id']}  sim={cos_sims[0]:.4f}")

        # ── Matching + re-ranking ──────────────────────────────────────
        best_score = -1.0; best_n = 0
        best_mkpts_q = best_mkpts_r = best_confs = None
        best_cand = candidates[0]; best_ref_rgb = None

        for rank, cand in enumerate(candidates):
            ref_rgb = cv2.cvtColor(cv2.imread(cand["rgb_path"]), cv2.COLOR_BGR2RGB)
            try:
                mkpts0, mkpts1, confs, n_good, score = _run_dedode(
                    detector, descriptor, matcher, query_rgb, ref_rgb, config, dev
                )
            except Exception as e:
                print(f"    Rank{rank+1} failed: {e}"); continue

            mean_conf = float(confs.mean()) if len(confs) > 0 else 0.0
            ret_sim   = cos_sims[rank] if rank < len(cos_sims) else 0.0
            print(f"    Rank{rank+1} #{cand['id']:>4}: {n_good:>4} matches  "
                  f"score={score:>7.2f}  mean_conf={mean_conf:.3f}  ret_sim={ret_sim:.4f}")

            if score > best_score:
                best_score = score; best_n = n_good
                best_mkpts_q = mkpts0; best_mkpts_r = mkpts1
                best_confs = confs; best_cand = cand; best_ref_rgb = ref_rgb

        if best_ref_rgb is None:
            best_ref_rgb = cv2.cvtColor(cv2.imread(candidates[0]["rgb_path"]), cv2.COLOR_BGR2RGB)
        if best_mkpts_q is None:
            best_mkpts_q = np.zeros((0, 2)); best_mkpts_r = np.zeros((0, 2))
            best_confs = np.array([])

        # ── 시각화 ─────────────────────────────────────────────────────
        h1, w1 = query_rgb.shape[:2]; h2, w2 = best_ref_rgb.shape[:2]
        th = max(h1, h2); sq, sr = 1.0, 1.0
        if h1 != th: sq = th/h1; query_rgb    = cv2.resize(query_rgb,    (int(w1*sq), th))
        if h2 != th: sr = th/h2; best_ref_rgb = cv2.resize(best_ref_rgb, (int(w2*sr), th))
        h1, w1 = query_rgb.shape[:2]; h2, w2 = best_ref_rgb.shape[:2]
        canvas = np.concatenate([query_rgb, best_ref_rgb], axis=1)

        fig, ax = plt.subplots(1, 1, figsize=(18, 6))
        ax.imshow(canvas)
        if best_n > 0:
            mkpts_q_v = best_mkpts_q * sq
            mkpts_r_v = best_mkpts_r * sr
            norm_c = best_confs / (best_confs.max() + 1e-8)
            cmap_v = plt.cm.RdYlGn(norm_c)
            step_v = max(1, best_n // 300)
            for i in range(0, best_n, step_v):
                ax.plot([mkpts_q_v[i, 0], mkpts_r_v[i, 0] + w1],
                        [mkpts_q_v[i, 1], mkpts_r_v[i, 1]],
                        c=cmap_v[i], alpha=0.5, linewidth=0.8)
            ax.scatter(mkpts_q_v[:, 0], mkpts_q_v[:, 1], c="cyan",   s=6, zorder=3)
            ax.scatter(mkpts_r_v[:, 0] + w1, mkpts_r_v[:, 1], c="yellow", s=6, zorder=3)

        ax.set_title(f"[DeDoDe] {stem}  →  best=#{best_cand['id']}  "
                     f"({best_n} matches, score={best_score:.1f})")
        ax.axis("off"); fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{stem}.png"), dpi=120); plt.close()
        print(f"    Saved: step6a_results/{stem}.png")

    print(f"\n  Done. {len(query_files)} results in {save_dir}/")


# ── 시각화 헬퍼 ───────────────────────────────────────────────────────────

def _save_viz(query_rgb, best_ref_rgb, best_mkpts_q, best_mkpts_r, best_confs,
              best_cand, best_n, best_score,
              candidates, all_match_counts, all_scores, output_dir):

    h1, w1 = query_rgb.shape[:2]; h2, w2 = best_ref_rgb.shape[:2]
    th = max(h1, h2); sq, sr = 1.0, 1.0
    if h1 != th: sq = th/h1; query_rgb    = cv2.resize(query_rgb,    (int(w1*sq), th))
    if h2 != th: sr = th/h2; best_ref_rgb = cv2.resize(best_ref_rgb, (int(w2*sr), th))
    h1, w1 = query_rgb.shape[:2]; h2, w2 = best_ref_rgb.shape[:2]
    canvas = np.concatenate([query_rgb, best_ref_rgb], axis=1)

    mkpts_q_v = best_mkpts_q * sq
    mkpts_r_v = best_mkpts_r * sr

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.imshow(canvas)
    if best_n > 0:
        norm_c = best_confs / (best_confs.max() + 1e-8)
        cmap_v = plt.cm.RdYlGn(norm_c)
        step_v = max(1, best_n // 200)
        for i in range(0, best_n, step_v):
            ax.plot([mkpts_q_v[i, 0], mkpts_r_v[i, 0] + w1],
                    [mkpts_q_v[i, 1], mkpts_r_v[i, 1]],
                    c=cmap_v[i], alpha=0.5, linewidth=0.8)
        ax.scatter(mkpts_q_v[:, 0], mkpts_q_v[:, 1], c="cyan",   s=8, zorder=3)
        ax.scatter(mkpts_r_v[:, 0] + w1, mkpts_r_v[:, 1], c="yellow", s=8, zorder=3)

    match_summary = "  |  ".join(f"R{r+1}:{n}" for r, n in enumerate(all_match_counts))
    ax.set_title(f"Step 6 [DeDoDe]: best=#{best_cand['id']} ({best_n} matches)\n"
                 f"{match_summary}")
    ax.axis("off"); fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "step6_matching.png"), dpi=150); plt.close()
    print(f"  Saved: step6_matching.png")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
    colors_bar = ["green" if c["id"] == best_cand["id"] else "steelblue"
                  for c in candidates[:len(all_match_counts)]]
    xlabels = [f"R{r+1}\n#{c['id']}" for r, c in enumerate(candidates[:len(all_match_counts)])]
    axes2[0].bar(xlabels, all_match_counts, color=colors_bar)
    axes2[0].set_ylabel("Match count"); axes2[0].set_title("Match count per candidate")
    axes2[1].bar(xlabels, all_scores, color=colors_bar)
    axes2[1].set_ylabel("Re-ranking score"); axes2[1].set_title("Re-ranking score (conf sum)")
    fig2.suptitle(f"Step 6 [DeDoDe]: best=#{best_cand['id']}  score={best_score:.2f}", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "step6_match_counts.png"), dpi=150); plt.close()
