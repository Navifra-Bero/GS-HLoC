"""STEP 6 Type-2: multi-view feature matching for dependent retrieval.

Uses step5_retrieval_type2 output:
  - final top-N candidates only (default N=5)
  - each rank has per-camera render entries in match_cam_top_results

For every final rank, all available camera views are matched. The rank with the
highest equal-weight average feature-matching score becomes the single result
passed to step7.
"""
import os
import pickle
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .step6_match import (
    _load_matcher,
    _run_single_match,
    _make_gray_tensor_fn,
    _make_loftr_gray_tensor_fn,
    _make_jamma_preprocess_fn,
)


def _draw_matches(ax, q_img, r_img, mkpts0, mkpts1, confs, title):
    """Draw a side-by-side match visualization on an axis."""
    h1, w1 = q_img.shape[:2]
    h2, w2 = r_img.shape[:2]
    th = max(h1, h2)
    sq = th / h1 if h1 != th else 1.0
    sr = th / h2 if h2 != th else 1.0
    qi = cv2.resize(q_img, (int(w1 * sq), th)) if sq != 1.0 else q_img
    ri = cv2.resize(r_img, (int(w2 * sr), th)) if sr != 1.0 else r_img
    canvas = np.concatenate([qi, ri], axis=1)
    wq = qi.shape[1]
    ax.imshow(canvas)

    if len(mkpts0) > 0:
        nc = confs / (confs.max() + 1e-8)
        cmap = plt.cm.RdYlGn(nc)
        step = max(1, len(mkpts0) // 200)
        for i in range(0, len(mkpts0), step):
            ax.plot([mkpts0[i, 0] * sq, mkpts1[i, 0] * sr + wq],
                    [mkpts0[i, 1] * sq, mkpts1[i, 1] * sr],
                    c=cmap[i], alpha=0.5, linewidth=0.8)
        ax.scatter(mkpts0[:, 0] * sq, mkpts0[:, 1] * sq,
                   c="cyan", s=6, zorder=3)
        ax.scatter(mkpts1[:, 0] * sr + wq, mkpts1[:, 1] * sr,
                   c="yellow", s=6, zorder=3)

    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _ordered_cam_ids(step5_data, config, cam_top_results):
    mc = config.get("multi_cam", {})
    main_cam = step5_data.get("dynamic_primary") or mc.get("main_cam") or mc.get("primary_cam")
    sub_cams = list(mc.get("sub_cams", []))
    cam_ids = []
    if main_cam and main_cam in cam_top_results:
        cam_ids.append(main_cam)
    for cam_id in sub_cams:
        if cam_id in cam_top_results and cam_id not in cam_ids:
            cam_ids.append(cam_id)
    for cam_id in cam_top_results:
        if cam_id not in cam_ids:
            cam_ids.append(cam_id)
    return cam_ids, main_cam


def step6_match_type2(step5_data, config, output_dir, save_images=True):
    """Match all type2 camera views for final top-5 and keep the best rank."""
    import torch

    print("\n" + "="*60 + "\nSTEP 6 (type2): Multi-view feature matching\n" + "="*60)

    fc = config["features"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_thresh = float(fc.get("match_conf_thresh", 0.2))
    eloftr_max_dim = int(fc.get("eloftr_max_dim", 840))
    vismatch_max_dim = int(fc.get("vismatch_max_dim", 840))

    query_images = step5_data.get("query_images") or {}
    cam_top_results = (step5_data.get("match_cam_top_results")
                       or step5_data.get("cam_top_results") or {})
    candidates = (step5_data.get("match_candidates")
                  or step5_data.get("candidates", [])[:5])
    cos_sims = (step5_data.get("match_cos_sims")
                or step5_data.get("cos_sims", [])[:len(candidates)])
    match_top_k = min(int(step5_data.get("match_top_k", 5)), len(candidates))
    candidates = candidates[:match_top_k]
    cos_sims = cos_sims[:match_top_k]

    if not candidates:
        raise ValueError("step6_match_type2 requires non-empty step5 match candidates")
    if not query_images or not cam_top_results:
        raise ValueError("step6_match_type2 requires query_images and cam_top_results from step5 type2")

    cam_ids, main_cam = _ordered_cam_ids(step5_data, config, cam_top_results)
    cam_ids = [c for c in cam_ids if c in query_images]
    if not cam_ids:
        raise ValueError("No matching camera ids found between query_images and cam_top_results")

    cam_rgbs = {}
    for cam_id in cam_ids:
        path = query_images[cam_id]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"query image not found for {cam_id}: {path}")
        cam_rgbs[cam_id] = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    weight = 1.0 / len(cam_ids)
    print(f"  Match top-{match_top_k}: cams={cam_ids}, weight={weight:.3f} each")

    matcher, matcher_type, info_str = _load_matcher(config, dev)
    print(f"  Matcher: {info_str}")
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    gray_tensor_fn = (_make_gray_tensor_fn(dev, eloftr_max_dim)
                      if matcher_type == "eloftr" else None)
    loftr_max_dim = int(fc.get("loftr_max_dim", 840))
    loftr_gray_fn = (_make_loftr_gray_tensor_fn(dev, loftr_max_dim)
                     if matcher_type == "loftr" else None)
    jamma_max_dim = int(fc.get("jamma_max_dim", 840))
    jamma_preprocess_fn = (_make_jamma_preprocess_fn(dev, jamma_max_dim)
                           if matcher_type == "jamma" else None)

    all_match_counts = []
    all_scores = []
    all_cam_scores = []
    all_cam_counts = []
    rank_results = []
    best_score = -1.0
    best_rank = 0

    for rank, cand in enumerate(candidates):
        cam_scores = {}
        cam_counts = {}
        cam_mkpts_q = {}
        cam_mkpts_r = {}
        cam_confs = {}
        cam_ref_rgbs = {}
        cam_entries = {}

        for cam_id in cam_ids:
            rows = cam_top_results.get(cam_id, [])
            entry = rows[rank][0] if rank < len(rows) else cand
            ref_rgb = cv2.cvtColor(cv2.imread(entry["rgb_path"]), cv2.COLOR_BGR2RGB)
            cam_ref_rgbs[cam_id] = ref_rgb
            cam_entries[cam_id] = entry

            try:
                mkpts0, mkpts1, confs, n_good, score = _run_single_match(
                    matcher_type, matcher, cam_rgbs[cam_id], ref_rgb,
                    conf_thresh, vismatch_max_dim,
                    gray_tensor_fn=gray_tensor_fn,
                    jamma_preprocess_fn=jamma_preprocess_fn,
                    loftr_gray_fn=loftr_gray_fn,
                )
            except Exception as e:
                print(f"    [{cam_id}] F{rank+1} failed: {e}")
                mkpts0 = np.zeros((0, 2))
                mkpts1 = np.zeros((0, 2))
                confs = np.array([])
                n_good = 0
                score = 0.0

            cam_scores[cam_id] = float(score)
            cam_counts[cam_id] = int(n_good)
            cam_mkpts_q[cam_id] = mkpts0
            cam_mkpts_r[cam_id] = mkpts1
            cam_confs[cam_id] = confs

        combined_score = float(sum(weight * cam_scores[c] for c in cam_ids))
        pnp_cam = max(cam_counts, key=cam_counts.get)
        pnp_count = cam_counts[pnp_cam]
        all_match_counts.append(pnp_count)
        all_scores.append(combined_score)
        all_cam_scores.append(dict(cam_scores))
        all_cam_counts.append(dict(cam_counts))
        rank_results.append({
            "cam_scores": cam_scores,
            "cam_counts": cam_counts,
            "cam_mkpts_q": cam_mkpts_q,
            "cam_mkpts_r": cam_mkpts_r,
            "cam_confs": cam_confs,
            "cam_ref_rgbs": cam_ref_rgbs,
            "cam_entries": cam_entries,
            "pnp_cam": pnp_cam,
        })

        cam_log = "  ".join(f"{c}:n={cam_counts[c]},s={cam_scores[c]:.1f}"
                            for c in cam_ids)
        ret_sim = cos_sims[rank] if rank < len(cos_sims) else 0.0
        print(f"  F{rank+1} #{cand['id']:>4}: combined={combined_score:>7.2f}  "
              f"pnp={pnp_cam}({pnp_count})  ret_sim={ret_sim:.4f}  [{cam_log}]")

        if combined_score > best_score:
            best_score = combined_score
            best_rank = rank

    best = rank_results[best_rank]
    best_cand = candidates[best_rank]
    best_cam_id = best["pnp_cam"]
    best_mkpts_q = best["cam_mkpts_q"][best_cam_id]
    best_mkpts_r = best["cam_mkpts_r"][best_cam_id]
    best_confs = best["cam_confs"][best_cam_id]
    best_n = best["cam_counts"][best_cam_id]
    best_ref_rgb = best["cam_ref_rgbs"][best_cam_id]
    best_cam_entries = best["cam_entries"]
    active_cams = [c for c in cam_ids if best["cam_counts"].get(c, 0) > 0]

    print(f"\n  Best: F{best_rank+1} #{best_cand['id']}  "
          f"combined={best_score:.2f}  pnp={best_cam_id}({best_n})")

    matcher_name = fc.get("matcher_name", "eloftr")

    if save_images:
        n_rows = len(cam_ids)
        fig, axes = plt.subplots(n_rows, 1, figsize=(16, 5.2 * n_rows),
                                 squeeze=False)
        for row, cam_id in enumerate(cam_ids):
            entry = best_cam_entries[cam_id]
            _draw_matches(
                axes[row][0],
                cam_rgbs[cam_id],
                best["cam_ref_rgbs"][cam_id],
                best["cam_mkpts_q"][cam_id],
                best["cam_mkpts_r"][cam_id],
                best["cam_confs"][cam_id],
                f"[{cam_id}] F{best_rank+1} render #{entry['id']}  "
                f"matches={best['cam_counts'][cam_id]}  "
                f"score={best['cam_scores'][cam_id]:.2f}",
            )
        fig.suptitle(f"Step 6 type2 [{matcher_name}]: best=F{best_rank+1} "
                     f"#{best_cand['id']}  combined={best_score:.2f}",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(output_dir, "step6_matching.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: step6_matching.png")

        x = np.arange(len(candidates))
        xlabels = [f"F{i+1}\n#{c['id']}" for i, c in enumerate(candidates)]
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
        colors = ["green" if i == best_rank else "steelblue"
                  for i in range(len(candidates))]
        axes2[0].bar(xlabels, all_match_counts, color=colors)
        axes2[0].set_ylabel(f"Match count ({best_cam_id} for best)")
        axes2[0].set_title("PnP cam match count")

        bar_w = 0.8 / (len(cam_ids) + 1)
        for ci, cam_id in enumerate(cam_ids):
            cam_sc = [d.get(cam_id, 0.0) * weight for d in all_cam_scores]
            axes2[1].bar(x + ci * bar_w, cam_sc, width=bar_w,
                         label=f"{cam_id}(x{weight:.2f})", alpha=0.8)
        axes2[1].bar(x + len(cam_ids) * bar_w, all_scores, width=bar_w,
                     label="combined", color="red", alpha=0.7)
        axes2[1].set_xticks(x + bar_w * len(cam_ids) / 2)
        axes2[1].set_xticklabels(xlabels)
        axes2[1].set_ylabel("Feature score")
        axes2[1].set_title("Per-view weighted score + combined")
        axes2[1].legend(fontsize=8)
        fig2.suptitle(f"Step 6 type2 [{matcher_name}]: best=F{best_rank+1} "
                      f"score={best_score:.2f}", fontsize=11)
        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, "step6_match_counts.png"),
                     dpi=150)
        plt.close(fig2)
        print("  Saved: step6_match_counts.png")

    data = {
        "matched_q_kps":    best_mkpts_q,
        "matched_r_kps":    best_mkpts_r,
        "confidences":      best_confs,
        "best_cand":        best_cand,
        "query_rgb":        cam_rgbs.get(main_cam, next(iter(cam_rgbs.values()))),
        "ref_rgb":          best_ref_rgb,
        "all_match_counts": all_match_counts,
        "all_scores":       all_scores,
        "all_cam_scores":   all_cam_scores,
        "all_cam_counts":   all_cam_counts,
        "candidates":       candidates,
        "matcher_name":     matcher_name,
        "best_cam_id":      best_cam_id,
        "best_rank":        best_rank,
        "best_final_rank":  best_rank + 1,
        "cam_mkpts_q":      best["cam_mkpts_q"],
        "cam_mkpts_r":      best["cam_mkpts_r"],
        "cam_entries":      best_cam_entries,
        "active_cams":      active_cams,
        "mc_primary":       main_cam,
        "retrieval_type":   "type2",
        "match_weights":    {c: weight for c in cam_ids},
    }
    if save_images:
        pickle.dump(data, open(os.path.join(output_dir, "step6_data.pkl"), "wb"))
    return data
