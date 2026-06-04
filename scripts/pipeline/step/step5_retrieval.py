import os, pickle
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .step3_global_desc import (_extract_megaloc_desc, _extract_megaloc_spatial,
                                _extract_mixvpr_desc, _extract_mixvpr_spatial,
                                _extract_depth_spatial, _load_query_depth,
                                _load_mixvpr_model)
from .multi_cam import infer_cam_id_from_path


def step5_retrieval(query_image_path, db, config, output_dir, save_images=True,
                    query_images=None):
    """Query 이미지 → global descriptor → KDTree top-K retrieval.

    Args:
        query_image_path: 단일 쿼리 이미지 경로 (primary cam).
        query_images:     {cam_id: path} 멀티캠 딕셔너리 (None = 단일 캠).
                          primary cam 경로는 query_image_path와 일치해야 함.
    """
    import torch
    print("\n" + "="*60 + "\nSTEP 5: Global retrieval\n" + "="*60)
    fc    = config["features"]
    top_k = config.get("matching", {}).get("top_k_retrieval", 10)
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_cam_id = infer_cam_id_from_path(
        query_image_path, config.get("multi_cam", {}).get("cam_ids"))

    # multi-cam 여부
    is_multi = query_images is not None and len(query_images) > 1
    if is_multi:
        print(f"  Mode: multi-cam {list(query_images.keys())}")

    if query_image_path and os.path.isfile(query_image_path):
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

    # ── 헬퍼: 단일 descriptor로 KDTree 쿼리 + depth fusion ───────────────
    def _query_kdtree(gd_norm, depth_desc=None):
        actual_k = min(expand_k, len(db["entries"]))
        dists_, idxs_ = db["kdtree"].query(gd_norm, k=actual_k)
        rgb_sims_ = 1.0 - dists_ ** 2 / 2.0
        depth_descs_db = db.get("depth_descs")
        if (depth_desc is not None and depth_descs_db is not None
                and db.get("has_depth", True)):
            depth_sims_ = depth_descs_db[idxs_] @ depth_desc
            sims_ = w_rgb * rgb_sims_ + w_depth * depth_sims_
        else:
            sims_ = rgb_sims_
        return idxs_, sims_

    # ── Step 1: KDTree 쿼리 (단일 / 멀티 캠) ────────────────────────────
    if is_multi:
        # ── 카메라별 가중치 계산 ──────────────────────────────────────────
        mc_cfg      = config.get("multi_cam", {})
        primary_cam = mc_cfg.get("primary_cam", list(query_images.keys())[0])
        primary_w   = float(mc_cfg.get("primary_cam_weight", 0.6))
        cam_list    = list(query_images.keys())
        N           = len(cam_list)
        if N == 1:
            cam_weights = {cam_list[0]: 1.0}
        else:
            other_w     = (1.0 - primary_w) / (N - 1)
            cam_weights = {c: (primary_w if c == primary_cam else other_w)
                           for c in cam_list}
        print(f"  Weights: " +
              "  ".join(f"{c}={w:.2f}" for c, w in cam_weights.items()))

        # ── 카메라별 KDTree 쿼리 → top-k sim 배열 (rank 순 정렬) ──────────
        cam_top_results = {}   # {cam_id: [(entry, sim), ...]} 시각화/로그용
        cam_sorted_sims = {}   # {cam_id: np.array of top-k sims, rank 순}

        for cam_id, cam_path in query_images.items():
            if not os.path.isfile(cam_path):
                raise FileNotFoundError(f"cam image not found: {cam_path}")
            cam_img = cv2.cvtColor(cv2.imread(cam_path), cv2.COLOR_BGR2RGB)
            if global_desc_method == "megaloc":
                cam_gd = (_extract_megaloc_spatial(cam_img, model, dev, grid_n)
                          if use_spatial else _extract_megaloc_desc(cam_img, model, dev))
            elif global_desc_method == "mixvpr":
                cam_gd = (_extract_mixvpr_spatial(cam_img, model, dev, grid_n)
                          if use_spatial else _extract_mixvpr_desc(cam_img, model, dev))
            else:
                raise ValueError(f"Unknown global_desc_method: {global_desc_method}")

            idxs_, sims_ = _query_kdtree(cam_gd, q_depth_desc)
            order_ = np.argsort(-sims_)[:top_k]
            s_idxs = idxs_[order_]
            s_sims = sims_[order_]
            cam_sorted_sims[cam_id] = s_sims
            cam_top_results[cam_id] = [
                (db["entries"][s_idxs[k]], float(s_sims[k]))
                for k in range(len(s_idxs))
            ]

        # ── Dynamic primary: 평균 sim이 높은 cam이 primary ─────────────────
        avg_sims = {cid: float(np.mean(ss))
                    for cid, ss in cam_sorted_sims.items()}
        dynamic_primary = max(avg_sims, key=avg_sims.get)
        if N > 1:
            other_w_dyn = (1.0 - primary_w) / (N - 1)
            cam_weights = {c: (primary_w if c == dynamic_primary else other_w_dyn)
                           for c in cam_list}
        else:
            cam_weights = {dynamic_primary: 1.0}

        cfg_primary = primary_cam
        if dynamic_primary != cfg_primary:
            print(f"  Dynamic primary: {dynamic_primary} ★ "
                  f"(avg_sim={avg_sims[dynamic_primary]:.4f}) "
                  f"← yaml 설정 '{cfg_primary}' "
                  f"(avg_sim={avg_sims[cfg_primary]:.4f}) 대체")
        else:
            print(f"  Dynamic primary: {dynamic_primary} ★ "
                  f"(avg_sim={avg_sims[dynamic_primary]:.4f}, yaml 설정과 동일)")

        for cam_id, s_sims in cam_sorted_sims.items():
            s_idxs_top = [e["id"] for e, _ in cam_top_results[cam_id]]
            print(f"    [{cam_id}] w={cam_weights[cam_id]:.2f}  "
                  f"top1=#{s_idxs_top[0]}  "
                  f"avg_sim={avg_sims[cam_id]:.4f}  "
                  f"top1_sim={cam_top_results[cam_id][0][1]:.4f}")

        # ── Rank-by-rank 합산: combined[k] = Σ_j w_j × sim_j[k] ──────────
        # dynamic primary의 top-k 후보를 기준으로 rank 위치마다 다른 카메라 sim 합산
        primary_entries = [e for e, _ in cam_top_results[dynamic_primary]]
        n_cands = len(primary_entries)
        combined = np.zeros(n_cands)
        for cam_id, s_sims in cam_sorted_sims.items():
            n = min(len(s_sims), n_cands)
            combined[:n] += cam_weights[cam_id] * s_sims[:n]

        # combined score 기준 재정렬
        rerank_order = np.argsort(-combined)
        candidates = [primary_entries[k] for k in rerank_order]
        cos_sims   = combined[rerank_order]
        print(f"  Rank-pairing: dynamic_primary={dynamic_primary}  "
              f"top-{top_k} re-ranked by combined score")

    else:
        # ── 단일 캠 기존 방식 ─────────────────────────────────────────
        actual_expand_k = min(expand_k, len(db["entries"]))
        dists, idxs = db["kdtree"].query(q_gd_norm, k=actual_expand_k)
        rgb_sims    = 1.0 - dists ** 2 / 2.0

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

        order      = np.argsort(-final_sims)[:top_k]
        top_idxs   = idxs[order]
        cos_sims   = final_sims[order]
        candidates = [db["entries"][i] for i in top_idxs]

    # ── Step 2 (공통): 최종 결과 출력 ────────────────────────────────────

    if is_multi:
        # ── 멀티캠 3열 테이블 출력 ──────────────────────────────────────
        cam_list_ordered = list(cam_top_results.keys())
        _primary = dynamic_primary
        # 각 cam별 id→rank 역인덱스 (변경 감지용)
        cam_id_rank = {}
        for cid, results in cam_top_results.items():
            cam_id_rank[cid] = {e["id"]: r+1 for r, (e, _) in enumerate(results)}

        col_w = 36
        sep   = " | "
        hdr_cams = [f"{'  ' + c + (' ★' if c == _primary else ''):^{col_w}}"
                    for c in cam_list_ordered]
        hdr_res  = f"{'combined (weighted avg)':^{col_w}}"
        print("\n  " + sep.join(hdr_cams + [hdr_res]))
        print("  " + "-" * (col_w * (len(cam_list_ordered)+1)
                             + len(sep) * len(cam_list_ordered)))

        n_rows_print = max(top_k, max(len(v) for v in cam_top_results.values()))
        for r in range(n_rows_print):
            cells = []
            for cid in cam_list_ordered:
                rows = cam_top_results[cid]
                if r < len(rows):
                    e, s = rows[r]
                    p = np.array(e["pose"])[:3, 3]
                    cells.append(
                        f"R{r+1} #{e['id']:<4} sim={s:.3f} "
                        f"({p[0]:.1f},{p[1]:.1f})"[:col_w].ljust(col_w))
                else:
                    cells.append(" " * col_w)

            # combined result 열: raw weighted sum + 기여 카메라 + 순위 변동
            if r < len(candidates):
                rc, rs = candidates[r], cos_sims[r]
                # 각 카메라별 기여분 계산 (cam_raw에서 후보 idx 조회)
                # primary cam 대비 순위 변동 감지
                p_rank = cam_id_rank[_primary].get(rc["id"])
                if p_rank is None:
                    chg = " ↑(new)"
                elif p_rank != r + 1:
                    arrow = "↑" if p_rank > r + 1 else "↓"
                    chg = f" {arrow}(p:R{p_rank})"
                else:
                    chg = ""
                gt_str = ""
                if gt_entry:
                    d = np.linalg.norm(np.array(rc["pose"])[:3, 3]
                                       - np.array(gt_entry["pose"])[:3, 3])
                    gt_str = f" GT={d:.1f}m"
                res_cell = f"R{r+1} #{rc['id']:<4} wt={rs:.3f}{gt_str}{chg}"
                cells.append(res_cell[:col_w].ljust(col_w))
            else:
                cells.append(" " * col_w)

            print("  " + sep.join(cells))

        print("  " + "-" * (col_w * (len(cam_list_ordered)+1)
                             + len(sep) * len(cam_list_ordered)))
    else:
        print(f"  Top-{top_k} results:")
        for rank, (cand, sim) in enumerate(zip(candidates, cos_sims)):
            gt_str = ""
            if gt_entry:
                d = np.linalg.norm(np.array(cand["pose"])[:3,3]
                                   - np.array(gt_entry["pose"])[:3,3])
                gt_str = f"  GT_dist={d:.2f}m"
            print(f"    Rank{rank+1}: #{cand['id']}  sim={sim:.4f}{gt_str}")

    if save_images:
        n_show = min(top_k, 5)
        n_cols = n_show + 1   # query 열 + 후보 열

        if is_multi and query_images:
            cam_list = list(query_images.keys())
            _primary = dynamic_primary
            # GridSpec: cam 행 + 구분선 행 + combined 결과 행
            n_cam_rows  = len(cam_list)
            n_total_gs  = n_cam_rows + 1 + 1   # cams + divider + result
            h_ratios    = [4] * n_cam_rows + [0.35] + [4]
            fig = plt.figure(figsize=(4*n_cols, 4*(n_cam_rows+1) + 0.5))
            gs  = matplotlib.gridspec.GridSpec(
                n_total_gs, n_cols,
                height_ratios=h_ratios, hspace=0.5, wspace=0.05,
                figure=fig)

            # ── 개별 cam 행 ───────────────────────────────────────────
            for row, cam_id in enumerate(cam_list):
                q_img   = cv2.cvtColor(cv2.imread(query_images[cam_id]),
                                       cv2.COLOR_BGR2RGB)
                q_label = f"QUERY\n[{cam_id}]"
                row_results = cam_top_results[cam_id][:n_show]

                ax = fig.add_subplot(gs[row, 0])
                ax.imshow(q_img); ax.set_title(q_label, color="blue", fontsize=10)
                ax.axis("off")
                for rank, (cand, sim) in enumerate(row_results):
                    ref_rgb = cv2.cvtColor(cv2.imread(cand["rgb_path"]),
                                           cv2.COLOR_BGR2RGB)
                    p   = np.array(cand["pose"])[:3, 3]
                    col = "green" if rank == 0 else "orange"
                    ax  = fig.add_subplot(gs[row, rank+1])
                    ax.imshow(ref_rgb)
                    ax.set_title(f"R{rank+1} #{cand['id']}  sim={sim:.3f}\n"
                                 f"({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})",
                                 color=col, fontsize=8)
                    ax.axis("off")

            # ── 구분선 행 ─────────────────────────────────────────────
            div_ax = fig.add_subplot(gs[n_cam_rows, :])
            div_ax.axhline(0.5, color="dimgray", lw=1.5)
            div_ax.text(0.5, 0.5,
                        "─── Combined Result (weighted avg) ───",
                        ha="center", va="center", fontsize=10,
                        color="dimgray", fontweight="bold",
                        transform=div_ax.transAxes)
            div_ax.axis("off")

            # ── combined 결과 행 ──────────────────────────────────────
            if _primary in query_images:
                pq_img = cv2.cvtColor(cv2.imread(query_images[_primary]),
                                      cv2.COLOR_BGR2RGB)
            else:
                pq_img = query_rgb
            ax = fig.add_subplot(gs[n_cam_rows+1, 0])
            ax.imshow(pq_img)
            ax.set_title(f"QUERY\n[{_primary}★]", color="darkblue",
                         fontsize=10, fontweight="bold")
            ax.axis("off")
            for rank, (cand, sim) in enumerate(
                    zip(candidates[:n_show], cos_sims[:n_show])):
                ref_rgb = cv2.cvtColor(cv2.imread(cand["rgb_path"]),
                                       cv2.COLOR_BGR2RGB)
                p   = np.array(cand["pose"])[:3, 3]
                col = "green" if rank == 0 else "orange"
                ax  = fig.add_subplot(gs[n_cam_rows+1, rank+1])
                ax.imshow(ref_rgb)
                ax.set_title(f"R{rank+1} #{cand['id']}  wt={sim:.3f}\n"
                             f"({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})",
                             color=col, fontsize=8)
                ax.axis("off")

            mode_str = f"multi-cam {list(query_images.keys())}"
        else:
            # ── 단일 캠: 기존 1행 레이아웃 ────────────────────────────
            fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4),
                                     squeeze=False)
            axes[0][0].imshow(query_rgb)
            axes[0][0].set_title("QUERY", color="blue", fontsize=10)
            axes[0][0].axis("off")
            for rank, (cand, sim) in enumerate(
                    zip(candidates[:n_show], cos_sims[:n_show])):
                ref_rgb = cv2.cvtColor(cv2.imread(cand["rgb_path"]),
                                       cv2.COLOR_BGR2RGB)
                p   = np.array(cand["pose"])[:3, 3]
                col = "green" if rank == 0 else "orange"
                axes[0][rank+1].imshow(ref_rgb)
                axes[0][rank+1].set_title(
                    f"Rank{rank+1} #{cand['id']}  sim={sim:.3f}\n"
                    f"({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})",
                    color=col, fontsize=8)
                axes[0][rank+1].axis("off")
            mode_str = "single-cam"

        fig.suptitle(f"Step 5: KDTree Retrieval — Top-{top_k}  [{mode_str}]",
                     fontsize=12)
        fig.savefig(os.path.join(output_dir, "step5_retrieval.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: step5_retrieval.png")

    data = {
        "query_rgb":        query_rgb,
        "query_gd_norm":    q_gd_norm,
        "candidates":       candidates,
        "cos_sims":         cos_sims.tolist(),
        "gt_entry":         gt_entry,
        "query_image_path": query_image_path,
        "query_cam_id":     query_cam_id,
        "query_images":     query_images,   # multi-cam: {cam_id: path} or None
        # multi-cam: 카메라별 개별 top-k 결과 → step6에서 각 cam이 자기 결과와 매칭
        "cam_top_results":  cam_top_results if is_multi else None,
        # dynamic primary: avg_sim 기준으로 선택된 PnP용 주 카메라
        "dynamic_primary":  dynamic_primary if is_multi else None,
    }
    if save_images:
        pickle.dump(data, open(os.path.join(output_dir,"step5_data.pkl"),"wb"))
    return data
