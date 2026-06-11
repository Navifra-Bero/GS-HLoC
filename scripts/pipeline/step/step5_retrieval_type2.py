"""STEP 5 Type-2: Dependent multi-camera retrieval.

Flow:
  1) main_cam → 전체 DB에서 top-K 리트리벌 (K = matching.top_k_retrieval)
  2) 각 결과 entry의 위치(pose translation) 추출 → 같은 viewpoint 그룹화
  3) sub_cam 후보 풀 = main top-K viewpoint들의 OTHER yaw 항목만
     (정확히 main과 같은 entry는 제외)
  4) sub_cam은 각 main rank의 viewpoint 풀에서 best yaw 선택
     → rank k에서 (main_cam 결과, sub_cam 결과)가 같은 viewpoint·다른 yaw

출력 데이터는 step6_match와 호환:
  - candidates: main top-K
  - cam_top_results[main_cam] = main top-K
  - cam_top_results[sub_cam]  = per-rank best at same viewpoint
  - dynamic_primary = main_cam (type2는 항상 main 고정)
"""
import os, pickle, time
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .step3_global_desc import (_extract_megaloc_desc, _extract_megaloc_spatial,
                                _extract_mixvpr_desc, _extract_mixvpr_spatial,
                                _extract_depth_spatial, _load_query_depth,
                                _load_mixvpr_model)
from .multi_cam import infer_cam_id_from_path


def _build_position_groups(entries, tol):
    """DB entries를 (rounded_x, rounded_y, rounded_z)로 그룹화.

    Returns:
        dict: pos_key → list of entry indices
        list: 각 entry index → pos_key (역인덱스)
    """
    groups = {}
    entry_to_key = []
    for i, e in enumerate(entries):
        if e.get("view_group_id"):
            key = ("group", str(e["view_group_id"]))
        else:
            pose = np.asarray(e["pose"], dtype=np.float64)
            x, y, z = pose[:3, 3]
            key = (round(float(x) / tol) * tol,
                   round(float(y) / tol) * tol,
                   round(float(z) / tol) * tol)
        groups.setdefault(key, []).append(i)
        entry_to_key.append(key)
    return groups, entry_to_key


def _extract_query_desc(rgb_img, method, model, dev, grid_n, use_spatial):
    """전역 디스크립터 추출 (megaloc / mixvpr 분기)."""
    if method == "megaloc":
        return (_extract_megaloc_spatial(rgb_img, model, dev, grid_n)
                if use_spatial else _extract_megaloc_desc(rgb_img, model, dev))
    elif method == "mixvpr":
        return (_extract_mixvpr_spatial(rgb_img, model, dev, grid_n)
                if use_spatial else _extract_mixvpr_desc(rgb_img, model, dev))
    else:
        raise ValueError(f"Unknown global_desc_method: {method}")


def step5_retrieval_type2(query_image_path, db, config, output_dir,
                          save_images=True, query_images=None):
    """Main cam 종속 sub cam 리트리벌.

    Args:
        query_image_path: main cam 쿼리 이미지 경로
        query_images:     {cam_id: path} 멀티캠 딕셔너리. main_cam 키 필수.
    """
    import torch
    def _sync_cuda():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    print("\n" + "="*60 +
          "\nSTEP 5: Type-2 dependent multi-camera retrieval"
          "\n" + "="*60)
    _sync_cuda()
    t_step = time.perf_counter()

    fc    = config["features"]
    mc    = config.get("multi_cam", {})
    top_k = config.get("matching", {}).get("top_k_retrieval", 30)
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main_cam = mc.get("main_cam") or mc.get("primary_cam")
    sub_cams = list(mc.get("sub_cams", []))
    pos_tol  = float(mc.get("position_tol", 0.05))

    if query_images is None or main_cam not in query_images:
        raise ValueError(f"type2 retrieval requires query_images dict with '{main_cam}' key")

    print(f"  main_cam={main_cam}  sub_cams={sub_cams}  top_k={top_k}  pos_tol={pos_tol}")

    # ── 공통 파라미터 ──────────────────────────────────────────────────────
    use_depth        = bool(fc.get("use_depth_desc", False))
    n_bins           = int(fc.get("depth_bins", 32))
    w_rgb            = float(fc.get("depth_rgb_weight", 0.7))
    w_depth          = 1.0 - w_rgb
    cam_h            = int(config.get("camera", {}).get("height", 1200))
    cam_w            = int(config.get("camera", {}).get("width", 1920))

    global_desc_method = db.get("global_desc_method", "mixvpr")

    # ── 모델 로드 (megaloc / mixvpr) ──────────────────────────────────────
    _cache = step5_retrieval_type2.__dict__
    if global_desc_method == "megaloc":
        grid_n      = int(fc.get("dino_grid_n", db["entries"][0].get("dino_grid_n", 1)
                                 if db["entries"] else 1))
        use_spatial = grid_n > 1
        if "_megaloc_model" not in _cache:
            print("  Loading MegaLoc …")
            model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
            model.eval().to(dev)
            _cache["_megaloc_model"] = model
        model = _cache["_megaloc_model"]
    elif global_desc_method == "mixvpr":
        ckpt_path   = fc.get("mixvpr_ckpt", "")
        out_dim     = int(fc.get("mixvpr_out_dim", 512))
        grid_n      = int(fc.get("grid_n", db["entries"][0].get("grid_n", 1)
                                  if db["entries"] else 1))
        use_spatial = grid_n > 1
        if ("_mixvpr_model" not in _cache
                or _cache.get("_mixvpr_ckpt") != ckpt_path
                or _cache.get("_mixvpr_out_dim") != out_dim):
            model = _load_mixvpr_model(ckpt_path, out_dim, dev)
            _cache["_mixvpr_model"]   = model
            _cache["_mixvpr_ckpt"]    = ckpt_path
            _cache["_mixvpr_out_dim"] = out_dim
        model = _cache["_mixvpr_model"]
    else:
        raise ValueError(f"Unsupported global_desc_method: {global_desc_method}")

    depth_grid       = grid_n if use_spatial else 1

    # ── 위치 그룹 인덱스 사전 구성 ────────────────────────────────────────
    pos_groups, entry_to_key = _build_position_groups(db["entries"], pos_tol)
    n_unique_vp = len(pos_groups)
    print(f"  DB: {len(db['entries'])} entries, {n_unique_vp} unique viewpoints")

    # ── Main cam 쿼리 디스크립터 + main_cam DB subset 검색 ───────────────
    # Type-2는 "main_cam 결과의 같은 rig/viewpoint에서 sub_cam을 고르는" 흐름이다.
    # main 후보를 전체 DB에서 뽑으면 cam_0 query가 cam_1/cam_2 DB entry를 main
    # rank로 선택할 수 있고, 그 sibling sub_cam이 엉뚱해 보인다.
    main_path = query_images[main_cam]
    if not os.path.isfile(main_path):
        raise FileNotFoundError(f"main cam image not found: {main_path}")
    main_rgb = cv2.cvtColor(cv2.imread(main_path), cv2.COLOR_BGR2RGB)

    main_gd = _extract_query_desc(main_rgb, global_desc_method,
                                  model, dev, grid_n, use_spatial)

    # depth descriptor (main cam만 사용)
    main_depth_desc = None
    if use_depth:
        depth_map, depth_path = _load_query_depth(main_path, cam_h, cam_w)
        if depth_map is not None:
            main_depth_desc = _extract_depth_spatial(depth_map, depth_grid, n_bins)
            print(f"  main depth loaded: {depth_path}")

    filter_main_by_cam = bool(mc.get("filter_main_by_cam", True))
    if filter_main_by_cam:
        main_search_idxs = np.array([
            i for i, e in enumerate(db["entries"])
            if e.get("cam_id") == main_cam
        ], dtype=np.int64)
        if len(main_search_idxs) == 0:
            print(f"  WARNING: no DB entries for main_cam={main_cam}; "
                  "falling back to all DB entries")
            main_search_idxs = np.arange(len(db["entries"]), dtype=np.int64)
    else:
        main_search_idxs = np.arange(len(db["entries"]), dtype=np.int64)

    main_descs = db["global_descs"][main_search_idxs]
    rgb_sims = main_descs @ main_gd
    depth_descs_db = db.get("depth_descs")
    if (main_depth_desc is not None and depth_descs_db is not None
            and db.get("has_depth", True)):
        depth_sims = depth_descs_db[main_search_idxs] @ main_depth_desc
        main_sims_full = w_rgb * rgb_sims + w_depth * depth_sims
    else:
        main_sims_full = rgb_sims

    main_order = np.argsort(-main_sims_full)[:top_k]
    main_top_idxs = main_search_idxs[main_order]
    main_top_sims = main_sims_full[main_order]

    main_top_results = [
        (db["entries"][int(idx)], float(main_top_sims[i]))
        for i, idx in enumerate(main_top_idxs)
    ]
    print(f"  [main {main_cam}] top-{top_k} retrieved  "
          f"from {len(main_search_idxs)} entries  "
          f"top1=#{db['entries'][int(main_top_idxs[0])]['id']}  "
          f"sim={main_top_sims[0]:.4f}")

    # ── Sub cam 후보 풀 구성: main top-K의 다른 yaw entries ──────────────
    sub_pool_per_rank = []   # rank별 [(db_idx, ...), ...] (other yaws만)
    pool_total = set()
    for idx in main_top_idxs:
        idx = int(idx)
        key = entry_to_key[idx]
        siblings = [j for j in pos_groups.get(key, []) if j != idx]
        sub_pool_per_rank.append(siblings)
        pool_total.update(siblings)
    print(f"  Sub pool: {len(pool_total)} unique entries "
          f"({sum(len(s) for s in sub_pool_per_rank)} with rank duplication)")

    # ── Sub cam 별 처리 ───────────────────────────────────────────────────
    cam_top_results = {main_cam: main_top_results}
    cam_sims_per_rank = {main_cam: main_top_sims.tolist()}

    for sub_cam in sub_cams:
        if sub_cam not in query_images:
            print(f"  WARNING: sub_cam '{sub_cam}' not in query_images, skip")
            continue
        sub_path = query_images[sub_cam]
        if not os.path.isfile(sub_path):
            print(f"  WARNING: {sub_path} not found, skip {sub_cam}")
            continue
        sub_rgb = cv2.cvtColor(cv2.imread(sub_path), cv2.COLOR_BGR2RGB)
        sub_gd  = _extract_query_desc(sub_rgb, global_desc_method,
                                       model, dev, grid_n, use_spatial)

        sub_depth_desc = None
        if use_depth:
            d_map, _ = _load_query_depth(sub_path, cam_h, cam_w)
            if d_map is not None:
                sub_depth_desc = _extract_depth_spatial(d_map, depth_grid, n_bins)

        # 각 main rank의 후보 풀에서 best 선택
        per_rank = []
        per_rank_sims = []
        for rank, siblings in enumerate(sub_pool_per_rank):
            if not siblings:
                # 같은 viewpoint에 다른 yaw가 없는 경우 → main 결과로 fallback
                per_rank.append(main_top_results[rank])
                per_rank_sims.append(main_top_sims[rank])
                continue
            cam_filtered = [
                j for j in siblings
                if db["entries"][j].get("cam_id") in (None, sub_cam)
            ]
            if cam_filtered:
                siblings = cam_filtered

            sib_descs = db["global_descs"][siblings]
            rgb_sub   = sib_descs @ sub_gd
            if (sub_depth_desc is not None and depth_descs_db is not None
                    and db.get("has_depth", True)):
                depth_sub = depth_descs_db[siblings] @ sub_depth_desc
                sub_sims  = w_rgb * rgb_sub + w_depth * depth_sub
            else:
                sub_sims = rgb_sub

            best_local = int(np.argmax(sub_sims))
            best_idx   = siblings[best_local]
            per_rank.append((db["entries"][best_idx], float(sub_sims[best_local])))
            per_rank_sims.append(float(sub_sims[best_local]))

        cam_top_results[sub_cam] = per_rank
        cam_sims_per_rank[sub_cam] = per_rank_sims
        print(f"  [sub {sub_cam}] per-rank best selected, "
              f"avg_sim={np.mean(per_rank_sims):.4f}  "
              f"top1_sim={per_rank_sims[0]:.4f}")

    # ── Main/sub similarity equal-weight 합산 후 최종 rank 재정렬 ─────────────
    # 위쪽 시각화에는 원래 per-cam retrieval 순서를 보여주기 위해 snapshot 보관.
    cam_top_results_original = {
        cam_id: list(results) for cam_id, results in cam_top_results.items()
    }
    active_cam_ids = [main_cam] + [c for c in sub_cams if c in cam_top_results]
    combined_weight = 1.0 / max(len(active_cam_ids), 1)

    combined_scores_original = []
    combined_components_original = []
    for rank in range(len(main_top_results)):
        components = {}
        weighted_sum = 0.0
        for cam_id in active_cam_ids:
            rows = cam_top_results.get(cam_id, [])
            sim = float(rows[rank][1]) if rank < len(rows) else 0.0
            components[cam_id] = sim
            weighted_sum += combined_weight * sim
        combined_scores_original.append(float(weighted_sum))
        combined_components_original.append(components)

    final_order = sorted(
        range(len(combined_scores_original)),
        key=lambda i: (-combined_scores_original[i], i),
    )
    cam_top_results = {
        cam_id: [results[i] for i in final_order if i < len(results)]
        for cam_id, results in cam_top_results.items()
    }
    main_top_results = cam_top_results[main_cam]
    combined_sims = [combined_scores_original[i] for i in final_order]
    combined_components = [combined_components_original[i] for i in final_order]
    combined_source_ranks = [i + 1 for i in final_order]

    print(f"  Combined re-rank: cams={active_cam_ids}, "
          f"weight={combined_weight:.3f} each")
    if main_top_results:
        print(f"  Final top1=#{main_top_results[0][0]['id']}  "
              f"sum_sim={combined_sims[0]:.4f}  "
              f"source_rank=R{combined_source_ranks[0]}")

    # ── 출력 + 시각화 ───────────────────────────────────────────────────────
    if query_image_path and os.path.isfile(query_image_path):
        query_rgb = cv2.cvtColor(cv2.imread(query_image_path), cv2.COLOR_BGR2RGB)
    else:
        query_rgb = main_rgb
    gt_entry = None  # type2는 self-test 미지원

    candidates = [e for e, _ in main_top_results]
    cos_sims   = list(combined_sims)
    match_top_k = min(int(mc.get("match_top_k", 5)), len(candidates))
    match_candidates = candidates[:match_top_k]
    match_cos_sims = cos_sims[:match_top_k]
    match_cam_top_results = {
        cam_id: rows[:match_top_k] for cam_id, rows in cam_top_results.items()
    }
    print(f"  Step6 matching candidates: final top-{match_top_k}")

    # 터미널 표 출력
    cam_list_ordered = [main_cam] + [c for c in sub_cams if c in cam_top_results]
    def _rank_shift_label(source_rank, final_rank):
        delta = int(source_rank) - int(final_rank)
        if delta > 0:
            return f"↑ {delta} step"
        if delta < 0:
            return f"↓ {abs(delta)} step"
        return "same"

    cam_col_w = 30
    final_col_w = 42
    col_widths = [cam_col_w] * len(cam_list_ordered) + [final_col_w]
    sep   = " | "
    hdr = [
        f"{'  ' + c + (' ★(main)' if c == main_cam else ''):^{cam_col_w}}"
        for c in cam_list_ordered
    ]
    hdr.append(f"{'  final combined':^{final_col_w}}")
    print("\n  " + sep.join(hdr))
    print("  " + "-" * (sum(col_widths) + len(sep) * (len(col_widths)-1)))
    for r in range(top_k):
        cells = []
        for cid in cam_list_ordered:
            rows = cam_top_results_original.get(cid, [])
            if r < len(rows):
                e, s = rows[r]
                cells.append(
                    f"R{r+1} #{e['id']:<4} sim={s:.3f}"[:cam_col_w].ljust(cam_col_w))
            else:
                cells.append(" " * cam_col_w)
        if r < len(candidates):
            source_rank = combined_source_ranks[r]
            shift = _rank_shift_label(source_rank, r + 1)
            final_cell = (
                f"F{r+1} #{candidates[r]['id']:<4} "
                f"sum={combined_sims[r]:.3f} (R{source_rank}, {shift})"
            )
            cells.append(final_cell[:final_col_w].ljust(final_col_w))
        else:
            cells.append(" " * final_col_w)
        print("  " + sep.join(cells))
    print("  " + "-" * (sum(col_widths) + len(sep) * (len(col_widths)-1)))

    if save_images:
        n_show = min(top_k, 5)
        n_cols = n_show + 1
        n_cam_rows = len(cam_list_ordered)
        n_rows = n_cam_rows + 2  # per-cam rows + separator + combined row

        fig = plt.figure(figsize=(3.4*n_cols, 2.65*n_cam_rows + 3.05))
        gs = fig.add_gridspec(
            n_rows, n_cols,
            height_ratios=([1.05] * n_cam_rows) + [0.18, 1.08],
            hspace=0.42,
            wspace=0.16,
        )

        for ri, cid in enumerate(cam_list_ordered):
            q_img = cv2.cvtColor(cv2.imread(query_images[cid]), cv2.COLOR_BGR2RGB) \
                    if cid in query_images else query_rgb
            ax = fig.add_subplot(gs[ri, 0])
            ax.imshow(q_img)
            tag = "★ main" if cid == main_cam else "sub"
            ax.set_title(f"QUERY [{cid}] {tag}", color="blue", fontsize=10, pad=9)
            ax.axis("off")
            for rank in range(n_show):
                ax = fig.add_subplot(gs[ri, rank+1])
                if rank >= len(cam_top_results_original[cid]):
                    ax.axis("off"); continue
                cand, sim = cam_top_results_original[cid][rank]
                ref = cv2.cvtColor(cv2.imread(cand["rgb_path"]), cv2.COLOR_BGR2RGB)
                col = "green" if rank == 0 else "orange"
                ax.imshow(ref)
                cand_cam = cand.get("cam_id") or "map"
                ax.set_title(
                    f"R{rank+1} {cand_cam}#{cand['id']}  sim={sim:.3f}",
                    color=col, fontsize=8, pad=8)
                ax.axis("off")

        sep_row = n_cam_rows
        sep_ax = fig.add_subplot(gs[sep_row, :])
        sep_ax.axis("off")
        sep_ax.plot([0.02, 0.98], [0.5, 0.5], color="black", linewidth=1.0,
                    transform=sep_ax.transAxes, clip_on=False)
        sep_ax.text(0.5, 0.62, "retrieval sum result", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", transform=sep_ax.transAxes,
                    bbox=dict(facecolor="white", edgecolor="none", pad=2.0))

        def _make_query_montage():
            imgs = []
            target_w = 360
            for cid in cam_list_ordered:
                path = query_images.get(cid) if query_images else None
                if not path or not os.path.isfile(path):
                    continue
                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                new_h = max(1, int(h * target_w / max(w, 1)))
                imgs.append(cv2.resize(img, (target_w, new_h),
                                       interpolation=cv2.INTER_AREA))
            return np.vstack(imgs) if imgs else query_rgb

        def _make_final_montage(rank):
            imgs = []
            target_w = 360
            for cid in cam_list_ordered:
                rows = cam_top_results.get(cid, [])
                if rank >= len(rows):
                    continue
                entry, _ = rows[rank]
                path = entry.get("rgb_path")
                if not path or not os.path.isfile(path):
                    continue
                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                new_h = max(1, int(h * target_w / max(w, 1)))
                imgs.append(cv2.resize(img, (target_w, new_h),
                                       interpolation=cv2.INTER_AREA))
            return np.vstack(imgs) if imgs else None

        def _final_id_label(rank):
            labels = []
            for cid in cam_list_ordered:
                rows = cam_top_results.get(cid, [])
                if rank < len(rows):
                    labels.append(f"{cid}#{rows[rank][0]['id']}")
            return " / ".join(labels)

        final_row = n_cam_rows + 1
        ax = fig.add_subplot(gs[final_row, 0])
        ax.imshow(_make_query_montage())
        ax.set_title(f"QUERY [combined]\nw={combined_weight:.3f} each",
                     color="purple", fontsize=10, pad=9)
        ax.axis("off")

        for rank in range(n_show):
            ax = fig.add_subplot(gs[final_row, rank+1])
            if rank >= len(candidates):
                ax.axis("off"); continue
            cand = candidates[rank]
            sim = combined_sims[rank]
            src_rank = combined_source_ranks[rank]
            shift = _rank_shift_label(src_rank, rank + 1)
            ref = _make_final_montage(rank)
            if ref is None:
                ax.axis("off"); continue
            col = "green" if rank == 0 else "purple"
            ax.imshow(ref)
            ax.set_title(
                f"F{rank+1} {cand.get('cam_id') or 'map'}#{cand['id']}  "
                f"sum={sim:.3f}  {_final_id_label(rank)}\n"
                f"src R{src_rank} ({shift})",
                color=col, fontsize=8, pad=8)
            ax.axis("off")

        fig.suptitle(f"Step 5 type2: main={main_cam} → sub={sub_cams} "
                     f"(top-{top_k}, pos_tol={pos_tol}m)", fontsize=12)
        fig.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.04)
        fig.savefig(os.path.join(output_dir, "step5_retrieval.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: step5_retrieval.png")

    data = {
        "query_rgb":        query_rgb,
        "query_gd_norm":    main_gd,
        "candidates":       candidates,
        "cos_sims":         cos_sims,
        "gt_entry":         gt_entry,
        "query_image_path": query_image_path,
        "query_cam_id":     infer_cam_id_from_path(query_image_path, [main_cam] + sub_cams) or main_cam,
        "query_images":     query_images,
        "cam_top_results":  cam_top_results,
        "match_top_k":      match_top_k,
        "match_candidates": match_candidates,
        "match_cos_sims":   match_cos_sims,
        "match_cam_top_results": match_cam_top_results,
        "dynamic_primary":  main_cam,        # type2는 main 고정
        "retrieval_type":   "type2",
        "combined_sims":    combined_sims,
        "combined_weights": {c: combined_weight for c in active_cam_ids},
        "combined_components": combined_components,
        "combined_source_ranks": combined_source_ranks,
        "timings": {
            "step5_retrieval_sec": None,
        },
    }
    _sync_cuda()
    data["timings"]["step5_retrieval_sec"] = time.perf_counter() - t_step
    if save_images:
        pickle.dump(data, open(os.path.join(output_dir, "step5_data.pkl"), "wb"))
    return data
