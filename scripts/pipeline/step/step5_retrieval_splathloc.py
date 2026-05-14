"""
SplatHLoc-style Adaptive Coarse-to-Fine Viewpoint Retrieval.

Paper: "Hierarchical Visual Relocalization with Nearest View Synthesis from
Feature Gaussian Splatting" (arXiv:2603.29185), Algorithm 1.

Pipeline:
    1) Coarse retrieval:
       - single-cam   : MixVPR → KDTree top-k1
       - multi-cam type2 : delegate to step5_retrieval_type2 → combined
         weighted ranking; pick **dynamic primary = cam with max avg_sim
         across top-k1** for the GV/fine stage (the rest of the cams are
         not used after this point).
    2) Geometric verification on (every gv_stride-th) candidate via
       sparse matcher (SuperPoint+LightGlue). In multi-cam, GV runs
       between ★cam.query and ★cam's per-rank yaw entry. Best by inlier
       count is tracked; loop exits early if it reaches threshold I.
    3) If best inlier count < threshold I:
         a) Generate k2 randomly perturbed poses around the chosen entry's
            pose (★cam's yaw) within (a°, b m).
         b) Render virtual views from the trained Feature Gaussian map.
         c) Re-extract MixVPR descriptors over virtual views, retrieve
            top-k3 by ★cam's query descriptor, and run GV on those.
         d) Adopt best virtual view if it has more inliers than coarse best;
            re-render RGB+depth at the chosen virtual pose for downstream
            PnP.

Inputs/outputs are compatible with the existing step5 interface so
downstream (step6/step7) plugs in unchanged. The function additionally
exposes per-cam ranking ("cam_top_results") and type2 metadata when
multi-cam is enabled.
"""
import os, pickle, math
from typing import Optional

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .step3_global_desc import (_extract_mixvpr_desc, _extract_mixvpr_spatial,
                                _load_mixvpr_model,
                                _extract_megaloc_desc, _extract_megaloc_spatial)
from .step5_retrieval_type2 import step5_retrieval_type2


# =============================================================================
# FGS runtime renderer (loads gaussians_feature.pt and rasterizes at any pose)
# =============================================================================
class _FGSRenderer:
    """Lightweight FGS rasterizer for inference-time virtual view synthesis."""

    def __init__(self, ckpt_path, device, gs_roll_180=True):
        import torch
        try:
            from gsplat import rasterization
        except ImportError as e:
            raise ImportError("gsplat 미설치: pip install gsplat") from e

        self.device = device
        self.gs_roll_180 = gs_roll_180
        self._rasterization = rasterization

        ckpt = torch.load(ckpt_path, map_location=device)
        self.means     = ckpt["means"].to(device)
        self.scales    = ckpt["scales"].to(device)
        self.quats     = ckpt["quats"].to(device)
        self.opacities = ckpt["opacities"].to(device)
        self.colors_sh = ckpt["colors_sh"].to(device)
        self.sh_degree = int(math.isqrt(self.colors_sh.shape[1])) - 1
        print(f"  [FGS-renderer] {ckpt_path}: {len(self.means):,} gaussians  "
              f"sh_degree={self.sh_degree}")

    @staticmethod
    def _roll_180(pose):
        """Step1 viewpoints use image-down=world-Z, GS training cams use +Z."""
        roll = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float32)
        return pose @ roll

    def render(self, c2w, K, W, H, near=0.3, far=100.0, apply_roll=None,
               with_depth=False):
        """Render RGB (and optional depth) at given c2w pose. Numpy in/out."""
        import torch
        import torch.nn.functional as F
        apply_roll = self.gs_roll_180 if apply_roll is None else apply_roll
        pose = c2w.astype(np.float32)
        if apply_roll:
            pose = self._roll_180(pose)
        viewmat = torch.from_numpy(np.linalg.inv(pose)).unsqueeze(0).float().to(self.device)
        K_t = torch.from_numpy(K.astype(np.float32)).unsqueeze(0).to(self.device)

        mode = "RGB+D" if with_depth else "RGB"
        with torch.no_grad():
            renders, _, _ = self._rasterization(
                means=self.means,
                quats=F.normalize(self.quats, dim=-1),
                scales=torch.exp(self.scales),
                opacities=torch.sigmoid(self.opacities),
                colors=self.colors_sh,
                viewmats=viewmat,
                Ks=K_t,
                width=int(W),
                height=int(H),
                sh_degree=self.sh_degree,
                near_plane=near,
                far_plane=far,
                render_mode=mode,
            )
        rgb = renders[0, ..., :3].clamp(0, 1).cpu().numpy()
        rgb_u8 = (rgb * 255).astype(np.uint8)
        if with_depth:
            depth = renders[0, ..., 3].cpu().numpy().astype(np.float32)
            return rgb_u8, depth
        return rgb_u8


def _find_fgs_ckpt(output_dir):
    """Resolve a feature-Gaussian checkpoint path under output_dir."""
    candidates = [
        os.path.join(output_dir, "gaussians_feature.pt"),
        os.path.join(output_dir, "gaussians.pt"),
    ]
    gs_dir = os.path.join(output_dir, "gaussian")
    if os.path.isdir(gs_dir):
        iter_dirs = sorted(d for d in os.listdir(gs_dir) if d.startswith("iter_"))
        if iter_dirs:
            candidates.append(os.path.join(gs_dir, iter_dirs[-1], "gaussians.pt"))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# =============================================================================
# Pose perturbation (paper §3.2, default strategy = "Random")
# =============================================================================
def _random_rotation_within(angle_deg, rng):
    """Uniform random axis × uniform angle in [0, angle_deg]."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    theta = math.radians(rng.uniform(0.0, angle_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def _perturb_poses(base_c2w, k, angle_deg, dist_m, seed=0):
    """Generate k Random-strategy perturbations of base_c2w (paper Table II)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        R_delta = _random_rotation_within(angle_deg, rng)
        # Uniform random unit-ball translation, scaled to within dist_m radius
        v = rng.normal(size=3)
        v *= rng.uniform(0.0, dist_m) / (np.linalg.norm(v) + 1e-12)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_delta @ base_c2w[:3, :3]
        T[:3, 3]  = base_c2w[:3, 3] + v
        out.append(T)
    return out


# =============================================================================
# Sparse matcher for geometric verification
# =============================================================================
_VISMATCH_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "vismatch")
)


def _load_sparse_matcher(name, device):
    import sys
    if _VISMATCH_ROOT not in sys.path and os.path.isdir(_VISMATCH_ROOT):
        sys.path.insert(0, _VISMATCH_ROOT)
    from vismatch import get_matcher
    return get_matcher(name, device=str(device))


def _gv_inliers(matcher, query_rgb, ref_rgb):
    """Return num_inliers (post-RANSAC) from a sparse matcher result."""
    import torch
    q = torch.from_numpy(query_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    r = torch.from_numpy(ref_rgb.astype(np.float32)   / 255.0).permute(2, 0, 1)
    res = matcher(q, r)
    return int(res.get("num_inliers", 0)), res


# =============================================================================
# Global descriptor helper (matches DB's method)
# =============================================================================
def _make_global_extractor(method, fc, db, dev):
    """Returns a callable img_rgb → np.float32 L2-normalized descriptor."""
    if method == "mixvpr":
        ckpt   = fc.get("mixvpr_ckpt", "")
        outdim = int(fc.get("mixvpr_out_dim", 512))
        gridn  = int(fc.get("grid_n", db["entries"][0].get("grid_n", 1)
                            if db["entries"] else 1))
        model  = _load_mixvpr_model(ckpt, outdim, dev)
        if gridn > 1:
            return lambda img: _extract_mixvpr_spatial(img, model, dev, gridn)
        return lambda img: _extract_mixvpr_desc(img, model, dev)

    if method == "megaloc":
        import torch
        gridn = int(fc.get("dino_grid_n", db["entries"][0].get("dino_grid_n", 1)
                           if db["entries"] else 1))
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        model.eval().to(dev)
        if gridn > 1:
            return lambda img: _extract_megaloc_spatial(img, model, dev, gridn)
        return lambda img: _extract_megaloc_desc(img, model, dev)

    raise ValueError(f"Unknown global_desc_method: {method}")


# =============================================================================
# Main entry
# =============================================================================
def step5_retrieval_splathloc(query_image_path, db, config, output_dir,
                              save_images=True, query_images=None):
    """Adaptive C2F viewpoint retrieval (paper Algorithm 1).

    Signature mirrors `step5_retrieval` so the rest of the pipeline plugs in
    unchanged. Currently single-cam; multi-cam fallback uses the primary cam.
    """
    import torch
    print("\n" + "="*60 + "\nSTEP 5: SplatHLoc Adaptive C2F Retrieval\n" + "="*60)

    fc  = config["features"]
    sh  = config.get("splathloc_retrieval", {})
    cam = config["camera"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k1 = int(sh.get("k1", 10))
    k2 = int(sh.get("k2", 150))
    k3 = int(sh.get("k3", 5))
    a_deg = float(sh.get("perturb_angle_deg", 5.0))
    b_m   = float(sh.get("perturb_dist_m",  0.5))
    I_thr = int(sh.get("inlier_threshold", 150))
    gv_stride = int(sh.get("gv_stride", 10))
    matcher_name = sh.get("sparse_matcher", "superpoint-lightglue")
    seed = int(sh.get("seed", 0))
    top_k_out = int(config.get("matching", {}).get("top_k_retrieval", k1))

    print(f"  Params: k1={k1}  k2={k2}  k3={k3}  a={a_deg}°  b={b_m}m  "
          f"I={I_thr}  stride={gv_stride}  matcher={matcher_name}")

    # ── Decide multi-cam type2 path vs single-cam path ────────────────────
    mc_cfg = config.get("multi_cam", {})
    use_type2 = (
        query_images is not None
        and len(query_images) > 1
        and mc_cfg.get("enabled", False)
        and mc_cfg.get("retrieval_type", "type1") == "type2"
        and (mc_cfg.get("main_cam") or mc_cfg.get("primary_cam")) in query_images
    )

    # ── Build / cache encoders ────────────────────────────────────────────
    cache = step5_retrieval_splathloc.__dict__
    method = db.get("global_desc_method", "mixvpr")
    extractor = cache.get(f"_extractor_{method}")
    if extractor is None:
        extractor = _make_global_extractor(method, fc, db, dev)
        cache[f"_extractor_{method}"] = extractor

    matcher = cache.get("_sparse_matcher")
    if matcher is None:
        print(f"  Loading sparse matcher: {matcher_name} …")
        matcher = _load_sparse_matcher(matcher_name, dev)
        cache["_sparse_matcher"] = matcher

    # ── (1) Coarse retrieval ─────────────────────────────────────────────
    #   multi-cam type2 → delegate to step5_retrieval_type2 for combined ranking
    #   single-cam      → MixVPR KDTree top-k1 (this file's own logic)
    cam_top_results = None
    type2_meta = None
    primary_cam = None
    if use_type2:
        main_cam = mc_cfg.get("main_cam") or mc_cfg.get("primary_cam")
        print(f"  Multi-cam type2 mode: main={main_cam}  "
              f"sub={mc_cfg.get('sub_cams', [])}")
        # Override top_k_retrieval temporarily so type2 returns exactly k1.
        cfg_copy = dict(config)
        cfg_copy["matching"] = dict(config.get("matching", {}))
        cfg_copy["matching"]["top_k_retrieval"] = k1
        t2 = step5_retrieval_type2(query_image_path, db, cfg_copy, output_dir,
                                   save_images=False, query_images=query_images)
        coarse_candidates = list(t2["candidates"])                 # length ≤ k1
        cam_top_results   = t2["cam_top_results"]                  # {cam: [(e,sim),..]}
        combined_sims     = list(t2.get("combined_sims",
                                        t2.get("cos_sims", [])))
        actual_k1 = len(coarse_candidates)
        top_sim   = np.asarray(combined_sims[:actual_k1], dtype=np.float32)
        type2_meta = {
            "combined_sims":         combined_sims,
            "combined_weights":      t2.get("combined_weights", {}),
            "combined_components":   t2.get("combined_components", []),
            "combined_source_ranks": t2.get("combined_source_ranks", []),
        }
        # Dynamic primary = cam with max average sim across top-k1 (paper sim of
        # GV/fine viewpoint, not type2's config-fixed main).
        avg_sims = {}
        for cid, rows in cam_top_results.items():
            sims = [s for (_e, s) in rows[:actual_k1]]
            if sims:
                avg_sims[cid] = float(np.mean(sims))
        dynamic_primary = max(avg_sims, key=avg_sims.get) if avg_sims else main_cam
        primary_cam = dynamic_primary
        for cid, v in sorted(avg_sims.items(), key=lambda kv: -kv[1]):
            star = " ★(GV)" if cid == dynamic_primary else ""
            print(f"    [{cid}] avg_sim(top-{actual_k1})={v:.4f}{star}")
        # Use the dynamic primary's query image for GV/fine
        primary_path = query_images[dynamic_primary]
        query_rgb = cv2.cvtColor(cv2.imread(primary_path), cv2.COLOR_BGR2RGB)
        # GV ref entries = dynamic primary's per-rank entries (not the combined
        # candidates, which are main_cam's yaw)
        gv_entries = [e for (e, _) in cam_top_results[dynamic_primary][:actual_k1]]
        # MixVPR descriptor for fine retrieval re-rank uses dynamic primary too
        q_desc = extractor(query_rgb)
        gt_entry = None
        print(f"  Coarse(type2): top-{actual_k1}  "
              f"combined_top1={top_sim[0]:.4f}  "
              f"combined_top{actual_k1}={top_sim[-1]:.4f}")
    else:
        # Single-cam: pick the primary path (or self-test)
        primary_path = query_image_path
        if not primary_path or not os.path.isfile(primary_path):
            qi = len(db["entries"]) // 3
            gt_entry = db["entries"][qi]
            query_rgb = cv2.cvtColor(cv2.imread(gt_entry["rgb_path"]),
                                     cv2.COLOR_BGR2RGB)
            print(f"  Query: DB entry #{gt_entry['id']} (self-test)")
        else:
            gt_entry = None
            query_rgb = cv2.cvtColor(cv2.imread(primary_path),
                                     cv2.COLOR_BGR2RGB)
            print(f"  Query: {primary_path} "
                  f"({query_rgb.shape[1]}×{query_rgb.shape[0]})")

        q_desc = extractor(query_rgb)
        actual_k1 = min(k1, len(db["entries"]))
        dists, idxs = db["kdtree"].query(q_desc, k=actual_k1)
        rgb_sims = 1.0 - dists ** 2 / 2.0
        order = np.argsort(-rgb_sims)
        top_idx  = idxs[order]
        top_sim  = rgb_sims[order]
        coarse_candidates = [db["entries"][i] for i in top_idx]
        gv_entries = coarse_candidates                              # 단일 cam
        dynamic_primary = None
        print(f"  Coarse: top-{actual_k1} retrieved  "
              f"sim_top1={top_sim[0]:.4f}  sim_top{actual_k1}={top_sim[-1]:.4f}")

    # ── (2) Geometric verification on every gv_stride-th coarse cand ──────
    #   gv_entries[i] = the reference rendered at the dynamic-primary cam's
    #   yaw (multi-cam) or simply coarse_candidates[i] (single-cam).
    best_inliers = 0
    best_idx_in_top = -1
    best_entry = gv_entries[0]
    gv_log = []
    print(f"  GV loop (every {gv_stride}-th of top-{actual_k1}"
          + (f", cam=★{dynamic_primary}" if dynamic_primary else "") + "):")
    for i, cand in enumerate(gv_entries):
        if i % gv_stride != 0 and i != actual_k1 - 1:
            continue
        ref_rgb = cv2.cvtColor(cv2.imread(cand["rgb_path"]), cv2.COLOR_BGR2RGB)
        n_in, _ = _gv_inliers(matcher, query_rgb, ref_rgb)
        gv_log.append({"rank": i + 1, "entry_id": cand["id"], "n_inliers": n_in})
        print(f"    rank{i+1:3d} #{cand['id']:<5} inliers={n_in:4d}")
        if n_in > best_inliers:
            best_inliers   = n_in
            best_idx_in_top = i
            best_entry      = cand
        if best_inliers >= I_thr:
            print(f"    → reached threshold I={I_thr}, stop coarse GV")
            break

    print(f"  Coarse best: rank{best_idx_in_top+1} #{best_entry['id']}  "
          f"inliers={best_inliers}")

    # ── (3) Fine retrieval via virtual views (only if low coverage) ───────
    fine_used = False
    fine_log  = []
    virtual_winner = None
    virtual_rgb    = None
    if best_inliers < I_thr:
        ckpt = _find_fgs_ckpt(output_dir)
        if ckpt is None:
            print("  Fine retrieval skipped: no FGS checkpoint found "
                  "(gaussians_feature.pt / gaussians.pt) under output_dir")
        else:
            renderer = cache.get("_fgs_renderer")
            if renderer is None or cache.get("_fgs_ckpt") != ckpt:
                renderer = _FGSRenderer(
                    ckpt, dev,
                    gs_roll_180=bool(config.get("rendering", {})
                                            .get("gs_roll_180", True)))
                cache["_fgs_renderer"] = renderer
                cache["_fgs_ckpt"]     = ckpt

            base_pose = np.asarray(best_entry["pose"], dtype=np.float64)
            virtual_poses = _perturb_poses(base_pose, k2, a_deg, b_m, seed=seed)

            # Intrinsics: rendered at the same resolution as DB renders
            W = int(cam["width"]); H = int(cam["height"])
            K = np.array([[cam["fx"], 0, cam["cx"]],
                          [0, cam["fy"], cam["cy"]],
                          [0, 0, 1]], dtype=np.float64)

            print(f"  Fine: rendering {k2} virtual views at "
                  f"({a_deg}°, {b_m}m) around #{best_entry['id']} …")
            virtual_imgs = []
            virtual_descs = []
            for j, pose in enumerate(virtual_poses):
                rgb = renderer.render(pose, K, W, H)
                virtual_imgs.append(rgb)
                virtual_descs.append(extractor(rgb))
                if (j + 1) % 25 == 0 or j + 1 == k2:
                    print(f"    rendered {j+1}/{k2}")

            virtual_descs = np.stack(virtual_descs, axis=0).astype(np.float32)
            v_sims = virtual_descs @ q_desc
            v_top  = np.argsort(-v_sims)[:k3]

            print(f"  Fine GV on top-{k3} virtual views:")
            for j in v_top:
                n_in, _ = _gv_inliers(matcher, query_rgb, virtual_imgs[j])
                fine_log.append({"virt_idx": int(j),
                                 "sim": float(v_sims[j]),
                                 "n_inliers": n_in})
                print(f"    virt#{j:3d}  sim={v_sims[j]:.4f}  "
                      f"inliers={n_in:4d}")
                if n_in > best_inliers:
                    best_inliers   = n_in
                    virtual_winner = j
                    virtual_rgb    = virtual_imgs[j]
                    best_entry = {
                        "id":         f"virt_{j}",
                        "pose":       virtual_poses[j],
                        "rgb_path":   None,           # in-memory only
                        "depth_path": None,
                        "is_virtual": True,
                        "source_id":  coarse_candidates[best_idx_in_top]["id"]
                                      if best_idx_in_top >= 0 else None,
                    }
                    # Optional FGS feature: skipped at runtime (no decoder load)
            fine_used = True
            print(f"  Final: inliers={best_inliers}  "
                  f"{'virtual' if virtual_winner is not None else 'coarse winner'}")

    # ── (4) Persist virtual artifact (always, since downstream needs files) ─
    if virtual_winner is not None and virtual_rgb is not None:
        vdir = os.path.join(output_dir, "step5_virtual")
        os.makedirs(vdir, exist_ok=True)
        vpath = os.path.join(vdir, f"virt_{virtual_winner:03d}.png")
        cv2.imwrite(vpath, cv2.cvtColor(virtual_rgb, cv2.COLOR_RGB2BGR))
        # depth at the same virtual pose (step7 lifts 2D→3D from this)
        renderer = cache.get("_fgs_renderer")
        dpath = None
        if renderer is not None:
            W = int(cam["width"]); H = int(cam["height"])
            K = np.array([[cam["fx"], 0, cam["cx"]],
                          [0, cam["fy"], cam["cy"]],
                          [0, 0, 1]], dtype=np.float64)
            _, vdepth = renderer.render(
                np.asarray(best_entry["pose"], dtype=np.float64),
                K, W, H, with_depth=True)
            dpath = os.path.join(vdir, f"virt_{virtual_winner:03d}.npy")
            np.save(dpath, vdepth)
        best_entry["rgb_path"]   = vpath
        best_entry["depth_path"] = dpath
        print(f"  Saved virtual reference: {vpath}"
              + (f"  + depth {dpath}" if dpath else "  (no depth)"))

    # ── (5) Prepare downstream-compatible output ──────────────────────────
    # Combined ranking pool: type2 다중캠이면 type2 candidates, 아니면 단일캠 결과.
    chosen_id = best_entry["id"]
    pool_sims = list(top_sim) if not isinstance(top_sim, list) else top_sim
    others = []
    others_sims = []
    for i, e in enumerate(coarse_candidates):
        if e["id"] != chosen_id:
            others.append(e)
            others_sims.append(float(pool_sims[i]) if i < len(pool_sims) else 0.0)
    n_keep = max(0, top_k_out - 1)
    final_candidates = [best_entry] + others[:n_keep]
    final_sims       = [float(best_inliers)] + others_sims[:n_keep]

    # Visualization
    if save_images:
        n_show = min(5, len(final_candidates))
        fig, ax = plt.subplots(1, n_show + 1, figsize=(4 * (n_show + 1), 4),
                                squeeze=False)
        ax[0][0].imshow(query_rgb)
        ax[0][0].set_title("QUERY", color="blue", fontsize=10); ax[0][0].axis("off")
        for k, e in enumerate(final_candidates[:n_show]):
            if e.get("is_virtual"):
                ax[0][k + 1].imshow(virtual_rgb if virtual_rgb is not None
                                    else np.zeros_like(query_rgb))
                ax[0][k + 1].set_title(f"R{k+1} {e['id']}\n(virtual)",
                                       color="red", fontsize=8)
            else:
                rgb = cv2.cvtColor(cv2.imread(e["rgb_path"]), cv2.COLOR_BGR2RGB)
                ax[0][k + 1].imshow(rgb)
                p = np.array(e["pose"])[:3, 3]
                col = "green" if k == 0 else "orange"
                title = f"R{k+1} #{e['id']}\n({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})"
                ax[0][k + 1].set_title(title, color=col, fontsize=8)
            ax[0][k + 1].axis("off")
        cam_tag = f"  ★cam={dynamic_primary}" if dynamic_primary else ""
        fig.suptitle(
            f"Step 5 SplatHLoc: coarse_best="
            f"{gv_log[0]['n_inliers'] if gv_log else 0}  "
            f"final_inliers={best_inliers}{cam_tag}  "
            f"{'(fine retrieval used)' if fine_used else '(coarse only)'}",
            fontsize=12)
        out_png = os.path.join(output_dir, "step5_retrieval_splathloc.png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {out_png}")

    data = {
        # Existing keys for compatibility with step6/step7
        "query_rgb":        query_rgb,
        "query_gd_norm":    q_desc,
        "candidates":       final_candidates,
        "cos_sims":         final_sims,
        "gt_entry":         gt_entry,
        "query_image_path": primary_path,
        "query_images":     query_images,
        "cam_top_results":  cam_top_results,         # type2 보존 (multi-cam일 때)
        "dynamic_primary":  primary_cam,             # ★cam (avg_sim max)
        "retrieval_type":   "splathloc",
        # SplatHLoc-specific
        "splathloc_gv_log":         gv_log,
        "splathloc_fine_used":      fine_used,
        "splathloc_fine_log":       fine_log,
        "splathloc_virtual_winner": virtual_winner,
        "splathloc_best_inliers":   int(best_inliers),
        "splathloc_params": {
            "k1": k1, "k2": k2, "k3": k3, "a_deg": a_deg, "b_m": b_m,
            "I": I_thr, "gv_stride": gv_stride, "matcher": matcher_name,
        },
    }
    if type2_meta is not None:
        data.update(type2_meta)
    if save_images:
        pickle.dump(data, open(os.path.join(output_dir, "step5_data.pkl"), "wb"))
    return data
