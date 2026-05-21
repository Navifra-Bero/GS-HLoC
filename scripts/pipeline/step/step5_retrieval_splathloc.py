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
from .step2_scaffold_render import (
    _ensure_sgs_path,
    _load_render_context,
    _load_model_to_map_transform,
    _make_camera,
    _filter_camera_records,
    _build_reference_camera_table,
    _select_reference_uid,
)


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
# Scaffold-GS runtime renderer (loads chkpnt_best.pth / iteration checkpoint)
# =============================================================================
def _resolve_scaffold_spec(config, output_dir):
    """Return (model_path, ckpt_path, iteration) for Scaffold-GS fine rendering."""
    sh = config.get("splathloc_retrieval", {})
    render_cfg = config.get("rendering", {})

    model_path = (
        sh.get("sgs_model_path")
        or sh.get("scaffold_gs_model_path")
        or render_cfg.get("sgs_model_path")
        or render_cfg.get("scaffold_gs_model_path")
    )
    ckpt_path = (
        sh.get("sgs_ckpt")
        or sh.get("sgs_ckpt_path")
        or sh.get("scaffold_gs_ckpt")
        or render_cfg.get("sgs_ckpt")
        or render_cfg.get("sgs_ckpt_path")
    )
    iteration = int(sh.get("sgs_iteration", render_cfg.get("sgs_iteration", -1)))

    if not model_path:
        return None
    model_path = os.path.abspath(os.path.expanduser(model_path))
    if ckpt_path:
        ckpt_path = os.path.abspath(os.path.expanduser(ckpt_path))
    if not os.path.isdir(model_path):
        print(f"  [SGS-fine] model path 없음: {model_path}")
        return None
    if ckpt_path and not os.path.exists(ckpt_path):
        print(f"  [SGS-fine] ckpt 없음: {ckpt_path}")
        return None
    return model_path, ckpt_path, iteration


class _ScaffoldGSRenderer:
    """Scaffold-GS renderer used for SplatHLoc fine virtual view synthesis."""

    kind = "scaffold_gs"

    def __init__(self, model_path, ckpt_path, iteration, config, output_dir, device):
        if str(device) != "cuda":
            raise RuntimeError("Scaffold-GS fine renderer requires CUDA.")
        import torch

        self.config = config
        self.output_dir = output_dir
        self.device = device
        self.model_path = model_path
        self.ckpt_path = ckpt_path
        self.iteration = int(iteration)

        _ensure_sgs_path(model_path)
        from gaussian_renderer import render as gs_render, prefilter_voxel
        self.gs_render = gs_render
        self.prefilter_voxel = prefilter_voxel

        step0_path = os.path.join(output_dir, "step0_data.pkl")
        if os.path.exists(step0_path):
            with open(step0_path, "rb") as f:
                step0_data = pickle.load(f)
            T_align = np.array(step0_data.get("T_align", np.eye(4)), dtype=np.float64)
        else:
            T_align = np.eye(4, dtype=np.float64)
            print("  [SGS-fine] step0_data.pkl 없음 → T_align=I 사용")

        T_model_to_map = _load_model_to_map_transform(config, output_dir)
        self.T_model_to_aligned = T_align @ T_model_to_map
        self.T_aligned_to_model = np.linalg.inv(self.T_model_to_aligned)
        if np.linalg.norm(T_model_to_map - np.eye(4)) > 1e-9:
            print("  [SGS-fine] pose: c2w_model = inv(T_align @ model_to_map) @ c2w_aligned")
        else:
            print("  [SGS-fine] pose: c2w_model = inv(T_align) @ c2w_aligned")

        ctx = _load_render_context(model_path, self.iteration, pth_path=ckpt_path)
        self.gaussians = ctx["gaussians"]
        self.pipeline = ctx["pipeline"]
        self.background = ctx["background"]
        self.loaded_iter = ctx["loaded_iter"]
        self.appearance_camera_count = ctx.get("appearance_camera_count")

        train_cameras = ctx.get("train_cameras")
        train_camera_records = ctx.get("train_camera_records")
        ref_source = train_cameras
        if ref_source is None:
            ref_source = _filter_camera_records(train_camera_records or [], split="train")
        if not ref_source:
            ref_source = train_camera_records or []
        self.appearance_refs = _build_reference_camera_table(ref_source)

        torch.cuda.empty_cache()
        print(f"  [SGS-fine] ready: iter={self.loaded_iter}, "
              f"anchors={self.gaussians.get_anchor.shape[0]:,}")

    def render(self, c2w, K, W, H, with_depth=False):
        import torch

        c2w_model = self.T_aligned_to_model @ np.asarray(c2w, dtype=np.float64)
        render_uid = 0
        if getattr(self.gaussians, "appearance_dim", 0) > 0:
            render_uid = _select_reference_uid(c2w_model, self.appearance_refs)
        if self.appearance_camera_count is not None and not (0 <= render_uid < self.appearance_camera_count):
            raise ValueError(
                f"appearance uid out of range: {render_uid} "
                f"(valid: 0..{self.appearance_camera_count - 1})"
            )

        cam = _make_camera(
            c2w_model,
            float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]),
            int(W), int(H),
            uid=int(render_uid),
            image_name="step5_virtual",
        )

        with torch.no_grad():
            voxel_mask = self.prefilter_voxel(
                cam, self.gaussians, self.pipeline, self.background)
            try:
                pkg = self.gs_render(
                    cam, self.gaussians, self.pipeline, self.background,
                    visible_mask=voxel_mask,
                    render_feature=False,
                    render_depth=with_depth)
            except TypeError:
                pkg = self.gs_render(
                    cam, self.gaussians, self.pipeline, self.background,
                    visible_mask=voxel_mask,
                    render_feature=False)

        rendering = torch.clamp(pkg["render"], 0.0, 1.0)
        rgb = rendering.permute(1, 2, 0).detach().cpu().numpy()
        rgb_u8 = (rgb * 255.0).astype(np.uint8)

        if with_depth:
            depth_tensor = pkg.get("depth", None)
            if depth_tensor is None:
                depth_tensor = pkg.get("rendered_depth", None)
            if depth_tensor is not None:
                depth = depth_tensor[0].detach().cpu().numpy().astype(np.float32)
            else:
                depth = None
            del cam, voxel_mask, pkg, rendering
            torch.cuda.empty_cache()
            return rgb_u8, depth

        del cam, voxel_mask, pkg, rendering
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rgb_u8


def _get_fine_renderer(config, output_dir, device, cache):
    """Resolve and cache the fine virtual renderer requested by config."""
    sh = config.get("splathloc_retrieval", {})
    renderer_type = str(sh.get("fine_renderer", "auto")).lower()

    if renderer_type in ("none", "off", "false", "disabled"):
        return None

    if renderer_type in ("auto", "scaffold", "scaffold_gs", "sgs"):
        spec = _resolve_scaffold_spec(config, output_dir)
        if spec is not None:
            if str(device) != "cuda":
                print("  Fine retrieval skipped: Scaffold-GS renderer requires CUDA")
                return None
            model_path, ckpt_path, iteration = spec
            key = (model_path, ckpt_path or "", int(iteration))
            renderer = cache.get("_sgs_fine_renderer")
            if renderer is None or cache.get("_sgs_fine_key") != key:
                renderer = _ScaffoldGSRenderer(
                    model_path, ckpt_path, iteration, config, output_dir, device)
                cache["_sgs_fine_renderer"] = renderer
                cache["_sgs_fine_key"] = key
            return renderer
        if renderer_type not in ("auto",):
            print("  Fine retrieval skipped: Scaffold-GS model/ckpt not configured")
            return None

    if renderer_type in ("auto", "fgs", "feature_gs"):
        ckpt = _find_fgs_ckpt(output_dir)
        if ckpt is not None:
            renderer = cache.get("_fgs_renderer")
            if renderer is None or cache.get("_fgs_ckpt") != ckpt:
                renderer = _FGSRenderer(
                    ckpt, device,
                    gs_roll_180=bool(config.get("rendering", {})
                                            .get("gs_roll_180", True)))
                cache["_fgs_renderer"] = renderer
                cache["_fgs_ckpt"] = ckpt
            renderer.kind = "fgs"
            return renderer

    print("  Fine retrieval skipped: no Scaffold-GS fine renderer configured "
          "and no FGS checkpoint found")
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
    fine_trigger = str(sh.get("fine_trigger", sh.get("gate_metric", "inlier"))).lower()
    sim_threshold = float(sh.get("similarity_threshold", 0.65))
    verify_virtual_with_gv = bool(sh.get(
        "verify_virtual_with_gv",
        fine_trigger in ("inlier", "gv", "geometric"),
    ))
    seed = int(sh.get("seed", 0))
    top_k_out = int(config.get("matching", {}).get("top_k_retrieval", k1))

    print(f"  Params: k1={k1}  k2={k2}  k3={k3}  a={a_deg}°  b={b_m}m  "
          f"I={I_thr}  stride={gv_stride}  matcher={matcher_name}")
    print(f"  Fine trigger: {fine_trigger}"
          + (f"  sim_thr={sim_threshold:.3f}" if fine_trigger in ("similarity", "sim", "retrieval") else "")
          + f"  virtual_gv={verify_virtual_with_gv}")

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

    matcher = None

    def _get_sparse_matcher():
        nonlocal matcher
        if matcher is None:
            matcher = cache.get("_sparse_matcher")
        if matcher is None:
            print(f"  Loading sparse matcher: {matcher_name} …")
            matcher = _load_sparse_matcher(matcher_name, dev)
            cache["_sparse_matcher"] = matcher
        return matcher

    # ── (1) Coarse retrieval ─────────────────────────────────────────────
    #   multi-cam type2 → delegate to step5_retrieval_type2 for combined ranking
    #   single-cam      → MixVPR KDTree top-k1 (this file's own logic)
    cam_top_results = None
    type2_meta = None
    primary_cam = None
    main_cam = None
    active_cam_ids = []
    query_rgb_by_cam = {}
    q_desc_by_cam = {}
    if use_type2:
        main_cam = mc_cfg.get("main_cam") or mc_cfg.get("primary_cam")
        print(f"  Multi-cam type2 mode: main={main_cam}  "
              f"sub={mc_cfg.get('sub_cams', [])}")
        # Override top_k_retrieval temporarily so type2 returns exactly k1.
        cfg_copy = dict(config)
        cfg_copy["matching"] = dict(config.get("matching", {}))
        cfg_copy["matching"]["top_k_retrieval"] = k1
        t2 = step5_retrieval_type2(query_image_path, db, cfg_copy, output_dir,
                                   save_images=save_images, query_images=query_images)
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
        active_cam_ids = [main_cam] + [
            c for c in mc_cfg.get("sub_cams", []) if c in cam_top_results
        ]
        # SplatHLoc GV/fine is anchored on the configured main camera. The
        # virtual fine stage still scores all rig-connected cameras jointly.
        avg_sims = {}
        for cid, rows in cam_top_results.items():
            sims = [s for (_e, s) in rows[:actual_k1]]
            if sims:
                avg_sims[cid] = float(np.mean(sims))
        dynamic_primary = main_cam
        primary_cam = main_cam
        for cid, v in sorted(avg_sims.items(), key=lambda kv: -kv[1]):
            star = " ★(main/GV)" if cid == dynamic_primary else ""
            print(f"    [{cid}] avg_sim(top-{actual_k1})={v:.4f}{star}")
        # Use main cam for the coarse GV gate; cache all cam query descriptors
        # for multi-view fine virtual ranking.
        for cid in active_cam_ids:
            if cid not in query_images:
                continue
            img = cv2.cvtColor(cv2.imread(query_images[cid]), cv2.COLOR_BGR2RGB)
            query_rgb_by_cam[cid] = img
            q_desc_by_cam[cid] = extractor(img)
        primary_path = query_images[main_cam]
        query_rgb = query_rgb_by_cam[main_cam]
        q_desc = q_desc_by_cam[main_cam]
        gv_entries = [e for (e, _) in cam_top_results[main_cam][:actual_k1]]
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

    # ── (2) Fine trigger gate ──────────────────────────────────────────────
    #   "inlier"     : paper-style GV with SuperPoint+LightGlue
    #   "similarity" : skip GV and trigger fine only when retrieval sim is low
    best_inliers = 0
    best_idx_in_top = 0
    best_entry = gv_entries[0]
    gv_log = []
    coarse_top_sim = float(top_sim[0]) if len(top_sim) else 0.0
    use_inlier_gate = fine_trigger in ("inlier", "gv", "geometric")
    if use_inlier_gate:
        print(f"  GV loop (every {gv_stride}-th of top-{actual_k1}"
              + (f", cam=★{dynamic_primary}" if dynamic_primary else "") + "):")
        matcher_obj = _get_sparse_matcher()
        for i, cand in enumerate(gv_entries):
            if i % gv_stride != 0 and i != actual_k1 - 1:
                continue
            ref_rgb = cv2.cvtColor(cv2.imread(cand["rgb_path"]), cv2.COLOR_BGR2RGB)
            n_in, _ = _gv_inliers(matcher_obj, query_rgb, ref_rgb)
            gv_log.append({"rank": i + 1, "entry_id": cand["id"], "n_inliers": n_in})
            print(f"    rank{i+1:3d} #{cand['id']:<5} inliers={n_in:4d}")
            if n_in > best_inliers:
                best_inliers   = n_in
                best_idx_in_top = i
                best_entry      = cand
            if best_inliers >= I_thr:
                print(f"    → reached threshold I={I_thr}, stop coarse GV")
                break
        fine_needed = best_inliers < I_thr
        print(f"  Coarse best: rank{best_idx_in_top+1} #{best_entry['id']}  "
              f"inliers={best_inliers}")
    elif fine_trigger in ("similarity", "sim", "retrieval"):
        fine_needed = coarse_top_sim < sim_threshold
        gv_log.append({
            "rank": 1,
            "entry_id": best_entry["id"],
            "similarity": coarse_top_sim,
            "gv_skipped": True,
        })
        print(f"  Coarse best: rank1 #{best_entry['id']}  sim={coarse_top_sim:.4f}")
        print(f"  Similarity gate: {coarse_top_sim:.4f} "
              f"{'<' if fine_needed else '>='} {sim_threshold:.4f} → "
              f"{'run fine' if fine_needed else 'skip fine'}")
    else:
        raise ValueError(f"Unknown splathloc_retrieval.fine_trigger: {fine_trigger}")

    # ── (3) Fine retrieval via virtual views (only if low coverage) ───────
    fine_used = False
    fine_log  = []
    virtual_winner = None
    virtual_rgb    = None
    fine_renderer_kind = None
    virtual_cam_imgs = {}
    virtual_cam_poses = {}
    virtual_cam_entries = {}
    virtual_top = []
    virtual_sims_by_cam = {}
    virtual_combined_sims = None
    if fine_needed:
        renderer = _get_fine_renderer(config, output_dir, dev, cache)
        if renderer is not None:
            fine_renderer_kind = getattr(renderer, "kind", renderer.__class__.__name__)
            base_pose = np.asarray(best_entry["pose"], dtype=np.float64)
            virtual_poses = _perturb_poses(base_pose, k2, a_deg, b_m, seed=seed)

            # Intrinsics: rendered at the same resolution as DB renders
            W = int(cam["width"]); H = int(cam["height"])
            K = np.array([[cam["fx"], 0, cam["cx"]],
                          [0, cam["fy"], cam["cy"]],
                          [0, 0, 1]], dtype=np.float64)

            print(f"  Fine({fine_renderer_kind}): rendering {k2} virtual views at "
                  f"({a_deg}°, {b_m}m) around #{best_entry['id']} …")

            if use_type2 and cam_top_results and active_cam_ids:
                base_poses = {}
                base_main = np.asarray(
                    cam_top_results[main_cam][best_idx_in_top][0]["pose"],
                    dtype=np.float64,
                )
                base_poses[main_cam] = base_main
                rel_from_main = {main_cam: np.eye(4, dtype=np.float64)}
                for cid in active_cam_ids:
                    if cid == main_cam:
                        continue
                    rows = cam_top_results.get(cid, [])
                    if best_idx_in_top < len(rows):
                        pose_c = np.asarray(rows[best_idx_in_top][0]["pose"],
                                            dtype=np.float64)
                        base_poses[cid] = pose_c
                        rel_from_main[cid] = np.linalg.inv(base_main) @ pose_c

                virtual_cam_imgs = {cid: [] for cid in rel_from_main}
                virtual_cam_poses = {cid: [] for cid in rel_from_main}
                virtual_descs_by_cam = {cid: [] for cid in rel_from_main}
                for j, pose_main in enumerate(virtual_poses):
                    for cid, rel in rel_from_main.items():
                        pose_c = pose_main @ rel
                        rgb = renderer.render(pose_c, K, W, H)
                        virtual_cam_imgs[cid].append(rgb)
                        virtual_cam_poses[cid].append(pose_c)
                        virtual_descs_by_cam[cid].append(extractor(rgb))
                    if (j + 1) % 25 == 0 or j + 1 == k2:
                        print(f"    rendered {j+1}/{k2} rig poses "
                              f"({len(rel_from_main)} cams)")

                weights = {}
                meta_weights = (type2_meta or {}).get("combined_weights", {})
                for cid in rel_from_main:
                    weights[cid] = float(meta_weights.get(cid, 1.0 / len(rel_from_main)))
                w_sum = sum(weights.values()) or 1.0
                weights = {cid: w / w_sum for cid, w in weights.items()}

                virtual_combined_sims = np.zeros(k2, dtype=np.float32)
                for cid, descs in virtual_descs_by_cam.items():
                    descs_np = np.stack(descs, axis=0).astype(np.float32)
                    sims = descs_np @ q_desc_by_cam.get(cid, q_desc)
                    virtual_sims_by_cam[cid] = sims
                    virtual_combined_sims += float(weights[cid]) * sims
                v_sims = virtual_combined_sims
                v_top = np.argsort(-v_sims)[:k3]
                virtual_top = [int(x) for x in v_top]

                print(f"  Fine multi-view rank: cams={list(rel_from_main.keys())}, "
                      + ", ".join(f"{c}=w{weights[c]:.3f}" for c in rel_from_main))
                if verify_virtual_with_gv:
                    print(f"  Fine GV on top-{k3} virtual rig views:")
                    matcher_obj = _get_sparse_matcher()
                    for j in v_top:
                        cam_inliers = {}
                        combined_inliers = 0.0
                        for cid in rel_from_main:
                            n_in, _ = _gv_inliers(
                                matcher_obj,
                                query_rgb_by_cam.get(cid, query_rgb),
                                virtual_cam_imgs[cid][j],
                            )
                            cam_inliers[cid] = int(n_in)
                            combined_inliers += float(weights[cid]) * float(n_in)
                        n_cmp = int(round(combined_inliers))
                        fine_log.append({
                            "virt_idx": int(j),
                            "sim": float(v_sims[j]),
                            "n_inliers": n_cmp,
                            "verified": True,
                            "cam_inliers": cam_inliers,
                            "cam_sims": {
                                cid: float(virtual_sims_by_cam[cid][j])
                                for cid in rel_from_main
                            },
                        })
                        cam_txt = "  ".join(f"{c}:sim={virtual_sims_by_cam[c][j]:.3f},"
                                            f"in={cam_inliers[c]}"
                                            for c in rel_from_main)
                        print(f"    virt#{j:3d}  sum={v_sims[j]:.4f}  "
                              f"inliers={n_cmp:4d}  [{cam_txt}]")
                        if n_cmp > best_inliers:
                            best_inliers = n_cmp
                            virtual_winner = int(j)
                            virtual_rgb = virtual_cam_imgs[main_cam][j]
                            best_entry = {
                                "id":         f"virt_{j}",
                                "pose":       virtual_cam_poses[main_cam][j],
                                "rgb_path":   None,
                                "depth_path": None,
                                "is_virtual": True,
                                "source_id":  coarse_candidates[best_idx_in_top]["id"]
                                              if best_idx_in_top >= 0 else None,
                                "fine_renderer": fine_renderer_kind,
                            }
                else:
                    print(f"  Fine selected by retrieval similarity (top-{k3}, GV skipped):")
                    for rank, j in enumerate(v_top, start=1):
                        fine_log.append({
                            "virt_idx": int(j),
                            "sim": float(v_sims[j]),
                            "n_inliers": 0,
                            "verified": False,
                            "cam_sims": {
                                cid: float(virtual_sims_by_cam[cid][j])
                                for cid in rel_from_main
                            },
                        })
                        cam_txt = "  ".join(f"{c}:sim={virtual_sims_by_cam[c][j]:.3f}"
                                            for c in rel_from_main)
                        print(f"    F{rank} virt#{j:3d}  sum={v_sims[j]:.4f}  [{cam_txt}]")
                    if len(v_top) > 0:
                        j = int(v_top[0])
                        virtual_winner = j
                        virtual_rgb = virtual_cam_imgs[main_cam][j]
                        best_entry = {
                            "id":         f"virt_{j}",
                            "pose":       virtual_cam_poses[main_cam][j],
                            "rgb_path":   None,
                            "depth_path": None,
                            "is_virtual": True,
                            "source_id":  coarse_candidates[best_idx_in_top]["id"]
                                          if best_idx_in_top >= 0 else None,
                            "fine_renderer": fine_renderer_kind,
                        }
            else:
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
                virtual_top = [int(x) for x in v_top]

                if verify_virtual_with_gv:
                    print(f"  Fine GV on top-{k3} virtual views:")
                    matcher_obj = _get_sparse_matcher()
                    for j in v_top:
                        n_in, _ = _gv_inliers(matcher_obj, query_rgb, virtual_imgs[j])
                        fine_log.append({"virt_idx": int(j),
                                         "sim": float(v_sims[j]),
                                         "n_inliers": n_in,
                                         "verified": True})
                        print(f"    virt#{j:3d}  sim={v_sims[j]:.4f}  "
                              f"inliers={n_in:4d}")
                        if n_in > best_inliers:
                            best_inliers = n_in
                            virtual_winner = int(j)
                            virtual_rgb = virtual_imgs[j]
                            best_entry = {
                                "id":         f"virt_{j}",
                                "pose":       virtual_poses[j],
                                "rgb_path":   None,           # in-memory only
                                "depth_path": None,
                                "is_virtual": True,
                                "source_id":  coarse_candidates[best_idx_in_top]["id"]
                                              if best_idx_in_top >= 0 else None,
                                "fine_renderer": fine_renderer_kind,
                            }
                else:
                    print(f"  Fine selected by retrieval similarity (top-{k3}, GV skipped):")
                    for rank, j in enumerate(v_top, start=1):
                        fine_log.append({"virt_idx": int(j),
                                         "sim": float(v_sims[j]),
                                         "n_inliers": 0,
                                         "verified": False})
                        print(f"    F{rank} virt#{j:3d}  sim={v_sims[j]:.4f}")
                    if len(v_top) > 0:
                        j = int(v_top[0])
                        virtual_winner = j
                        virtual_rgb = virtual_imgs[j]
                        best_entry = {
                            "id":         f"virt_{j}",
                            "pose":       virtual_poses[j],
                            "rgb_path":   None,
                            "depth_path": None,
                            "is_virtual": True,
                            "source_id":  coarse_candidates[best_idx_in_top]["id"]
                                          if best_idx_in_top >= 0 else None,
                            "fine_renderer": fine_renderer_kind,
                        }
            fine_used = True
            print(f"  Final: inliers={best_inliers}  "
                  f"{'virtual' if virtual_winner is not None else 'coarse winner'}")

    # ── (4) Persist virtual artifact (always, since downstream needs files) ─
    if virtual_winner is not None and virtual_rgb is not None:
        vdir = os.path.join(output_dir, "step5_virtual")
        os.makedirs(vdir, exist_ok=True)
        if fine_renderer_kind in ("scaffold_gs", "_ScaffoldGSRenderer"):
            renderer = cache.get("_sgs_fine_renderer")
        else:
            renderer = cache.get("_fgs_renderer")
        W = int(cam["width"]); H = int(cam["height"])
        K = np.array([[cam["fx"], 0, cam["cx"]],
                      [0, cam["fy"], cam["cy"]],
                      [0, 0, 1]], dtype=np.float64)

        if use_type2 and virtual_cam_imgs:
            for cid, imgs in virtual_cam_imgs.items():
                if virtual_winner >= len(imgs):
                    continue
                suffix = f"virt_{virtual_winner:03d}_{cid}"
                vpath = os.path.join(vdir, f"{suffix}.png")
                cv2.imwrite(vpath, cv2.cvtColor(imgs[virtual_winner], cv2.COLOR_RGB2BGR))
                dpath = None
                pose_c = np.asarray(virtual_cam_poses[cid][virtual_winner],
                                    dtype=np.float64)
                if renderer is not None:
                    _, vdepth = renderer.render(pose_c, K, W, H, with_depth=True)
                    if vdepth is not None:
                        dpath = os.path.join(vdir, f"{suffix}.npy")
                        np.save(dpath, vdepth)
                entry = {
                    "id": f"virt_{virtual_winner}_{cid}",
                    "pose": pose_c,
                    "rgb_path": vpath,
                    "depth_path": dpath,
                    "is_virtual": True,
                    "source_id": coarse_candidates[best_idx_in_top]["id"]
                                 if best_idx_in_top >= 0 else None,
                    "fine_renderer": fine_renderer_kind,
                    "cam_id": cid,
                }
                virtual_cam_entries[cid] = entry
            if main_cam in virtual_cam_entries:
                best_entry = virtual_cam_entries[main_cam]
                virtual_rgb = virtual_cam_imgs[main_cam][virtual_winner]
            print(f"  Saved virtual rig reference: {vdir}/virt_{virtual_winner:03d}_*.png")
        else:
            vpath = os.path.join(vdir, f"virt_{virtual_winner:03d}.png")
            cv2.imwrite(vpath, cv2.cvtColor(virtual_rgb, cv2.COLOR_RGB2BGR))
            dpath = None
            if renderer is not None:
                _, vdepth = renderer.render(
                    np.asarray(best_entry["pose"], dtype=np.float64),
                    K, W, H, with_depth=True)
                if vdepth is not None:
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

    match_cam_top_results = None
    match_candidates = None
    match_cos_sims = None
    match_top_k = None
    if use_type2 and cam_top_results:
        match_top_k = min(
            int(config.get("multi_cam", {}).get("match_top_k", top_k_out)),
            len(final_candidates),
        )
        match_candidates = final_candidates[:match_top_k]
        match_cos_sims = final_sims[:match_top_k]
        match_cam_top_results = {}
        for cid, rows in cam_top_results.items():
            rebuilt = []
            if virtual_cam_entries and cid in virtual_cam_entries:
                sim = 0.0
                if virtual_sims_by_cam and cid in virtual_sims_by_cam and virtual_winner is not None:
                    sim = float(virtual_sims_by_cam[cid][virtual_winner])
                rebuilt.append((virtual_cam_entries[cid], sim))
            elif final_candidates and final_candidates[0].get("is_virtual"):
                rebuilt.append((final_candidates[0], final_sims[0]))

            for e, s in rows:
                if len(rebuilt) >= match_top_k:
                    break
                if any(str(e["id"]) == str(prev[0]["id"]) for prev in rebuilt):
                    continue
                rebuilt.append((e, s))
            match_cam_top_results[cid] = rebuilt[:match_top_k]

    # Visualization
    if save_images:
        out_png = os.path.join(output_dir, "step5_retrieval_splathloc.png")
        if use_type2 and virtual_cam_imgs and virtual_top:
            show_idxs = virtual_top[:min(5, len(virtual_top))]
            cam_rows = [c for c in active_cam_ids if c in virtual_cam_imgs]
            n_cols = len(show_idxs) + 1
            n_cam_rows = len(cam_rows)
            n_rows = n_cam_rows + 2
            fig = plt.figure(figsize=(3.4 * n_cols, 2.65 * n_cam_rows + 3.05))
            gs = fig.add_gridspec(
                n_rows, n_cols,
                height_ratios=([1.05] * n_cam_rows) + [0.18, 1.08],
                hspace=0.42,
                wspace=0.16,
            )

            for ri, cid in enumerate(cam_rows):
                ax0 = fig.add_subplot(gs[ri, 0])
                ax0.imshow(query_rgb_by_cam.get(cid, query_rgb))
                tag = "★ main" if cid == main_cam else "sub"
                ax0.set_title(f"QUERY [{cid}] {tag}", color="blue", fontsize=10, pad=9)
                ax0.axis("off")
                for col, vidx in enumerate(show_idxs, start=1):
                    axv = fig.add_subplot(gs[ri, col])
                    axv.imshow(virtual_cam_imgs[cid][vidx])
                    sim = virtual_sims_by_cam.get(cid, np.zeros(k2))[vidx]
                    color = "green" if vidx == virtual_winner else "orange"
                    axv.set_title(f"V{col} virt_{vidx}\nsim={sim:.3f}",
                                  color=color, fontsize=8, pad=8)
                    axv.axis("off")

            sep_row = n_cam_rows
            sep_ax = fig.add_subplot(gs[sep_row, :])
            sep_ax.axis("off")
            sep_ax.plot([0.02, 0.98], [0.5, 0.5], color="black", linewidth=1.0,
                        transform=sep_ax.transAxes, clip_on=False)
            sep_ax.text(0.5, 0.62, "fine virtual sum result", ha="center",
                        va="bottom", fontsize=11, fontweight="bold",
                        transform=sep_ax.transAxes,
                        bbox=dict(facecolor="white", edgecolor="none", pad=2.0))

            final_row = n_cam_rows + 1
            axq = fig.add_subplot(gs[final_row, 0])
            montage = []
            target_w = 360
            for cid in cam_rows:
                img = query_rgb_by_cam.get(cid)
                if img is None:
                    continue
                h0, w0 = img.shape[:2]
                nh = max(1, int(h0 * target_w / max(w0, 1)))
                montage.append(cv2.resize(img, (target_w, nh),
                                          interpolation=cv2.INTER_AREA))
            axq.imshow(np.vstack(montage) if montage else query_rgb)
            axq.set_title("QUERY [combined]", color="purple", fontsize=10, pad=9)
            axq.axis("off")

            for col, vidx in enumerate(show_idxs, start=1):
                axc = fig.add_subplot(gs[final_row, col])
                axc.imshow(virtual_cam_imgs[main_cam][vidx])
                sim = float(virtual_combined_sims[vidx]) if virtual_combined_sims is not None else 0.0
                log = next((x for x in fine_log if x.get("virt_idx") == int(vidx)), {})
                inl = int(log.get("n_inliers", 0))
                color = "green" if vidx == virtual_winner else "purple"
                axc.set_title(f"F{col} virt_{vidx}  sum={sim:.3f}\ninliers={inl}",
                              color=color, fontsize=8, pad=8)
                axc.axis("off")

            fig.suptitle(
                f"Step 5 SplatHLoc [type2 fine]: main={main_cam}  cams={cam_rows}  "
                f"coarse_best={gv_log[0]['n_inliers'] if gv_log else 0}  "
                f"final_inliers={best_inliers}",
                fontsize=12,
            )
            fig.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.04)
        else:
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
        "match_top_k":      match_top_k,
        "match_candidates": match_candidates,
        "match_cos_sims":   match_cos_sims,
        "match_cam_top_results": match_cam_top_results,
        "dynamic_primary":  primary_cam,             # ★cam (main 고정)
        "retrieval_type":   "splathloc",
        # SplatHLoc-specific
        "splathloc_gv_log":         gv_log,
        "splathloc_fine_used":      fine_used,
        "splathloc_fine_renderer":  fine_renderer_kind,
        "splathloc_fine_log":       fine_log,
        "splathloc_virtual_winner": virtual_winner,
        "splathloc_virtual_top":    virtual_top,
        "splathloc_best_inliers":   int(best_inliers),
        "splathloc_params": {
            "k1": k1, "k2": k2, "k3": k3, "a_deg": a_deg, "b_m": b_m,
            "I": I_thr, "gv_stride": gv_stride, "matcher": matcher_name,
            "fine_trigger": fine_trigger,
            "similarity_threshold": sim_threshold,
            "verify_virtual_with_gv": verify_virtual_with_gv,
        },
    }
    if type2_meta is not None:
        data.update(type2_meta)
    if save_images:
        pickle.dump(data, open(os.path.join(output_dir, "step5_data.pkl"), "wb"))
    return data
