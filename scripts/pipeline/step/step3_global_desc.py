import os, pickle
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .step2_render import slim_rendered


def _resolve_render_path(path, output_dir):
    """Resolve render paths saved from different working directories."""
    if not path:
        return path
    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(os.path.abspath(path))
        norm = os.path.normpath(path)
        parts = norm.split(os.sep)
        if "output" in parts:
            idx = parts.index("output")
            candidates.append(os.path.join(os.getcwd(), *parts[idx:]))
        base = os.path.basename(output_dir.rstrip(os.sep))
        if base in parts:
            idx = parts.index(base)
            candidates.append(os.path.join(output_dir, *parts[idx + 1:]))

    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return path


def _load_render_rgb(record, output_dir):
    rgb_path = _resolve_render_path(record.get("rgb_path", ""), output_dir)
    img = cv2.imread(rgb_path)
    if img is None:
        raise FileNotFoundError(f"render RGB not found/readable: {record.get('rgb_path')} -> {rgb_path}")
    record["rgb_path"] = rgb_path
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_render_depth(record, output_dir):
    depth_path = _resolve_render_path(record.get("depth_path", ""), output_dir)
    if not depth_path or not os.path.exists(depth_path):
        return None
    record["depth_path"] = depth_path
    return np.load(depth_path)


def _attach_render_feature_metadata(record, output_dir, rgb_shape=None):
    """Keep step2 dense feature-map references alive through step3/step4."""
    feature_path = _resolve_render_path(record.get("feature_path", ""), output_dir)
    if (not feature_path or not os.path.exists(feature_path)) and record.get("id") is not None:
        feature_dir = os.path.join(output_dir, "rendered", "feature")
        rid = int(record["id"])
        for name in (f"{rid:05d}.npy", f"{rid:06d}.npy"):
            candidate = os.path.join(feature_dir, name)
            if os.path.exists(candidate):
                feature_path = candidate
                break
    if not feature_path or not os.path.exists(feature_path):
        return False

    record["feature_path"] = feature_path
    if record.get("feature_shape") is None or record.get("feature_stride") is None:
        feat = np.load(feature_path, mmap_mode="r")
        record["feature_shape"] = tuple(int(x) for x in feat.shape)
        if record.get("feature_stride") is None and rgb_shape is not None and len(feat.shape) >= 3:
            rgb_h = int(rgb_shape[0])
            feat_h = int(feat.shape[-2])
            record["feature_stride"] = int(round(rgb_h / max(1, feat_h)))
    if not record.get("feature_type"):
        record["feature_type"] = "rendered_feature"
    return True


# =============================================================================
# Descriptor helpers
# =============================================================================
def build_regional_descriptor(rgb_img, extract_fn, grid_rows, grid_cols):
    """이미지를 grid 분할하여 공간 정보 보존 descriptor 반환."""
    H, W = rgb_img.shape[:2]
    parts = [extract_fn(rgb_img)]
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0 = int(H * r / grid_rows)
            y1 = int(H * (r + 1) / grid_rows)
            x0 = int(W * c / grid_cols)
            x1 = int(W * (c + 1) / grid_cols)
            cell = rgb_img[y0:y1, x0:x1]
            parts.append(extract_fn(cell))
    desc = np.concatenate(parts).astype(np.float32)
    norm = np.linalg.norm(desc)
    return desc / (norm + 1e-8)


def _load_mixvpr_model(ckpt_path, out_dim, dev):
    """MixVPR 모델 로드.

    사전 준비:
        git clone https://github.com/amaralibey/MixVPR third_party/MixVPR

    ckpt_path : 학습된 가중치 경로 (비워두면 ImageNet pretrained ResNet50만 사용)
    out_dim   : 512 (권장) / 128 / 4096
    """
    import sys, torch, torch.nn as nn
    mixvpr_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'third_party', 'MixVPR')
    )
    if not os.path.isdir(mixvpr_root):
        raise RuntimeError(
            f"MixVPR not found at {mixvpr_root}\n"
            "Please run:\n"
            "  git clone https://github.com/amaralibey/MixVPR third_party/MixVPR"
        )
    if mixvpr_root not in sys.path:
        sys.path.insert(0, mixvpr_root)

    from models.helper import get_backbone, get_aggregator

    # out_dim → (out_channels, out_rows)
    _cfg = {128: (64, 2), 512: (256, 2), 4096: (1024, 4)}
    out_channels, out_rows = _cfg.get(out_dim, (256, 2))

    has_ckpt = ckpt_path and os.path.exists(ckpt_path)
    # pretrained=True는 ckpt가 없을 때만 의미 있음
    backbone   = get_backbone('resnet50', pretrained=not has_ckpt,
                              layers_to_freeze=2, layers_to_crop=[4])
    aggregator = get_aggregator('MixVPR', agg_config={
        'in_channels': 1024, 'in_h': 20, 'in_w': 20,
        'out_channels': out_channels, 'mix_depth': 4,
        'mlp_ratio': 1, 'out_rows': out_rows,
    })

    class _Model(nn.Module):
        def __init__(self, bb, agg):
            super().__init__()
            self.backbone = bb; self.aggregator = agg
        def forward(self, x):
            return self.aggregator(self.backbone(x))

    model = _Model(backbone, aggregator)

    if has_ckpt:
        sd = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in sd:   # Lightning checkpoint
            sd = {k.replace('model.', '', 1): v for k, v in sd['state_dict'].items()}
        model.load_state_dict(sd, strict=True)
        print(f"  MixVPR weights loaded: {ckpt_path}")
    else:
        print("  MixVPR: ImageNet pretrained ResNet50 backbone (no VPR checkpoint)")

    return model.eval().to(dev)


def _mixvpr_preprocess(img_rgb):
    """RGB (H,W,3) uint8 → tensor (1,3,320,320)"""
    import torch, torchvision.transforms as T
    tf = T.Compose([
        T.ToPILImage(),
        T.Resize((320, 320), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(img_rgb).unsqueeze(0)


def _extract_mixvpr_desc(img_rgb, model, dev):
    """MixVPR 전역 디스크립터 추출 → L2 정규화된 float32 벡터"""
    import torch
    t = _mixvpr_preprocess(img_rgb).to(dev)
    with torch.no_grad():
        desc = model(t).squeeze(0).cpu().numpy().astype(np.float32)
    return desc / (np.linalg.norm(desc) + 1e-8)


def _extract_mixvpr_spatial(img_rgb, model, dev, grid_n):
    """MixVPR spatial: 전체 + N개 수직 스트립(좌→우) → concat + L2 정규화

    grid_n=2 → 전체 + 좌 + 우            = 3 patches, dim = 3 × out_dim
    grid_n=3 → 전체 + 좌 + 중앙 + 우    = 4 patches, dim = 4 × out_dim
    dim = (1 + grid_n) × out_dim
    """
    H, W = img_rgb.shape[:2]
    parts = [_extract_mixvpr_desc(img_rgb, model, dev)]
    for c in range(grid_n):
        x0, x1 = int(W * c / grid_n), int(W * (c + 1) / grid_n)
        parts.append(_extract_mixvpr_desc(img_rgb[:, x0:x1], model, dev))
    desc = np.concatenate(parts).astype(np.float32)
    return desc / (np.linalg.norm(desc) + 1e-8)


def _extract_depth_spatial(depth_map, grid_n, n_bins=32):
    """Depth map → spatial histogram descriptor (L2 normalized)

    depth_map : (H, W) float32, meters (0 = invalid)
    Returns   : (1 + grid_n) × n_bins dimensional L2-normalized descriptor

    수직 스트립(좌→우)로 나눠 각 스트립의 depth 분포를 히스토그램으로 표현.
    조명/텍스처에 무관한 공간 구조 정보만 담음.
    """
    H, W = depth_map.shape
    valid_all = depth_map[depth_map > 0]
    if len(valid_all) < 10:
        return np.zeros((1 + grid_n) * n_bins, dtype=np.float32)

    d_min = np.percentile(valid_all, 2)
    d_max = np.percentile(valid_all, 98)
    if d_max <= d_min:
        d_max = d_min + 1.0
    bins = np.linspace(d_min, d_max, n_bins + 1)

    def cell_hist(patch):
        v = patch[(patch > 0) & (patch < d_max * 1.1)]
        if len(v) < 5:
            return np.zeros(n_bins, dtype=np.float32)
        h, _ = np.histogram(v, bins=bins)
        h = h.astype(np.float32)
        return h / (h.sum() + 1e-8)

    parts = [cell_hist(depth_map)]   # 전체 global
    for c in range(grid_n):
        x0, x1 = int(W * c / grid_n), int(W * (c + 1) / grid_n)
        parts.append(cell_hist(depth_map[:, x0:x1]))

    desc = np.concatenate(parts).astype(np.float32)
    norm = np.linalg.norm(desc)
    return desc / (norm + 1e-8)


def _load_query_depth(image_path, cam_h, cam_w):
    """쿼리 이미지 경로에서 대응하는 depth 파일 로드.

    images/xxx.jpg → depths/xxx.depth (raw float32 binary)
    """
    depth_path = image_path.replace('/images/', '/depths/')
    depth_path = os.path.splitext(depth_path)[0] + '.depth'
    if not os.path.exists(depth_path):
        return None, depth_path
    arr = np.fromfile(depth_path, dtype=np.float32)
    if arr.size == cam_h * cam_w:
        return arr.reshape(cam_h, cam_w), depth_path
    return None, depth_path


def _plot_sim_by_yaw(rendered, desc_key, num_yaw_angles, title, output_path):
    """Yaw angle별로 분리된 similarity matrix 시각화.

    num_yaw_angles개의 서브플롯을 grid(최대 3열)로 배치.
    각 서브플롯: 해당 yaw angle의 모든 뷰포인트(모든 height 포함) 간 cosine 유사도.

    Args:
        rendered       : step3 결과 list. 각 원소에 desc_key 필드가 있어야 함.
        desc_key       : 'global_descriptor' 또는 'depth_descriptor'
        num_yaw_angles : config의 sampling.num_yaw_angles
        title          : figure suptitle 텍스트
        output_path    : 저장 경로
    """
    # ── yaw angle별 그룹핑 ────────────────────────────────────────────────
    def _get_yaw(r):
        """r["yaw"] 필드가 있으면 사용, 없으면 pose rotation에서 추출.
        step1: R = [right | -up | forward], forward = R[:, 2]
        yaw = atan2(forward_y, forward_x)
        """
        yaw = r.get("yaw", None)
        if yaw is not None:
            return float(yaw)
        pose = r.get("pose")
        if pose is not None:
            R = np.array(pose)[:3, :3]
            fwd = R[:, 2]   # 3rd column = forward direction
            return float(np.arctan2(fwd[1], fwd[0]))  # [-pi, pi]
        return 0.0

    groups = [[] for _ in range(num_yaw_angles)]
    for i, r in enumerate(rendered):
        if r.get(desc_key) is None:
            continue
        yaw = _get_yaw(r)
        # [-pi, pi] → [0, 2pi)
        yaw = yaw % (2 * np.pi)
        yi = int(round(yaw * num_yaw_angles / (2 * np.pi))) % num_yaw_angles
        groups[yi].append(i)

    # ── 레이아웃: 최대 3열 ────────────────────────────────────────────────
    ncols = min(3, num_yaw_angles)
    nrows = int(np.ceil(num_yaw_angles / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    all_means = []
    for yi, indices in enumerate(groups):
        ax = axes_flat[yi]
        yaw_deg = 360.0 * yi / num_yaw_angles

        # (x, y) 좌표 기준 오름차순 정렬 → 공간 패턴이 블록으로 보임
        indices.sort(key=lambda i: (rendered[i]["pose"][0, 3], rendered[i]["pose"][1, 3]))

        if len(indices) < 2:
            ax.set_title(f"Yaw {yaw_deg:.0f}°  (n={len(indices)})", fontsize=10)
            ax.axis("off")
            continue

        dm = np.array([rendered[i][desc_key] for i in indices], dtype=np.float32)
        dm /= np.linalg.norm(dm, axis=1, keepdims=True) + 1e-8
        sim = dm @ dm.T
        off = sim[~np.eye(len(sim), dtype=bool)]
        all_means.append(off.mean())

        im = ax.imshow(sim, cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"Yaw {yaw_deg:.0f}°  (n={len(indices)}, dim={dm.shape[1]})",
                     fontsize=10)
        ax.set_xlabel("Viewpoint", fontsize=8)
        ax.set_ylabel("Viewpoint", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cosine sim", fontsize=8)
        ax.text(0.02, 0.98,
                f"mean={off.mean():.3f}\nmin={off.min():.3f}\nmax={off.max():.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                color="cyan",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.35))

    # 남는 서브플롯 숨기기
    for ai in range(num_yaw_angles, len(axes_flat)):
        axes_flat[ai].set_visible(False)

    overall = f"  overall mean={np.mean(all_means):.3f}" if all_means else ""
    fig.suptitle(f"Step 3: {title}{overall}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def _megaloc_preprocess(img_rgb, resize=518):
    """MegaLoc 입력 전처리"""
    import torch, torchvision.transforms as T
    tf = T.Compose([
        T.ToPILImage(),
        T.Resize((resize, resize)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(img_rgb).unsqueeze(0)


def _extract_megaloc_desc(img_rgb, model, dev):
    """MegaLoc 전역 디스크립터 추출 (L2 정규화 포함)"""
    import torch
    t = _megaloc_preprocess(img_rgb).to(dev)
    with torch.no_grad():
        desc = model(t).cpu().numpy().flatten().astype(np.float32)
    norm = np.linalg.norm(desc)
    return desc / (norm + 1e-8)


def _extract_megaloc_spatial(img_rgb, model, dev, grid_n):
    """MegaLoc spatial descriptor: 전체 + N개 수직 스트립(좌→우) → concat + L2 정규화

    grid_n=2 → 전체 + 좌 + 우            = 3 patches, dim = 3 × 8448
    grid_n=3 → 전체 + 좌 + 중앙 + 우    = 4 patches, dim = 4 × 8448
    MegaLoc이 각 셀을 518×518으로 resize하므로 crop 크기에 무관하게 동작함.
    """
    H, W = img_rgb.shape[:2]
    parts = [_extract_megaloc_desc(img_rgb, model, dev)]   # 전체 global

    for c in range(grid_n):
        x0, x1 = int(W * c / grid_n), int(W * (c + 1) / grid_n)
        parts.append(_extract_megaloc_desc(img_rgb[:, x0:x1], model, dev))

    desc = np.concatenate(parts).astype(np.float32)
    norm = np.linalg.norm(desc)
    return desc / (norm + 1e-8)


# =============================================================================
# Step 3 main
# =============================================================================
def step3_global_desc(rendered, config, output_dir):
    """Global descriptor 추출 (MegaLoc 또는 DINOv2+VLAD)."""
    import torch
    fc        = config["features"]
    method    = fc.get("global_desc_method", "megaloc")
    dev       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_yaw   = int(config.get("sampling", {}).get("num_yaw_angles", 1))

    # ── MegaLoc (+ optional spatial grid + optional depth) 분기
    if method == "megaloc":
        use_depth  = fc.get("use_depth_desc", True)
        grid_n     = int(fc.get("dino_grid_n", 3))   # 0 또는 1 = global only
        n_bins     = int(fc.get("depth_bins", 32))
        w_rgb      = float(fc.get("depth_rgb_weight", 0.7))
        use_spatial = grid_n > 1
        depth_dim  = (1 + grid_n * grid_n) * n_bins if use_spatial else n_bins
        megaloc_base = 8448
        rgb_dim    = (1 + grid_n) * megaloc_base if use_spatial else megaloc_base

        parts = []
        if use_spatial: parts.append(f"{grid_n} vertical strips")
        if use_depth:   parts.append("depth histogram")
        label = "MegaLoc" + (f" ({', '.join(parts)})" if parts else "")

        print("\n" + "="*60 + f"\nSTEP 3: Global descriptors ({label})\n" + "="*60)
        print("  Loading MegaLoc from torch.hub …")
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        model.eval().to(dev)
        total_dim = rgb_dim + depth_dim if use_depth else rgb_dim
        print(f"  MegaLoc loaded  →  RGB dim={rgb_dim}"
              + (f"  depth dim={depth_dim}  total={total_dim}  w_rgb={w_rgb:.1f}"
                 if use_depth else f"  total={total_dim}"))

        depth_grid = grid_n if use_spatial else 1
        n_depth_ok = 0
        n_feature_ok = 0
        for i, r in enumerate(rendered):
            img_rgb  = _load_render_rgb(r, output_dir)
            if _attach_render_feature_metadata(r, output_dir, rgb_shape=img_rgb.shape):
                n_feature_ok += 1
            rgb_desc = (_extract_megaloc_spatial(img_rgb, model, dev, grid_n)
                        if use_spatial else _extract_megaloc_desc(img_rgb, model, dev))

            depth_map = _load_render_depth(r, output_dir) if use_depth else None
            if depth_map is not None:
                depth_desc = _extract_depth_spatial(depth_map, depth_grid, n_bins)
                n_depth_ok += 1
            else:
                depth_desc = None

            # RGB / depth 분리 저장 → step4에서 late fusion용 별도 KDTree/행렬 구성
            r["global_descriptor"] = rgb_desc        # RGB-only (KDTree용)
            r["depth_descriptor"]  = depth_desc      # depth-only (re-ranking용), None 가능
            r["global_desc_method"] = "megaloc"
            r["use_depth_desc"]     = use_depth
            r["dino_grid_n"]        = grid_n
            r["use_spatial"]        = use_spatial
            if (i + 1) % 50 == 0 or i == 0:
                print(f"    {i+1}/{len(rendered)}")

        if use_depth:
            print(f"  Depth used: {n_depth_ok}/{len(rendered)} entries")
        print(f"  Feature maps kept: {n_feature_ok}/{len(rendered)} entries")

        if len(rendered) >= 2:
            _plot_sim_by_yaw(rendered, "global_descriptor", num_yaw,
                             f"{label} (RGB)",
                             os.path.join(output_dir, "step3_global_desc.png"))
            if use_depth and any(r.get("depth_descriptor") is not None for r in rendered):
                _plot_sim_by_yaw(rendered, "depth_descriptor", num_yaw,
                                 f"{label} (Depth)",
                                 os.path.join(output_dir, "step3_global_desc_depth.png"))

        slim = slim_rendered(rendered)
        pickle.dump({"rendered": slim},
                    open(os.path.join(output_dir, "step3_data.pkl"), "wb"))
        return rendered, None

    # ── MixVPR 분기
    if method == "mixvpr":
        ckpt_path   = fc.get("mixvpr_ckpt", "")
        out_dim     = int(fc.get("mixvpr_out_dim", 512))
        grid_n      = int(fc.get("grid_n", 2))
        use_depth   = bool(fc.get("use_depth_desc", False))
        n_bins      = int(fc.get("depth_bins", 32))
        use_spatial = grid_n > 1
        depth_grid  = grid_n if use_spatial else 1
        rgb_dim     = (1 + grid_n) * out_dim if use_spatial else out_dim

        label = f"MixVPR (out_dim={out_dim}" + \
                (f", {grid_n} vertical strips" if use_spatial else "") + \
                (" + depth" if use_depth else "") + ")"
        print("\n" + "="*60 + f"\nSTEP 3: Global descriptors ({label})\n" + "="*60)

        model = _load_mixvpr_model(ckpt_path, out_dim, dev)
        print(f"  RGB desc dim: {rgb_dim}")

        n_depth_ok = 0
        n_feature_ok = 0
        for i, r in enumerate(rendered):
            rgb      = _load_render_rgb(r, output_dir)
            if _attach_render_feature_metadata(r, output_dir, rgb_shape=rgb.shape):
                n_feature_ok += 1
            rgb_desc = (_extract_mixvpr_spatial(rgb, model, dev, grid_n)
                        if use_spatial else _extract_mixvpr_desc(rgb, model, dev))

            depth_map = _load_render_depth(r, output_dir) if use_depth else None
            if depth_map is not None:
                depth_desc = _extract_depth_spatial(depth_map, depth_grid, n_bins)
                n_depth_ok += 1
            else:
                depth_desc = None

            r["global_descriptor"]  = rgb_desc
            r["depth_descriptor"]   = depth_desc
            r["global_desc_method"] = "mixvpr"
            r["use_depth_desc"]     = use_depth
            r["grid_n"]             = grid_n
            r["use_spatial"]        = use_spatial
            if (i + 1) % 100 == 0 or i == 0:
                print(f"    {i+1}/{len(rendered)}")

        if use_depth:
            print(f"  Depth used: {n_depth_ok}/{len(rendered)} entries")
        print(f"  Feature maps kept: {n_feature_ok}/{len(rendered)} entries")

        if len(rendered) >= 2:
            _plot_sim_by_yaw(rendered, "global_descriptor", num_yaw,
                             f"{label} (RGB)",
                             os.path.join(output_dir, "step3_global_desc.png"))
            if use_depth and any(r.get("depth_descriptor") is not None for r in rendered):
                _plot_sim_by_yaw(rendered, "depth_descriptor", num_yaw,
                                 f"{label} (Depth)",
                                 os.path.join(output_dir, "step3_global_desc_depth.png"))

        slim = slim_rendered(rendered)
        pickle.dump({"rendered": slim},
                    open(os.path.join(output_dir, "step3_data.pkl"), "wb"))
        return rendered, None

    # ── 알 수 없는 method
    if method not in ("megaloc", "mixvpr"):
        raise ValueError(f"Unknown global_desc_method: '{method}'. "
                         "Choose from: megaloc, mixvpr")
