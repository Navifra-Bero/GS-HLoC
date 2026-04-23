#!/usr/bin/env python3
"""Step3 global descriptor 테스트 스크립트.

이미지 폴더를 지정하면 해당 폴더의 이미지들로 descriptor를 추출하고
similarity matrix를 시각화합니다.

Usage:
    python3 scripts/test_step3.py \
        --image_dir output/gs_test/rendered_gs/rgb \
        --method dinov2 \
        --output_dir output/gs_test/test_step3

    # 여러 method 비교
    python3 scripts/test_step3.py \
        --image_dir output/gs_test/rendered_gs/rgb \
        --method dinov2 megaloc dinov2_vlad \
        --output_dir output/gs_test/test_step3
"""
import os, sys, glob, argparse, pickle
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.pipeline.config import load_config, default_config
from scripts.pipeline.step.step3_global_desc import (
    _extract_megaloc_desc, _extract_megaloc_spatial,
    _extract_dino_cls, _extract_dino_spatial, _extract_dino_patches,
    _extract_dinov3_cls, _extract_dinov3_spatial,
    _extract_dino_vlad, _extract_dino_vlad_spatial,
    _extract_depth_spatial, _compute_vlad,
)


def load_images(image_dir, max_images=None):
    """이미지 폴더에서 파일 목록 로드."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
    files = sorted(files)
    if max_images and len(files) > max_images:
        idx = np.linspace(0, len(files) - 1, max_images, dtype=int)
        files = [files[i] for i in idx]
    print(f"  Loaded {len(files)} images from {image_dir}")
    return files


def extract_descs(files, method, config):
    """지정된 method로 descriptor 추출."""
    import torch
    fc = config.get("features", {})
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dino_name = fc.get("dino_model", "dinov2_vitb14")
    img_size = int(fc.get("dino_img_size", 322))
    grid_n = int(fc.get("dino_grid_n", 1))
    use_spatial = grid_n > 1
    n_clusters = int(fc.get("vlad_clusters", 64))

    descs = []

    if method == "megaloc":
        print("  Loading MegaLoc …")
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        model.eval().to(dev)
        for i, f in enumerate(files):
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            if use_spatial:
                d = _extract_megaloc_spatial(img, model, dev, grid_n)
            else:
                d = _extract_megaloc_desc(img, model, dev)
            descs.append(d)
            if (i + 1) % 50 == 0 or i == 0:
                print(f"    {i+1}/{len(files)}")

    elif method == "dinov2":
        print(f"  Loading DINOv2: {dino_name} …")
        model = torch.hub.load("facebookresearch/dinov2", dino_name, pretrained=True)
        model.eval().to(dev)
        for i, f in enumerate(files):
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            if use_spatial:
                d = _extract_dino_spatial(img, model, dev, img_size, grid_n)
            else:
                d = _extract_dino_cls(img, model, dev, img_size)
            descs.append(d)
            if (i + 1) % 50 == 0 or i == 0:
                print(f"    {i+1}/{len(files)}")

    elif method == "dinov3":
        dinov3_name = fc.get("dinov3_model", "facebook/dinov3-vitb16-pretrain-lvd1689m")
        print(f"  Loading DINOv3: {dinov3_name} …")
        from transformers import AutoModel, AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(dinov3_name)
        model = AutoModel.from_pretrained(dinov3_name)
        model.eval().to(dev)
        for i, f in enumerate(files):
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            if use_spatial:
                d = _extract_dinov3_spatial(img, model, processor, dev, grid_n)
            else:
                d = _extract_dinov3_cls(img, model, processor, dev)
            descs.append(d)
            if (i + 1) % 50 == 0 or i == 0:
                print(f"    {i+1}/{len(files)}")

    elif method == "dinov2_vlad":
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import PCA
        pca_dim = int(fc.get("vlad_pca_dim", 512))

        print(f"  Loading DINOv2: {dino_name} …")
        model = torch.hub.load("facebookresearch/dinov2", dino_name, pretrained=True)
        model.eval().to(dev)
        feat_dim = model.embed_dim

        # 1) patch 추출
        print(f"  Extracting patches …")
        all_patches = []
        per_image = []
        for i, f in enumerate(files):
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            pf = _extract_dino_patches(img, model, dev, img_size)
            per_image.append(pf)
            all_patches.append(pf)
            if (i + 1) % 50 == 0 or i == 0:
                print(f"    {i+1}/{len(files)}")

        # 2) vocabulary
        all_feats = np.vstack(all_patches)
        max_sample = 200_000
        if len(all_feats) > max_sample:
            sample = all_feats[np.random.choice(len(all_feats), max_sample, replace=False)]
        else:
            sample = all_feats
        print(f"  Fitting VLAD vocabulary: K={n_clusters} on {len(sample)} patches …")
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                                 n_init=5, batch_size=4096, max_iter=300)
        kmeans.fit(sample)
        centers = kmeans.cluster_centers_.astype(np.float32)

        # 3) VLAD
        raw = []
        for i, f in enumerate(files):
            if use_spatial:
                img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
                d = _extract_dino_vlad_spatial(img, model, dev, img_size, centers, grid_n)
            else:
                d = _compute_vlad(per_image[i], centers)
            raw.append(d)
            if (i + 1) % 50 == 0 or i == 0:
                print(f"    VLAD {i+1}/{len(files)}")
        raw = np.array(raw, dtype=np.float32)

        # 4) PCA
        actual_dim = min(pca_dim, raw.shape[1], raw.shape[0])
        print(f"  PCA: {raw.shape[1]} → {actual_dim}")
        pca = PCA(n_components=actual_dim, svd_solver='randomized', random_state=42)
        reduced = pca.fit_transform(raw).astype(np.float32)
        norms = np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-8
        descs = list(reduced / norms)

    else:
        raise ValueError(f"Unknown method: {method}")

    return np.array(descs, dtype=np.float32)


def plot_similarity(descs, method, n_images, output_path):
    """Similarity matrix 시각화."""
    dm = descs.copy()
    dm /= np.linalg.norm(dm, axis=1, keepdims=True) + 1e-8
    sim = dm @ dm.T
    off = sim[~np.eye(len(sim), dtype=bool)]

    print(f"  [{method}] dim={descs.shape[1]}  "
          f"off-diag: min={off.min():.3f}  mean={off.mean():.3f}  max={off.max():.3f}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(sim, cmap="hot", vmin=0, vmax=1)
    ax.set_title(f"{method} similarity (n={n_images}, dim={descs.shape[1]})", fontsize=13)
    ax.set_xlabel("Image index")
    ax.set_ylabel("Image index")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")

    return {"method": method, "dim": descs.shape[1],
            "min": float(off.min()), "mean": float(off.mean()), "max": float(off.max())}


def main():
    parser = argparse.ArgumentParser(description="Step3 descriptor test")
    parser.add_argument("--image_dir", required=True, help="이미지 폴더 경로")
    parser.add_argument("--method", nargs="+", default=["dinov2"],
                        help="테스트할 method(들): megaloc, dinov2, dinov3, dinov2_vlad")
    parser.add_argument("--config", default=None, help="config yaml 경로")
    parser.add_argument("--output_dir", default=None, help="결과 저장 폴더")
    parser.add_argument("--max_images", type=int, default=100, help="최대 이미지 수 (기본 100)")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else default_config()
    output_dir = args.output_dir or os.path.join(args.image_dir, "test_step3")
    os.makedirs(output_dir, exist_ok=True)

    files = load_images(args.image_dir, args.max_images)
    if not files:
        print("ERROR: 이미지가 없습니다.")
        return

    results = []
    for method in args.method:
        print(f"\n{'='*60}\n  Method: {method}\n{'='*60}")
        descs = extract_descs(files, method, config)
        out_path = os.path.join(output_dir, f"sim_{method}.png")
        r = plot_similarity(descs, method, len(files), out_path)
        results.append(r)

    # 비교 요약
    if len(results) > 1:
        print(f"\n{'='*60}\n  Summary\n{'='*60}")
        print(f"  {'Method':<15} {'Dim':>6} {'Min':>7} {'Mean':>7} {'Max':>7}")
        print(f"  {'-'*15} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
        for r in results:
            print(f"  {r['method']:<15} {r['dim']:>6} {r['min']:>7.3f} {r['mean']:>7.3f} {r['max']:>7.3f}")


if __name__ == "__main__":
    main()
