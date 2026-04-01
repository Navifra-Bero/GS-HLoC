import os, pickle
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .step2_render import slim_rendered


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


def _dino_preprocess(img_rgb, img_size):
    """RGB uint8 → normalized tensor (1,3,H,W)"""
    import torch
    t = cv2.resize(img_rgb, (img_size, img_size)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    t = (t - mean) / std
    return torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0)


def _extract_dino_patches(img_rgb, model, dev, img_size):
    """DINOv2 patch feature 추출: (n_patches, feat_dim) float32"""
    import torch
    t = _dino_preprocess(img_rgb, img_size).to(dev)
    with torch.no_grad():
        feats = model.get_intermediate_layers(t, n=1)[0]
    return feats[0].cpu().numpy().astype(np.float32)


def _compute_vlad(patch_feats, centers):
    """VLAD descriptor 계산 (intra-normalized + L2 global)"""
    K, D = centers.shape
    diff = patch_feats[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    labels = dists.argmin(axis=1)

    vlad = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        mask = labels == k
        if mask.any():
            vlad[k] = (patch_feats[mask] - centers[k]).sum(axis=0)

    norms = np.linalg.norm(vlad, axis=1, keepdims=True) + 1e-8
    vlad /= norms
    vlad = vlad.flatten()
    vlad /= (np.linalg.norm(vlad) + 1e-8)
    return vlad


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
    """MegaLoc 전역 디스크립터 추출"""
    import torch
    t = _megaloc_preprocess(img_rgb).to(dev)
    with torch.no_grad():
        desc = model(t).cpu().numpy().flatten()
    return desc.astype(np.float32)


# =============================================================================
# Step 3 main
# =============================================================================
def step3_global_desc(rendered, config, output_dir):
    """Global descriptor 추출 (MegaLoc 또는 DINOv2+VLAD)."""
    import torch
    fc     = config["features"]
    method = fc.get("global_desc_method", "megaloc")
    dev    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── MegaLoc 분기
    if method == "megaloc":
        print("\n" + "="*60 + "\nSTEP 3: Global descriptors (MegaLoc)\n" + "="*60)
        print("  Loading MegaLoc from torch.hub …")
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        model.eval().to(dev)
        print(f"  MegaLoc loaded  →  descriptor dim={fc.get('megaloc_dim', 8448)}")

        for i, r in enumerate(rendered):
            img_bgr = cv2.imread(r["rgb_path"])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            r["global_descriptor"] = _extract_megaloc_desc(img_rgb, model, dev)
            r["global_desc_method"] = "megaloc"
            if (i + 1) % 50 == 0 or i == 0:
                print(f"    {i+1}/{len(rendered)}")

        nv = min(100, len(rendered))
        if nv >= 2:
            idx = np.linspace(0, len(rendered)-1, nv, dtype=int)
            dm  = np.array([rendered[i]["global_descriptor"] for i in idx], dtype=np.float32)
            sim = dm @ dm.T
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            ax.imshow(sim, cmap="hot", vmin=0, vmax=1)
            ax.set_title(f"Step 3: MegaLoc similarity  (dim={dm.shape[1]})", fontsize=13)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "step3_global_desc.png"), dpi=150); plt.close()
            print(f"  Saved: step3_global_desc.png")

        slim = slim_rendered(rendered)
        pickle.dump({"rendered": slim},
                    open(os.path.join(output_dir, "step3_data.pkl"), "wb"))
        return rendered, None

    # ── AnyLoc 분기 (DINOv2+VLAD)
    print("\n" + "="*60 + "\nSTEP 3: Global descriptors (DINOv2 + VLAD)\n" + "="*60)
    from sklearn.cluster import MiniBatchKMeans

    dino_name = fc.get("dino_model", "dinov2_vitb14")
    img_size  = int(fc.get("dino_img_size", 322))
    n_clusters = int(fc.get("vlad_clusters", 64))

    print(f"  Loading DINOv2: {dino_name}  (img_size={img_size})")
    model = torch.hub.load("facebookresearch/dinov2", dino_name, pretrained=True)
    model.eval().to(dev)
    feat_dim = model.embed_dim
    vlad_dim = n_clusters * feat_dim
    print(f"  feat_dim={feat_dim}, clusters={n_clusters}  →  VLAD dim={vlad_dim}")

    print(f"  Extracting patch features from {len(rendered)} images …")
    all_patches = []
    per_image_patches = []
    for i, r in enumerate(rendered):
        rgb = r["rgb"] if isinstance(r.get("rgb"), np.ndarray) \
              else cv2.cvtColor(cv2.imread(r["rgb_path"]), cv2.COLOR_BGR2RGB)
        pf = _extract_dino_patches(rgb, model, dev, img_size)
        per_image_patches.append(pf)
        all_patches.append(pf)
        if (i+1) % 100 == 0:
            print(f"    {i+1}/{len(rendered)}")

    all_feats = np.vstack(all_patches)
    max_sample = 200_000
    if len(all_feats) > max_sample:
        idx = np.random.choice(len(all_feats), max_sample, replace=False)
        sample = all_feats[idx]
    else:
        sample = all_feats
    print(f"  Fitting VLAD vocabulary: {n_clusters} clusters on {len(sample)} patches …")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                             n_init=5, batch_size=4096, max_iter=300)
    kmeans.fit(sample)
    centers = kmeans.cluster_centers_.astype(np.float32)
    print(f"  Vocabulary fitted.")

    for i, (r, pf) in enumerate(zip(rendered, per_image_patches)):
        r["global_descriptor"] = _compute_vlad(pf, centers)
        r["vlad_vocab"] = centers
        r["global_desc_method"] = "anyloc"

    nv  = min(100, len(rendered))
    idx = np.linspace(0, len(rendered)-1, nv, dtype=int)
    dm  = np.array([rendered[i]["global_descriptor"] for i in idx], dtype=np.float32)
    dm /= np.linalg.norm(dm, axis=1, keepdims=True) + 1e-8

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    im = ax[0].imshow(dm @ dm.T, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax[0].set_title(f"Similarity {nv}×{nv}")
    plt.colorbar(im, ax=ax[0], fraction=0.046)

    from sklearn.decomposition import PCA
    c2  = PCA(n_components=2).fit_transform(dm)
    ps_ = np.array([rendered[i]["pose"][:3, 3] for i in idx])
    ax[1].scatter(c2[:, 0], c2[:, 1], c=ps_[:, 0], cmap="viridis", s=10)
    ax[1].set_title("PCA (color=X)")

    fig.suptitle(f"Step 3: Global descriptors (DINOv2+VLAD, K={n_clusters}, dim={vlad_dim})",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "step3_global_desc.png"), dpi=150); plt.close()
    print(f"  Saved: step3_global_desc.png")

    slim = slim_rendered(rendered)
    pickle.dump({"rendered": slim, "vlad_centers": centers},
                open(os.path.join(output_dir, "step3_data.pkl"), "wb"))
    return rendered, centers
