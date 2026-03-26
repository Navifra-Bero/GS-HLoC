#!/usr/bin/env python3
"""
RenderLoc Model Setup
=====================
Downloads pretrained SuperPoint, SuperGlue, and NetVLAD models
and exports them to TorchScript format for C++ inference.

Usage:
  python setup_models.py --output_dir models/

Note: SuperPoint and SuperGlue require cloning their repos first.
This script handles the full setup.
"""

import argparse
import os
import sys
import subprocess

# Use the same Python interpreter that's running this script
PYTHON = sys.executable


def run(cmd, cwd=None):
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)


def setup_superpoint(output_dir, cache_dir):
    """Download and export SuperPoint to TorchScript."""
    print("\n=== SuperPoint ===")
    sp_dir = os.path.join(cache_dir, "SuperPointPretrainedNetwork")

    if not os.path.exists(sp_dir):
        run(f"git clone https://github.com/magicleap/SuperPointPretrainedNetwork.git {sp_dir}")

    # SuperPoint weights
    weights_path = os.path.join(sp_dir, "superpoint_v1.pth")
    if not os.path.exists(weights_path):
        run(f"wget -O {weights_path} https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth")

    out_pt = os.path.join(output_dir, "superpoint_v1.pt")

    # Export to TorchScript inline (no subprocess, no temp file)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SuperPointNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
            self.conv1a = nn.Conv2d(1, c1, 3, padding=1)
            self.conv1b = nn.Conv2d(c1, c1, 3, padding=1)
            self.conv2a = nn.Conv2d(c1, c2, 3, padding=1)
            self.conv2b = nn.Conv2d(c2, c2, 3, padding=1)
            self.conv3a = nn.Conv2d(c2, c3, 3, padding=1)
            self.conv3b = nn.Conv2d(c3, c3, 3, padding=1)
            self.conv4a = nn.Conv2d(c3, c4, 3, padding=1)
            self.conv4b = nn.Conv2d(c4, c4, 3, padding=1)
            self.convPa = nn.Conv2d(c4, c5, 3, padding=1)
            self.convPb = nn.Conv2d(c5, 65, 1)
            self.convDa = nn.Conv2d(c4, c5, 3, padding=1)
            self.convDb = nn.Conv2d(c5, 256, 1)

        def forward(self, x):
            x = self.relu(self.conv1a(x))
            x = self.relu(self.conv1b(x))
            x = self.pool(x)
            x = self.relu(self.conv2a(x))
            x = self.relu(self.conv2b(x))
            x = self.pool(x)
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool(x)
            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x))
            # Detector head
            cPa = self.relu(self.convPa(x))
            scores = self.convPb(cPa)
            scores = F.softmax(scores, dim=1)[:, :-1]
            b, c, h, w = scores.shape
            scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
            scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
            # Descriptor head
            cDa = self.relu(self.convDa(x))
            desc = self.convDb(cDa)
            desc = F.normalize(desc, p=2, dim=1)
            return scores, desc

    model = SuperPointNet()
    weights = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(weights)
    model.eval()

    # Trace forward only (scores_map, descriptor_map)
    # NMS + keypoint extraction will be done in Python/C++ post-processing
    example = torch.randn(1, 1, 480, 640)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    traced.save(out_pt)
    size_mb = os.path.getsize(out_pt) / 1e6
    print(f"  Exported: {out_pt} ({size_mb:.1f} MB)")
    print("  Output: (score_map [B,H,W], desc_map [B,256,H/8,W/8])")
    print("  SuperPoint ready!")


def setup_superglue(output_dir, cache_dir):
    """Download and export SuperGlue to TorchScript."""
    print("\n=== SuperGlue ===")
    sg_dir = os.path.join(cache_dir, "SuperGluePretrainedNetwork")

    if not os.path.exists(sg_dir):
        run(f"git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git {sg_dir}")

    weights_indoor = os.path.join(sg_dir, "models", "weights", "superglue_indoor.pth")
    weights_outdoor = os.path.join(sg_dir, "models", "weights", "superglue_outdoor.pth")

    # Copy weights to output
    for w, name in [(weights_indoor, "superglue_indoor.pth"),
                     (weights_outdoor, "superglue_outdoor.pth")]:
        if os.path.exists(w):
            dst = os.path.join(output_dir, name)
            run(f"cp {w} {dst}")
            print(f"  {name} copied")

    print("  SuperGlue ready!")
    print("  Note: SuperGlue TorchScript export requires custom wrapping.")
    print("  For C++ inference, use the Python weights with LibTorch or ONNX.")


def setup_netvlad(output_dir, cache_dir):
    """
    실제 NetVLAD global descriptor 모델 설정.

    논문 Section III.E:
    "a pre-trained NetVLAD backbone based on VGG16, trained on the Pitts30K dataset"

    전략 (우선순위 순):
    1. EigenPlaces (torch.hub) - pip install 없이 동작, 뛰어난 성능
    2. CosPlace   (torch.hub) - EigenPlaces 실패 시 fallback
    3. VGG16+NetVLAD inline   - 위 둘 다 실패 시, pretrained VGG16 + VLAD 레이어
    """
    import torch

    print("\n=== Global Descriptor (NetVLAD / EigenPlaces) ===")
    out_pt = os.path.join(output_dir, "netvlad.pt")

    # ── 전략 1: EigenPlaces via torch.hub ───────────────────────────────
    print("  [1/3] EigenPlaces (torch.hub)...")
    try:
        model = torch.hub.load(
            "gmberton/eigenplaces", "get_trained_model",
            backbone="ResNet50", fc_output_dim=512,
            trust_repo=True
        )
        model.eval()
        ex = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            traced = torch.jit.trace(model, ex)
        traced.save(out_pt)
        print(f"  EigenPlaces ready: {out_pt} ({os.path.getsize(out_pt)/1e6:.1f} MB)")
        print("  (EigenPlaces: ResNet50 trained on SF-XL, 512-dim, outperforms NetVLAD)")
        return
    except Exception as e:
        print(f"    EigenPlaces failed: {e}")

    # ── 전략 2: CosPlace via torch.hub ──────────────────────────────────
    print("  [2/3] CosPlace (torch.hub)...")
    try:
        model = torch.hub.load(
            "gmberton/cosplace", "get_trained_model",
            backbone="ResNet50", fc_output_dim=512,
            trust_repo=True
        )
        model.eval()
        ex = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            traced = torch.jit.trace(model, ex)
        traced.save(out_pt)
        print(f"  CosPlace ready: {out_pt} ({os.path.getsize(out_pt)/1e6:.1f} MB)")
        return
    except Exception as e:
        print(f"    CosPlace failed: {e}")

    # ── 전략 3: VGG16 + NetVLAD inline ──────────────────────────────────
    print("  [3/3] VGG16 + NetVLAD (inline implementation)...")
    _build_vgg16_netvlad(cache_dir, out_pt)


def _build_vgg16_netvlad(cache_dir, out_pt):
    """
    VGG16 백본 + NetVLAD 집계 레이어를 inline으로 구현.

    논문과 동일한 구조:
    - VGG16 features[:23] → conv4_3 출력 (512채널)
    - NetVLAD layer: 64 clusters, intra-normalize + L2-normalize
    - FC compression: 64*512=32768 → 512
    - ImageNet 정규화 포함 (step_by_step.py에서 별도 처리 불필요)

    pretrained checkpoint 다운로드 시도:
    1. Nanne/pytorch-NetVlad (Google Drive, gdown 사용)
    2. 실패 시 pretrained VGG16 features만 사용 (VLAD centers는 random)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as tvm

    # ── NetVLAD 레이어 (vectorized, TorchScript 호환) ─────────────────
    class NetVLADLayer(nn.Module):
        """
        Efficient vectorized VLAD aggregation.
        Memory: O(N*K*C) avoided by decomposing sum:
          vlad[k] = sum_hw(soft[k,hw] * x[hw]) - sum_hw(soft[k,hw]) * centroid[k]
                  = (soft @ x_flat.T)  -  soft.sum() * centroid[k]
        """
        def __init__(self, num_clusters: int = 64, dim: int = 512):
            super().__init__()
            self.K = num_clusters
            self.D = dim
            self.conv      = nn.Conv2d(dim, num_clusters, kernel_size=1, bias=True)
            self.centroids = nn.Parameter(torch.randn(num_clusters, dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            N, C, H, W = x.shape
            # Soft assignment: (N, K, H*W)
            soft = F.softmax(self.conv(x), dim=1).view(N, self.K, H * W)
            # Flattened features: (N, H*W, C)
            x_flat = x.view(N, C, H * W).permute(0, 2, 1)
            # Positive part: soft-weighted feature sum  (N, K, C)
            vlad_pos = torch.bmm(soft, x_flat)
            # Negative part: soft-weighted centroid sum  (N, K, C)
            soft_sum = soft.sum(dim=2, keepdim=True)              # (N, K, 1)
            vlad_neg = soft_sum * self.centroids.unsqueeze(0)     # (N, K, C)
            # VLAD descriptor
            vlad = vlad_pos - vlad_neg                            # (N, K, C)
            vlad = F.normalize(vlad, p=2, dim=2)                  # intra-normalize
            vlad = vlad.view(N, self.K * self.D)                  # flatten
            vlad = F.normalize(vlad, p=2, dim=1)                  # L2-normalize
            return vlad

    # ── 전체 모델 ─────────────────────────────────────────────────────
    class VGG16NetVLAD(nn.Module):
        """VGG16(conv4_3) + NetVLAD + FC(512) + ImageNet normalization."""
        def __init__(self, num_clusters: int = 64, out_dim: int = 512):
            super().__init__()
            # ImageNet 정규화 (step_by_step.py에서 /255만 하면 됨)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
            self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
            # VGG16 features up to conv4_3 (index 22, before pool4)
            vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT)
            self.encoder = nn.Sequential(*list(vgg.features.children())[:23])
            # NetVLAD
            self.vlad = NetVLADLayer(num_clusters=num_clusters, dim=512)
            # Compression: 64*512=32768 → out_dim
            self.compress = nn.Linear(num_clusters * 512, out_dim, bias=True)
            nn.init.normal_(self.compress.weight, std=0.01)
            nn.init.zeros_(self.compress.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # ImageNet normalize (x is in [0,1])
            x = (x - self.mean) / self.std
            feat = self.encoder(x)          # (N, 512, H', W')
            desc = self.vlad(feat)          # (N, 64*512)
            desc = self.compress(desc)      # (N, 512)
            desc = F.normalize(desc, p=2, dim=1)
            return desc

    model = VGG16NetVLAD(num_clusters=64, out_dim=512)

    # ── pretrained checkpoint 다운로드 시도 ──────────────────────────
    ckpt_path = os.path.join(cache_dir, "netvlad_pitts30k.pth.tar")
    checkpoint_loaded = False

    if not os.path.exists(ckpt_path):
        print("    Nanne/pytorch-NetVlad checkpoint 다운로드 시도...")
        try:
            run(f"{PYTHON} -m pip install -q gdown")
            # Nanne repo Google Drive file ID (mapillary_WPCA512.pth.tar)
            run(f"{PYTHON} -m gdown --fuzzy "
                f"'https://drive.google.com/file/d/17luTjZFCX639guSVy6TWh4C_PUoJlCWH' "
                f"-O {ckpt_path}")
        except Exception as e:
            print(f"    다운로드 실패: {e}")

    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)

            # VGG16 encoder weights 로드
            enc_state = {}
            for k, v in state.items():
                stripped = k.replace("module.", "")
                if stripped.startswith("encoder."):
                    enc_state[stripped[len("encoder."):]] = v
            if enc_state:
                model.encoder.load_state_dict(enc_state, strict=False)
                print("    VGG16 encoder weights 로드 완료")

            # VLAD cluster centers / conv 로드
            vlad_state = {}
            for k, v in state.items():
                stripped = k.replace("module.", "")
                if stripped.startswith("pool."):
                    vlad_state[stripped[len("pool."):]] = v
            if vlad_state:
                model.vlad.load_state_dict(vlad_state, strict=False)
                print("    NetVLAD cluster centers 로드 완료")

            checkpoint_loaded = True
        except Exception as e:
            print(f"    Checkpoint 로드 실패: {e}")

    if not checkpoint_loaded:
        print("    pretrained weights 없음 → pretrained VGG16 + random VLAD centers 사용")
        print("    품질 향상을 위해 수동으로 checkpoint를 받아주세요:")
        print("    https://github.com/Nanne/pytorch-NetVlad")

    # ── TorchScript export ────────────────────────────────────────────
    model.eval()
    ex = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(model, ex)
    traced.save(out_pt)
    size_mb = os.path.getsize(out_pt) / 1e6
    ckpt_str = "pretrained" if checkpoint_loaded else "random VLAD centers"
    print(f"  VGG16+NetVLAD exported: {out_pt} ({size_mb:.1f} MB, {ckpt_str})")
    print("  Output: 512-dim L2-normalized descriptor")


def main():
    parser = argparse.ArgumentParser(description="Setup RenderLoc pretrained models")
    parser.add_argument("--output_dir", default="models/")
    parser.add_argument("--cache_dir", default="/tmp/renderloc_cache")
    parser.add_argument("--skip_superpoint", action="store_true")
    parser.add_argument("--skip_superglue", action="store_true")
    parser.add_argument("--skip_global", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    print("=== RenderLoc Model Setup ===")
    print(f"Output: {args.output_dir}")

    if not args.skip_superpoint:
        setup_superpoint(args.output_dir, args.cache_dir)
    if not args.skip_superglue:
        setup_superglue(args.output_dir, args.cache_dir)
    if not args.skip_global:
        setup_netvlad(args.output_dir, args.cache_dir)

    print("\n=== Setup Complete ===")
    print(f"Models in: {args.output_dir}")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith((".pt", ".pth")):
            size_mb = os.path.getsize(os.path.join(args.output_dir, f)) / 1e6
            print(f"  {f:40s} {size_mb:.1f} MB")


if __name__ == "__main__":
    main()