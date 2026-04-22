#!/usr/bin/env python3
"""
gs_to_dense_ply.py
==================
Gaussian Splatting .pt 체크포인트를 dense point cloud PLY로 변환.

각 Gaussian 타원체에서 포인트를 샘플링해서 sparse PLY보다 훨씬 dense하고
Gaussian의 형태(크기, 방향)를 반영한 포인트클라우드를 생성합니다.

Usage:
    python3 scripts/gs_to_dense_ply.py \
        --ckpt output/gs_test/gaussians.pt \
        --output output/gs_test/gaussian_dense.ply \
        --samples_per_gs 3 \
        --opacity_thresh 0.1 \
        --scale_multiplier 1.0
"""
import argparse
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F


def quat_to_rotmat(quats):
    """(N,4) wxyz quaternion → (N,3,3) rotation matrix"""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w),
          2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w),
          2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y),
    ], dim=1).reshape(-1, 3, 3)
    return R


def gaussian_to_points(means, scales, quats, colors, opacities,
                       samples_per_gs=3, opacity_thresh=0.1,
                       scale_multiplier=1.0):
    """
    각 Gaussian 타원체에서 포인트를 샘플링해 dense point cloud 생성.

    Args:
        means:       (N, 3) Gaussian 중심
        scales:      (N, 3) Gaussian scale (이미 exp 적용된 양수값)
        quats:       (N, 4) wxyz quaternion
        colors:      (N, 3) RGB [0,1]
        opacities:   (N,)   opacity [0,1]
        samples_per_gs: Gaussian당 샘플 포인트 수
        opacity_thresh: 이 값 미만 Gaussian 제거
        scale_multiplier: scale 배율 (1.0 = 1-sigma, 2.0 = 2-sigma)

    Returns:
        xyz:  (M, 3) numpy
        rgb:  (M, 3) uint8
    """
    # Opacity 필터링
    mask = opacities > opacity_thresh
    means    = means[mask]
    scales   = scales[mask]
    quats    = quats[mask]
    colors   = colors[mask]
    opacities = opacities[mask]
    N = len(means)
    print(f"  Gaussians after opacity filter: {N:,} (thresh={opacity_thresh})")

    if N == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

    # Rotation matrices
    R = quat_to_rotmat(F.normalize(quats, dim=-1))   # (N, 3, 3)

    # Covariance: Σ = R @ diag(s²) @ R^T  →  L = R @ diag(s)  (Cholesky-like)
    s = scales * scale_multiplier   # (N, 3)
    L = R * s.unsqueeze(1)          # (N, 3, 3): L[:, :, i] = R[:, :, i] * s[:, i]

    all_pts  = []
    all_cols = []

    chunk = 50000
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        n   = end - start

        # 단위구에서 샘플링 후 타원체로 변환
        # samples_per_gs개 샘플을 가우시안 분포에서 뽑기
        noise = torch.randn(n, samples_per_gs, 3, device=means.device)  # (n, K, 3)

        # L @ noise: (n, 3, 3) @ (n, K, 3, 1) → (n, K, 3)
        L_chunk = L[start:end]    # (n, 3, 3)
        # einsum: 'nij, nkj -> nki'
        pts = torch.einsum('nij,nkj->nki', L_chunk, noise)  # (n, K, 3)
        pts = pts + means[start:end].unsqueeze(1)            # (n, K, 3) + (n, 1, 3)
        pts = pts.reshape(-1, 3)                             # (n*K, 3)

        # 색상: opacity 가중 밝기
        col = colors[start:end]         # (n, 3)
        opa = opacities[start:end]      # (n,)
        # opacity에 비례해서 밝기 조정 (너무 어두운 Gaussian은 흐리게)
        col_weighted = col * opa.unsqueeze(1).clamp(0.3, 1.0)
        col_rep = col_weighted.unsqueeze(1).expand(-1, samples_per_gs, -1).reshape(-1, 3)

        all_pts.append(pts.cpu())
        all_cols.append(col_rep.cpu())

    xyz_t = torch.cat(all_pts, dim=0)
    rgb_t = torch.cat(all_cols, dim=0)

    xyz_np = xyz_t.numpy().astype(np.float32)
    rgb_np = (rgb_t.numpy().clip(0, 1) * 255).astype(np.uint8)

    return xyz_np, rgb_np


def load_checkpoint(ckpt_path, device):
    """gaussians.pt 로드 → means, scales, quats, colors, opacities"""
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"  Checkpoint keys: {list(ckpt.keys())}")

    means     = ckpt["means"].float()
    scales_raw = ckpt["scales"].float()
    quats     = ckpt["quats"].float()
    opacities_raw = ckpt["opacities"].float()

    # activation 적용
    scales    = torch.exp(scales_raw)          # log-scale → scale
    quats     = F.normalize(quats, dim=-1)
    opacities = torch.sigmoid(opacities_raw)   # logit → [0,1]

    # 색상: SH degree=0 (colors_sh) 또는 raw colors
    if "colors_sh" in ckpt:
        SH_C0 = 0.28209479177387814
        colors = (ckpt["colors_sh"][:, 0, :].float() * SH_C0 + 0.5).clamp(0, 1)
    elif "colors" in ckpt:
        colors = ckpt["colors"].float().clamp(0, 1)
    else:
        colors = torch.full((len(means), 3), 0.7)

    print(f"  Gaussians: {len(means):,}")
    print(f"  Opacity stats: mean={opacities.mean():.3f}  "
          f"max={opacities.max():.3f}  >0.1: {(opacities>0.1).sum():,}")
    print(f"  Scale stats: mean={scales.mean():.4f}  max={scales.max():.4f}")

    return means, scales, quats, colors, opacities


def main():
    parser = argparse.ArgumentParser(description="GS checkpoint → dense PLY")
    parser.add_argument("--ckpt",    required=True,
                        help="gaussians.pt 경로")
    parser.add_argument("--output",  required=True,
                        help="출력 PLY 경로")
    parser.add_argument("--samples_per_gs", type=int, default=3,
                        help="Gaussian당 샘플 포인트 수 (default: 3)")
    parser.add_argument("--opacity_thresh", type=float, default=0.1,
                        help="이 값 미만 Gaussian 제거 (default: 0.1)")
    parser.add_argument("--scale_multiplier", type=float, default=1.0,
                        help="타원체 크기 배율 1=1σ, 2=2σ (default: 1.0)")
    parser.add_argument("--voxel_size", type=float, default=0.0,
                        help="최종 PLY voxel 다운샘플 (0=사용안함, 예: 0.02)")
    parser.add_argument("--max_scale", type=float, default=0.5,
                        help="이 값 초과 Gaussian 제거 (needle 방지, default: 0.5m)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading: {args.ckpt}")

    means, scales, quats, colors, opacities = load_checkpoint(args.ckpt, device)

    # 너무 큰 Gaussian (needle/artifact) 추가 필터링
    if args.max_scale > 0:
        scale_max = scales.max(dim=-1).values
        scale_mask = scale_max <= args.max_scale
        n_before = len(means)
        means, scales, quats, colors, opacities = (
            means[scale_mask], scales[scale_mask], quats[scale_mask],
            colors[scale_mask], opacities[scale_mask])
        print(f"  Scale filter (max<={args.max_scale}m): "
              f"{n_before:,} → {len(means):,}")

    print(f"\nSampling {args.samples_per_gs} pts/Gaussian "
          f"(opacity>{args.opacity_thresh}, scale×{args.scale_multiplier})...")
    xyz, rgb = gaussian_to_points(
        means, scales, quats, colors, opacities,
        samples_per_gs=args.samples_per_gs,
        opacity_thresh=args.opacity_thresh,
        scale_multiplier=args.scale_multiplier,
    )
    print(f"  Sampled points: {len(xyz):,}")

    # Open3D PLY로 저장
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)

    if args.voxel_size > 0:
        n_before = len(pcd.points)
        pcd = pcd.voxel_down_sample(args.voxel_size)
        print(f"  Voxel downsample ({args.voxel_size}m): "
              f"{n_before:,} → {len(pcd.points):,}")

    o3d.io.write_point_cloud(args.output, pcd, write_ascii=False, compressed=False)
    size_mb = __import__('os').path.getsize(args.output) / 1e6
    print(f"\nSaved: {args.output}  ({len(pcd.points):,} pts, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
