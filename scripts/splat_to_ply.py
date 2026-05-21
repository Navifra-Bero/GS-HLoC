#!/usr/bin/env python3
"""
Convert a WebGL-style .splat file into an Open3D-readable PLY proxy.

The output is meant for geometry/debug tasks such as viewpoint sampling.  It
keeps the .splat Gaussian centers by default, and can optionally sample a few
points from each Gaussian ellipsoid when a denser proxy is useful.
"""

import argparse
import os

import numpy as np
import open3d as o3d


def _quat_wxyz_to_rotmat(q):
    q = q.astype(np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)


def load_splat(path):
    raw = np.fromfile(path, dtype=np.uint8)
    if len(raw) % 32 != 0:
        raise ValueError(f"{path} is not a 32-byte .splat stream: {len(raw)} bytes")
    rec = raw.reshape(-1, 32)
    xyz = rec[:, 0:12].copy().view(np.float32).reshape(-1, 3)
    scale = rec[:, 12:24].copy().view(np.float32).reshape(-1, 3)
    rgba = rec[:, 24:28].astype(np.float32) / 255.0
    quat = rec[:, 28:32].astype(np.float32) / 127.5 - 1.0
    return xyz, scale, rgba, quat


def sample_gaussians(xyz, scale, rgba, quat, samples_per_splat, scale_multiplier, seed):
    if samples_per_splat <= 1:
        return xyz, rgba[:, :3]

    rng = np.random.default_rng(seed)
    n = len(xyz)
    rot = _quat_wxyz_to_rotmat(quat)
    noise = rng.normal(size=(n, samples_per_splat, 3)).astype(np.float32)
    local = noise * (scale[:, None, :] * float(scale_multiplier))
    pts = np.einsum("nij,nkj->nki", rot, local) + xyz[:, None, :]
    rgb = np.repeat(rgba[:, None, :3], samples_per_splat, axis=1)
    return pts.reshape(-1, 3), rgb.reshape(-1, 3)


def main():
    parser = argparse.ArgumentParser(description=".splat -> PLY proxy")
    parser.add_argument("--splat", required=True, help="Input .splat path")
    parser.add_argument("--output", required=True, help="Output PLY path")
    parser.add_argument("--opacity_thresh", type=float, default=0.02)
    parser.add_argument("--max_scale", type=float, default=1.0,
                        help="Drop Gaussians whose largest scale exceeds this; <=0 disables")
    parser.add_argument("--samples_per_splat", type=int, default=1,
                        help="1 keeps centers only; >1 samples ellipsoid points")
    parser.add_argument("--scale_multiplier", type=float, default=1.0)
    parser.add_argument("--voxel_size", type=float, default=0.03,
                        help="Voxel downsample size in meters; <=0 disables")
    parser.add_argument("--max_points", type=int, default=3000000,
                        help="Randomly subsample after filtering; <=0 disables")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    xyz, scale, rgba, quat = load_splat(args.splat)
    print(f"Loaded splats: {len(xyz):,}")

    mask = np.isfinite(xyz).all(axis=1) & np.isfinite(scale).all(axis=1)
    mask &= rgba[:, 3] >= args.opacity_thresh
    if args.max_scale > 0:
        mask &= np.max(scale, axis=1) <= args.max_scale
    xyz, scale, rgba, quat = xyz[mask], scale[mask], rgba[mask], quat[mask]
    print(f"After filter : {len(xyz):,}  "
          f"(opacity>={args.opacity_thresh}, max_scale<={args.max_scale})")

    if args.max_points > 0 and len(xyz) > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(xyz), size=args.max_points, replace=False))
        xyz, scale, rgba, quat = xyz[idx], scale[idx], rgba[idx], quat[idx]
        print(f"Subsample    : {len(xyz):,}")

    pts, rgb = sample_gaussians(
        xyz, scale, rgba, quat,
        args.samples_per_splat,
        args.scale_multiplier,
        args.seed,
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0.0, 1.0).astype(np.float64))
    if args.voxel_size > 0:
        before = len(pcd.points)
        pcd = pcd.voxel_down_sample(args.voxel_size)
        print(f"Voxel        : {before:,} -> {len(pcd.points):,} "
              f"(size={args.voxel_size}m)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    o3d.io.write_point_cloud(args.output, pcd, write_ascii=False, compressed=False)
    mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Saved        : {args.output} ({len(pcd.points):,} pts, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
