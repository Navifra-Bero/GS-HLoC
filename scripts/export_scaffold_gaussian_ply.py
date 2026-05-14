#!/usr/bin/env python3
"""Bake a Scaffold-GS sparse checkpoint into a regular 3DGS-style PLY.

Scaffold-GS stores anchors, offsets, and view-dependent MLPs. This script
evaluates those MLPs from one or more camera centers and writes the resulting
Gaussians as ordinary PLY fields:

  x y z, nx ny nz, f_dc_0..2, opacity, scale_0..2, rot_0..3

The output is compatible with the gaussian_ply loader in step2_render.py.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData


SH_C0 = 0.28209479177387814


PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("f_dc_0", "<f4"),
        ("f_dc_1", "<f4"),
        ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"),
        ("scale_1", "<f4"),
        ("scale_2", "<f4"),
        ("rot_0", "<f4"),
        ("rot_1", "<f4"),
        ("rot_2", "<f4"),
        ("rot_3", "<f4"),
    ]
)


def _read_cfg_args(model_path: Path) -> dict:
    cfg_path = model_path / "cfg_args"
    if not cfg_path.exists():
        return {}
    text = cfg_path.read_text(encoding="utf-8").strip()
    if text.startswith("Namespace(") and text.endswith(")"):
        text = text[len("Namespace(") : -1]
    cfg = {}
    for item in text.split(","):
        if "=" not in item:
            continue
        key, value = item.strip().split("=", 1)
        try:
            cfg[key.strip()] = ast.literal_eval(value.strip())
        except Exception:
            cfg[key.strip()] = value.strip()
    return cfg


def _sorted_fields(names: tuple[str, ...], prefix: str) -> list[str]:
    fields = [name for name in names if name.startswith(prefix)]
    return sorted(fields, key=lambda name: int(name.rsplit("_", 1)[1]))


def _load_sparse_ply(path: Path):
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()

    anchors = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

    offset_names = _sorted_fields(names, "f_offset_")
    feat_names = _sorted_fields(names, "f_anchor_feat_")
    scale_names = _sorted_fields(names, "scale_")

    offsets = np.stack([vertex[name] for name in offset_names], axis=1).astype(np.float32)
    if offsets.shape[1] % 3 != 0:
        raise ValueError(f"offset field count is not divisible by 3: {offsets.shape[1]}")
    n_offsets = offsets.shape[1] // 3
    offsets = offsets.reshape(offsets.shape[0], 3, n_offsets).transpose(0, 2, 1).copy()

    anchor_feat = np.stack([vertex[name] for name in feat_names], axis=1).astype(np.float32)
    scaling_raw = np.stack([vertex[name] for name in scale_names], axis=1).astype(np.float32)
    if scaling_raw.shape[1] != 6:
        raise ValueError(f"expected 6 scale fields in Scaffold-GS PLY, got {scaling_raw.shape[1]}")

    return anchors, offsets, anchor_feat, scaling_raw


def _load_camera_centers(model_path: Path, num_views: int) -> np.ndarray:
    cameras_path = model_path / "cameras.json"
    if not cameras_path.exists():
        raise FileNotFoundError(f"cameras.json not found: {cameras_path}")
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    if not cameras:
        raise ValueError(f"no cameras in {cameras_path}")
    count = min(max(1, int(num_views)), len(cameras))
    indices = np.linspace(0, len(cameras) - 1, count, dtype=int)
    centers = [cameras[i]["position"] for i in indices]
    return np.asarray(centers, dtype=np.float32)


def _load_mlp(path: Path):
    module = torch.jit.load(str(path), map_location="cpu")
    module.eval()
    return module


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def _write_header(out, count: int):
    fields = "\n".join(f"property float {name}" for name in PLY_DTYPE.names)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        f"{fields}\n"
        "end_header\n"
    )
    out.write(header.encode("ascii"))


def _make_chunk_records(
    xyz: torch.Tensor,
    color: torch.Tensor,
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    rot: torch.Tensor,
) -> np.ndarray:
    n = int(xyz.shape[0])
    records = np.empty(n, dtype=PLY_DTYPE)
    xyz_np = xyz.cpu().numpy().astype(np.float32, copy=False)
    color_np = color.cpu().numpy().astype(np.float32, copy=False)
    opacity_raw_np = _inverse_sigmoid(opacity.reshape(-1)).cpu().numpy().astype(np.float32, copy=False)
    scale_log_np = torch.log(scaling.clamp_min(1e-8)).cpu().numpy().astype(np.float32, copy=False)
    rot_np = rot.cpu().numpy().astype(np.float32, copy=False)

    records["x"] = xyz_np[:, 0]
    records["y"] = xyz_np[:, 1]
    records["z"] = xyz_np[:, 2]
    records["nx"] = 0.0
    records["ny"] = 0.0
    records["nz"] = 0.0
    records["f_dc_0"] = (color_np[:, 0] - 0.5) / SH_C0
    records["f_dc_1"] = (color_np[:, 1] - 0.5) / SH_C0
    records["f_dc_2"] = (color_np[:, 2] - 0.5) / SH_C0
    records["opacity"] = opacity_raw_np
    records["scale_0"] = scale_log_np[:, 0]
    records["scale_1"] = scale_log_np[:, 1]
    records["scale_2"] = scale_log_np[:, 2]
    records["rot_0"] = rot_np[:, 0]
    records["rot_1"] = rot_np[:, 1]
    records["rot_2"] = rot_np[:, 2]
    records["rot_3"] = rot_np[:, 3]
    return records


def export_gaussian_ply(
    model_path: Path,
    iteration: int,
    output_path: Path,
    num_bake_views: int,
    batch_anchors: int,
    opacity_min: float,
):
    cfg = _read_cfg_args(model_path)
    add_opacity_dist = bool(cfg.get("add_opacity_dist", True))
    add_cov_dist = bool(cfg.get("add_cov_dist", True))
    add_color_dist = bool(cfg.get("add_color_dist", True))
    use_feat_bank = bool(cfg.get("use_feat_bank", False))
    if use_feat_bank:
        raise NotImplementedError("CPU exporter does not yet support use_feat_bank=True")
    if int(cfg.get("appearance_dim", 0)) > 0:
        raise NotImplementedError("CPU exporter does not yet support appearance_dim > 0")

    ckpt_dir = model_path / "point_cloud" / f"iteration_{iteration}"
    sparse_ply = ckpt_dir / "point_cloud.ply"
    if not sparse_ply.exists():
        raise FileNotFoundError(f"Sparse Scaffold-GS PLY not found: {sparse_ply}")

    print(f"Loading sparse Scaffold-GS checkpoint: {ckpt_dir}")
    anchors_np, offsets_np, anchor_feat_np, scaling_raw_np = _load_sparse_ply(sparse_ply)
    n_anchors, n_offsets = offsets_np.shape[:2]
    feat_dim = anchor_feat_np.shape[1]
    print(f"Anchors: {n_anchors:,}, offsets: {n_offsets}, feature dim: {feat_dim}")

    opacity_mlp = _load_mlp(ckpt_dir / "opacity_mlp.pt")
    cov_mlp = _load_mlp(ckpt_dir / "cov_mlp.pt")
    color_mlp = _load_mlp(ckpt_dir / "color_mlp.pt")
    camera_centers = _load_camera_centers(model_path, num_bake_views)
    print(f"Baking with {len(camera_centers)} camera center(s)")

    # Scaffold-GS normalizes anchor-camera distance by the mean distance of
    # every visible anchor for that view. When exporting in chunks, using a
    # per-chunk mean changes the MLP input distribution and produces very
    # different opacity/covariance. Precompute the same global mean per bake
    # camera before the chunked pass.
    mean_dists = []
    with torch.no_grad():
        anchors_all = torch.from_numpy(anchors_np)
        for center_np in camera_centers:
            center = torch.from_numpy(center_np).reshape(1, 3)
            dist_sum = 0.0
            count = 0
            for start in range(0, n_anchors, batch_anchors):
                end = min(start + batch_anchors, n_anchors)
                d = torch.linalg.norm(anchors_all[start:end] - center, dim=1)
                dist_sum += float(d.sum().item())
                count += int(d.numel())
            mean_dists.append(dist_sum / max(count, 1))
    print("Distance means:", ", ".join(f"{d:.4f}" for d in mean_dists))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=output_path.name + ".", suffix=".body", dir=str(output_path.parent))
    total_written = 0

    try:
        with os.fdopen(tmp_fd, "wb") as body:
            with torch.no_grad():
                for start in range(0, n_anchors, batch_anchors):
                    end = min(start + batch_anchors, n_anchors)
                    b = end - start
                    anchors = torch.from_numpy(anchors_np[start:end])
                    offsets = torch.from_numpy(offsets_np[start:end])
                    feat = torch.from_numpy(anchor_feat_np[start:end])
                    grid_scaling = torch.exp(torch.from_numpy(scaling_raw_np[start:end]))

                    m = b * n_offsets
                    color_sum = torch.zeros((m, 3), dtype=torch.float32)
                    opacity_sum = torch.zeros((m, 1), dtype=torch.float32)
                    scaling_sum = torch.zeros((m, 3), dtype=torch.float32)
                    rot_sum = torch.zeros((m, 4), dtype=torch.float32)
                    counts = torch.zeros((m, 1), dtype=torch.float32)
                    xyz_ref = None
                    rot_ref = None

                    for center_np, mean_dist in zip(camera_centers, mean_dists):
                        center = torch.from_numpy(center_np).reshape(1, 3)
                        ob_view = anchors - center
                        ob_dist = torch.linalg.norm(ob_view, dim=1, keepdim=True).clamp_min(1e-8)
                        ob_view = ob_view / ob_dist
                        ob_dist = ob_dist / (float(mean_dist) + 1e-8)

                        if add_opacity_dist:
                            opacity_input = torch.cat([feat, ob_view, ob_dist], dim=1)
                        else:
                            opacity_input = torch.cat([feat, ob_view], dim=1)
                        if add_color_dist:
                            color_input = torch.cat([feat, ob_view, ob_dist], dim=1)
                        else:
                            color_input = torch.cat([feat, ob_view], dim=1)
                        if add_cov_dist:
                            cov_input = torch.cat([feat, ob_view, ob_dist], dim=1)
                        else:
                            cov_input = torch.cat([feat, ob_view], dim=1)

                        neural_opacity = opacity_mlp(opacity_input).reshape(-1, 1)
                        valid = neural_opacity[:, 0] > float(opacity_min)
                        if not torch.any(valid):
                            continue

                        color = color_mlp(color_input).reshape(m, 3)
                        scale_rot = cov_mlp(cov_input).reshape(m, 7)

                        scaling_repeat = grid_scaling[:, 3:].repeat_interleave(n_offsets, dim=0)
                        offset_scale = grid_scaling[:, :3].repeat_interleave(n_offsets, dim=0)
                        repeat_anchor = anchors.repeat_interleave(n_offsets, dim=0)
                        flat_offsets = offsets.reshape(m, 3)

                        xyz = repeat_anchor + flat_offsets * offset_scale
                        scaling = scaling_repeat * torch.sigmoid(scale_rot[:, :3])
                        rot = torch.nn.functional.normalize(scale_rot[:, 3:7], dim=-1)

                        if xyz_ref is None:
                            xyz_ref = xyz
                            rot_ref = rot
                        else:
                            sign = torch.where((rot * rot_ref).sum(dim=1, keepdim=True) < 0, -1.0, 1.0)
                            rot = rot * sign

                        mask = valid.reshape(-1, 1)
                        color_sum += color * mask
                        opacity_sum += neural_opacity.clamp(0.0, 1.0) * mask
                        scaling_sum += scaling * mask
                        rot_sum += rot * mask
                        counts += mask.float()

                    keep = counts[:, 0] > 0
                    if torch.any(keep):
                        denom = counts[keep].clamp_min(1.0)
                        xyz_out = xyz_ref[keep]
                        color_out = (color_sum[keep] / denom).clamp(0.0, 1.0)
                        opacity_out = (opacity_sum[keep] / denom).clamp(1e-6, 1.0 - 1e-6)
                        scaling_out = (scaling_sum[keep] / denom).clamp_min(1e-8)
                        rot_out = torch.nn.functional.normalize(rot_sum[keep] / denom, dim=-1)
                        records = _make_chunk_records(
                            xyz_out, color_out, opacity_out, scaling_out, rot_out
                        )
                        body.write(records.tobytes())
                        total_written += int(records.shape[0])

                    print(
                        f"  anchors {end:,}/{n_anchors:,} -> gaussians {total_written:,}",
                        flush=True,
                    )

        with output_path.open("wb") as out:
            _write_header(out, total_written)
            with open(tmp_name, "rb") as body:
                while True:
                    chunk = body.read(1024 * 1024 * 64)
                    if not chunk:
                        break
                    out.write(chunk)

    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Saved: {output_path} ({total_written:,} gaussians, {size_mb:.1f} MiB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        default="scaffold_gs_result/2026-04-30_15:53:27",
        help="Scaffold-GS model directory.",
    )
    parser.add_argument("--iteration", type=int, default=120000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num_bake_views", type=int, default=1)
    parser.add_argument("--batch_anchors", type=int, default=32768)
    parser.add_argument("--opacity_min", type=float, default=0.0)
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else model_path / "point_cloud" / f"iteration_{args.iteration}" / "gaussian_baked.ply"
    )
    export_gaussian_ply(
        model_path=model_path,
        iteration=args.iteration,
        output_path=output,
        num_bake_views=args.num_bake_views,
        batch_anchors=args.batch_anchors,
        opacity_min=args.opacity_min,
    )


if __name__ == "__main__":
    main()
