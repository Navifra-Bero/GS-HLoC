#!/usr/bin/env python3
"""
Visualize a height slice of an aligned PLY map.

This is intended for sparse SfM / Gaussian PLY debugging before viewpoint
sampling.  It reads PLY directly with plyfile, so Gaussian fields such as
opacity/scale/rotation are not lost through Open3D conversion.

Example:
  python3 scripts/visualize_height_slice.py \
      --ply output/colmap_sgs_test/aligned_map.ply \
      --output output/colmap_sgs_test/height_slice_0p5_1p5.png \
      --z_min 0.5 --z_max 1.5
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData, PlyElement


def _read_xyz_and_fields(path):
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    names = {p.name for p in vertex.properties}

    fields = {}
    if "opacity" in names:
        opacity_raw = np.asarray(vertex["opacity"], dtype=np.float64)
        fields["opacity"] = 1.0 / (1.0 + np.exp(-np.clip(opacity_raw, -60.0, 60.0)))
    for name in ("scale_0", "scale_1", "scale_2"):
        if name in names:
            fields[name] = np.asarray(vertex[name], dtype=np.float64)
    return ply, xyz, fields


def _load_floor_z(step0_pkl, default):
    if not step0_pkl:
        return default
    if not os.path.exists(step0_pkl):
        raise FileNotFoundError(step0_pkl)
    data = pickle.load(open(step0_pkl, "rb"))
    # step0 aligned_map is shifted so floor is z=0.  Keep this hook for future
    # nonzero floors or manual experiments.
    return float(data.get("floor_z_aligned", default))


def _sample_indices(n, max_points, seed):
    if max_points <= 0 or n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _write_filtered_ply(ply, mask, output_path):
    vertex = np.array(ply["vertex"].data, copy=True)[mask]
    elements = [PlyElement.describe(vertex, "vertex")]
    elements.extend([el for el in ply.elements if el.name != "vertex"])
    PlyData(
        elements,
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    ).write(output_path)


def main():
    parser = argparse.ArgumentParser(description="Top-down visualization of a PLY height slice")
    parser.add_argument("--ply", required=True, help="Input aligned PLY")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--filtered_ply", default=None, help="Optional output PLY containing only the slice")
    parser.add_argument("--step0_pkl", default=None, help="Optional step0_data.pkl")
    parser.add_argument("--floor_z", type=float, default=0.0, help="Aligned floor height")
    parser.add_argument("--z_min", type=float, default=0.1, help="Slice min height above floor")
    parser.add_argument("--z_max", type=float, default=1.5, help="Slice max height above floor")
    parser.add_argument("--max_points", type=int, default=2500000, help="Max points to draw; <=0 draws all")
    parser.add_argument("--point_size", type=float, default=0.1)
    parser.add_argument("--color_by", choices=("z", "opacity", "scale_0", "density"), default="z")
    parser.add_argument("--grid_resolution", type=float, default=0.2, help="Density image resolution in meters")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    floor_z = _load_floor_z(args.step0_pkl, args.floor_z)
    z0 = floor_z + args.z_min
    z1 = floor_z + args.z_max

    ply, xyz, fields = _read_xyz_and_fields(args.ply)
    mask = (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1)
    sliced = xyz[mask]
    if len(sliced) == 0:
        raise RuntimeError(f"No points in height slice z=[{z0:.3f}, {z1:.3f}]")

    if args.filtered_ply:
        os.makedirs(os.path.dirname(os.path.abspath(args.filtered_ply)), exist_ok=True)
        _write_filtered_ply(ply, mask, args.filtered_ply)
        print(f"  Saved filtered PLY: {args.filtered_ply} ({len(sliced):,} points)")

    draw_idx = _sample_indices(len(sliced), args.max_points, args.seed)
    draw = sliced[draw_idx]
    global_idx = np.flatnonzero(mask)[draw_idx]

    if args.color_by == "opacity" and "opacity" in fields:
        color = fields["opacity"][global_idx]
        cmap = "magma"
        color_label = "opacity"
    elif args.color_by.startswith("scale") and args.color_by in fields:
        color = fields[args.color_by][global_idx]
        cmap = "viridis"
        color_label = args.color_by
    else:
        color = draw[:, 2] - floor_z
        cmap = "turbo"
        color_label = "height above floor (m)"

    x_min, y_min = sliced[:, :2].min(axis=0)
    x_max, y_max = sliced[:, :2].max(axis=0)
    gr = float(args.grid_resolution)
    iw = max(1, int(np.ceil((x_max - x_min) / gr)))
    ih = max(1, int(np.ceil((y_max - y_min) / gr)))
    px = np.clip(((sliced[:, 0] - x_min) / gr).astype(int), 0, iw - 1)
    py = np.clip(((sliced[:, 1] - y_min) / gr).astype(int), 0, ih - 1)
    density = np.zeros((ih, iw), dtype=np.float32)
    np.add.at(density, (py, px), 1.0)
    density_vis = np.log1p(density)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    ax = axes[0]
    sc = ax.scatter(draw[:, 0], draw[:, 1], c=color, s=args.point_size, cmap=cmap, alpha=0.75)
    ax.set_title(f"Top-down slice: z={args.z_min:.2f}~{args.z_max:.2f}m above floor")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=color_label)

    ax = axes[1]
    im = ax.imshow(
        density_vis,
        origin="lower",
        extent=[x_min, x_min + iw * gr, y_min, y_min + ih * gr],
        cmap="gray",
    )
    ax.set_title(f"Top-down density (log, {gr:.2f}m/px)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log(1 + points/bin)")

    ax = axes[2]
    side_idx = _sample_indices(len(xyz), args.max_points, args.seed)
    side = xyz[side_idx]
    ax.scatter(side[:, 0], side[:, 2], c="lightgray", s=args.point_size, alpha=0.35)
    ax.scatter(draw[:, 0], draw[:, 2], c="tab:red", s=args.point_size * 2.0, alpha=0.65)
    ax.axhline(floor_z, color="green", ls="--", lw=1.5, label="floor")
    ax.axhline(z0, color="tab:blue", ls="--", lw=1.2, label=f"+{args.z_min:.2f}m")
    ax.axhline(z1, color="tab:blue", ls="--", lw=1.2, label=f"+{args.z_max:.2f}m")
    ax.set_title("Side view sanity check")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.legend(loc="upper right")

    fig.suptitle(
        f"{os.path.basename(args.ply)} | selected {len(sliced):,}/{len(xyz):,} points "
        f"({100.0 * len(sliced) / len(xyz):.1f}%)",
        fontsize=14,
    )
    fig.tight_layout()

    output = args.output
    if output is None:
        stem = os.path.splitext(os.path.basename(args.ply))[0]
        output = os.path.join(
            os.path.dirname(args.ply),
            f"{stem}_height_slice_{args.z_min:.2f}_{args.z_max:.2f}.png",
        )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(f"  Input points    : {len(xyz):,}")
    print(f"  Slice z range   : [{z0:.3f}, {z1:.3f}]")
    print(f"  Selected points : {len(sliced):,} ({100.0 * len(sliced) / len(xyz):.1f}%)")
    print(f"  XY bbox         : x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
    print(f"  Saved PNG       : {output}")


if __name__ == "__main__":
    main()
