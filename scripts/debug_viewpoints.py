"""
Visualize 10 viewpoints overlaid on both PLYs (floor vs ceil).
4 subplots: top-down and side view for each PLY.
"""
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pickle, os, sys

OUT = "output/step_by_step"
FLOOR_PLY = os.path.join(OUT, "aligned_map_floor.ply")
CEIL_PLY  = os.path.join(OUT, "aligned_map.ply")
VP_PKL    = os.path.join(OUT, "step1_data.pkl")
N_VP      = 10
SUB_PTS   = 80000   # scatter point budget per PLY

# ── Load viewpoints ──────────────────────────────────────────────────────────
data   = pickle.load(open(VP_PKL, "rb"))
all_vp = data["viewpoints"]
vps    = all_vp[:N_VP]
vp_pos = np.array([v["pose"][:3, 3] for v in vps])          # (N,3) world pos
vp_dir = np.array([v["pose"][:3, 2] for v in vps])          # (N,3) forward (+Z cam axis)
print(f"Loaded {len(all_vp)} viewpoints, showing first {N_VP}")
print(f"Camera positions Z range: {vp_pos[:,2].min():.2f} ~ {vp_pos[:,2].max():.2f}")

# ── Load both PLYs ────────────────────────────────────────────────────────────
plys = {}
for tag, path in [("floor (no ceiling)", FLOOR_PLY), ("ceil (with ceiling)", CEIL_PLY)]:
    print(f"Loading {tag} …")
    pcd  = o3d.io.read_point_cloud(path)
    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None
    sub  = max(1, len(pts) // SUB_PTS)
    plys[tag] = {"pts": pts[::sub], "cols": cols[::sub] if cols is not None else None,
                 "n_total": len(pts)}
    print(f"  {len(pts):,} pts → subsampled {len(pts[::sub]):,} for plot")
    print(f"  XYZ: X[{pts[:,0].min():.1f}, {pts[:,0].max():.1f}] "
          f"Y[{pts[:,1].min():.1f}, {pts[:,1].max():.1f}] "
          f"Z[{pts[:,2].min():.1f}, {pts[:,2].max():.1f}]")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(22, 16))
tags = list(plys.keys())

ARROW_LEN = 1.5   # metres

for col, tag in enumerate(tags):
    pts  = plys[tag]["pts"]
    cols = plys[tag]["cols"]
    c    = cols if cols is not None else pts[:, 2]

    # ── Top-down (X-Y) ───────────────────────────────────────────────────
    ax = axes[0, col]
    ax.scatter(pts[:, 0], pts[:, 1], c=c, s=0.2, alpha=0.4,
               rasterized=True, linewidths=0)
    ax.scatter(vp_pos[:, 0], vp_pos[:, 1], c="red", s=120,
               marker="x", linewidths=2, zorder=5, label="viewpoints")
    for i, (p, d) in enumerate(zip(vp_pos, vp_dir)):
        ax.annotate(str(i), (p[0], p[1]), fontsize=7, color="red",
                    xytext=(2, 2), textcoords="offset points")
        ax.arrow(p[0], p[1], d[0]*ARROW_LEN, d[1]*ARROW_LEN,
                 head_width=0.4, head_length=0.3, fc="orange", ec="orange",
                 linewidth=1.2, zorder=6)
    ax.set_title(f"Top-down (X-Y)\n{tag}  [{plys[tag]['n_total']:,} pts]", fontsize=11)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

    # ── Side (X-Z) ──────────────────────────────────────────────────────
    ax = axes[1, col]
    ax.scatter(pts[:, 0], pts[:, 2], c="steelblue", s=0.2, alpha=0.3,
               rasterized=True, linewidths=0)
    ax.scatter(vp_pos[:, 0], vp_pos[:, 2], c="red", s=120,
               marker="x", linewidths=2, zorder=5, label="viewpoints")
    for i, (p, d) in enumerate(zip(vp_pos, vp_dir)):
        ax.annotate(str(i), (p[0], p[2]), fontsize=7, color="red",
                    xytext=(2, 2), textcoords="offset points")
        ax.arrow(p[0], p[2], d[0]*ARROW_LEN, d[2]*ARROW_LEN,
                 head_width=0.15, head_length=0.12, fc="orange", ec="orange",
                 linewidth=1.2, zorder=6)
    # Mark Z=0 floor
    ax.axhline(0, color="green", ls="--", lw=1, alpha=0.7, label="Z=0 floor")
    ax.set_title(f"Side (X-Z)\n{tag}", fontsize=11)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Z [m]")
    ax.legend(loc="upper right", fontsize=8)

fig.suptitle(f"First {N_VP} viewpoints vs two PLYs", fontsize=14, fontweight="bold")
plt.tight_layout()
out_png = os.path.join(OUT, "debug_viewpoints.png")
plt.savefig(out_png, dpi=150)
print(f"\nSaved → {out_png}")
