#!/usr/bin/env python3
"""가우시안 PLY → .splat 변환 (antimatter15 / @mkkellogg/gaussian-splats-3d 포맷).

splat 한 점당 32바이트:
  - position : float32 × 3   (12B)
  - scale    : float32 × 3   (12B, exp(scale_i) = 선형 스케일)
  - color    : uint8   × 4   (4B, RGB=0.5+C0·f_dc, A=sigmoid(opacity))
  - rotation : uint8   × 4   (4B, 정규화 quat(w,x,y,z)*128+128)

중요도(스케일 부피 × opacity) 내림차순으로 정렬해 렌더 품질을 높인다.

사용:
  python3 scripts/ros/ply_to_splat.py input.ply output.splat
"""
import argparse
import sys

import numpy as np
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def ply_to_splat_bytes(ply_path, z_min=None, z_max=None):
    """z_min/z_max 가 주어지면 정렬된 맵의 해당 Z 범위(바닥=z=0 기준)만 남긴다.
    top-down 뷰용 '바닥+N m 슬라이스' splat 생성에 사용."""
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    names = v.dtype.names

    x = np.asarray(v["x"], np.float32)
    y = np.asarray(v["y"], np.float32)
    z = np.asarray(v["z"], np.float32)
    pos = np.stack([x, y, z], axis=1)
    n = len(pos)

    # scale: 3DGS는 log-scale 저장 → exp
    if all(f"scale_{i}" in names for i in range(3)):
        s = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
        scale = np.exp(s)
    else:
        scale = np.full((n, 3), 0.01, np.float32)

    # opacity: logit → sigmoid
    if "opacity" in names:
        op = np.asarray(v["opacity"], np.float32)
        alpha = 1.0 / (1.0 + np.exp(-op))
    else:
        alpha = np.ones(n, np.float32)

    # color: SH DC → RGB
    if all(f"f_dc_{i}" in names for i in range(3)):
        f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
        rgb = 0.5 + SH_C0 * f_dc
    elif all(c in names for c in ("red", "green", "blue")):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32) / 255.0
    else:
        rgb = np.full((n, 3), 0.7, np.float32)
    rgba = np.concatenate([rgb, alpha[:, None]], axis=1)
    rgba_u8 = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)

    # rotation: (rot_0..3) = (w,x,y,z), 정규화 후 *128+128
    if all(f"rot_{i}" in names for i in range(4)):
        rot = np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32)
        norm = np.linalg.norm(rot, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        rot = rot / norm
    else:
        rot = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
    rot_u8 = np.clip(rot * 128.0 + 128.0, 0, 255).astype(np.uint8)

    # Z 범위 슬라이싱 (top-down 뷰용). 바닥이 z=0 으로 정렬돼 있다고 가정.
    if z_min is not None or z_max is not None:
        m = np.ones(n, dtype=bool)
        if z_min is not None:
            m &= pos[:, 2] >= float(z_min)
        if z_max is not None:
            m &= pos[:, 2] <= float(z_max)
        pos, scale, alpha = pos[m], scale[m], alpha[m]
        rgba_u8, rot_u8 = rgba_u8[m], rot_u8[m]
        n = int(m.sum())

    # 중요도 내림차순 정렬: exp(Σscale_log) * sigmoid(opacity)
    importance = scale.prod(axis=1) * alpha
    order = np.argsort(-importance)

    dt = np.dtype([
        ("pos", "<f4", 3),
        ("scale", "<f4", 3),
        ("rgba", "u1", 4),
        ("rot", "u1", 4),
    ])
    out = np.zeros(n, dt)
    out["pos"] = pos[order]
    out["scale"] = scale[order]
    out["rgba"] = rgba_u8[order]
    out["rot"] = rot_u8[order]
    return out.tobytes(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_ply")
    ap.add_argument("output_splat")
    ap.add_argument("--z-min", type=float, default=None,
                    help="이 Z(바닥=0 기준 m) 미만 splat 제거")
    ap.add_argument("--z-max", type=float, default=None,
                    help="이 Z(바닥=0 기준 m) 초과 splat 제거 (top-down 슬라이스)")
    args = ap.parse_args()

    data, n = ply_to_splat_bytes(args.input_ply, z_min=args.z_min, z_max=args.z_max)
    with open(args.output_splat, "wb") as f:
        f.write(data)
    print(f"{args.input_ply} → {args.output_splat}  ({n} splats, {len(data)/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
