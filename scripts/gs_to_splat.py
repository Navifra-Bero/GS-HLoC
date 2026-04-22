"""
Scaffold-GS → .splat 변환기 (standalone, 학습 데이터 불필요)

학습된 모델 폴더의 point_cloud.ply + MLP 체크포인트를 로드하고,
cameras.json 의 학습 카메라들에서 Gaussian 을 "bake" (여러 뷰 평균) 하여
.splat 바이너리 파일로 저장합니다.

.splat 포맷 (32 bytes/Gaussian):
  pos    : 3 × float32  (12 bytes)
  scale  : 3 × float32  (12 bytes)  ← exp-activated 값
  color  : RGBA 4 × uint8 ( 4 bytes)
  rot    : wxyz 4 × uint8 ( 4 bytes)  ← [-1,1] → [0,255]

Usage:
  python3 scripts/gs_to_splat.py \
      -m scaffold_gs_result/2026-04-14_17:55:21 \
      [--iteration 120000] \
      [--num_bake_views 30] \
      [--output output/map.splat]
"""

import os, sys, ast, json
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm

# ── scaffold_gs 경로 ──────────────────────────────────────────────────────────
_SGS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "third_party", "scaffold_gs")
)
if _SGS_ROOT not in sys.path:
    sys.path.insert(0, _SGS_ROOT)


# ── 모델 로드 (Scene 없이) ─────────────────────────────────────────────────────
def _read_cfg_args(model_path: str) -> dict:
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        return {}
    text = open(cfg_path).read().strip()
    text = text.replace("Namespace(", "").rstrip(")")
    params = {}
    for item in text.split(","):
        if "=" not in item:
            continue
        k, _, v = item.strip().partition("=")
        try:
            params[k.strip()] = ast.literal_eval(v.strip())
        except Exception:
            params[k.strip()] = v.strip()
    return params


def load_gaussians(model_path: str, iteration: int):
    from scene.gaussian_model import GaussianModel
    from scene import searchForMaxIteration

    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(model_path, "point_cloud"))

    cfg = _read_cfg_args(model_path)
    def g(k, d): return cfg.get(k, d)

    gaussians = GaussianModel(
        feat_dim               = g("feat_dim",               64),
        n_offsets              = g("n_offsets",              15),
        voxel_size             = g("voxel_size",             0.0),
        update_depth           = g("update_depth",           3),
        update_init_factor     = g("update_init_factor",     16),
        update_hierachy_factor = g("update_hierachy_factor", 4),
        use_feat_bank          = g("use_feat_bank",          False),
        appearance_dim         = g("appearance_dim",         0),
        ratio                  = g("ratio",                  1),
        add_opacity_dist       = g("add_opacity_dist",       False),
        add_cov_dist           = g("add_cov_dist",           False),
        add_color_dist         = g("add_color_dist",         True),
    )

    ckpt_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"checkpoint 없음: {ckpt_dir}")

    gaussians.load_ply_sparse_gaussian(os.path.join(ckpt_dir, "point_cloud.ply"))
    gaussians.load_mlp_checkpoints(ckpt_dir)
    gaussians.eval()
    print(f"  [load] iter={iteration}, anchors: {gaussians.get_anchor.shape[0]:,}")
    return gaussians, iteration


# ── cameras.json → Camera 객체 목록 ───────────────────────────────────────────
def load_bake_cameras(model_path: str, num_bake_views: int):
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov

    cam_json = os.path.join(model_path, "cameras.json")
    if not os.path.exists(cam_json):
        raise FileNotFoundError(f"cameras.json 없음: {cam_json}")

    all_cams = json.load(open(cam_json))
    idxs = np.linspace(0, len(all_cams) - 1, min(num_bake_views, len(all_cams)), dtype=int)
    selected = [all_cams[i] for i in idxs]
    print(f"  [bake] {len(all_cams)} cameras → {len(selected)} 개 선택 (baking)")

    cameras = []
    for uid, cam in enumerate(selected):
        pos = np.array(cam["position"], dtype=np.float64)   # camera center (C2W col3)
        R_c2w = np.array(cam["rotation"], dtype=np.float64) # C2W rotation
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R_c2w.T
        w2c[:3,  3] = -(R_c2w.T @ pos)
        R_w2c = w2c[:3, :3]
        T_w2c = w2c[:3,  3]

        fx = cam.get("fx", 1039.0)
        fy = cam.get("fy", 1041.0)
        w  = cam.get("width",  1920)
        h  = cam.get("height", 1200)
        cx = cam.get("cx", w / 2)
        cy = cam.get("cy", h / 2)

        c = Camera(
            colmap_id    = uid,
            R            = R_w2c.T,
            T            = T_w2c,
            FoVx         = focal2fov(fx, w),
            FoVy         = focal2fov(fy, h),
            image        = torch.zeros((3, h, w), dtype=torch.float32),
            gt_alpha_mask= None,
            image_name   = str(uid),
            uid          = uid,
            data_device  = "cuda",
        )
        cameras.append(c)

    return cameras


# ── Gaussian baking ───────────────────────────────────────────────────────────
def bake_gaussians(gaussians, cameras, opacity_threshold: float = 0.005):
    from gaussian_renderer import generate_neural_gaussians

    n_anchors = gaussians.get_anchor.shape[0]
    all_visible = torch.ones(n_anchors, dtype=torch.bool, device="cuda")

    color_sum   = None
    opacity_sum = None
    xyz_ref = scaling_ref = rot_ref = None
    count = 0

    with torch.no_grad():
        for cam in tqdm(cameras, desc="  Baking"):
            try:
                xyz, color, opacity, scaling, rot = generate_neural_gaussians(
                    cam, gaussians, all_visible, is_training=False
                )
            except Exception as e:
                print(f"    skip cam {cam.uid}: {e}")
                continue

            if color_sum is None:
                color_sum   = color.clone()
                opacity_sum = opacity.clone()
                xyz_ref     = xyz.clone()
                scaling_ref = scaling.clone()
                rot_ref     = rot.clone()
                count = 1
            elif color.shape[0] == color_sum.shape[0]:
                color_sum   += color
                opacity_sum += opacity
                count += 1
            # shape mismatch (opacity mask 차이) → skip

    if count == 0:
        raise RuntimeError("모든 카메라에서 baking 실패")

    color_avg   = (color_sum   / count).clamp(0.0, 1.0)
    opacity_avg = (opacity_sum / count).clamp(0.0, 1.0)

    # opacity threshold 로 pruning
    mask = (opacity_avg[:, 0] >= opacity_threshold)
    print(f"  [bake] {count} views 평균, Gaussians: {mask.shape[0]:,} → {mask.sum():,} (opacity≥{opacity_threshold})")

    xyz_np     = xyz_ref[mask].cpu().numpy().astype(np.float32)
    scaling_np = scaling_ref[mask].cpu().numpy().astype(np.float32)
    rot_np     = rot_ref[mask].cpu().numpy().astype(np.float32)
    color_np   = color_avg[mask].cpu().numpy()
    opacity_np = opacity_avg[mask, 0].cpu().numpy()

    return xyz_np, scaling_np, rot_np, color_np, opacity_np


# ── .splat 저장 ───────────────────────────────────────────────────────────────
def save_splat(xyz, scaling, rot, color, opacity, output_path: str):
    """
    .splat 바이너리: 32 bytes/Gaussian
      [0:12]  pos   float32 × 3
      [12:24] scale float32 × 3  (exp-activated)
      [24:28] RGBA  uint8   × 4
      [28:32] rot   uint8   × 4  (wxyz, [-1,1]→[0,255])
    """
    n = xyz.shape[0]
    out = np.zeros((n, 32), dtype=np.uint8)

    out[:, 0:12]  = xyz.view(np.uint8).reshape(n, 12)
    out[:, 12:24] = scaling.view(np.uint8).reshape(n, 12)

    color_u8   = (color * 255).clip(0, 255).astype(np.uint8)
    opacity_u8 = (opacity * 255).clip(0, 255).astype(np.uint8)
    out[:, 24] = color_u8[:, 0]   # R
    out[:, 25] = color_u8[:, 1]   # G
    out[:, 26] = color_u8[:, 2]   # B
    out[:, 27] = opacity_u8       # A

    rot_u8 = ((rot + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    out[:, 28] = rot_u8[:, 0]   # w
    out[:, 29] = rot_u8[:, 1]   # x
    out[:, 30] = rot_u8[:, 2]   # y
    out[:, 31] = rot_u8[:, 3]   # z

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out.tofile(output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  [save] {n:,} Gaussians → {output_path}  ({size_mb:.1f} MB)")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = ArgumentParser(description="Scaffold-GS → .splat (standalone)")
    parser.add_argument("-m", "--model_path", required=True,
                        help="학습된 모델 폴더 (cfg_args, point_cloud/, cameras.json 포함)")
    parser.add_argument("--iteration", type=int, default=-1,
                        help="사용할 iteration (-1 = 최신)")
    parser.add_argument("--num_bake_views", type=int, default=30,
                        help="색상 baking 에 사용할 카메라 수 (많을수록 안정적)")
    parser.add_argument("--opacity_threshold", type=float, default=0.005,
                        help="이 값 미만 opacity Gaussian 제거 (default: 0.005)")
    parser.add_argument("--output", type=str, default=None,
                        help="저장 경로 (default: <model_path>/point_cloud_120k.splat)")
    args = parser.parse_args()

    output = args.output
    if output is None:
        iter_str = f"{args.iteration}" if args.iteration != -1 else "latest"
        output = os.path.join(args.model_path, f"point_cloud_{iter_str}.splat")

    print(f"\n=== Scaffold-GS → .splat ===")
    print(f"  model : {args.model_path}")
    print(f"  iter  : {args.iteration}")
    print(f"  views : {args.num_bake_views}")
    print(f"  output: {output}\n")

    gaussians, loaded_iter = load_gaussians(args.model_path, args.iteration)
    cameras = load_bake_cameras(args.model_path, args.num_bake_views)
    xyz, scaling, rot, color, opacity = bake_gaussians(
        gaussians, cameras, opacity_threshold=args.opacity_threshold
    )
    save_splat(xyz, scaling, rot, color, opacity, output)
    print(f"\nDone → {output}")


if __name__ == "__main__":
    main()
