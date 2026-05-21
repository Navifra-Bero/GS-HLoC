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
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
from tqdm import tqdm
from plyfile import PlyData

# ── scaffold_gs 경로: 모델 backup/ 우선, 없으면 third_party/scaffold_gs ──────
_DEFAULT_SGS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "third_party", "scaffold_gs")
)


def _ensure_sgs_path(model_path: str = None) -> str:
    root = _DEFAULT_SGS_ROOT
    if model_path:
        backup = os.path.normpath(os.path.join(model_path, "backup"))
        if os.path.isfile(os.path.join(backup, "scene", "gaussian_model.py")):
            root = backup
    # 기존 경로 제거 후 올바른 경로 삽입
    sys.path[:] = [p for p in sys.path if os.path.normpath(p) != os.path.normpath(_DEFAULT_SGS_ROOT)]
    if root not in sys.path:
        sys.path.insert(0, root)
    print(f"  [SGS] 코드 경로: {root}")
    return root


# ── 모델 로드 (Scene 없이) ─────────────────────────────────────────────────────
class BakedGaussianModel:
    """Minimal Scaffold-GS model wrapper for export.

    The training GaussianModel hard-codes CUDA in several places.  For export we
    only need tensors from point_cloud.ply plus the TorchScript MLPs, so this
    wrapper keeps the converter usable on CPU-only machines too.
    """

    def __init__(self, cfg: dict, device: torch.device):
        def g(k, d): return cfg.get(k, d)

        self.feat_dim = g("feat_dim", 64)
        self.n_offsets = g("n_offsets", 15)
        self.use_feat_bank = g("use_feat_bank", False)
        self.appearance_dim = g("appearance_dim", 0)
        self.add_opacity_dist = g("add_opacity_dist", False)
        self.add_cov_dist = g("add_cov_dist", False)
        self.add_color_dist = g("add_color_dist", True)
        # Newer local Scaffold-GS backups may include the SplatHLoc feature
        # field. Static .splat baking only needs RGB/opacity/covariance, so keep
        # the feature path disabled while still exposing the attributes expected
        # by gaussian_renderer.generate_neural_gaussians().
        self.feat_field_dim = 0
        self.feat_gt_dim = g("feat_gt_dim", 256)
        self.device = device

        self._anchor = None
        self._offset = None
        self._anchor_feat = None
        self._scaling = None
        self._rotation = None
        self.mlp_opacity = None
        self.mlp_cov = None
        self.mlp_color = None
        self.mlp_feature_bank = None
        self.mlp_feature = None
        self.feat_decoder = None
        self.embedding_appearance = None

        self._build_mlp_modules()

    def _build_mlp_modules(self):
        opacity_dist_dim = 1 if self.add_opacity_dist else 0
        cov_dist_dim = 1 if self.add_cov_dist else 0
        color_dist_dim = 1 if self.add_color_dist else 0

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(4, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3),
                nn.Softmax(dim=1),
            ).to(self.device)
        if self.appearance_dim > 0:
            # chkpnt_best.pth 로드 시 실제 카메라 개수에 맞춰 다시 만든다.
            self.embedding_appearance = nn.Embedding(1, self.appearance_dim).to(self.device)

        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim + 3 + opacity_dist_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh(),
        ).to(self.device)

        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim + 3 + cov_dist_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7 * self.n_offsets),
        ).to(self.device)

        self.mlp_color = nn.Sequential(
            nn.Linear(self.feat_dim + 3 + color_dist_dim + self.appearance_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3 * self.n_offsets),
            nn.Sigmoid(),
        ).to(self.device)

    def load_pth_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not (isinstance(checkpoint, (tuple, list)) and len(checkpoint) == 2):
            raise ValueError(f"예상치 못한 checkpoint 형식: {path}")

        model_params, iteration = checkpoint
        if len(model_params) == 10:
            (anchor, offset, scaling, rotation, opacity,
             max_radii2D, offset_denom, _opt_dict, spatial_lr_scale, extras) = model_params
        else:
            (anchor, offset, scaling, rotation, opacity,
             max_radii2D, offset_denom, _opt_dict, spatial_lr_scale) = model_params
            extras = {}

        def tensor(x):
            return x.detach().to(self.device) if torch.is_tensor(x) else x

        self._anchor = tensor(anchor)
        self._offset = tensor(offset)
        self._scaling = tensor(scaling)
        self._rotation = tensor(rotation)
        self._opacity = tensor(opacity)
        if "_anchor_feat" in extras:
            self._anchor_feat = tensor(extras["_anchor_feat"])

        if "mlp_opacity" in extras:
            self.mlp_opacity.load_state_dict(extras["mlp_opacity"])
        if "mlp_cov" in extras:
            self.mlp_cov.load_state_dict(extras["mlp_cov"])
        if "mlp_color" in extras:
            self.mlp_color.load_state_dict(extras["mlp_color"])
        if "mlp_feature_bank" in extras and self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(extras["mlp_feature_bank"])
        # Deliberately ignore optional SplatHLoc feature extras. They are useful
        # for descriptor rendering, but not for RGB .splat export.
        if "embedding_appearance" in extras and self.appearance_dim > 0:
            emb_state = extras["embedding_appearance"]
            weight = emb_state.get("weight") if isinstance(emb_state, dict) else None
            if weight is not None:
                self.embedding_appearance = nn.Embedding(
                    weight.shape[0], weight.shape[1]
                ).to(self.device)
                self.embedding_appearance.load_state_dict(emb_state)

        if self._anchor_feat is None:
            raise ValueError(f"checkpoint 안에 _anchor_feat가 없습니다: {path}")
        return iteration

    @property
    def get_anchor(self):
        return self._anchor

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @staticmethod
    def rotation_activation(rot):
        return F.normalize(rot, dim=-1)

    def eval(self):
        for module in (
            self.mlp_opacity,
            self.mlp_cov,
            self.mlp_color,
            self.mlp_feature_bank,
            self.mlp_feature,
            self.feat_decoder,
            self.embedding_appearance,
        ):
            if module is not None:
                module.eval()

    def load_ply_sparse_gaussian(self, path: str):
        plydata = PlyData.read(path)
        vertex = plydata.elements[0]

        anchor = np.stack((
            np.asarray(vertex["x"]),
            np.asarray(vertex["y"]),
            np.asarray(vertex["z"]),
        ), axis=1).astype(np.float32)

        scale_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        scales = np.stack([np.asarray(vertex[name]) for name in scale_names], axis=1).astype(np.float32)

        rot_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("rot")],
            key=lambda x: int(x.split("_")[-1]),
        )
        rots = np.stack([np.asarray(vertex[name]) for name in rot_names], axis=1).astype(np.float32)

        feat_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("f_anchor_feat")],
            key=lambda x: int(x.split("_")[-1]),
        )
        anchor_feats = np.stack([np.asarray(vertex[name]) for name in feat_names], axis=1).astype(np.float32)

        offset_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("f_offset")],
            key=lambda x: int(x.split("_")[-1]),
        )
        offsets = np.stack([np.asarray(vertex[name]) for name in offset_names], axis=1).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1)).transpose(0, 2, 1).copy()

        self._anchor = torch.from_numpy(anchor).to(self.device)
        self._offset = torch.from_numpy(offsets).to(self.device)
        self._anchor_feat = torch.from_numpy(anchor_feats).to(self.device)
        self._scaling = torch.from_numpy(scales).to(self.device)
        self._rotation = torch.from_numpy(rots).to(self.device)

    def load_mlp_checkpoints(self, path: str):
        def load_jit(name):
            module = torch.jit.load(os.path.join(path, name), map_location=self.device)
            return module.to(self.device)

        self.mlp_opacity = load_jit("opacity_mlp.pt")
        self.mlp_cov = load_jit("cov_mlp.pt")
        self.mlp_color = load_jit("color_mlp.pt")
        if self.use_feat_bank:
            self.mlp_feature_bank = load_jit("feature_bank_mlp.pt")
        if self.appearance_dim > 0:
            self.embedding_appearance = load_jit("embedding_appearance.pt")


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
    _ensure_sgs_path(model_path)
    from scene import searchForMaxIteration

    cfg = _read_cfg_args(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gaussians = BakedGaussianModel(cfg, device)
    print(f"  [device] {device}")

    pth_path = os.path.join(model_path, "chkpnt_best.pth")
    if os.path.exists(pth_path):
        loaded_iter = gaussians.load_pth_checkpoint(pth_path)
        gaussians.eval()
        print(f"  [load] pth={pth_path}")
        print(f"  [load] iter={loaded_iter}, anchors: {gaussians.get_anchor.shape[0]:,}")
        return gaussians, loaded_iter

    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(model_path, "point_cloud"))

    ckpt_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"checkpoint 없음: {ckpt_dir}")

    gaussians.load_ply_sparse_gaussian(os.path.join(ckpt_dir, "point_cloud.ply"))
    gaussians.load_mlp_checkpoints(ckpt_dir)
    gaussians.eval()
    print(f"  [load] iter={iteration}, anchors: {gaussians.get_anchor.shape[0]:,}")
    return gaussians, iteration


# ── prefilter_voxel용 lightweight Camera ─────────────────────────────────────
class _LiteCamera:
    """prefilter_voxel / GaussianRasterizer 가 요구하는 최소 필드만 가진 lightweight camera."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _make_full_camera(cam_dict: dict, uid: int, device) -> _LiteCamera:
    """cameras.json entry → 풀 transform Camera (prefilter_voxel용).

    cameras.json 에는 rotation/position/fx/fy/width/height 가 들어있다.
    step2_scaffold_render._make_camera 와 같은 변환 규칙을 따른다.
    """
    from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2

    width  = int(cam_dict["width"])
    height = int(cam_dict["height"])
    fx     = float(cam_dict["fx"])
    fy     = float(cam_dict["fy"])

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.array(cam_dict["rotation"], dtype=np.float64)
    c2w[:3,  3] = np.array(cam_dict["position"], dtype=np.float64)
    w2c   = np.linalg.inv(c2w)
    R_w2c = w2c[:3, :3]
    T_w2c = w2c[:3,  3]

    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)
    world_view_transform = torch.tensor(
        getWorld2View2(R_w2c.T, T_w2c, np.array([0.0, 0.0, 0.0]), 1.0)
    ).transpose(0, 1).to(device)
    projection_matrix = getProjectionMatrix(
        znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy
    ).transpose(0, 1).to(device)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    return _LiteCamera(
        uid=uid,
        colmap_id=uid,
        image_name=str(cam_dict.get("img_name", f"{uid:05d}")),
        image_width=width,
        image_height=height,
        FoVx=fovx,
        FoVy=fovy,
        znear=0.01,
        zfar=100.0,
        world_view_transform=world_view_transform,
        projection_matrix=projection_matrix,
        full_proj_transform=full_proj_transform,
        camera_center=camera_center,
        R=R_w2c.T,
        T=T_w2c,
        camera_center_np=np.array(c2w[:3, 3], dtype=np.float32),
    )


# ── cameras.json → Camera 객체 목록 ───────────────────────────────────────────
def load_bake_cameras(model_path: str, num_bake_views: int, device=None):
    cam_json = os.path.join(model_path, "cameras.json")
    if not os.path.exists(cam_json):
        raise FileNotFoundError(f"cameras.json 없음: {cam_json}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_cams = json.load(open(cam_json))
    idxs = np.linspace(0, len(all_cams) - 1, min(num_bake_views, len(all_cams)), dtype=int)
    selected = [all_cams[i] for i in idxs]
    print(f"  [bake] {len(all_cams)} cameras → {len(selected)} 개 선택 (baking)")

    cameras = []
    n_full = n_pos_only = 0
    for uid, cam in enumerate(selected):
        try:
            cameras.append(_make_full_camera(cam, uid, device))
            n_full += 1
        except KeyError as e:
            print(f"    [warn] cam {uid}: {e} 누락 → position-only fallback (prefilter 불가)")
            cameras.append(_LiteCamera(
                uid=uid,
                camera_center_np=np.array(cam["position"], dtype=np.float32),
            ))
            n_pos_only += 1
    print(f"  [bake] full transforms: {n_full},  position-only: {n_pos_only}")
    return cameras


# ── Gaussian baking ───────────────────────────────────────────────────────────
def bake_gaussians(
    gaussians,
    cameras,
    opacity_threshold: float = 0.005,
):
    """Bake view-adaptive Scaffold-GS outputs into one static .splat map.

    Scaffold-GS renders only offsets whose MLP opacity is positive.  The
    opacity MLP ends with Tanh(), so positive values are already valid alpha
    values in (0, 1); applying sigmoid here would incorrectly keep negative
    offsets alive with alpha around 0.5.
    """
    anchor      = gaussians.get_anchor      # [N, 3]
    device      = anchor.device
    anchor_feat = gaussians._anchor_feat    # [N, C]
    offsets     = gaussians._offset         # [N, k, 3]
    grid_scale  = gaussians.get_scaling     # [N, 6]
    N, k        = anchor.shape[0], gaussians.n_offsets
    Nk          = N * k

    anchor_rep   = anchor.unsqueeze(1).expand(-1, k, -1).reshape(Nk, 3)
    gscale_rep   = grid_scale.unsqueeze(1).expand(-1, k, -1).reshape(Nk, 6)
    offsets_flat = offsets.reshape(Nk, 3)

    color_sum = torch.zeros(Nk, 3, device=device)
    opacity_sum = torch.zeros(Nk, 1, device=device)
    scaling_sum = torch.zeros(Nk, 3, device=device)
    rot_sum = torch.zeros(Nk, 4, device=device)
    weight_sum = torch.zeros(Nk, 1, device=device)
    xyz_ref = rot_ref = None
    count = 0

    with torch.no_grad():
        for cam in tqdm(cameras, desc="  Baking"):
            try:
                if not hasattr(cam, "camera_center"):
                    cam.camera_center = torch.from_numpy(cam.camera_center_np).to(device)
                ob_view = anchor - cam.camera_center          # [N, 3]
                ob_dist = ob_view.norm(dim=1, keepdim=True)
                ob_view = ob_view / (ob_dist + 1e-8)
                ob_dist = ob_dist / (ob_dist.detach().mean() + 1e-8)

                cat_wd  = torch.cat([anchor_feat, ob_view, ob_dist], dim=1)
                cat_nod = torch.cat([anchor_feat, ob_view],           dim=1)

                # color: mlp_color 마지막에 Sigmoid() 있음 -> [0,1]
                c_in  = cat_wd if gaussians.add_color_dist else cat_nod
                color = gaussians.get_color_mlp(c_in).reshape(Nk, 3)

                # opacity: original renderer uses Tanh output directly.
                o_in    = cat_wd if gaussians.add_opacity_dist else cat_nod
                opacity = gaussians.get_opacity_mlp(o_in).reshape(Nk, 1)

                s_in      = cat_wd if gaussians.add_cov_dist else cat_nod
                scale_rot = gaussians.get_cov_mlp(s_in).reshape(Nk, 7)
                scaling   = gscale_rep[:, 3:] * torch.sigmoid(scale_rot[:, :3])
                rot       = gaussians.rotation_activation(scale_rot[:, 3:7])

                if xyz_ref is None:
                    xyz_ref = anchor_rep + offsets_flat * gscale_rep[:, :3]
                    rot_ref = rot

                # Quaternion q and -q represent the same rotation.  Align signs
                # before averaging so rotations do not cancel each other out.
                sign = torch.where((rot * rot_ref).sum(dim=1, keepdim=True) < 0, -1.0, 1.0)
                rot = rot * sign

                weight = opacity.clamp(min=0.0)
                color_sum += color.clamp(0, 1) * weight
                opacity_sum += weight
                scaling_sum += scaling * weight
                rot_sum += rot * weight
                weight_sum += weight

                count += 1
            except Exception as e:
                print(f"    skip cam {cam.uid}: {e}")

    if count == 0:
        raise RuntimeError("모든 카메라에서 baking 실패")

    valid_weight = weight_sum.clamp(min=1e-8)
    color_avg = (color_sum / valid_weight).clamp(0.0, 1.0)
    scaling_avg = scaling_sum / valid_weight
    rot_avg = F.normalize(rot_sum / valid_weight, dim=-1)
    opacity_avg = (opacity_sum / count).clamp(0.0, 1.0)

    mask = (opacity_avg[:, 0] >= opacity_threshold)
    print(f"  [bake] {count} views, "
          f"Gaussians: {Nk:,} → {mask.sum().item():,} (opacity≥{opacity_threshold})")
    print(f"  scale range: min={scaling_avg[mask].min():.4f}  "
          f"max={scaling_avg[mask].max():.4f}  "
          f"mean={scaling_avg[mask].mean():.4f}")

    return (xyz_ref[mask].cpu().numpy().astype(np.float32),
            scaling_avg[mask].cpu().numpy().astype(np.float32),
            rot_avg[mask].cpu().numpy().astype(np.float32),
            color_avg[mask].cpu().numpy(),
            opacity_avg[mask, 0].cpu().numpy())


def bake_gaussians_renderer(
    gaussians,
    cameras,
    opacity_threshold: float = 0.005,
    model_path: str = None,
):
    """Bake only Gaussians that pass Scaffold-GS' renderer visibility path.

    This follows step2_scaffold_render.py more closely than the direct MLP bake:
    prefilter_voxel first culls anchors for the current camera, then
    generate_neural_gaussians applies the same opacity mask used by render().
    """
    if not torch.cuda.is_available():
        raise RuntimeError("--bake_mode renderer 는 CUDA 환경이 필요합니다.")

    _ensure_sgs_path(model_path)
    from gaussian_renderer import generate_neural_gaussians, prefilter_voxel

    device = gaussians.get_anchor.device
    if device.type != "cuda":
        raise RuntimeError("renderer bake는 CUDA에 로드된 GaussianModel이 필요합니다.")

    pipe = type("Pipeline", (), {
        "debug": False,
        "compute_cov3D_python": False,
    })()
    background = torch.zeros(3, dtype=torch.float32, device=device)

    N = gaussians.get_anchor.shape[0]
    k = gaussians.n_offsets
    Nk = N * k

    # Keep accumulators on CPU to reduce GPU memory pressure.
    color_sum = torch.zeros(Nk, 3, dtype=torch.float32)
    opacity_sum = torch.zeros(Nk, 1, dtype=torch.float32)
    scaling_sum = torch.zeros(Nk, 3, dtype=torch.float32)
    rot_sum = torch.zeros(Nk, 4, dtype=torch.float32)
    weight_sum = torch.zeros(Nk, 1, dtype=torch.float32)
    seen_count = torch.zeros(Nk, 1, dtype=torch.float32)
    xyz_ref = torch.empty(Nk, 3, dtype=torch.float32)
    xyz_seen = torch.zeros(Nk, dtype=torch.bool)

    count = 0
    used_total = 0

    with torch.no_grad():
        for cam in tqdm(cameras, desc="  Renderer baking"):
            if not hasattr(cam, "world_view_transform"):
                print(f"    skip cam {cam.uid}: full camera transform 없음")
                continue

            voxel_mask = prefilter_voxel(cam, gaussians, pipe, background)
            if voxel_mask.sum().item() == 0:
                count += 1
                continue

            generated = generate_neural_gaussians(
                cam, gaussians, visible_mask=voxel_mask, is_training=True
            )
            if len(generated) == 8:
                xyz, color, opacity, scaling, rot, neural_opacity, offset_mask, _feat = generated
            else:
                xyz, color, opacity, scaling, rot, neural_opacity, offset_mask = generated
            if xyz.shape[0] == 0:
                count += 1
                continue

            visible_anchor_idx = torch.nonzero(voxel_mask, as_tuple=False).flatten()
            full_idx = (
                visible_anchor_idx[:, None] * k
                + torch.arange(k, device=device)[None, :]
            ).reshape(-1)
            full_idx = full_idx[offset_mask].detach().cpu().long()

            xyz_cpu = xyz.detach().cpu().float()
            color_cpu = color.detach().cpu().float().clamp(0, 1)
            opacity_cpu = opacity.detach().cpu().float().clamp(min=0)
            scaling_cpu = scaling.detach().cpu().float()
            rot_cpu = rot.detach().cpu().float()

            # Quaternion q and -q are equivalent.  Align against the current
            # accumulated direction for stable averaging.
            prev_w = weight_sum[full_idx]
            has_prev = prev_w[:, 0] > 0
            if has_prev.any():
                prev_rot = rot_sum[full_idx[has_prev]] / prev_w[has_prev].clamp(min=1e-8)
                flip = (rot_cpu[has_prev] * prev_rot).sum(dim=1, keepdim=True) < 0
                rot_cpu[has_prev] = torch.where(flip, -rot_cpu[has_prev], rot_cpu[has_prev])

            weighted_color = color_cpu * opacity_cpu
            weighted_scaling = scaling_cpu * opacity_cpu
            weighted_rot = rot_cpu * opacity_cpu

            color_sum.index_add_(0, full_idx, weighted_color)
            opacity_sum.index_add_(0, full_idx, opacity_cpu)
            scaling_sum.index_add_(0, full_idx, weighted_scaling)
            rot_sum.index_add_(0, full_idx, weighted_rot)
            weight_sum.index_add_(0, full_idx, opacity_cpu)
            seen_count.index_add_(0, full_idx, torch.ones_like(opacity_cpu))

            unseen = ~xyz_seen[full_idx]
            if unseen.any():
                xyz_ref[full_idx[unseen]] = xyz_cpu[unseen]
                xyz_seen[full_idx[unseen]] = True

            used_total += int(xyz.shape[0])
            count += 1

            del voxel_mask, xyz, color, opacity, scaling, rot, neural_opacity, offset_mask
            torch.cuda.empty_cache()

    if count == 0:
        raise RuntimeError("renderer bake에 사용할 수 있는 카메라가 없습니다.")

    valid_weight = weight_sum.clamp(min=1e-8)
    color_avg = (color_sum / valid_weight).clamp(0.0, 1.0)
    scaling_avg = scaling_sum / valid_weight
    rot_avg = F.normalize(rot_sum / valid_weight, dim=-1)
    opacity_avg = (opacity_sum / seen_count.clamp(min=1.0)).clamp(0.0, 1.0)

    mask = xyz_seen & (opacity_avg[:, 0] >= opacity_threshold)
    if mask.sum().item() == 0:
        raise RuntimeError(
            f"renderer bake 결과가 0개입니다. opacity_threshold={opacity_threshold} 를 낮춰보세요."
        )

    print(f"  [renderer bake] {count} views, generated {used_total:,} view-Gaussians")
    print(f"  [renderer bake] unique Gaussians: {xyz_seen.sum().item():,} "
          f"→ {mask.sum().item():,} (opacity≥{opacity_threshold})")
    print(f"  scale range: min={scaling_avg[mask].min():.4f}  "
          f"max={scaling_avg[mask].max():.4f}  "
          f"mean={scaling_avg[mask].mean():.4f}")

    return (xyz_ref[mask].numpy().astype(np.float32),
            scaling_avg[mask].numpy().astype(np.float32),
            rot_avg[mask].numpy().astype(np.float32),
            color_avg[mask].numpy(),
            opacity_avg[mask, 0].numpy())


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
    parser.add_argument("--bake_mode", choices=("renderer", "direct"), default="renderer",
                        help="renderer=step2 렌더러 visibility 경로 사용, direct=MLP 직접 평균")
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
    print(f"  mode  : {args.bake_mode}")
    print(f"  output: {output}\n")

    gaussians, loaded_iter = load_gaussians(args.model_path, args.iteration)
    cameras = load_bake_cameras(
        args.model_path,
        args.num_bake_views,
        device=gaussians.get_anchor.device,
    )
    if args.bake_mode == "renderer":
        xyz, scaling, rot, color, opacity = bake_gaussians_renderer(
            gaussians,
            cameras,
            opacity_threshold=args.opacity_threshold,
            model_path=args.model_path,
        )
    else:
        xyz, scaling, rot, color, opacity = bake_gaussians(
            gaussians,
            cameras,
            opacity_threshold=args.opacity_threshold,
        )
    save_splat(xyz, scaling, rot, color, opacity, output)
    print(f"\nDone → {output}")


if __name__ == "__main__":
    main()
