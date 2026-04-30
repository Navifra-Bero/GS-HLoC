"""
STEP 2: Rendering — Pre-trained Scaffold-GS model

사전 학습된 Scaffold-GS 모델에서 step1 viewpoints 를 렌더링합니다.
train.py 의 render_set 과 동일한 방식(prefilter_voxel → render)을 사용합니다.

좌표계:
  step0_align 이 출력하는 T_align 은 GS 모델 좌표 → floor-aligned 좌표 변환:
    p_aligned  = T_align @ p_model
  렌더링에는 모델 좌표가 필요하므로 역변환:
    c2w_model  = inv(T_align) @ c2w_aligned

use_train_cameras=True 모드:
  Scene.getTrainCameras() 로 실제 학습 카메라를 그대로 사용합니다.
  즉, train.py / render.py 와 같은 카메라 로딩 경로를 탑니다.

Usage (main.py):
  python3 scripts/main.py \\
      --ply_map output/sgs_test/aligned_map.ply \\
      --output_dir output/sgs_test \\
      --step 2_render \\
      --render_mode scaffold_gs \\
      --sgs_model_path /path/to/scaffold_gs_result/TIMESTAMP \\
      --sgs_iteration 20000

  # train.py / render.py 방식 (실제 학습 카메라 사용):
  python3 scripts/main.py ... --sgs_use_train_cameras
"""
import os, sys, ast, pickle, time
import numpy as np
import torch
import torchvision
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import cv2
import open3d as o3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace

_DEFAULT_SGS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "scaffold_gs")
)
_ACTIVE_SGS_ROOT = None


# ── 내부 유틸 ──────────────────────────────────────────────────────────────────

class _LiteCamera:
    """렌더러가 요구하는 최소 필드만 가진 lightweight camera."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def _model_backup_root(model_path: str) -> str:
    return os.path.normpath(os.path.join(model_path, "backup"))


def _is_valid_sgs_root(root: str) -> bool:
    required = (
        os.path.join(root, "gaussian_renderer", "__init__.py"),
        os.path.join(root, "scene", "__init__.py"),
        os.path.join(root, "arguments", "__init__.py"),
    )
    return all(os.path.exists(path) for path in required)


def _resolve_sgs_root(model_path: str = None) -> str:
    if model_path:
        backup_root = _model_backup_root(model_path)
        if _is_valid_sgs_root(backup_root):
            return backup_root
    return _DEFAULT_SGS_ROOT


def _module_under_root(module, root: str) -> bool:
    mod_file = getattr(module, "__file__", None)
    if not mod_file:
        return False
    mod_file = os.path.normpath(os.path.abspath(mod_file))
    root = os.path.normpath(os.path.abspath(root))
    return mod_file == root or mod_file.startswith(root + os.sep)


def _ensure_sgs_path(model_path: str = None) -> str:
    global _ACTIVE_SGS_ROOT

    root = os.path.normpath(os.path.abspath(_resolve_sgs_root(model_path)))
    used_backup = bool(model_path) and root == os.path.normpath(os.path.abspath(_model_backup_root(model_path)))
    if _ACTIVE_SGS_ROOT == root:
        if root not in sys.path:
            sys.path.insert(0, root)
        return root

    roots_to_clear = {os.path.normpath(os.path.abspath(_DEFAULT_SGS_ROOT))}
    if model_path:
        roots_to_clear.add(os.path.normpath(os.path.abspath(_model_backup_root(model_path))))
    if _ACTIVE_SGS_ROOT is not None:
        roots_to_clear.add(os.path.normpath(os.path.abspath(_ACTIVE_SGS_ROOT)))

    for name, module in list(sys.modules.items()):
        if any(_module_under_root(module, candidate) for candidate in roots_to_clear):
            sys.modules.pop(name, None)

    sys.path[:] = [p for p in sys.path
                   if os.path.normpath(os.path.abspath(p)) not in roots_to_clear]
    sys.path.insert(0, root)
    _ACTIVE_SGS_ROOT = root

    print(f"  [SGS] 코드 경로: {root}")
    if model_path and not used_backup:
        print("  [SGS] model backup code 없음 → third_party/scaffold_gs fallback 사용")
    return root


def _read_cfg_args(model_path: str) -> dict:
    """model_path/cfg_args 에서 학습 파라미터를 dict 로 반환."""
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        text = f.read().strip()
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


def _build_dataset_args(model_path: str) -> Namespace:
    cfg = _read_cfg_args(model_path)
    if not cfg:
        raise FileNotFoundError(f"cfg_args 없음: {model_path}")
    cfg["model_path"] = model_path
    cfg.setdefault("images", "images")
    cfg.setdefault("resolution", 1)
    cfg.setdefault("white_background", False)
    cfg.setdefault("data_device", "cuda")
    cfg.setdefault("eval", False)
    cfg.setdefault("lod", 0)
    return Namespace(**cfg)


def _make_gaussian_model(dataset):
    from scene import GaussianModel

    return GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        dataset.appearance_dim,
        dataset.ratio,
        dataset.add_opacity_dist,
        dataset.add_cov_dist,
        dataset.add_color_dist,
    )


def _read_best_ckpt_meta(model_path: str) -> dict | None:
    """chkpnt_best.json → {"iteration": N, "psnr": P}. 없으면 None."""
    import json
    meta_path = os.path.join(model_path, "chkpnt_best.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_gaussians_from_pth(model_path: str, pth_path: str, dataset=None):
    """Render-only load from capture()-format .pth checkpoint (optimizer 초기화 생략).

    Returns
    -------
    (gaussians, iteration, appearance_camera_count)
    """
    if dataset is None:
        dataset = _build_dataset_args(model_path)

    gaussians = _make_gaussian_model(dataset)
    print(f"  [SGS] .pth 로드: {pth_path}")
    checkpoint = torch.load(pth_path, map_location="cuda", weights_only=False)
    if not (isinstance(checkpoint, (tuple, list)) and len(checkpoint) == 2):
        raise ValueError(f"예상치 못한 checkpoint 형식 (tuple(model_params, iter) 기대): {pth_path}")

    model_params, iteration = checkpoint
    if len(model_params) == 10:
        (anchor, offset, scaling, rotation, opacity,
         max_radii2D, offset_denom, _opt_dict, spatial_lr_scale, extras) = model_params
    else:
        (anchor, offset, scaling, rotation, opacity,
         max_radii2D, offset_denom, _opt_dict, spatial_lr_scale) = model_params
        extras = {}

    gaussians._anchor        = anchor
    gaussians._offset        = offset
    gaussians._scaling       = scaling
    gaussians._rotation      = rotation
    gaussians._opacity       = opacity
    gaussians.max_radii2D    = max_radii2D
    gaussians.spatial_lr_scale = float(spatial_lr_scale)
    if offset_denom is not None:
        gaussians.offset_denom = offset_denom
    if "_anchor_feat" in extras:
        gaussians._anchor_feat = extras["_anchor_feat"]
    if "mlp_opacity" in extras:
        gaussians.mlp_opacity.load_state_dict(extras["mlp_opacity"])
    if "mlp_cov" in extras:
        gaussians.mlp_cov.load_state_dict(extras["mlp_cov"])
    if "mlp_color" in extras:
        gaussians.mlp_color.load_state_dict(extras["mlp_color"])
    if "mlp_feature_bank" in extras and getattr(gaussians, "use_feat_bank", False):
        gaussians.mlp_feature_bank.load_state_dict(extras["mlp_feature_bank"])

    appearance_camera_count = None
    if "embedding_appearance" in extras and getattr(gaussians, "appearance_dim", 0) > 0:
        n_cams = extras["embedding_appearance"]["embedding.weight"].shape[0]
        gaussians.set_appearance(n_cams)
        gaussians.embedding_appearance.load_state_dict(extras["embedding_appearance"])
        appearance_camera_count = n_cams

    gaussians.eval()
    print(f"  [SGS] .pth 로드 완료 (iter={int(iteration)}, "
          f"anchors={gaussians.get_anchor.shape[0]:,}"
          + (f", appearance={appearance_camera_count}" if appearance_camera_count else "") + ")")
    return gaussians, int(iteration), appearance_camera_count


def _resolve_iteration_and_pth(model_path: str, iteration: int) -> tuple[int, str | None]:
    """iteration과 사용할 .pth 경로를 결정한다.

    우선순위:
      1. point_cloud/iteration_N/ 존재 → (N, None)
      2. chkpnt_best.json의 iteration이 일치하고 chkpnt_best.pth 존재 → (N, pth_path)
      3. chkpntN.pth 존재 → (N, pth_path)
      4. iteration=-1이면 chkpnt_best.json → (best_iter, pth_path)
      5. 실패시 FileNotFoundError
    """
    import json
    from utils.system_utils import searchForMaxIteration

    best_meta = _read_best_ckpt_meta(model_path)

    # iteration 결정
    if iteration == -1:
        if best_meta:
            iteration = int(best_meta["iteration"])
            print(f"  [SGS] chkpnt_best.json → 최고 iteration={iteration} "
                  f"(PSNR={best_meta.get('psnr', 'N/A')})")
        else:
            iteration = searchForMaxIteration(os.path.join(model_path, "point_cloud"))

    # point_cloud 디렉토리 우선
    ckpt_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    if os.path.exists(ckpt_dir):
        return iteration, None

    # .pth 폴백
    candidates = [
        os.path.join(model_path, f"chkpnt{iteration}.pth"),
    ]
    if best_meta and int(best_meta.get("iteration", -1)) == iteration:
        candidates.insert(0, os.path.join(model_path, "chkpnt_best.pth"))
    for pth in candidates:
        if os.path.exists(pth):
            print(f"  [SGS] point_cloud/iteration_{iteration} 없음 → {pth} 사용")
            return iteration, pth

    raise FileNotFoundError(
        f"checkpoint 없음: {ckpt_dir}\n"
        f"  .pth 후보도 없음: {candidates}"
    )


def _load_gaussians_only(model_path: str, iteration: int, dataset=None, pth_path: str = None):
    if dataset is None:
        dataset = _build_dataset_args(model_path)

    if pth_path:
        gaussians, loaded_iter, _ = _load_gaussians_from_pth(model_path, pth_path, dataset)
        return gaussians, loaded_iter

    iteration, pth_path = _resolve_iteration_and_pth(model_path, iteration)
    if pth_path:
        gaussians, loaded_iter, _ = _load_gaussians_from_pth(model_path, pth_path, dataset)
        return gaussians, loaded_iter

    gaussians = _make_gaussian_model(dataset)
    ckpt_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    gaussians.load_ply_sparse_gaussian(os.path.join(ckpt_dir, "point_cloud.ply"))
    gaussians.load_mlp_checkpoints(ckpt_dir)
    gaussians.eval()
    return gaussians, iteration


def _get_appearance_camera_count(gaussians) -> int | None:
    if getattr(gaussians, "appearance_dim", 0) <= 0:
        return None
    appearance = getattr(gaussians, "embedding_appearance", None)
    if appearance is None:
        return None
    state = appearance.state_dict()
    weight = state.get("embedding.weight")
    if weight is None:
        return None
    return int(weight.shape[0])


def _load_camera_records_from_json(model_path: str, appearance_camera_count: int | None = None):
    import json

    cam_json = os.path.join(model_path, "cameras.json")
    if not os.path.exists(cam_json):
        raise FileNotFoundError(f"cameras.json 없음: {cam_json}")

    with open(cam_json, encoding="utf-8") as f:
        cams = json.load(f)

    records = []
    total_count = len(cams)
    test_count = None
    if appearance_camera_count is not None:
        test_count = total_count - appearance_camera_count
        if test_count < 0:
            raise ValueError(
                f"appearance camera count({appearance_camera_count}) > cameras.json count({total_count})"
            )
    for i, cam in enumerate(cams):
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = np.array(cam["rotation"], dtype=np.float64)
        c2w[:3,  3] = np.array(cam["position"], dtype=np.float64)
        uid = int(cam.get("id", i))
        render_uid = uid
        split = "all"
        if test_count is not None:
            if i < test_count:
                render_uid = i
                split = "test"
            else:
                render_uid = i - test_count
                split = "train"
        records.append({
            "uid": uid,
            "render_uid": int(render_uid),
            "split": split,
            "img_name": str(cam.get("img_name", f"{uid:05d}")),
            "width": int(cam["width"]),
            "height": int(cam["height"]),
            "fx": float(cam["fx"]),
            "fy": float(cam["fy"]),
            "cx": 0.5 * float(cam["width"]),
            "cy": 0.5 * float(cam["height"]),
            "c2w_model": c2w,
        })

    print(f"  [SGS] cameras.json fallback 로드: {len(records)} camera records")
    if test_count is not None:
        print(f"  [SGS] appearance uid 매핑: test={test_count}, train={appearance_camera_count}")
    return records


def _load_render_context(model_path: str, iteration: int,
                         pth_path: str = None) -> dict:
    """render.py / train.py 와 동일한 Scene 경로로 렌더링 컨텍스트를 로드.

    pth_path 가 주어지면 Scene 로드를 건너뛰고 .pth → cameras.json 경로를 사용한다.
    """
    _ensure_sgs_path(model_path)
    from scene import Scene
    from arguments import PipelineParams

    dataset = _build_dataset_args(model_path)
    scene = None
    train_cameras = None
    train_camera_records = None
    appearance_camera_count = None

    # ── .pth 직접 로드 경로 ────────────────────────────────────────────────────
    if pth_path:
        gaussians, loaded_iter, appearance_camera_count = _load_gaussians_from_pth(
            model_path, pth_path, dataset)
        train_camera_records = _load_camera_records_from_json(
            model_path, appearance_camera_count=appearance_camera_count)
    else:
        # iteration 결정 + point_cloud 없을 때 자동 .pth 폴백
        resolved_iter, auto_pth = _resolve_iteration_and_pth(model_path, iteration)
        if auto_pth:
            gaussians, loaded_iter, appearance_camera_count = _load_gaussians_from_pth(
                model_path, auto_pth, dataset)
            train_camera_records = _load_camera_records_from_json(
                model_path, appearance_camera_count=appearance_camera_count)
        else:
            try:
                gaussians = _make_gaussian_model(dataset)
                scene = Scene(dataset, gaussians, load_iteration=resolved_iter, shuffle=False)
                gaussians.eval()
                loaded_iter = scene.loaded_iter
                train_cameras = scene.getTrainCameras()
                if getattr(gaussians, "appearance_dim", 0) > 0:
                    appearance_camera_count = len(train_cameras)
            except AssertionError as e:
                msg = str(e)
                if "Could not recognize scene type!" not in msg:
                    raise
                print("  [SGS] Scene 로드 실패: source_path 구조를 찾지 못했습니다.")
                print("  [SGS] fallback: cameras.json + checkpoint 직접 로드")
                gaussians, loaded_iter = _load_gaussians_only(
                    model_path, resolved_iter, dataset)
                appearance_camera_count = _get_appearance_camera_count(gaussians)
                train_camera_records = _load_camera_records_from_json(
                    model_path, appearance_camera_count=appearance_camera_count)

    parser = ArgumentParser()
    pipeline = PipelineParams(parser).extract(parser.parse_args([]))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ckpt_src = pth_path or os.path.join(model_path, "point_cloud", f"iteration_{loaded_iter}")
    print(f"  [SGS] 모델 로드 (iter={loaded_iter}), "
          f"anchors: {gaussians.get_anchor.shape[0]:,}  ← {ckpt_src}")

    return {
        "dataset": dataset,
        "scene": scene,
        "train_cameras": train_cameras,
        "train_camera_records": train_camera_records,
        "gaussians": gaussians,
        "pipeline": pipeline,
        "background": background,
        "loaded_iter": loaded_iter,
        "appearance_camera_count": appearance_camera_count,
    }


def _make_camera(c2w: np.ndarray,
                 fx: float, fy: float, cx: float, cy: float,
                 width: int, height: int,
                 uid: int, image_name: str):
    """4×4 c2w (모델 좌표계) → Scaffold-GS Camera 객체."""
    from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2

    w2c   = np.linalg.inv(c2w)
    R_w2c = w2c[:3, :3]
    T_w2c = w2c[:3,  3]

    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)
    world_view_transform = torch.tensor(
        getWorld2View2(R_w2c.T, T_w2c, np.array([0.0, 0.0, 0.0]), 1.0)
    ).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(
        znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy
    ).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    return _LiteCamera(
        uid=uid,
        colmap_id=uid,
        image_name=image_name,
        image_width=int(width),
        image_height=int(height),
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
        cx=float(cx),
        cy=float(cy),
        _c2w_model=np.array(c2w, dtype=np.float64),
    )


def _camera_to_c2w_model(cam) -> np.ndarray:
    if hasattr(cam, "_c2w_model"):
        return np.array(cam._c2w_model, dtype=np.float64)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = np.array(cam.R, dtype=np.float64).T
    w2c[:3,  3] = np.array(cam.T, dtype=np.float64)
    return np.linalg.inv(w2c)


def _scene_cameras_as_viewpoints(cameras, T_align: np.ndarray) -> list:
    """
    Scene 가 만든 실제 Camera 객체를 pipeline 좌표계 viewpoint 로 변환.
    렌더링은 Scene camera 를 그대로 쓰고, 반환값 저장/비교용으로 aligned pose 도 유지한다.
    """
    viewpoints = []
    for i, cam in enumerate(cameras):
        c2w_model = _camera_to_c2w_model(cam)
        c2w_aligned = T_align @ c2w_model
        viewpoints.append({
            "id":       i,
            "pose":     c2w_aligned,
            "floor":    0,
            "yaw":      0.0,
            "img_name": getattr(cam, "image_name", f"{i:05d}"),
            "_c2w_model": c2w_model,
            "_camera":  cam,
        })
    return viewpoints


def _camera_records_as_viewpoints(records, T_align: np.ndarray) -> list:
    viewpoints = []
    for i, rec in enumerate(records):
        c2w_model = np.array(rec["c2w_model"], dtype=np.float64)
        c2w_aligned = T_align @ c2w_model
        viewpoints.append({
            "id": i,
            "pose": c2w_aligned,
            "floor": 0,
            "yaw": 0.0,
            "img_name": rec["img_name"],
            "_c2w_model": c2w_model,
            "_uid": rec["render_uid"],
            "_fx": rec["fx"],
            "_fy": rec["fy"],
            "_cx": rec["cx"],
            "_cy": rec["cy"],
            "_width": rec["width"],
            "_height": rec["height"],
        })
    return viewpoints


def _filter_camera_records(records, split: str | None = None):
    if split is None:
        return list(records)
    return [rec for rec in records if rec.get("split") == split]


def _build_reference_camera_table(cameras) -> list:
    refs = []
    for cam in cameras:
        if isinstance(cam, dict):
            c2w = np.array(cam["c2w_model"], dtype=np.float64)
            uid = int(cam.get("render_uid", cam["uid"]))
        else:
            c2w = _camera_to_c2w_model(cam)
            uid = int(cam.uid)
        refs.append({
            "uid": uid,
            "center": c2w[:3, 3],
            "forward": c2w[:3, 2] / (np.linalg.norm(c2w[:3, 2]) + 1e-8),
        })
    return refs


def _select_reference_uid(c2w_model: np.ndarray, refs: list) -> int:
    if not refs:
        return 0

    center = c2w_model[:3, 3]
    forward = c2w_model[:3, 2]
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    best_uid = refs[0]["uid"]
    best_score = None
    for ref in refs:
        dist = np.linalg.norm(center - ref["center"])
        align = 1.0 - np.clip(np.dot(forward, ref["forward"]), -1.0, 1.0)
        score = dist + 0.25 * align
        if best_score is None or score < best_score:
            best_score = score
            best_uid = ref["uid"]
    return best_uid


# ── 메인 함수 ──────────────────────────────────────────────────────────────────

def _make_o3d_renderer(ply_path: str, config: dict):
    """aligned_map.ply 를 로드해 Open3D OffscreenRenderer 를 반환."""
    cam_cfg = config.get("camera", {})
    w = cam_cfg.get("width",  1920)
    h = cam_cfg.get("height", 1200)

    pcd = o3d.io.read_point_cloud(ply_path)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.point_size = config.get("rendering", {}).get("point_size", 1.0)
    mat.shader     = "defaultUnlit"

    ren = o3d.visualization.rendering.OffscreenRenderer(w, h)
    ren.scene.set_background([0.0, 0.0, 0.0, 1.0])
    ren.scene.add_geometry("map", pcd, mat)
    print(f"  [PLY] loaded {len(np.asarray(pcd.points)):,} pts → O3D renderer ({w}×{h})")
    return ren


def _render_ply_o3d(ren, pose: np.ndarray, config: dict,
                    with_rgb: bool = True,
                    with_depth: bool = False):
    """O3D OffscreenRenderer 로 aligned-space pose 에서 PLY RGB / depth 렌더."""
    cam_cfg = config.get("camera", {})
    fx  = cam_cfg.get("fx",  1039.045981)
    fy  = cam_cfg.get("fy",  1041.496942)
    cx  = cam_cfg.get("cx",   937.044077)
    cy  = cam_cfg.get("cy",   560.826738)
    w   = cam_cfg.get("width",       1920)
    h   = cam_cfg.get("height",      1200)
    intr     = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    extrinsic = np.linalg.inv(pose)
    ren.setup_camera(intr, extrinsic)
    rgb = None
    depth = None
    if with_rgb:
        rgb = np.asarray(ren.render_to_image()).copy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb.astype(np.float32), 0, 1) * 255).astype(np.uint8)
    if with_depth:
        depth = np.asarray(ren.render_to_depth_image(z_in_view_space=True)).astype(np.float32).copy()
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0.0] = 0.0
    return rgb, depth


def step2_scaffold_render(ply_path: str,
                          viewpoints: list,
                          config: dict,
                          output_dir: str,
                          step0_data: dict = None,
                          sgs_model_path: str = None,
                          sgs_iteration: int = -1,
                          sgs_ckpt_path: str = None,
                          use_train_cameras: bool = False,
                          save_ply_compare: bool = True,
                          save_ply_depth: bool = True,
                          **kwargs):
    """
    사전 학습된 Scaffold-GS 모델로 step1 viewpoints 렌더링.
    save_ply_compare=True 이면 같은 viewpoint 에서 PLY 렌더도 함께 저장.

    Parameters
    ----------
    ply_path          : aligned_map.ply 경로
    viewpoints        : step1 viewpoints (floor-aligned 좌표, pose = 4×4 c2w)
    config            : pipeline config
    output_dir        : 결과 저장 폴더
    step0_data        : step0 출력 dict (T_align 포함)
    sgs_model_path    : scaffold_gs 모델 폴더 (필수)
    sgs_iteration     : 사용할 iteration (-1 = 자동: chkpnt_best.json → max iteration 순)
    sgs_ckpt_path     : .pth 체크포인트 직접 지정 (예: chkpnt_best.pth 경로).
                        지정 시 sgs_iteration 무시. None이면 자동 결정.
    use_train_cameras : True → Scene.getTrainCameras() 사용
    save_ply_compare  : True → rendered/ply_rgb/ 에 PLY 렌더도 함께 저장
    save_ply_depth    : True → GS depth 부재 시 aligned PLY depth 를 rendered/depth/ 에 저장

    Returns
    -------
    rendered : list[dict]  각 viewpoint 의 id, pose, rgb_path, ply_rgb_path, depth_path
    """
    if sgs_model_path is None:
        raise ValueError("sgs_model_path 를 지정해주세요 (--sgs_model_path).")

    _ensure_sgs_path(sgs_model_path)
    from gaussian_renderer import render as gs_render, prefilter_voxel

    # ── T_align ────────────────────────────────────────────────────────────────
    if step0_data is not None and "T_align" in step0_data:
        T_align = np.array(step0_data["T_align"], dtype=np.float64)
    else:
        T_align = np.eye(4, dtype=np.float64)
    T_inv = np.linalg.inv(T_align)

    # ── 모델/Scene 로드 (render.py render_sets 와 동일한 경로) ──────────────────
    ctx = _load_render_context(sgs_model_path, sgs_iteration, pth_path=sgs_ckpt_path)
    gaussians = ctx["gaussians"]
    pipeline = ctx["pipeline"]
    background = ctx["background"]
    loaded_iter = ctx["loaded_iter"]
    train_cameras = ctx["train_cameras"]
    train_camera_records = ctx.get("train_camera_records")
    appearance_camera_count = ctx.get("appearance_camera_count")
    ref_source = train_cameras if train_cameras is not None else train_camera_records
    appearance_ref_source = train_cameras
    if appearance_ref_source is None:
        appearance_ref_source = _filter_camera_records(train_camera_records or [], split="train")
    if not appearance_ref_source:
        appearance_ref_source = ref_source
    appearance_refs = _build_reference_camera_table(appearance_ref_source or [])
    torch.cuda.empty_cache()

    # ── 뷰포인트 결정 ─────────────────────────────────────────────────────────
    if use_train_cameras:
        if train_cameras is not None:
            print(f"  [SGS] train.py 방식: Scene.getTrainCameras() 사용 ({len(train_cameras)} cams)")
            viewpoints = _scene_cameras_as_viewpoints(train_cameras, T_align)
        else:
            train_only_records = _filter_camera_records(train_camera_records, split="train")
            if not train_only_records:
                train_only_records = train_camera_records
            print(f"  [SGS] train.py fallback: cameras.json train pose 사용 ({len(train_only_records)} cams)")
            viewpoints = _camera_records_as_viewpoints(train_only_records, T_align)
    else:
        if not viewpoints:
            raise ValueError("viewpoints 가 비어 있습니다. "
                             "step1 을 먼저 실행하거나 --sgs_use_train_cameras 를 사용하세요.")
        print(f"  [SGS] step1 viewpoints 사용: {len(viewpoints)} 개")

    # ── 카메라 파라미터 ────────────────────────────────────────────────────────
    cam_cfg = config.get("camera", {})
    fx  = cam_cfg.get("fx",     1039.045981)
    fy  = cam_cfg.get("fy",     1041.496942)
    cx  = cam_cfg.get("cx",      937.044077)
    cy  = cam_cfg.get("cy",      560.826738)
    w   = cam_cfg.get("width",        1920)
    h   = cam_cfg.get("height",       1200)

    # ── 출력 폴더 ─────────────────────────────────────────────────────────────
    renders_dir = os.path.join(output_dir, "rendered", "rgb")
    depth_dir   = os.path.join(output_dir, "rendered", "depth")
    ply_dir     = os.path.join(output_dir, "rendered", "ply_rgb")
    os.makedirs(renders_dir, exist_ok=True)
    os.makedirs(depth_dir,   exist_ok=True)

    # ── PLY renderer (aligned space) ──────────────────────────────────────────
    o3d_ren = None
    need_ply_renderer = save_ply_compare or save_ply_depth
    if need_ply_renderer:
        aligned_ply = ply_path
        if step0_data is not None and "aligned_ply_path" in step0_data:
            aligned_ply = step0_data["aligned_ply_path"]
        if not os.path.exists(aligned_ply):
            aligned_ply = os.path.join(output_dir, "aligned_map.ply")
        if os.path.exists(aligned_ply):
            os.makedirs(ply_dir, exist_ok=True)
            o3d_ren = _make_o3d_renderer(aligned_ply, config)
        else:
            print(f"  [PLY] aligned_map.ply 없음 — PLY 렌더 스킵: {aligned_ply}")

    # ── PLY renderer 를 SGS 렌더와 분리 (GPU OOM 방지) ──────────────────────────
    # SGS 모델과 O3D EGL renderer 를 동시에 GPU 에 올리면 OOM 발생.
    # Pass-1 에서는 O3D 를 로드하지 않고 SGS RGB 만 렌더링한 뒤,
    # Pass-2 에서 GPU 캐시를 비우고 O3D 를 로드해 PLY depth/compare 를 채운다.
    o3d_ren_deferred = o3d_ren
    need_ply_pass = o3d_ren_deferred is not None and (save_ply_compare or save_ply_depth)
    if need_ply_pass:
        o3d_ren = None   # pass-1 에선 O3D 사용 안 함

    # ── Pass-1: SGS 렌더링 (RGB + native depth) ───────────────────────────────
    torch.cuda.empty_cache()
    rendered = []
    t_list   = []
    n        = len(viewpoints)
    depth_supported = None
    print(f"  [SGS] 렌더링 시작: {n} viewpoints  (PLY compare: {save_ply_compare and o3d_ren_deferred is not None}, "
          f"PLY depth fallback: {save_ply_depth and o3d_ren_deferred is not None})")

    for idx, vp in enumerate(viewpoints):
        if "_camera" in vp:
            cam = vp["_camera"]
            c2w_model = np.array(vp["_c2w_model"], dtype=np.float64)
        else:
            c2w_model = T_inv @ np.array(vp["pose"], dtype=np.float64)
            vp_has_uid = "_uid" in vp
            render_uid = int(vp.get("_uid", idx))
            if gaussians.appearance_dim > 0 and not vp_has_uid:
                render_uid = _select_reference_uid(c2w_model, appearance_refs)
            if appearance_camera_count is not None and not (0 <= render_uid < appearance_camera_count):
                raise ValueError(
                    f"appearance uid out of range: {render_uid} "
                    f"(valid: 0..{appearance_camera_count - 1})"
                )
            img_name = vp.get("img_name", f"{vp['id']:05d}")
            render_fx = float(vp.get("_fx", fx))
            render_fy = float(vp.get("_fy", fy))
            render_cx = float(vp.get("_cx", cx))
            render_cy = float(vp.get("_cy", cy))
            render_w = int(vp.get("_width", w))
            render_h = int(vp.get("_height", h))
            cam = _make_camera(c2w_model, render_fx, render_fy, render_cx, render_cy, render_w, render_h,
                               uid=render_uid, image_name=str(img_name))

        torch.cuda.synchronize(); t0 = time.time()

        with torch.no_grad():
            voxel_mask = prefilter_voxel(cam, gaussians, pipeline, background)
            pkg        = gs_render(cam, gaussians, pipeline, background,
                                   visible_mask=voxel_mask)
        if depth_supported is None:
            depth_supported = "depth" in pkg
            if not depth_supported:
                print("  [SGS] depth 미저장: 현재 renderer가 depth tensor를 반환하지 않습니다.")
                if save_ply_depth and o3d_ren_deferred is not None:
                    print("  [SGS] depth fallback: aligned PLY metric depth 를 저장합니다. (pass-2)")
                elif save_ply_depth:
                    print("  [SGS] depth fallback 불가: aligned PLY renderer를 준비하지 못했습니다.")
                else:
                    print("  [SGS] depth fallback 비활성화: --sgs_no_ply_depth")

        torch.cuda.synchronize()
        t_list.append(time.time() - t0)

        # RGB 저장
        rendering = torch.clamp(pkg["render"], 0.0, 1.0)
        name      = f"{vp['id']:05d}"
        rgb_path  = os.path.join(renders_dir, f"{name}.png")
        torchvision.utils.save_image(rendering, rgb_path)

        # Depth 저장 (native gaussian depth 가 있을 때만)
        dep_path = ""
        depth_source = ""
        if "depth" in pkg:
            depth    = pkg["depth"][0].cpu().numpy()
            dep_path = os.path.join(depth_dir, f"{name}.npy")
            np.save(dep_path, depth.astype(np.float32))
            depth_source = "gaussian"

        rendered.append({
            "id":           vp["id"],
            "pose":         vp["pose"],     # floor-aligned 좌표 유지
            "floor":        vp.get("floor", 0),
            "yaw":          vp.get("yaw", 0.0),
            "rgb_path":     rgb_path,
            "ply_rgb_path": "",
            "depth_path":   dep_path,
            "depth_source": depth_source,
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == n:
            recent = t_list[-100:]
            fps    = len(recent) / sum(recent)
            print(f"    {idx+1}/{n}  ({fps:.1f} fps)")

        del cam, voxel_mask, pkg, rendering
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

    avg_fps = len(t_list) / sum(t_list) if t_list else 0
    print(f"  [SGS] pass-1 완료: {len(rendered)}/{n}  avg {avg_fps:.2f} fps")

    # ── Pass-2: PLY depth / compare (O3D) ────────────────────────────────────
    # if need_ply_pass:
    #     torch.cuda.empty_cache()
    #     o3d_ren = o3d_ren_deferred
    #     need_any_ply_rgb   = save_ply_compare
    #     need_any_ply_depth = save_ply_depth and not depth_supported
    #     print(f"  [SGS] pass-2 PLY 렌더 시작 (rgb={need_any_ply_rgb}, depth={need_any_ply_depth})")
    #     for idx, (vp, rec) in enumerate(zip(viewpoints, rendered)):
    #         need_ply_rgb   = need_any_ply_rgb
    #         need_ply_depth = need_any_ply_depth and not rec["depth_path"]
    #         if not need_ply_rgb and not need_ply_depth:
    #             continue
    #         ply_rgb, ply_depth = _render_ply_o3d(
    #             o3d_ren,
    #             np.array(vp["pose"], dtype=np.float64),
    #             config,
    #             with_rgb=need_ply_rgb,
    #             with_depth=need_ply_depth,
    #         )
    #         name = f"{vp['id']:05d}"
    #         if need_ply_rgb and ply_rgb is not None:
    #             os.makedirs(ply_dir, exist_ok=True)
    #             ply_rgb_path = os.path.join(ply_dir, f"{name}.png")
    #             cv2.imwrite(ply_rgb_path, cv2.cvtColor(ply_rgb, cv2.COLOR_RGB2BGR))
    #             rec["ply_rgb_path"] = ply_rgb_path
    #         if need_ply_depth and ply_depth is not None:
    #             dep_path = os.path.join(depth_dir, f"{name}.npy")
    #             np.save(dep_path, ply_depth.astype(np.float32))
    #             rec["depth_path"]   = dep_path
    #             rec["depth_source"] = "ply"
    #         if (idx + 1) % 100 == 0 or (idx + 1) == n:
    #             print(f"    PLY {idx+1}/{n}")
    #     print(f"  [SGS] pass-2 완료")

    # ── 샘플 시각화 ───────────────────────────────────────────────────────────
    if rendered:
        ns       = min(8, len(rendered))
        idx_list = np.linspace(0, len(rendered) - 1, ns, dtype=int)

        nrows = 2   # SGS / Depth
        fig, axes = plt.subplots(nrows, ns, figsize=(4 * ns, 4 * nrows))
        if ns == 1:
            axes = axes.reshape(nrows, 1)

        row_labels = ["SGS", "Depth"]

        for c, ii in enumerate(idx_list):
            r = rendered[ii]
            sgs = cv2.cvtColor(cv2.imread(r["rgb_path"]), cv2.COLOR_BGR2RGB)
            axes[0, c].imshow(sgs)
            axes[0, c].set_title(f"#{r['id']}", fontsize=9)
            axes[0, c].axis("off")

            dep_path = r.get("depth_path", "")
            if dep_path and os.path.exists(dep_path):
                depth = np.load(dep_path).astype(np.float32) if dep_path.endswith(".npy") \
                        else cv2.imread(dep_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                valid = depth[np.isfinite(depth) & (depth > 0)]
                if valid.size > 0:
                    vmin, vmax = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
                else:
                    vmin, vmax = 0.0, 1.0
                axes[1, c].imshow(depth, cmap="turbo", vmin=vmin, vmax=vmax)
            else:
                axes[1, c].text(0.5, 0.5, "no depth", ha="center", va="center",
                                transform=axes[1, c].transAxes, fontsize=10, color="gray")
            axes[1, c].axis("off")

        for row, lbl in enumerate(row_labels):
            axes[row, 0].set_ylabel(lbl, fontsize=11)

        mode_str = "train-cams" if use_train_cameras else "step1-vps"
        fig.suptitle(f"Step2-SGS [{mode_str}]: {len(rendered)} renders (iter={loaded_iter})",
                     fontsize=13)
        fig.tight_layout()
        vis_path = os.path.join(output_dir, "step2_sgs_rendered.png")
        fig.savefig(vis_path, dpi=150)
        plt.close()
        print(f"  [SGS] 시각화: {vis_path}")

    # ── step2_data.pkl ────────────────────────────────────────────────────────
    pkl_path = os.path.join(output_dir, "step2_data.pkl")
    pickle.dump(rendered, open(pkl_path, "wb"))
    print(f"  [SGS] step2_data.pkl → {pkl_path}  ({len(rendered)} entries)")

    return rendered
