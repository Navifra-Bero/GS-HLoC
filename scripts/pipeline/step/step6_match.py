import os, sys, pickle, glob
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .step3_global_desc import (_extract_megaloc_desc,
                                _extract_mixvpr_desc, _load_mixvpr_model)


# ── EfficientLoFTR helpers ─────────────────────────────────────────────────

def _load_eloftr(config, dev):
    """EfficientLoFTR 로드 헬퍼."""
    import torch

    eloftr_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "EfficientLoFTR"))
    if eloftr_root not in sys.path:
        sys.path.insert(0, eloftr_root)

    from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

    fc = config["features"]
    ckpt_path = fc.get("eloftr_ckpt") or os.path.join(
        eloftr_root, "weights", "ELoFTR", "weights", "eloftr_outdoor.ckpt")
    use_opt = fc.get("eloftr_opt", False)

    _cfg = (opt_default_cfg if use_opt else full_default_cfg).copy()
    matcher = LoFTR(config=_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    matcher.load_state_dict(state["state_dict"])
    matcher = reparameter(matcher).eval().to(dev)
    return matcher, ckpt_path, use_opt


def _make_gray_tensor_fn(dev, max_dim):
    """to_gray_tensor 클로저 생성."""
    import torch

    def to_gray_tensor(rgb_img):
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        orig_h, orig_w = gray.shape
        h, w = orig_h, orig_w
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            nw = int(w * s); nh = int(h * s)
            gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
            h, w = gray.shape
        scale_x = orig_w / w
        scale_y = orig_h / h
        ph = ((h + 31) // 32) * 32
        pw = ((w + 31) // 32) * 32
        if ph != h or pw != w:
            padded = np.zeros((ph, pw), dtype=np.float32)
            padded[:h, :w] = gray
            gray = padded
        return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(dev), scale_x, scale_y

    return to_gray_tensor


# ── vismatch helper ────────────────────────────────────────────────────────

def _load_vismatch(matcher_name, dev):
    """vismatch 모델 로드.
    third_party/vismatch 가 있으면 그걸 우선 사용, 없으면 pip 패키지를 시도.
    """
    vismatch_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "vismatch"))
    if os.path.isdir(vismatch_root) and vismatch_root not in sys.path:
        sys.path.insert(0, vismatch_root)

    try:
        from vismatch import get_matcher  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            f"vismatch를 찾을 수 없습니다. 아래 중 하나를 실행하세요:\n"
            f"  pip install vismatch\n"
            f"  git clone https://github.com/gmberton/vismatch third_party/vismatch\n"
            f"원래 에러: {e}"
        )

    device_str = str(dev)  # "cpu", "cuda", "cuda:0" 등 그대로 사용
    matcher = get_matcher(matcher_name, device=device_str)
    return matcher


def _run_vismatch(matcher, query_rgb, ref_rgb, max_dim):
    """
    vismatch로 매칭 수행.
    Returns:
        mkpts0: (M, 2) 원본 좌표계 기준 query keypoints
        mkpts1: (M, 2) 원본 좌표계 기준 ref keypoints
        confs : (M,) uniform 1.0 (vismatch는 per-match confidence 미제공)
        n_good: matched pair 수
        score : re-ranking 점수 (= n_good, confidence 없으므로)

    NOTE: vismatch load_image(resize=int) resizes to a square (int × int).
    We must pass a (H, W) tuple with aspect-preserving dimensions so that the
    scale-back to original coords is correct.
    """
    def _aspect_resize_dims(orig_h, orig_w, max_d):
        """longest-edge resize → new (H, W). max_d=None → 원본 크기 유지."""
        if max_d is None or max(orig_h, orig_w) <= max_d:
            return orig_h, orig_w
        s = max_d / max(orig_h, orig_w)
        return int(orig_h * s), int(orig_w * s)

    q_h, q_w = query_rgb.shape[:2]
    r_h, r_w = ref_rgb.shape[:2]
    q_new_h, q_new_w = _aspect_resize_dims(q_h, q_w, max_dim)
    r_new_h, r_new_w = _aspect_resize_dims(r_h, r_w, max_dim)

    import io
    from PIL import Image as _PILImage

    def _to_bytes(rgb_arr):
        buf = io.BytesIO()
        _PILImage.fromarray(rgb_arr).save(buf, format="PNG")
        buf.seek(0)
        return buf

    # resize 인수를 (H, W) 튜플로 전달 → 비율 유지 리사이즈
    img0 = matcher.load_image(_to_bytes(query_rgb), resize=(q_new_h, q_new_w))
    img1 = matcher.load_image(_to_bytes(ref_rgb),   resize=(r_new_h, r_new_w))

    result = matcher(img0, img1)

    mkpts0 = result.get("matched_kpts0", np.zeros((0, 2)))
    mkpts1 = result.get("matched_kpts1", np.zeros((0, 2)))

    if len(mkpts0) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.array([]), 0, 0.0

    # resize 좌표 → 원본 좌표 (aspect-preserving scale)
    q_sx = q_w / q_new_w if q_new_w > 0 else 1.0
    q_sy = q_h / q_new_h if q_new_h > 0 else 1.0
    r_sx = r_w / r_new_w if r_new_w > 0 else 1.0
    r_sy = r_h / r_new_h if r_new_h > 0 else 1.0

    mkpts0 = mkpts0 * np.array([q_sx, q_sy], dtype=np.float32)
    mkpts1 = mkpts1 * np.array([r_sx, r_sy], dtype=np.float32)

    n_good = len(mkpts0)
    confs  = np.ones(n_good, dtype=np.float32)
    score  = float(n_good)

    return mkpts0, mkpts1, confs, n_good, score


# ── LoFTR (kornia) direct helper ──────────────────────────────────────────

def _load_loftr(config, dev):
    """kornia LoFTR 직접 로드. indoor/outdoor 선택 가능."""
    from kornia.feature import LoFTR

    fc = config["features"]
    variant = fc.get("loftr_variant", "indoor").strip().lower()  # "indoor" | "outdoor"
    matcher = LoFTR(pretrained=variant).eval().to(dev)
    return matcher, variant


def _make_loftr_gray_tensor_fn(dev, max_dim):
    """LoFTR용 grayscale tensor 전처리. EfficientLoFTR과 동일한 방식."""
    import torch

    def to_gray_tensor(rgb_img):
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        orig_h, orig_w = gray.shape
        h, w = orig_h, orig_w
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            nw, nh = int(w * s), int(h * s)
            gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
            h, w = gray.shape
        scale_x = orig_w / w
        scale_y = orig_h / h
        ph = ((h + 7) // 8) * 8
        pw = ((w + 7) // 8) * 8
        if ph != h or pw != w:
            padded = np.zeros((ph, pw), dtype=np.float32)
            padded[:h, :w] = gray
            gray = padded
        return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(dev), scale_x, scale_y

    return to_gray_tensor


def _run_loftr(matcher, query_rgb, ref_rgb, conf_thresh, gray_tensor_fn):
    """kornia LoFTR 단일 쌍 매칭."""
    import torch

    q_tensor, q_sx, q_sy = gray_tensor_fn(query_rgb)
    r_tensor, r_sx, r_sy = gray_tensor_fn(ref_rgb)

    with torch.no_grad():
        result = matcher({"image0": q_tensor, "image1": r_tensor})

    mkpts0 = result["keypoints0"].cpu().numpy()
    mkpts1 = result["keypoints1"].cpu().numpy()
    confs  = result["confidence"].cpu().numpy()

    mask   = confs >= conf_thresh
    mkpts0 = mkpts0[mask] * np.array([q_sx, q_sy], dtype=np.float32)
    mkpts1 = mkpts1[mask] * np.array([r_sx, r_sy], dtype=np.float32)
    confs  = confs[mask]

    n_good = len(mkpts0)
    score  = float(confs.sum())
    return mkpts0, mkpts1, confs, n_good, score


# ── JamMa helper ──────────────────────────────────────────────────────────

def _lower_config(yacs_cfg):
    """yacs CfgNode → plain dict (모든 키를 lowercase). pytorch_lightning 의존 회피."""
    from yacs.config import CfgNode as CN
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): _lower_config(v) for k, v in yacs_cfg.items()}


def _load_jamma(config, dev):
    """JamMa 모델 로드."""
    import torch

    jamma_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "JamMa"))
    if jamma_root not in sys.path:
        sys.path.insert(0, jamma_root)

    # PassThroughProfiler 대용 (pytorch_lightning 의존 회피)
    import contextlib

    class _NoOpProfiler:
        @contextlib.contextmanager
        def profile(self, action_name):
            yield action_name

    # profiler 모듈을 먼저 패치하여 pytorch_lightning import 방지
    import importlib, types
    _profiler_mod = types.ModuleType("src.utils.profiler")
    _profiler_mod.PassThroughProfiler = _NoOpProfiler
    sys.modules["src.utils.profiler"] = _profiler_mod

    from src.config.default import get_cfg_defaults
    from src.jamma.jamma import JamMa
    from src.jamma.backbone import CovNextV2_nano

    # build config with inference mode
    jamma_cfg = get_cfg_defaults()
    jamma_cfg.JAMMA.MATCH_COARSE.INFERENCE = True
    jamma_cfg.JAMMA.FINE.INFERENCE = True
    jamma_cfg.JAMMA.MATCH_COARSE.USE_SM = True

    fc = config["features"]
    coarse_thr = float(fc.get("jamma_coarse_thr", 0.2))
    fine_thr = float(fc.get("jamma_fine_thr", 0.1))
    jamma_cfg.JAMMA.MATCH_COARSE.THR = coarse_thr
    jamma_cfg.JAMMA.FINE.THR = fine_thr

    _config = _lower_config(jamma_cfg)

    backbone = CovNextV2_nano()
    matcher = JamMa(config=_config['jamma'])

    ckpt_path = fc.get("jamma_ckpt", "")
    if not ckpt_path:
        ckpt_path = os.path.join(jamma_root, "weights", "jamma.ckpt")
    # 상대 경로면 프로젝트 루트 기준으로 resolve
    if not os.path.isabs(ckpt_path):
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        ckpt_path = os.path.join(project_root, ckpt_path)

    if os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    else:
        # official weight from hub
        state_dict = torch.hub.load_state_dict_from_url(
            'https://github.com/leoluxxx/JamMa/releases/download/v0.1/jamma.ckpt',
            file_name='jamma.ckpt')['state_dict']

    # PL_JamMa wraps backbone + matcher; split the state dict
    backbone_sd = {}
    matcher_sd = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            backbone_sd[k[len("backbone."):]] = v
        elif k.startswith("matcher."):
            matcher_sd[k[len("matcher."):]] = v

    backbone.load_state_dict(backbone_sd, strict=True)
    matcher.load_state_dict(matcher_sd, strict=True)

    backbone = backbone.eval().to(dev)
    matcher = matcher.eval().to(dev)

    return backbone, matcher, ckpt_path


def _make_jamma_preprocess_fn(dev, max_dim):
    """JamMa용 이미지 전처리 클로저. RGB → resize → square pad → ImageNet normalize.

    JamMa backbone은 query/ref를 batch dim으로 concat해서 처리하므로 두 이미지가
    반드시 동일한 (H, W)를 가져야 함. 따라서 longest-edge를 max_dim으로 맞추고
    bottom-right zero padding으로 max_dim × max_dim square로 만든다.
    """
    import torch
    import torchvision.transforms as T

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    # max_dim을 8의 배수로 정렬 (backbone이 1/8 downsample)
    tgt = ((max_dim + 7) // 8) * 8

    def preprocess(rgb_img):
        """Returns: (tensor, scale_x, scale_y, valid_w, valid_h).
        valid_w/valid_h = padded square 내 실제 이미지 영역 크기 (border filter용).
        """
        orig_h, orig_w = rgb_img.shape[:2]
        # longest-edge = tgt 되도록 resize (up/down 모두)
        s = tgt / max(orig_h, orig_w)
        nw, nh = max(1, int(round(orig_w * s))), max(1, int(round(orig_h * s)))
        nw, nh = min(nw, tgt), min(nh, tgt)
        rgb_img = cv2.resize(rgb_img, (nw, nh), interpolation=cv2.INTER_AREA)

        scale_x = orig_w / nw
        scale_y = orig_h / nh

        # bottom-right zero padding → tgt × tgt square
        padded = np.zeros((tgt, tgt, 3), dtype=np.uint8)
        padded[:nh, :nw, :] = rgb_img

        img_t = torch.from_numpy(padded.astype(np.float32) / 255.0).permute(2, 0, 1)
        img_t = normalize(img_t).unsqueeze(0).to(dev)  # (1,3,tgt,tgt)
        return img_t, scale_x, scale_y, nw, nh

    return preprocess


def _run_jamma(backbone, matcher, query_rgb, ref_rgb, preprocess_fn):
    """
    JamMa 단일 쌍 매칭.
    Returns:
        mkpts0, mkpts1, confs, n_good, score
    """
    import torch

    q_tensor, q_sx, q_sy, q_vw, q_vh = preprocess_fn(query_rgb)
    r_tensor, r_sx, r_sy, r_vw, r_vh = preprocess_fn(ref_rgb)

    data = {
        "imagec_0": q_tensor,
        "imagec_1": r_tensor,
    }

    with torch.no_grad():
        backbone(data)
        matcher(data, mode='test')

    mkpts0 = data["mkpts0_f"].cpu().numpy()
    mkpts1 = data["mkpts1_f"].cpu().numpy()
    confs = data["mconf_f"].cpu().numpy()

    if len(mkpts0) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.array([]), 0, 0.0

    # padded square 내 실제 이미지 영역 밖 매칭 제거 (bottom-right padding)
    valid = ((mkpts0[:, 0] < q_vw) & (mkpts0[:, 1] < q_vh) &
             (mkpts1[:, 0] < r_vw) & (mkpts1[:, 1] < r_vh))
    mkpts0 = mkpts0[valid]
    mkpts1 = mkpts1[valid]
    confs = confs[valid]

    if len(mkpts0) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.array([]), 0, 0.0

    # scale back to original image coords
    mkpts0 = mkpts0 * np.array([q_sx, q_sy], dtype=np.float32)
    mkpts1 = mkpts1 * np.array([r_sx, r_sy], dtype=np.float32)

    n_good = len(mkpts0)
    score = float(confs.sum())
    return mkpts0, mkpts1, confs, n_good, score


# ── RoMa direct helper ────────────────────────────────────────────────────

def _load_roma(config, dev):
    """RoMa 직접 로드 (vismatch 우회)."""
    try:
        from romatch import roma_outdoor, tiny_roma_v1_outdoor
    except ImportError:
        raise ImportError("romatch가 없습니다: pip install romatch")

    fc         = config["features"]
    roma_size  = fc.get("roma_size", "full")   # "full" | "tiny"
    if roma_size == "tiny":
        model = tiny_roma_v1_outdoor(device=dev)
        label = "RoMa-tiny"
    else:
        model = roma_outdoor(device=dev)
        label = "RoMa"
    return model, label


def _match_roma(model, query_rgb, ref_rgb, num_samples=5000, certainty_thresh=0.02):
    """
    RoMa 직접 inference.
    match()는 파일 경로를 받으므로 임시 파일로 저장 후 전달.
    Returns:
        mkpts0: (M, 2) query keypoints (original pixel coords)
        mkpts1: (M, 2) ref   keypoints (original pixel coords)
        confs : (M,) certainty scores
        n_good: 매칭 수
        score : re-ranking 점수 (certainty sum)
    """
    import tempfile

    def _save_temp(rgb_arr):
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(path, cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR))
        return path

    H_q, W_q = query_rgb.shape[:2]
    H_r, W_r = ref_rgb.shape[:2]

    path_q = _save_temp(query_rgb)
    path_r = _save_temp(ref_rgb)
    try:
        warp, certainty = model.match(path_q, path_r)
        matches, cert   = model.sample(warp, certainty, num=num_samples)
    finally:
        os.unlink(path_q)
        os.unlink(path_r)

    # certainty threshold 필터링
    if hasattr(cert, "cpu"):
        cert_np = cert.cpu().numpy()
    else:
        cert_np = np.asarray(cert)

    mask = cert_np > certainty_thresh
    if mask.sum() == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.array([]), 0, 0.0

    matches_f = matches[mask]
    confs     = cert_np[mask]

    kptsA, kptsB = model.to_pixel_coordinates(matches_f, H_q, W_q, H_r, W_r)
    mkpts0 = kptsA.cpu().numpy() if hasattr(kptsA, "cpu") else np.asarray(kptsA)
    mkpts1 = kptsB.cpu().numpy() if hasattr(kptsB, "cpu") else np.asarray(kptsB)

    n_good = len(mkpts0)
    score  = float(confs.sum())
    return mkpts0, mkpts1, confs, n_good, score


# ── Unified matcher loader ─────────────────────────────────────────────────

def _load_matcher(config, dev):
    """
    config['features']['matcher_name'] 에 따라 적절한 matcher를 로드.
    Returns:
        matcher_obj  : 모델 객체 (jamma의 경우 (backbone, matcher) 튜플)
        matcher_type : "eloftr" | "roma" | "jamma" | "vismatch"
        info_str     : 로그용 문자열
    """
    fc           = config["features"]
    matcher_name = fc.get("matcher_name", "eloftr").strip().lower()

    if matcher_name == "eloftr":
        matcher, ckpt_path, use_opt = _load_eloftr(config, dev)
        info = (f"EfficientLoFTR  ckpt={os.path.basename(ckpt_path)}  "
                f"mode={'opt' if use_opt else 'full'}  device={dev}")
        return matcher, "eloftr", info
    elif matcher_name == "loftr":
        matcher, variant = _load_loftr(config, dev)
        info = f"LoFTR [{variant}]  device={dev}"
        return matcher, "loftr", info
    elif matcher_name == "jamma":
        backbone, matcher, ckpt_path = _load_jamma(config, dev)
        info = f"JamMa  ckpt={os.path.basename(ckpt_path)}  device={dev}"
        return (backbone, matcher), "jamma", info
    elif matcher_name in ("roma", "roma-tiny", "tiny-roma"):
        config["features"]["roma_size"] = "tiny" if "tiny" in matcher_name else "full"
        matcher, label = _load_roma(config, dev)
        info = f"{label}  device={dev}"
        return matcher, "roma", info
    else:
        matcher = _load_vismatch(matcher_name, dev)
        info = f"vismatch [{matcher_name}]  device={dev}"
        return matcher, "vismatch", info


# ── Unified single-pair matching ───────────────────────────────────────────

def _run_single_match(matcher_type, matcher, query_rgb, ref_rgb,
                      conf_thresh, vismatch_max_dim,
                      gray_tensor_fn=None, jamma_preprocess_fn=None,
                      loftr_gray_fn=None):
    """
    단일 이미지 쌍 매칭. 백엔드를 통일된 인터페이스로 추상화.
    Returns:
        mkpts0  : (M, 2) query keypoints (original image coords)
        mkpts1  : (M, 2) ref   keypoints (original image coords)
        confs   : (M,) confidence (vismatch는 uniform 1.0)
        n_good  : 유효 매칭 수
        score   : re-ranking 점수
    """
    if matcher_type == "eloftr":
        q_tensor, q_sx, q_sy = gray_tensor_fn(query_rgb)
        r_tensor, r_sx, r_sy = gray_tensor_fn(ref_rgb)
        import torch
        batch = {"image0": q_tensor, "image1": r_tensor}
        with torch.no_grad():
            matcher(batch)
        mkpts0 = batch["mkpts0_f"].cpu().numpy() * np.array([q_sx, q_sy], dtype=np.float32)
        mkpts1 = batch["mkpts1_f"].cpu().numpy() * np.array([r_sx, r_sy], dtype=np.float32)
        confs  = batch["mconf"].cpu().numpy()
        mask   = confs >= conf_thresh
        n_good = int(mask.sum())
        score  = float(confs[mask].sum())
        return mkpts0[mask], mkpts1[mask], confs[mask], n_good, score
    elif matcher_type == "loftr":
        return _run_loftr(matcher, query_rgb, ref_rgb, conf_thresh, loftr_gray_fn)
    elif matcher_type == "jamma":
        backbone, jamma_matcher = matcher
        return _run_jamma(backbone, jamma_matcher, query_rgb, ref_rgb, jamma_preprocess_fn)
    elif matcher_type == "roma":
        return _match_roma(matcher, query_rgb, ref_rgb)
    else:
        return _run_vismatch(matcher, query_rgb, ref_rgb, vismatch_max_dim)


# ── step6_match ────────────────────────────────────────────────────────────

def step6_match(step5_data, config, output_dir, save_images=True):
    """EfficientLoFTR or vismatch: query vs top-K candidates → best match."""
    import torch

    print("\n" + "="*60 + "\nSTEP 6: Feature matching\n" + "="*60)

    fc               = config["features"]
    dev              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_thresh      = float(fc.get("match_conf_thresh", 0.2))
    eloftr_max_dim   = int(fc.get("eloftr_max_dim", 840))
    vismatch_max_dim = int(fc.get("vismatch_max_dim", 840))

    query_rgb  = step5_data["query_rgb"]
    candidates = step5_data["candidates"]
    cos_sims   = step5_data.get("cos_sims", [1.0] * len(candidates))

    matcher, matcher_type, info_str = _load_matcher(config, dev)
    print(f"  Matcher: {info_str}")

    if dev.type == "cuda":
        torch.cuda.empty_cache()

    gray_tensor_fn = (_make_gray_tensor_fn(dev, eloftr_max_dim)
                      if matcher_type == "eloftr" else None)
    loftr_max_dim = int(fc.get("loftr_max_dim", 840))
    loftr_gray_fn = (_make_loftr_gray_tensor_fn(dev, loftr_max_dim)
                     if matcher_type == "loftr" else None)
    jamma_max_dim = int(fc.get("jamma_max_dim", 840))
    jamma_preprocess_fn = (_make_jamma_preprocess_fn(dev, jamma_max_dim)
                           if matcher_type == "jamma" else None)

    best_mkpts_q  = np.zeros((0, 2))
    best_mkpts_r  = np.zeros((0, 2))
    best_confs    = np.array([])
    best_cand     = candidates[0]
    best_ref_rgb  = cv2.cvtColor(cv2.imread(candidates[0]["rgb_path"]), cv2.COLOR_BGR2RGB)
    best_score    = -1.0
    best_n        = 0
    all_match_counts = []
    all_scores       = []

    for rank, cand in enumerate(candidates):
        ref_rgb = cv2.cvtColor(cv2.imread(cand["rgb_path"]), cv2.COLOR_BGR2RGB)

        try:
            mkpts0, mkpts1, confs, n_good, score = _run_single_match(
                matcher_type, matcher, query_rgb, ref_rgb,
                conf_thresh, vismatch_max_dim,
                gray_tensor_fn=gray_tensor_fn,
                jamma_preprocess_fn=jamma_preprocess_fn,
                loftr_gray_fn=loftr_gray_fn,
            )
        except Exception as e:
            print(f"  Rank{rank+1} matching failed: {e}")
            all_match_counts.append(0)
            all_scores.append(0.0)
            continue

        mean_conf = float(confs.mean()) if len(confs) > 0 else 0.0
        ret_sim   = cos_sims[rank] if rank < len(cos_sims) else 0.0
        all_match_counts.append(n_good)
        all_scores.append(score)

        print(f"  Rank{rank+1} #{cand['id']:>4}: {n_good:>3} good  "
              f"score={score:>7.2f}  mean_conf={mean_conf:.3f}  ret_sim={ret_sim:.4f}")

        if score > best_score:
            best_score   = score
            best_n       = n_good
            best_mkpts_q = mkpts0
            best_mkpts_r = mkpts1
            best_confs   = confs
            best_cand    = cand
            best_ref_rgb = ref_rgb

    print(f"\n  Best: #{best_cand['id']}  matches={best_n}  score={best_score:.2f}")

    matcher_name = fc.get("matcher_name", "eloftr")

    if save_images:
        # ── 시각화 ───────────────────────────────────────────────────────
        h1, w1 = query_rgb.shape[:2]; h2, w2 = best_ref_rgb.shape[:2]
        th = max(h1, h2)
        sq, sr = 1.0, 1.0
        if h1 != th:
            sq = th / h1; query_rgb    = cv2.resize(query_rgb,    (int(w1*sq), th))
        if h2 != th:
            sr = th / h2; best_ref_rgb = cv2.resize(best_ref_rgb, (int(w2*sr), th))
        h1, w1 = query_rgb.shape[:2]; h2, w2 = best_ref_rgb.shape[:2]
        canvas = np.concatenate([query_rgb, best_ref_rgb], axis=1)
        mkpts_q_v = best_mkpts_q * sq
        mkpts_r_v = best_mkpts_r * sr

        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        ax.imshow(canvas)
        if len(mkpts_q_v) > 0:
            norm_c = best_confs / (best_confs.max() + 1e-8)
            cmap_v = plt.cm.RdYlGn(norm_c)
            step_v = max(1, len(mkpts_q_v) // 200)
            for i in range(0, len(mkpts_q_v), step_v):
                ax.plot([mkpts_q_v[i,0], mkpts_r_v[i,0]+w1],
                        [mkpts_q_v[i,1], mkpts_r_v[i,1]],
                        c=cmap_v[i], alpha=0.5, linewidth=0.8)
            ax.scatter(mkpts_q_v[:,0], mkpts_q_v[:,1], c="cyan",   s=8, zorder=3)
            ax.scatter(mkpts_r_v[:,0]+w1, mkpts_r_v[:,1], c="yellow", s=8, zorder=3)

        match_summary = "  |  ".join(
            f"R{r+1}:{n}" for r, n in enumerate(all_match_counts))
        ax.set_title(f"Step 6 [{matcher_name}]: best=#{best_cand['id']} ({best_n} matches)\n"
                     f"{match_summary}")
        ax.axis("off"); fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "step6_matching.png"), dpi=150); plt.close()
        print(f"  Saved: step6_matching.png")

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
        colors_bar = ["green" if c["id"] == best_cand["id"] else "steelblue"
                      for c in candidates[:len(all_match_counts)]]
        xlabels = [f"R{r+1}\n#{c['id']}" for r, c in enumerate(candidates[:len(all_match_counts)])]
        axes2[0].bar(xlabels, all_match_counts, color=colors_bar)
        axes2[0].set_ylabel("Match count"); axes2[0].set_title("Match count per candidate")
        axes2[1].bar(xlabels, all_scores, color=colors_bar)
        axes2[1].set_ylabel("Re-ranking score")
        axes2[1].set_title("Re-ranking score per candidate\n(eloftr: conf_sum / vismatch: count)")
        fig2.suptitle(f"Step 6 [{matcher_name}]: Re-ranking → best=#{best_cand['id']}  "
                      f"score={best_score:.2f}", fontsize=11)
        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, "step6_match_counts.png"), dpi=150); plt.close()

    data = {
        "matched_q_kps":    best_mkpts_q,
        "matched_r_kps":    best_mkpts_r,
        "confidences":      best_confs,
        "best_cand":        best_cand,
        "query_rgb":        query_rgb,
        "ref_rgb":          best_ref_rgb,
        "all_match_counts": all_match_counts,
        "all_scores":       all_scores,
        "candidates":       candidates,
        "matcher_name":     matcher_name,
    }
    if save_images:
        pickle.dump(data, open(os.path.join(output_dir, "step6_data.pkl"), "wb"))
    return data


# ── step6a_match_viz ───────────────────────────────────────────────────────

def step6a_match_viz(query_dir, db, config, output_dir):
    """배치 retrieval + matching 시각화."""
    import torch

    print("\n" + "="*60 + "\nSTEP 6a: Batch retrieval + match viz\n" + "="*60)
    print(f"  Query dir : {query_dir}")

    fc               = config["features"]
    dev              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_thresh      = float(fc.get("match_conf_thresh", 0.2))
    eloftr_max_dim   = int(fc.get("eloftr_max_dim", 840))
    vismatch_max_dim = int(fc.get("vismatch_max_dim", 840))
    onl              = config.get("online", {})
    top_k            = onl.get("top_k", 5)

    query_files = sorted(
        glob.glob(os.path.join(query_dir, "*.jpg")) +
        glob.glob(os.path.join(query_dir, "*.png")) +
        glob.glob(os.path.join(query_dir, "*.jpeg"))
    )
    if not query_files:
        print(f"  ERROR: {query_dir} 에 이미지 없음"); return
    print(f"  {len(query_files)} query images found")

    global_desc_method = db.get("global_desc_method", "anyloc")
    retr_model = None
    if global_desc_method == "megaloc":
        print("  Loading MegaLoc …")
        retr_model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        retr_model.eval().to(dev)

    matcher, matcher_type, info_str = _load_matcher(config, dev)
    print(f"  Matcher: {info_str}")

    if dev.type == "cuda":
        torch.cuda.empty_cache()

    gray_tensor_fn = (_make_gray_tensor_fn(dev, eloftr_max_dim)
                      if matcher_type == "eloftr" else None)
    loftr_max_dim = int(fc.get("loftr_max_dim", 840))
    loftr_gray_fn = (_make_loftr_gray_tensor_fn(dev, loftr_max_dim)
                     if matcher_type == "loftr" else None)
    jamma_max_dim = int(fc.get("jamma_max_dim", 840))
    jamma_preprocess_fn = (_make_jamma_preprocess_fn(dev, jamma_max_dim)
                           if matcher_type == "jamma" else None)

    save_dir = os.path.join(output_dir, "step6a_results")
    os.makedirs(save_dir, exist_ok=True)

    matcher_name = fc.get("matcher_name", "eloftr")

    for qi, qpath in enumerate(query_files):
        stem = os.path.splitext(os.path.basename(qpath))[0]
        print(f"\n  [{qi+1}/{len(query_files)}] {stem}")

        query_rgb = cv2.cvtColor(cv2.imread(qpath), cv2.COLOR_BGR2RGB)

        # ── Retrieval ──────────────────────────────────────────────────
        if global_desc_method == "megaloc":
            q_gd = _extract_megaloc_desc(query_rgb, retr_model, dev)
        elif global_desc_method == "mixvpr":
            from .step5_retrieval import step5_retrieval
            _sc = step5_retrieval.__dict__
            if "_mixvpr_model" not in _sc:
                ckpt_path = fc.get("mixvpr_ckpt", "")
                out_dim   = int(fc.get("mixvpr_out_dim", 512))
                _sc["_mixvpr_model"] = _load_mixvpr_model(ckpt_path, out_dim, dev)
            q_gd = _extract_mixvpr_desc(query_rgb, _sc["_mixvpr_model"], dev)
        else:
            raise ValueError(f"Unknown global_desc_method: '{global_desc_method}'")

        q_gd = q_gd / (np.linalg.norm(q_gd) + 1e-8)
        dists, idxs = db["kdtree"].query(q_gd, k=top_k)
        cos_sims    = 1.0 - dists**2 / 2.0
        candidates  = [db["entries"][i] for i in idxs]
        print(f"    Retrieval top1: #{candidates[0]['id']}  sim={cos_sims[0]:.4f}")

        # ── Matching + re-ranking ──────────────────────────────────────
        best_score = -1.0; best_n = 0
        best_mkpts_q = best_mkpts_r = best_confs = None
        best_cand = candidates[0]; best_ref_rgb = None

        for rank, cand in enumerate(candidates):
            ref_rgb = cv2.cvtColor(cv2.imread(cand["rgb_path"]), cv2.COLOR_BGR2RGB)
            try:
                mkpts0, mkpts1, confs, n_good, score = _run_single_match(
                    matcher_type, matcher, query_rgb, ref_rgb,
                    conf_thresh, vismatch_max_dim,
                    gray_tensor_fn=gray_tensor_fn,
                    loftr_gray_fn=loftr_gray_fn,
                    jamma_preprocess_fn=jamma_preprocess_fn,
                )
            except Exception as e:
                print(f"    Rank{rank+1} failed: {e}"); continue

            mean_conf = float(confs.mean()) if len(confs) > 0 else 0.0
            ret_sim   = cos_sims[rank] if rank < len(cos_sims) else 0.0
            print(f"    Rank{rank+1} #{cand['id']:>4}: {n_good:>3} good  "
                  f"score={score:>7.2f}  mean_conf={mean_conf:.3f}  ret_sim={ret_sim:.4f}")

            if score > best_score:
                best_score = score; best_n = n_good
                best_mkpts_q = mkpts0; best_mkpts_r = mkpts1
                best_confs = confs; best_cand = cand; best_ref_rgb = ref_rgb

        if best_ref_rgb is None:
            best_ref_rgb = cv2.cvtColor(cv2.imread(candidates[0]["rgb_path"]), cv2.COLOR_BGR2RGB)

        # ── 시각화 ─────────────────────────────────────────────────────
        h1, w1 = query_rgb.shape[:2]; h2, w2 = best_ref_rgb.shape[:2]
        th = max(h1, h2); sq, sr = 1.0, 1.0
        if h1 != th: sq = th/h1; query_rgb    = cv2.resize(query_rgb,    (int(w1*sq), th))
        if h2 != th: sr = th/h2; best_ref_rgb = cv2.resize(best_ref_rgb, (int(w2*sr), th))
        h1, w1 = query_rgb.shape[:2]; h2, w2 = best_ref_rgb.shape[:2]
        canvas = np.concatenate([query_rgb, best_ref_rgb], axis=1)

        fig, ax = plt.subplots(1, 1, figsize=(18, 6))
        ax.imshow(canvas)
        if best_n > 0 and best_mkpts_q is not None:
            mkpts_q_v = best_mkpts_q * sq
            mkpts_r_v = best_mkpts_r * sr
            norm_c = best_confs / (best_confs.max() + 1e-8)
            cmap_v = plt.cm.RdYlGn(norm_c)
            step_v = max(1, best_n // 300)
            for i in range(0, best_n, step_v):
                ax.plot([mkpts_q_v[i,0], mkpts_r_v[i,0]+w1],
                        [mkpts_q_v[i,1], mkpts_r_v[i,1]],
                        c=cmap_v[i], alpha=0.5, linewidth=0.8)
            ax.scatter(mkpts_q_v[:,0], mkpts_q_v[:,1], c="cyan",   s=6, zorder=3)
            ax.scatter(mkpts_r_v[:,0]+w1, mkpts_r_v[:,1], c="yellow", s=6, zorder=3)

        ax.set_title(f"[{matcher_name}] {stem}  →  best=#{best_cand['id']}  "
                     f"({best_n} matches, score={best_score:.1f})")
        ax.axis("off"); fig.tight_layout()
        out_png = os.path.join(save_dir, f"{stem}.png")
        fig.savefig(out_png, dpi=120); plt.close()
        print(f"    Saved: step6a_results/{stem}.png")

    print(f"\n  Done. {len(query_files)} results in {save_dir}/")
