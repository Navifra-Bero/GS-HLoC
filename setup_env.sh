#!/usr/bin/env bash
# =============================================================================
# RenderLoc 환경 설정 스크립트
# =============================================================================
# 대상 GPU  : RTX 5070 / 5080 / 5090 (Blackwell SM 12.0) — CUDA 12.8 필요
# 사용 모델 : MixVPR + MegaLoc (global desc) / EfficientLoFTR (matcher)
#             Scaffold-GS Feature Gaussian Splatting (SplatHLoc 스타일)
#
# 사전 요구사항:
#   - miniforge3 또는 mambaforge 설치
#   - CUDA 12.8 드라이버 (nvidia-smi 로 확인)
#   - third_party/Scaffold-GS 가 이미 체크아웃되어 있어야 함
#     (rasterizer 빌드에 필요. submodule 또는 직접 clone)
#
# 사용법:
#   bash setup_env.sh          # 기본 (환경명: render_loc)
#   bash setup_env.sh myenv    # 환경명 지정
# =============================================================================

set -e  # 오류 발생 시 즉시 중단

ENV_NAME="${1:-render_loc}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="$REPO_ROOT/third_party"

# Blackwell (RTX 50xx) 아키텍처. 다른 GPU 도 같이 받으려면 세미콜론으로 추가.
CUDA_ARCHES="12.0"

CONDA_BASE="$(conda info --base)"
ENV_PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
ENV_PIP="$CONDA_BASE/envs/$ENV_NAME/bin/pip"
# PYTHONNOUSERSITE=1: ~/.local 패키지 차단 → 환경 완전 격리
PIP="env PYTHONNOUSERSITE=1 $ENV_PIP"

echo "======================================================"
echo "  RenderLoc 환경 설정"
echo "  ENV_NAME       : $ENV_NAME"
echo "  REPO_ROOT      : $REPO_ROOT"
echo "  CUDA_ARCHES    : $CUDA_ARCHES (Blackwell sm_120)"
echo "======================================================"

# ── 1. mamba 환경 생성 ────────────────────────────────────────────
echo ""
echo "[1/8] mamba 환경 생성 (python=3.10)"
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "  기존 환경 '$ENV_NAME' 발견 → 삭제 후 재생성"
    mamba env remove -n "$ENV_NAME" -y
fi
mamba create -n "$ENV_NAME" python=3.10 -y

# ── 2. CUDA 툴킷 (gsplat / Scaffold-GS 빌드용) ────────────────────
echo ""
echo "[2/8] CUDA 12.8 툴킷 설치"
mamba install -n "$ENV_NAME" -y \
    -c nvidia/label/cuda-12.8.0 \
    cuda-toolkit

# ── 3. PyTorch (CUDA 12.8, Blackwell 지원) ───────────────────────
echo ""
echo "[3/8] PyTorch 설치 (cu128 nightly — Blackwell sm_120 필수)"
$PIP install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# ── 4. 핵심 패키지 ───────────────────────────────────────────────
echo ""
echo "[4/8] 핵심 패키지 설치"
$PIP install \
    "open3d>=0.18.0" \
    "opencv-python>=4.5.0" \
    "numpy==1.26.4" \
    "pyyaml>=6.0" \
    "matplotlib>=3.5.0" \
    "scipy>=1.7.0" \
    "scikit-learn>=1.0.0" \
    "scikit-image>=0.18.0" \
    "Pillow>=9.0" \
    "tqdm>=4.60" \
    "einops>=0.6.0" \
    "kornia>=0.6.0" \
    "yacs>=0.1.8" \
    "loguru>=0.6" \
    "h5py>=3.6" \
    "timm>=0.6.12" \
    "setuptools<81"

# ── 5. gsplat (CUDA 컴파일 — 수 분 소요) ─────────────────────────
echo ""
echo "[5/8] gsplat 설치 (CUDA 커널 컴파일 — 5~15분 소요)"
TORCH_CUDA_ARCH_LIST="$CUDA_ARCHES" \
    CUDA_HOME="$CONDA_BASE/envs/$ENV_NAME" \
    $PIP install gsplat

# ── 6. Scaffold-GS feature training 의존성 ──────────────────────
echo ""
echo "[6/8] Scaffold-GS feature training 의존성"
$PIP install \
    "plyfile>=0.8" \
    "lpips>=0.1.4" \
    "wandb>=0.16"

# torch-scatter: PyG wheel index 에서 torch 2.11 + cu128 매칭 휠 설치
TORCH_VERSION="$($ENV_PY -c 'import torch;print(torch.__version__.split("+")[0])')"
echo "  torch-scatter: matching torch ${TORCH_VERSION} + cu128"
$PIP install torch-scatter \
    -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu128.html" || {
        echo "  ⚠  binary wheel 실패 — 소스 빌드로 fallback (5~10분 소요)"
        TORCH_CUDA_ARCH_LIST="$CUDA_ARCHES" \
            CUDA_HOME="$CONDA_BASE/envs/$ENV_NAME" \
            $PIP install --no-binary torch-scatter torch-scatter
    }

# ── 7. Scaffold-GS CUDA rasterizers (sm_120 포함) ───────────────
echo ""
echo "[7/8] Scaffold-GS rasterizer 빌드 (CUDA — 각 5~10분)"

SGS_ROOT="$THIRD_PARTY/Scaffold-GS"
if [ ! -d "$SGS_ROOT" ]; then
    echo "  ⚠  $SGS_ROOT 없음 — Scaffold-GS submodule 먼저 체크아웃하세요."
    echo "     예: git clone https://github.com/city-super/Scaffold-GS $SGS_ROOT"
    echo "     (또는 사내 fork)."
    echo "     그 후 이 스크립트를 다시 실행하거나 7단계 명령을 수동 실행하세요."
else
    # 빌드 환경: conda env 의 CUDA 사용 (system gcc 와 충돌 방지 위해 CC 명시)
    BUILD_ENV=(env
        PYTHONNOUSERSITE=1
        TORCH_CUDA_ARCH_LIST="$CUDA_ARCHES"
        CUDA_HOME="$CONDA_BASE/envs/$ENV_NAME"
        CC="/usr/bin/gcc"
        CXX="/usr/bin/g++"
    )

    for SUB in diff-gaussian-rasterization diff-gaussian-rasterization-feat simple-knn; do
        SUBDIR="$SGS_ROOT/submodules/$SUB"
        if [ ! -d "$SUBDIR" ]; then
            echo "  ⚠  $SUBDIR 없음 — skip"
            continue
        fi
        echo "  [7.$SUB] 빌드 시작"
        rm -rf "$SUBDIR/build"
        find "$SUBDIR" -maxdepth 2 -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
        "${BUILD_ENV[@]}" "$ENV_PIP" install "$SUBDIR" --no-build-isolation \
            && echo "       ✓ $SUB 설치 완료" \
            || { echo "       ✗ $SUB 빌드 실패 — 로그 확인"; exit 1; }
    done
fi

# ── 8. third_party 모델 / 가중치 ────────────────────────────────
echo ""
echo "[8/8] third_party 모델 / 가중치"

mkdir -p "$REPO_ROOT/models"
$PIP install gdown --quiet

# ── MixVPR ───────────────────────────────────────────────────────
echo "  [8a] MixVPR"
MIXVPR_DIR="$THIRD_PARTY/MixVPR"
if [ ! -d "$MIXVPR_DIR" ]; then
    echo "       클론 중: $MIXVPR_DIR"
    git clone https://github.com/amaralibey/MixVPR "$MIXVPR_DIR"
else
    echo "       이미 존재: $MIXVPR_DIR (skip)"
fi
$PIP install pytorch-metric-learning

# MixVPR 가중치
MIXVPR_CKPT_4096="$REPO_ROOT/models/mixvpr_resnet50_4096.ckpt"
if [ -f "$MIXVPR_CKPT_4096" ]; then
    echo "       MixVPR 4096 가중치 이미 존재 (skip)"
else
    echo "       MixVPR ResNet50 4096-dim 가중치 다운로드..."
    "$CONDA_BASE/envs/$ENV_NAME/bin/gdown" "1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L" \
        -O "$MIXVPR_CKPT_4096" && echo "       저장: $MIXVPR_CKPT_4096" \
        || echo "  ⚠  MixVPR 4096 다운로드 실패 — 수동 저장 필요"
fi
MIXVPR_CKPT_512="$REPO_ROOT/models/mixvpr_resnet50_512.ckpt"
if [ -f "$MIXVPR_CKPT_512" ]; then
    echo "       MixVPR 512 가중치 이미 존재 (skip)"
else
    echo "       MixVPR ResNet50 512-dim 가중치 다운로드..."
    "$CONDA_BASE/envs/$ENV_NAME/bin/gdown" "1khiTUNzZhfV2UUupZoIsPIbsMRBYVDqj" \
        -O "$MIXVPR_CKPT_512" && echo "       저장: $MIXVPR_CKPT_512" \
        || echo "  ⚠  MixVPR 512 다운로드 실패"
fi

# ── EfficientLoFTR ───────────────────────────────────────────────
echo "  [8b] EfficientLoFTR 의존성"
ELOFTR_DIR="$THIRD_PARTY/EfficientLoFTR"
if [ -d "$ELOFTR_DIR" ]; then
    $PIP install "joblib>=1.0.1"
    CKPT="$ELOFTR_DIR/weights/ELoFTR/weights/eloftr_outdoor.ckpt"
    if [ -f "$CKPT" ]; then
        echo "       가중치 확인: $CKPT"
    else
        echo "  ⚠  EfficientLoFTR 가중치 없음! 수동 다운로드 필요:"
        echo "     https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q"
        echo "     → $CKPT"
    fi
else
    echo "  ⚠  third_party/EfficientLoFTR 없음 — skip"
fi

# ── SuperPoint (Scaffold-GS feature 학습 supervision 용) ─────────
echo "  [8c] SuperPoint TorchScript export"
SP_PT="$REPO_ROOT/models/superpoint_v1.pt"
if [ -f "$SP_PT" ]; then
    echo "       SuperPoint 가중치 이미 존재 (skip): $SP_PT"
else
    echo "       setup_models.py 로 SuperPoint TorchScript 생성..."
    env PYTHONNOUSERSITE=1 "$ENV_PY" "$REPO_ROOT/scripts/setup_models.py" \
        --output_dir "$REPO_ROOT/models" \
        --cache_dir "$REPO_ROOT/.cache_models" \
        --skip_superglue --skip_netvlad 2>/dev/null \
        || echo "  ⚠  SuperPoint export 실패 — scripts/setup_models.py 직접 실행하세요"
fi

# ── MegaLoc: torch.hub 로 자동 다운로드 (별도 설치 불필요) ─────────

# ── numpy 버전 고정 (마지막에 다시 한 번) ────────────────────────
$PIP install --force-reinstall "numpy==1.26.4"

# ── 설치 확인 ────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  설치 확인"
echo "======================================================"
env PYTHONNOUSERSITE=1 "$ENV_PY" - <<'PYEOF'
import sys
print(f"  Python  : {sys.version.split()[0]}")

items = [
    ("torch",                              lambda: __import__("torch").__version__),
    ("torchvision",                        lambda: __import__("torchvision").__version__),
    ("numpy",                              lambda: __import__("numpy").__version__),
    ("open3d",                             lambda: __import__("open3d").__version__),
    ("gsplat",                             lambda: __import__("gsplat").__version__),
    ("cv2",                                lambda: __import__("cv2").__version__),
    ("scipy",                              lambda: __import__("scipy").__version__),
    ("sklearn",                            lambda: __import__("sklearn").__version__),
    ("timm",                               lambda: __import__("timm").__version__),
    ("kornia",                             lambda: __import__("kornia").__version__),
    ("einops",                             lambda: __import__("einops").__version__),
    ("plyfile",                            lambda: __import__("plyfile").__version__ if hasattr(__import__("plyfile"), "__version__") else "ok"),
    ("lpips",                              lambda: "ok" if __import__("lpips") else "?"),
    ("wandb",                              lambda: __import__("wandb").__version__),
    ("torch_scatter",                      lambda: __import__("torch_scatter").__version__),
    ("diff_gaussian_rasterization",        lambda: "ok" if __import__("diff_gaussian_rasterization") else "?"),
    ("diff_gaussian_rasterization_feat",   lambda: "ok" if __import__("diff_gaussian_rasterization_feat") else "?"),
    ("simple_knn",                         lambda: "ok" if __import__("simple_knn") else "?"),
]
for name, fn in items:
    try:
        print(f"  ✓ {name:<36} {fn()}")
    except Exception as e:
        print(f"  ✗ {name:<36} MISSING  ({e})")

import torch
avail = torch.cuda.is_available()
print(f"\n  CUDA available : {avail}")
if avail:
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    print(f"  Compute cap.   : {torch.cuda.get_device_capability(0)}")
    print(f"  CUDA version   : {torch.version.cuda}")
PYEOF

# ── 완료 ─────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  완료!"
echo ""
echo "  환경 활성화:"
echo "    conda activate $ENV_NAME"
echo ""
echo "  Scaffold-GS feature 학습:"
echo "    cd third_party/Scaffold-GS"
echo "    # single_train.sh 상단의 scene/exp_name/cam_filter 본인 데이터에 맞게 수정"
echo "    bash single_train.sh"
echo ""
echo "  파이프라인 실행 (offline + scaffold_gs 렌더):"
echo "    python scripts/main.py \\"
echo "        --ply_map <map.ply> \\"
echo "        --output_dir output/run1 \\"
echo "        --config config/render_loc.yaml \\"
echo "        --render_mode scaffold_gs \\"
echo "        --sgs_model_path <model_dir> \\"
echo "        --sgs_ckpt <model_dir>/chkpnt_best.pth \\"
echo "        --step offline"
echo "======================================================"
