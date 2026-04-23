#!/usr/bin/env bash
# =============================================================================
# RenderLoc 환경 설정 스크립트
# =============================================================================
# 대상 GPU  : RTX 5070 (Blackwell SM 12.0) — CUDA 12.8 필요
# 사용 모델 : MixVPR + MegaLoc (global desc) / EfficientLoFTR (matcher)
#
# 사전 요구사항:
#   - miniforge3 또는 mambaforge 설치
#   - CUDA 12.8 드라이버 (nvidia-smi 로 확인)
#
# 사용법:
#   bash setup_env.sh          # 기본 (환경명: render_loc)
#   bash setup_env.sh myenv    # 환경명 지정
# =============================================================================

set -e  # 오류 발생 시 즉시 중단

ENV_NAME="${1:-render_loc}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="$REPO_ROOT/third_party"

CONDA_BASE="$(conda info --base)"
ENV_PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
ENV_PIP="$CONDA_BASE/envs/$ENV_NAME/bin/pip"
# PYTHONNOUSERSITE=1: ~/.local 패키지 차단 → 환경 완전 격리
PIP="env PYTHONNOUSERSITE=1 $ENV_PIP"

echo "======================================================"
echo "  RenderLoc 환경 설정"
echo "  ENV_NAME  : $ENV_NAME"
echo "  REPO_ROOT : $REPO_ROOT"
echo "======================================================"

# ── 1. mamba 환경 생성 ────────────────────────────────────────────
echo ""
echo "[1/6] mamba 환경 생성 (python=3.10)"
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "  기존 환경 '$ENV_NAME' 발견 → 삭제 후 재생성"
    mamba env remove -n "$ENV_NAME" -y
fi
mamba create -n "$ENV_NAME" python=3.10 -y

# ── 2. CUDA 툴킷 (gsplat 빌드용) ─────────────────────────────────
echo ""
echo "[2/6] CUDA 12.8 툴킷 설치"
mamba install -n "$ENV_NAME" -y \
    -c nvidia/label/cuda-12.8.0 \
    cuda-toolkit

# ── 3. PyTorch (CUDA 12.8, Blackwell 지원) ───────────────────────
echo ""
echo "[3/6] PyTorch 설치 (cu128 nightly — RTX 5070 Blackwell 필수)"
$PIP install --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# ── 4. 핵심 패키지 ───────────────────────────────────────────────
echo ""
echo "[4/6] 핵심 패키지 설치"
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
echo "[5/6] gsplat 설치 (CUDA 커널 컴파일 — 5~15분 소요)"
$PIP install gsplat

# ── 6. third_party 준비 ──────────────────────────────────────────
echo ""
echo "[6/6] third_party 준비"

# ── MixVPR ───────────────────────────────────────────────────────
echo "  [6a] MixVPR"
MIXVPR_DIR="$THIRD_PARTY/MixVPR"
if [ ! -d "$MIXVPR_DIR" ]; then
    echo "       클론 중: $MIXVPR_DIR"
    git clone https://github.com/amaralibey/MixVPR "$MIXVPR_DIR"
else
    echo "       이미 존재: $MIXVPR_DIR (skip)"
fi
# MixVPR 추가 의존성 (timm 은 위에서 설치 완료)
$PIP install pytorch-metric-learning

# ── MixVPR 가중치 다운로드 (ResNet50, 512-dim, GSV-Cities trained) ──
mkdir -p "$REPO_ROOT/models"
$PIP install gdown --quiet
# ResNet50 4096-dim (최고 성능) — GSV-Cities trained
MIXVPR_CKPT_4096="$REPO_ROOT/models/mixvpr_resnet50_4096.ckpt"
if [ -f "$MIXVPR_CKPT_4096" ]; then
    echo "       MixVPR 4096 가중치 이미 존재: $MIXVPR_CKPT_4096 (skip)"
else
    echo "       MixVPR ResNet50 4096-dim 가중치 다운로드 중..."
    "$CONDA_BASE/envs/$ENV_NAME/bin/gdown" "1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L" \
        -O "$MIXVPR_CKPT_4096" && echo "       저장: $MIXVPR_CKPT_4096" \
        || echo "  ⚠  MixVPR 4096 가중치 다운로드 실패 — 수동으로 다운받아 models/ 에 저장하세요"
fi
# ResNet50 512-dim (경량 옵션)
MIXVPR_CKPT_512="$REPO_ROOT/models/mixvpr_resnet50_512.ckpt"
if [ -f "$MIXVPR_CKPT_512" ]; then
    echo "       MixVPR 512 가중치 이미 존재: $MIXVPR_CKPT_512 (skip)"
else
    echo "       MixVPR ResNet50 512-dim 가중치 다운로드 중..."
    "$CONDA_BASE/envs/$ENV_NAME/bin/gdown" "1khiTUNzZhfV2UUupZoIsPIbsMRBYVDqj" \
        -O "$MIXVPR_CKPT_512" && echo "       저장: $MIXVPR_CKPT_512" \
        || echo "  ⚠  MixVPR 512 가중치 다운로드 실패"
fi

# ── EfficientLoFTR ───────────────────────────────────────────────
# step6_match.py 가 sys.path 에 직접 추가하므로 pip 설치 불필요
# 의존성 패키지만 설치
echo "  [6b] EfficientLoFTR 의존성"
ELOFTR_DIR="$THIRD_PARTY/EfficientLoFTR"
if [ -d "$ELOFTR_DIR" ]; then
    # pytorch-lightning / torchmetrics 는 학습 전용 → 추론에 불필요, 스킵
    $PIP install "joblib>=1.0.1"

    CKPT="$ELOFTR_DIR/weights/ELoFTR/weights/eloftr_outdoor.ckpt"
    if [ -f "$CKPT" ]; then
        echo "       가중치 확인: $CKPT"
    else
        echo "  ⚠  EfficientLoFTR 가중치 없음!"
        echo "     다음 링크에서 다운로드 후 아래 경로에 저장하세요:"
        echo "     https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q"
        echo "     → $CKPT"
    fi
else
    echo "  ⚠  third_party/EfficientLoFTR 없음 — skip"
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
    ("torch",         lambda: __import__("torch").__version__),
    ("torchvision",   lambda: __import__("torchvision").__version__),
    ("numpy",         lambda: __import__("numpy").__version__),
    ("open3d",        lambda: __import__("open3d").__version__),
    ("gsplat",        lambda: __import__("gsplat").__version__),
    ("cv2",           lambda: __import__("cv2").__version__),
    ("scipy",         lambda: __import__("scipy").__version__),
    ("sklearn",       lambda: __import__("sklearn").__version__),
    ("timm",          lambda: __import__("timm").__version__),
    ("kornia",        lambda: __import__("kornia").__version__),
    ("einops",        lambda: __import__("einops").__version__),
]
for name, fn in items:
    try:
        print(f"  ✓ {name:<16} {fn()}")
    except Exception as e:
        print(f"  ✗ {name:<16} MISSING  ({e})")

import torch
avail = torch.cuda.is_available()
print(f"\n  CUDA available : {avail}")
if avail:
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
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
echo "  파이프라인 실행 (offline):"
echo "    python scripts/main.py \\"
echo "        --ply_map <map.ply> \\"
echo "        --output_dir output/run1 \\"
echo "        --config config/render_loc.yaml \\"
echo "        --render_mode scaffold_gs \\"
echo "        --sgs_model_path <model_dir> \\"
echo "        --step offline"
echo ""
echo "  MixVPR 사전학습 가중치 (선택, 성능 향상):"
echo "    https://github.com/amaralibey/MixVPR#trained-models"
echo "    → config/render_loc.yaml 의 mixvpr_ckpt 에 경로 지정"
echo "======================================================"
