#!/usr/bin/env python3
"""
진단용 스크립트: 학습 카메라 pose를 그대로 ns-render camera-path JSON으로 export.

목적:
- step1 viewpoint를 쓸 때 흐릿한 게 진짜 "novel view" 문제인지 확인
- 학습 카메라 그대로 넣어서 렌더링해도 흐리면 → 좌표계/변환 문제
- 학습 카메라는 깔끔하면 → 진짜 novel view 일반화 문제

사용:
    python3 export_train_camera_path.py \
        --config nerfstudio/outputs/images_metric/splatfacto/<run>/config.yml \
        --output output/test_train_camera_path.json \
        --num-cameras 50

그 다음:
    cd nerfstudio
    TORCH_COMPILE_DISABLE=1 ns-render camera-path \
        --load-config outputs/images_metric/splatfacto/<run>/config.yml \
        --camera-path-filename ../output/test_train_camera_path.json \
        --output-path ../output/test_train_render/ \
        --output-format images
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from nerfstudio.utils.eval_utils import eval_setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="출력 JSON 경로")
    parser.add_argument("--num-cameras", type=int, default=50,
                        help="export할 학습 카메라 개수 (전체에서 균등 샘플)")
    args = parser.parse_args()

    # 모델 로드 (학습 카메라 정보만 필요)
    print(f"Loading config: {args.config}")
    config, pipeline, _, _ = eval_setup(Path(args.config), test_mode="inference")

    train_cameras = pipeline.datamanager.train_dataset.cameras
    n_total = len(train_cameras)
    print(f"Train cameras: {n_total}")

    # 균등하게 샘플
    indices = np.linspace(0, n_total - 1, min(args.num_cameras, n_total), dtype=int)
    print(f"Selecting {len(indices)} cameras: {list(indices[:5])}...{list(indices[-3:])}")

    cam_list = []
    H_out, W_out = None, None
    for idx in indices:
        i = int(idx)
        c2w_3x4 = train_cameras.camera_to_worlds[i].cpu().numpy()  # (3, 4)
        c2w = np.eye(4)
        c2w[:3, :4] = c2w_3x4

        H = int(train_cameras.height[i].item())
        W = int(train_cameras.width[i].item())
        fy = float(train_cameras.fy[i].item())
        if H_out is None:
            H_out, W_out = H, W

        # 학습 카메라의 fov 계산 (vertical, degrees)
        vfov_deg = math.degrees(2.0 * math.atan(H / (2.0 * fy)))
        aspect = W / H

        cam_list.append({
            "camera_to_world": c2w.flatten().tolist(),
            "fov": vfov_deg,
            "aspect": aspect,
        })

    n = len(cam_list)
    out = {
        "camera_type": "perspective",
        "render_height": H_out,
        "render_width": W_out,
        "fps": 1,
        "seconds": max(1, n),
        "is_cycle": False,
        "smoothness_value": 0.0,
        "camera_path": cam_list,
        "keyframes": [
            {
                "matrix": c["camera_to_world"],
                "fov": c["fov"],
                "aspect": c["aspect"],
                "override_transition_enabled": False,
                "override_transition_sec": None,
            }
            for c in cam_list
        ],
        "default_fov": cam_list[0]["fov"],
        "default_transition_sec": 1.0,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved {n} train cameras to {args.output}")
    print(f"  Resolution: {W_out}×{H_out}")
    print(f"  Sample fov: {cam_list[0]['fov']:.2f}°")
    print(f"\n다음 명령으로 렌더링 후 비교하세요:")
    print(f"  cd nerfstudio")
    print(f"  TORCH_COMPILE_DISABLE=1 ns-render camera-path \\")
    print(f"      --load-config {args.config} \\")
    print(f"      --camera-path-filename ../{args.output} \\")
    print(f"      --output-path ../output/test_train_render/ \\")
    print(f"      --output-format images")


if __name__ == "__main__":
    main()
