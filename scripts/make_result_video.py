"""
각 프레임 폴더의 step5/step6 결과 이미지를 4Hz 영상으로 저장합니다.
출력: step5_result.mp4, step6_result.mp4  (results_dir 안에 생성)
"""

import os
import glob
import cv2

RESULTS_DIR = (
    "/home/park/loc_ws/src/render_loc/output/gs_test/"
    "test_results/cam_3_ransac_no_refine"
)
FPS = 4

TARGETS = [
    ("step5_retrieval.png", "step5_result.mp4"),
    ("step6_matching.png",  "step6_result.mp4"),
]

# 타임스탬프 기준 정렬
frame_dirs = sorted(
    d for d in glob.glob(os.path.join(RESULTS_DIR, "*"))
    if os.path.isdir(d)
)
print(f"Total frame dirs: {len(frame_dirs)}")

for img_name, out_name in TARGETS:
    img_paths = []
    for d in frame_dirs:
        p = os.path.join(d, img_name)
        if os.path.exists(p):
            img_paths.append(p)

    if not img_paths:
        print(f"[SKIP] No '{img_name}' found in any folder.")
        continue

    # 첫 이미지로 해상도 결정
    first = cv2.imread(img_paths[0])
    h, w = first.shape[:2]
    out_path = os.path.join(RESULTS_DIR, out_name)
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (w, h),
    )

    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  [WARN] Failed to read: {p}")
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)

    writer.release()
    print(f"[OK] {out_name}  ({len(img_paths)} frames, {w}x{h} @ {FPS}fps) -> {out_path}")
