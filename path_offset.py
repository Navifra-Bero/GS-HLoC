import json
import math
import numpy as np
from pathlib import Path

transforms_path = Path("images_metric/transforms.json")
out_path = Path("../output/colmap_sgs_test/step1_camera_path_offset_5cm.json")

render_width = 1920
render_height = 1200
fps = 30
seconds = 10
num_path_frames = fps * seconds

# camera local coordinate 기준 offset
# x: 오른쪽, y: 아래/위는 convention에 따라 다를 수 있음, z: forward/backward convention 주의
max_translation_offset_m = 0.20
max_rotation_offset_deg = 5.0


def random_local_translation(max_offset_m):
    return np.random.uniform(-max_offset_m, max_offset_m, size=3)


def random_local_rotation(max_angle_deg):
    axis = np.random.normal(size=3)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        return np.eye(3)

    axis /= axis_norm
    angle = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    kx, ky, kz = axis
    K = np.array([
        [0.0, -kz, ky],
        [kz, 0.0, -kx],
        [-ky, kx, 0.0],
    ])

    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)

with open(transforms_path, "r") as f:
    data = json.load(f)

frames = data["frames"]
w = data.get("w", render_width)
h = data.get("h", render_height)
fl_x = data.get("fl_x", None)

if fl_x is not None:
    fov = 2.0 * math.atan(w / (2.0 * fl_x)) * 180.0 / math.pi
else:
    fov = 60.0

if len(frames) <= num_path_frames:
    selected = frames
else:
    idxs = [
        round(i * (len(frames) - 1) / (num_path_frames - 1))
        for i in range(num_path_frames)
    ]
    selected = [frames[i] for i in idxs]

camera_path = []
for fr in selected:
    mat = np.array(fr["transform_matrix"], dtype=float)

    R = mat[:3, :3]
    # local offset을 world offset으로 변환
    mat[:3, 3] += R @ random_local_translation(max_translation_offset_m)
    mat[:3, :3] = R @ random_local_rotation(max_rotation_offset_deg)

    c2w_3x4 = mat[:4, :4].reshape(-1).tolist()

    camera_path.append({
        "camera_to_world": c2w_3x4,
        "fov": fov,
        "aspect": float(w) / float(h),
    })

out = {
    "camera_type": "perspective",
    "render_height": int(h),
    "render_width": int(w),
    "seconds": float(seconds),
    "fps": int(fps),
    "camera_path": camera_path,
}

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print("saved:", out_path)
print("num camera path frames:", len(camera_path))
print("max_translation_offset_m:", max_translation_offset_m)
print("max_rotation_offset_deg:", max_rotation_offset_deg)
print("fov:", fov)
