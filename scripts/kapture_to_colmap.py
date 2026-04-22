#!/usr/bin/env python3
"""
trajectories.txt 변환:
  입력: T_cam_world (cam -> world) — 이미 카메라 포즈
  출력: T_world_cam (world -> cam) — kapture/COLMAP 표준
"""
import numpy as np
from pathlib import Path

SRC_DIR = Path.home() / "loc_ws/src/render_loc/kapture/sensors"
DST_DIR = Path.home() / "loc_ws/src/render_loc/kapture_cam3/sensors"
TARGET_CAM = "cam_3"

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

def R_to_quat(R):
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr > 0:
        S = np.sqrt(tr+1.0)*2
        return np.array([0.25*S,(R[2,1]-R[1,2])/S,(R[0,2]-R[2,0])/S,(R[1,0]-R[0,1])/S])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
        return np.array([(R[2,1]-R[1,2])/S,0.25*S,(R[0,1]+R[1,0])/S,(R[0,2]+R[2,0])/S])
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
        return np.array([(R[0,2]-R[2,0])/S,(R[0,1]+R[1,0])/S,0.25*S,(R[1,2]+R[2,1])/S])
    else:
        S = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
        return np.array([(R[1,0]-R[0,1])/S,(R[0,2]+R[2,0])/S,(R[1,2]+R[2,1])/S,0.25*S])

DST_DIR.mkdir(parents=True, exist_ok=True)
n = 0
with open(SRC_DIR / "trajectories.txt") as fi, \
     open(DST_DIR / "trajectories.txt", "w") as fo:
    fo.write("# kapture format: 1.1\n")
    fo.write("# timestamp, device_id, qw, qx, qy, qz, tx, ty, tz\n")
    for line in fi:
        s = line.strip()
        if not s or s.startswith("#"): continue
        p = [x.strip() for x in s.split(",")]
        ts, sid = p[0], p[1]
        if sid != TARGET_CAM: continue

        q = np.array(list(map(float, p[2:6])))
        t = np.array(list(map(float, p[6:9])))

        # T_cam_world -> T_world_cam (inverse)
        R = quat_to_R(q)
        R_inv = R.T
        t_inv = -R_inv @ t
        q_inv = R_to_quat(R_inv)

        fo.write(f"{ts}, {TARGET_CAM}, "
                 f"{q_inv[0]:.9f}, {q_inv[1]:.9f}, {q_inv[2]:.9f}, {q_inv[3]:.9f}, "
                 f"{t_inv[0]:.9f}, {t_inv[1]:.9f}, {t_inv[2]:.9f}\n")
        n += 1

print(f"wrote {n} lines")