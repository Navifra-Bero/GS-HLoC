#!/usr/bin/env python3
"""
Multi-frame fusion - 4가지 케이스를 정확하게.
각 케이스에서 점 구름과 카메라 경로(빨강)를 함께 ply에 저장.
"""
import numpy as np
from pathlib import Path
from PIL import Image

KAPTURE = Path.home() / "loc_ws/src/render_loc/kapture/sensors"
OUT_DIR = Path.home() / "loc_ws/src/render_loc/ply_test"
CAM = "cam_3"

FRAME_STRIDE = 20
DEPTH_MIN, DEPTH_MAX = 0.3, 30.0
PIXEL_STRIDE = 4
VOXEL_SIZE = 0.05

fx, fy, cx, cy = 1039.045981, 1041.496942, 937.044077, 560.826738
W, H = 1920, 1200

# Calibration
R_il = np.array([[-1,0,0],[0,1,0],[0,0,-1]], dtype=float)
t_il = np.array([-0.00845, 0.00004, -0.09992])
R_lc = np.array([
    [ 0.01396, -0.99990, -0.00000],
    [ 0.05582,  0.00078, -0.99844],
    [ 0.99834,  0.01394,  0.05582],
])
t_lc = np.array([0.0, -0.13736, -0.2284])
R_ic = R_il @ R_lc
t_ic = R_il @ t_lc + t_il

def make_T(R, t):
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=t; return T

def Tinv(T):
    R, t = T[:3,:3], T[:3,3]
    Ti = np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3]=-R.T@t
    return Ti

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

T_imu_cam = make_T(R_ic, t_ic)        # cam->imu (4x4)
T_cam_imu = Tinv(T_imu_cam)           # imu->cam

# === 4가지 케이스: 각각 traj_T -> T_cam_world (cam->world, 4x4) ===
def caseA(T_traj):
    """traj = T_world_cam (kapture 표준)"""
    return Tinv(T_traj)

def caseB(T_traj):
    """traj = T_cam_world (반대 규약)"""
    return T_traj

def caseC(T_traj):
    """traj = T_world_imu (imu 포즈, kapture 표준)
       T_cam_world = T_imu_world @ T_imu_cam = inv(T_world_imu) @ T_imu_cam"""
    return Tinv(T_traj) @ T_imu_cam

def caseD(T_traj):
    """traj = T_imu_world (imu가 world에서 어디 있는지)
       T_cam_world = T_imu_world @ T_imu_cam"""
    return T_traj @ T_imu_cam

cases = {"A_world_cam": caseA, "B_cam_world": caseB,
         "C_world_imu": caseC, "D_imu_world": caseD}

# === records / trajectories 읽기 ===
def load_records():
    out = {}
    with open(KAPTURE / "records_camera.txt") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            p = [x.strip() for x in s.split(",")]
            if p[1] == CAM: out[p[0]] = p[2]
    return out

def load_trajs():
    out = {}
    with open(KAPTURE / "trajectories.txt") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            p = [x.strip() for x in s.split(",")]
            if p[1] != CAM: continue
            q = np.array(list(map(float, p[2:6])))
            t = np.array(list(map(float, p[6:9])))
            out[p[0]] = make_T(quat_to_R(q), t)
    return out

records = load_records()
trajs = load_trajs()
common = sorted(set(records) & set(trajs))[::FRAME_STRIDE]
print(f"frames: {len(common)}")

# 픽셀 그리드
us = np.arange(0, W, PIXEL_STRIDE)
vs = np.arange(0, H, PIXEL_STRIDE)
uu, vv = np.meshgrid(us, vs)
uu_f = (uu - cx) / fx
vv_f = (vv - cy) / fy

def load_frame(ts, img_relpath):
    dp = KAPTURE / "records_data" / CAM / "depths" / f"{ts}.depth"
    if not dp.exists(): return None, None
    raw = np.fromfile(dp, dtype=np.float32)
    if raw.size != W*H: return None, None
    depth = raw.reshape(H, W)
    z = depth[vs[:,None], us[None,:]]
    valid = (z > DEPTH_MIN) & (z < DEPTH_MAX) & np.isfinite(z)
    if not valid.any(): return None, None
    pts = np.stack([uu_f*z, vv_f*z, z], axis=-1)[valid]  # (N,3) cam local
    rp = KAPTURE / "records_data" / img_relpath
    if rp.exists():
        rgb = np.array(Image.open(rp).convert("RGB"))
        cols = rgb[vs[:,None], us[None,:]][valid]
    else:
        cols = np.full((pts.shape[0], 3), 200, dtype=np.uint8)
    return pts.astype(np.float32), cols.astype(np.uint8)

# === 각 케이스마다 누적 ===
buckets = {n: {"pts": [], "cols": [], "traj": []} for n in cases}

for i, ts in enumerate(common):
    pts_cam, cols = load_frame(ts, records[ts])
    if pts_cam is None: continue
    T_traj = trajs[ts]
    pts_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0],1), dtype=np.float32)])

    for name, fn in cases.items():
        T_cw = fn(T_traj)   # cam->world
        # 점 변환: p_world = T_cw @ p_cam_h
        pts_w = (pts_h @ T_cw.T)[:, :3]
        # 카메라 중심 = T_cw의 t
        cam_center = T_cw[:3, 3]
        buckets[name]["pts"].append(pts_w.astype(np.float32))
        buckets[name]["cols"].append(cols)
        buckets[name]["traj"].append(cam_center.astype(np.float32))

    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(common)}")

# === voxel downsample ===
def voxel_ds(pts, cols, v):
    if v is None: return pts, cols
    k = np.floor(pts / v).astype(np.int64)
    h = k[:,0]*73856093 ^ k[:,1]*19349663 ^ k[:,2]*83492791
    _, idx = np.unique(h, return_index=True)
    return pts[idx], cols[idx]

# === 저장 ===
OUT_DIR.mkdir(parents=True, exist_ok=True)
def write_ply(path, pts, cols):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {pts.shape[0]}\n"
                 f"property float x\nproperty float y\nproperty float z\n"
                 f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 f"end_header\n").encode())
        a = np.empty(pts.shape[0], dtype=[("x","<f4"),("y","<f4"),("z","<f4"),
                                           ("r","u1"),("g","u1"),("b","u1")])
        a["x"], a["y"], a["z"] = pts[:,0], pts[:,1], pts[:,2]
        a["r"], a["g"], a["b"] = cols[:,0], cols[:,1], cols[:,2]
        f.write(a.tobytes())

for name, b in buckets.items():
    if not b["pts"]:
        print(f"{name}: empty"); continue
    pts = np.concatenate(b["pts"])
    cols = np.concatenate(b["cols"])
    pts, cols = voxel_ds(pts, cols, VOXEL_SIZE)
    # 카메라 경로(빨강) 추가
    traj_pts = np.array(b["traj"], dtype=np.float32)
    traj_cols = np.tile(np.array([255,0,0], dtype=np.uint8), (traj_pts.shape[0],1))
    pts_all = np.vstack([pts, traj_pts])
    cols_all = np.vstack([cols, traj_cols])
    out = OUT_DIR / f"fused_{name}.ply"
    write_ply(out, pts_all, cols_all)
    print(f"  {out.name}: scene={pts.shape[0]:,} + traj={traj_pts.shape[0]} pts")

print("\n4개 ply를 MeshLab에서 비교. 카메라 경로(빨강)가 점 구름의 바닥 위로 부드럽게 흐르는 게 정답.")