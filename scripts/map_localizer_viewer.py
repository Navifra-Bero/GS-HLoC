#!/usr/bin/env python3
"""
map_localizer_viewer.py
=======================
aligned_map.ply를 로드하고, localization trajectory를 따라
카메라 가시 영역을 실시간으로 강조하는 ROS2 시각화 노드.

퍼블리시 토픽:
  /map_cloud        (PointCloud2, latching) — 전체 맵 (super downsample + 85% 밝기, 1회 발행)
  /visible_region   (PointCloud2)           — 현재 가시 영역 (원본 색 100%, 원본 해상도)
  /visible_bbox     (MarkerArray)           — 가시 영역 3D AABB 경계박스
  /frustum_image    (PointCloud2)           — frustum 끝 평면에 현재 카메라 이미지 투영
  /matching_points  (PointCloud2, latching) — step6 매칭 3D 포인트 (빨간색)
  /camera_frustum   (MarkerArray)           — 카메라 절두체 (초록)
  /camera_pose      (PoseStamped)           — 현재 카메라 위치/방향
  /trajectory_line  (MarkerArray)           — 누적 경로 (파란색, 스플라인 곡선)

Usage:
  python3 scripts/map_localizer_viewer.py --ros-args \\
      -p aligned_ply:=output/step_by_step/aligned_map.ply \\
      -p trajectory_json:=output/step_by_step/trajectory_poses.json \\
      -p images_dir:=test_data/cam_1/images \\
      -p fps:=5.0

또는 launch 파일:
  ros2 launch render_loc map_localizer_viewer.launch.py
"""

import os, sys, json, time
import numpy as np
import cv2
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import message_filters

from sensor_msgs.msg import PointCloud2, PointField, Image
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros


# ─────────────────────────────────────────────────────────────────────────────
# QoS 프로파일
# ─────────────────────────────────────────────────────────────────────────────
LATCHING = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
)
VOLATILE = QoSProfile(depth=5)


# ─────────────────────────────────────────────────────────────────────────────
# PointCloud2 helper
# ─────────────────────────────────────────────────────────────────────────────
def make_pointcloud2(xyz, rgb, frame_id, stamp):
    """(N,3) float32 + (N,3) uint8 → PointCloud2 with rgb field"""
    N = len(xyz)
    if N == 0:
        msg = PointCloud2()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp
        msg.height = 1
        msg.width = 0
        return msg

    xyz_f = xyz.astype(np.float32)
    r = rgb[:, 0].astype(np.uint32)
    g = rgb[:, 1].astype(np.uint32)
    b = rgb[:, 2].astype(np.uint32)
    rgb_packed = ((r << 16) | (g << 8) | b).view(np.float32)

    dt = np.dtype([('x', np.float32), ('y', np.float32),
                   ('z', np.float32), ('rgb', np.float32)])
    buf = np.zeros(N, dtype=dt)
    buf['x'] = xyz_f[:, 0]
    buf['y'] = xyz_f[:, 1]
    buf['z'] = xyz_f[:, 2]
    buf['rgb'] = rgb_packed

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = N
    msg.fields = [
        PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * N
    msg.data = buf.tobytes()
    msg.is_dense = True
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# Frustum image marker — frustum 끝 사각형에 카메라 이미지를 TRIANGLE_LIST로 렌더링
# ─────────────────────────────────────────────────────────────────────────────
def _build_frustum_image_data(T_WC, fx, fy, cx, cy, W, H, depth, img_rgb,
                              grid_w=80, grid_h=50):
    """frustum 이미지의 무거운 기하/색상 계산만 먼저 수행."""
    if img_rgb is None:
        return None

    gH, gW = grid_h, grid_w
    us = np.linspace(0, W - 1, gW)
    vs = np.linspace(0, H - 1, gH)
    ug, vg = np.meshgrid(us, vs)  # (gH, gW)

    # 카메라 좌표 → 월드 좌표
    x_cam = (ug - cx) / fx * depth
    y_cam = (vg - cy) / fy * depth
    pts_cam = np.stack([x_cam, y_cam, np.full_like(x_cam, depth)], axis=-1)  # (gH,gW,3)
    R = T_WC[:3, :3]
    t_pos = T_WC[:3, 3]
    pts_world = pts_cam @ R.T + t_pos  # (gH, gW, 3)

    # 이미지 컬러 샘플링
    h_img, w_img = img_rgb.shape[:2]
    u_int = np.clip((ug * w_img / W).astype(int), 0, w_img - 1)
    v_int = np.clip((vg * h_img / H).astype(int), 0, h_img - 1)
    colors_grid = img_rgb[v_int, u_int].astype(np.float32) / 255.0  # (gH, gW, 3)

    # quad → 2 삼각형 인덱스 (벡터화)
    j_idx, i_idx = np.meshgrid(np.arange(gH - 1), np.arange(gW - 1), indexing='ij')
    j_idx = j_idx.ravel(); i_idx = i_idx.ravel()
    N = len(j_idx)

    p00 = pts_world[j_idx,     i_idx    ]
    p10 = pts_world[j_idx,     i_idx + 1]
    p01 = pts_world[j_idx + 1, i_idx    ]
    p11 = pts_world[j_idx + 1, i_idx + 1]
    c00 = colors_grid[j_idx,     i_idx    ]
    c10 = colors_grid[j_idx,     i_idx + 1]
    c01 = colors_grid[j_idx + 1, i_idx    ]
    c11 = colors_grid[j_idx + 1, i_idx + 1]

    # tri1: p00,p10,p11 / tri2: p00,p11,p01 → (N*6, 3)
    tri_pts = np.empty((N * 6, 3), dtype=np.float64)
    tri_pts[0::6] = p00; tri_pts[1::6] = p10; tri_pts[2::6] = p11
    tri_pts[3::6] = p00; tri_pts[4::6] = p11; tri_pts[5::6] = p01
    tri_col = np.empty((N * 6, 3), dtype=np.float32)
    tri_col[0::6] = c00; tri_col[1::6] = c10; tri_col[2::6] = c11
    tri_col[3::6] = c00; tri_col[4::6] = c11; tri_col[5::6] = c01

    return tri_pts, tri_col


def _make_frustum_image_marker(frust_img_data, frame_id, stamp):
    ma = MarkerArray()
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = "frustum_image"
    m.id = 0
    m.type = Marker.TRIANGLE_LIST
    m.action = Marker.ADD
    m.scale.x = m.scale.y = m.scale.z = 1.0
    m.color.a = 1.0  # per-vertex colors 사용 시에도 필요

    if frust_img_data is None:
        m.action = Marker.DELETE
        ma.markers.append(m)
        return ma

    tri_pts, tri_col = frust_img_data
    m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in tri_pts]
    m.colors = [ColorRGBA(r=float(c[0]), g=float(c[1]), b=float(c[2]), a=1.0)
                for c in tri_col]
    ma.markers.append(m)
    return ma


def make_frustum_image_marker(T_WC, fx, fy, cx, cy, W, H, depth, img_rgb,
                              frame_id, stamp, grid_w=80, grid_h=50):
    """현재 카메라 이미지를 frustum 끝 사각형에 삼각형 메쉬로 렌더링.
    grid_w x grid_h 격자 → (grid_w-1)*(grid_h-1)*2 삼각형으로 solid 이미지 표현.
    """
    frust_img_data = _build_frustum_image_data(
        T_WC, fx, fy, cx, cy, W, H, depth, img_rgb,
        grid_w=grid_w, grid_h=grid_h)
    return _make_frustum_image_marker(frust_img_data, frame_id, stamp)


# ─────────────────────────────────────────────────────────────────────────────
# Frustum culling
# ─────────────────────────────────────────────────────────────────────────────
def frustum_cull(map_xyz, T_WC, fx, fy, cx, cy, W, H, depth_max):
    """
    T_WC: (4,4) world←camera  |  반환: visible 포인트 인덱스 배열
    """
    R = T_WC[:3, :3]
    t = T_WC[:3, 3]
    pts_cam = (map_xyz - t) @ R          # (N, 3)
    z = pts_cam[:, 2]
    mask_front = z > 0.1
    if not np.any(mask_front):
        return np.array([], dtype=np.int64)

    pf = pts_cam[mask_front]
    idx_f = np.where(mask_front)[0]
    zf = pf[:, 2]
    u = pf[:, 0] / zf * fx + cx
    v = pf[:, 1] / zf * fy + cy
    mask = (zf < depth_max) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return idx_f[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Frustum marker (카메라 절두체 LINE_LIST, 초록)
# ─────────────────────────────────────────────────────────────────────────────
def make_frustum_marker(T_WC, fx, fy, cx, cy, W, H,
                        depth, frame_id, stamp,
                        color=(0.0, 1.0, 0.0), line_width=0.04):
    corners_cam = np.array([
        [(0   - cx) / fx * depth, (0   - cy) / fy * depth, depth],
        [(W-1 - cx) / fx * depth, (0   - cy) / fy * depth, depth],
        [(W-1 - cx) / fx * depth, (H-1 - cy) / fy * depth, depth],
        [(0   - cx) / fx * depth, (H-1 - cy) / fy * depth, depth],
    ], dtype=np.float64)

    R = T_WC[:3, :3]
    origin = T_WC[:3, 3]
    corners_w = (R @ corners_cam.T).T + origin

    lines = []
    for c in corners_w:
        lines += [origin, c]
    for i in range(4):
        lines += [corners_w[i], corners_w[(i + 1) % 4]]

    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = "frustum"
    m.id = 0
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = line_width
    m.color.r, m.color.g, m.color.b, m.color.a = color[0], color[1], color[2], 1.0
    for pt in lines:
        p = Point(); p.x, p.y, p.z = float(pt[0]), float(pt[1]), float(pt[2])
        m.points.append(p)

    ma = MarkerArray()
    ma.markers.append(m)
    return ma


# ─────────────────────────────────────────────────────────────────────────────
# Image helper — RGB ndarray → sensor_msgs/Image
# ─────────────────────────────────────────────────────────────────────────────
def make_image_msg(rgb_img, frame_id, stamp):
    msg = Image()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    h, w = rgb_img.shape[:2]
    msg.height = h
    msg.width = w
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = w * 3
    msg.data = rgb_img.tobytes()
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# AABB 경계박스 marker (LINE_LIST, 12 edges)
# ─────────────────────────────────────────────────────────────────────────────
def make_bbox_marker(xyz_visible, frame_id, stamp,
                     color=(1.0, 1.0, 1.0), line_width=0.08):
    ma = MarkerArray()
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = "visible_bbox"
    m.id = 0

    if len(xyz_visible) == 0:
        m.action = Marker.DELETE
        ma.markers.append(m)
        return ma

    mn = xyz_visible.min(axis=0)
    mx = xyz_visible.max(axis=0)
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = line_width
    m.color.r, m.color.g, m.color.b, m.color.a = color[0], color[1], color[2], 1.0
    for a, b in edges:
        for pt in [corners[a], corners[b]]:
            p = Point(); p.x, p.y, p.z = float(pt[0]), float(pt[1]), float(pt[2])
            m.points.append(p)
    ma.markers.append(m)
    return ma


# ─────────────────────────────────────────────────────────────────────────────
# 누적 trajectory marker — 스플라인 보간으로 부드러운 곡선
# ─────────────────────────────────────────────────────────────────────────────
def make_trajectory_marker(positions, frame_id, stamp, interp_pts=8):
    ma = MarkerArray()
    pts = np.array(positions, dtype=np.float64)

    if len(pts) >= 2:
        try:
            from scipy.interpolate import CubicSpline
            t_param = np.zeros(len(pts))
            for i in range(1, len(pts)):
                t_param[i] = t_param[i-1] + np.linalg.norm(pts[i] - pts[i-1])
            if t_param[-1] > 1e-6:
                _, uniq = np.unique(t_param, return_index=True)
                if len(uniq) >= 2:
                    cs = CubicSpline(t_param[uniq], pts[uniq])
                    t_fine = np.linspace(t_param[uniq[0]], t_param[uniq[-1]],
                                         max(2, (len(uniq) - 1) * interp_pts + 1))
                    pts = cs(t_fine)
        except ImportError:
            pass

    m_line = Marker()
    m_line.header.frame_id = frame_id
    m_line.header.stamp = stamp
    m_line.ns = "trajectory"
    m_line.id = 0
    m_line.type = Marker.LINE_STRIP
    m_line.action = Marker.ADD
    m_line.scale.x = 0.06
    m_line.color.r = 0.2
    m_line.color.g = 0.5
    m_line.color.b = 1.0
    m_line.color.a = 0.9
    for p3 in pts:
        p = Point(); p.x, p.y, p.z = float(p3[0]), float(p3[1]), float(p3[2])
        m_line.points.append(p)
    ma.markers.append(m_line)

    m_sph = Marker()
    m_sph.header.frame_id = frame_id
    m_sph.header.stamp = stamp
    m_sph.ns = "trajectory"
    m_sph.id = 1
    m_sph.type = Marker.SPHERE_LIST
    m_sph.action = Marker.ADD
    m_sph.scale.x = m_sph.scale.y = m_sph.scale.z = 0.15
    m_sph.color.r = 0.0
    m_sph.color.g = 0.6
    m_sph.color.b = 1.0
    m_sph.color.a = 1.0
    for pos in positions:
        p = Point(); p.x, p.y, p.z = float(pos[0]), float(pos[1]), float(pos[2])
        m_sph.points.append(p)
    ma.markers.append(m_sph)

    return ma


# ─────────────────────────────────────────────────────────────────────────────
# FOV footprint marker — 카메라 FOV를 바닥면에 투영한 2D 사다리꼴
# ─────────────────────────────────────────────────────────────────────────────
def make_view_footprint_marker(T_WC, fx, fy, cx, cy, W, H,
                               depth, frame_id, stamp,
                               color=(1.0, 0.9, 0.0), line_width=0.06):
    """
    카메라 앞 depth 거리의 far-plane 사각형(4코너)을 LINE_LIST로 생성.
    이미지 4개 모서리를 카메라 공간에서 depth 만큼 투영 → world 변환.
    """
    R = T_WC[:3, :3]
    t = T_WC[:3, 3]

    corners_img = [(0, 0), (W-1, 0), (W-1, H-1), (0, H-1)]
    corners_world = []
    for u, v in corners_img:
        # 카메라 공간에서 depth 거리의 3D 점
        pt_cam = np.array([(u - cx) / fx * depth,
                           (v - cy) / fy * depth,
                           depth])
        pt_world = R @ pt_cam + t
        corners_world.append(pt_world)

    c0, c1, c2, c3 = corners_world

    # 코너 4개를 이어 닫힌 사각형
    segments = [
        (c0, c1), (c1, c2), (c2, c3), (c3, c0),
    ]

    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp    = stamp
    m.ns    = "view_footprint"
    m.id    = 0
    m.type  = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = line_width
    m.color.r, m.color.g, m.color.b, m.color.a = color[0], color[1], color[2], 0.85
    m.pose.orientation.w = 1.0
    for a, b in segments:
        for pt in (a, b):
            m.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2])))

    ma = MarkerArray()
    ma.markers.append(m)
    return ma


# ─────────────────────────────────────────────────────────────────────────────
# Rotation matrix → quaternion
# ─────────────────────────────────────────────────────────────────────────────
def rot2quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s, 0.25/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s, (R[1,0]-R[0,1])/s


# ─────────────────────────────────────────────────────────────────────────────
# Main Node
# ─────────────────────────────────────────────────────────────────────────────
class MapLocalizerViewer(Node):

    def __init__(self):
        super().__init__('map_localizer_viewer')

        base = '/home/park/loc_ws/src/render_loc'
        out  = os.path.join(base, 'output/MegaLoc')

        self.declare_parameter('aligned_ply',
            os.path.join(out, 'aligned_map.ply'))
        self.declare_parameter('vis_ply', '')   # 0.1m pre-downsampled PLY (없으면 aligned_ply에서 계산)
        self.declare_parameter('bg_ply',  '')   # 0.4m pre-downsampled PLY (없으면 aligned_ply에서 계산)
        self.declare_parameter('trajectory_json',
            os.path.join(out, 'trajectory_poses.json'))
        self.declare_parameter('images_dir',
            os.path.join(base, 'test_data/cam_1/images'))
        self.declare_parameter('fps',            5.0)
        self.declare_parameter('frame_id',       'map')
        self.declare_parameter('map_voxel',      0.3)   # 배경 맵 다운샘플 (빡세게)
        self.declare_parameter('fx',             1028.487260)
        self.declare_parameter('fy',             1030.283620)
        self.declare_parameter('cx',             949.476264)
        self.declare_parameter('cy',             597.274302)
        self.declare_parameter('img_width',      1920)
        self.declare_parameter('img_height',     1200)
        self.declare_parameter('depth_max',      20.0)
        self.declare_parameter('frustum_depth',  2.0)
        self.declare_parameter('frustum_grid_w', 80)
        self.declare_parameter('frustum_grid_h', 50)
        self.declare_parameter('view_footprint_range', 15.0)  # FOV footprint 투영 거리 (m)
        self.declare_parameter('view_footprint_floor_z', 0.0) # 바닥면 z 높이 (m)
        self.declare_parameter('step6_results_dir', '')  # test_results/cam_X 경로 (비워두면 스킵)
        self.declare_parameter('match_line_alpha',  0.2) # 연두색 라인 투명도 (0.0~1.0)
        self.declare_parameter('live_image_topic', '')   # 설정 시 subscriber 동기화 모드
        self.declare_parameter('live_pose_topic',  '/camera_pose')
        self.declare_parameter('sync_slop',        0.15) # ApproximateTimeSynchronizer slop (초)

        ply_path   = self.get_parameter('aligned_ply').value
        vis_ply    = self.get_parameter('vis_ply').value
        bg_ply     = self.get_parameter('bg_ply').value
        traj_path  = self.get_parameter('trajectory_json').value
        self.img_dir      = self.get_parameter('images_dir').value
        self.fps          = float(self.get_parameter('fps').value)
        self.frame_id     = self.get_parameter('frame_id').value
        map_voxel         = float(self.get_parameter('map_voxel').value)
        self.fx           = float(self.get_parameter('fx').value)
        self.fy           = float(self.get_parameter('fy').value)
        self.cx           = float(self.get_parameter('cx').value)
        self.cy           = float(self.get_parameter('cy').value)
        self.img_W        = int(self.get_parameter('img_width').value)
        self.img_H        = int(self.get_parameter('img_height').value)
        self.depth_max    = float(self.get_parameter('depth_max').value)
        self.frust_d      = float(self.get_parameter('frustum_depth').value)
        self.frust_grid_w = max(2, int(self.get_parameter('frustum_grid_w').value))
        self.frust_grid_h = max(2, int(self.get_parameter('frustum_grid_h').value))
        self._fp_range    = float(self.get_parameter('view_footprint_range').value)
        self._fp_floor_z  = float(self.get_parameter('view_footprint_floor_z').value)
        step6_results_dir       = self.get_parameter('step6_results_dir').value
        self._match_line_alpha  = float(self.get_parameter('match_line_alpha').value)
        live_image_topic        = self.get_parameter('live_image_topic').value
        live_pose_topic    = self.get_parameter('live_pose_topic').value
        self._sync_slop    = float(self.get_parameter('sync_slop').value)
        self._live_mode    = bool(live_image_topic)

        # ── PLY 로드 ──────────────────────────────────────────────────
        # vis_ply + bg_ply 둘 다 있으면 aligned_ply(100M pts) 로드 생략
        vis_ply_ok = bool(vis_ply and os.path.exists(vis_ply))
        bg_ply_ok  = bool(bg_ply  and os.path.exists(bg_ply))

        if vis_ply_ok and bg_ply_ok:
            self.get_logger().info("vis_ply + bg_ply 모두 존재 → aligned_ply 로드 생략")
            pcd_full = None
        else:
            self.get_logger().info(f"Loading PLY: {ply_path}")
            pcd_full = o3d.io.read_point_cloud(ply_path)
            self.get_logger().info(f"Full map: {len(pcd_full.points)} points")

        # 가시 영역용 map — frustum culling + visible_region 발행
        if vis_ply_ok:
            self.get_logger().info(f"Loading vis PLY: {vis_ply}")
            pcd_vis = o3d.io.read_point_cloud(vis_ply)
        else:
            self.get_logger().info("vis_ply 없음 → aligned_ply에서 0.1m downsample 계산 중...")
            pcd_vis = pcd_full.voxel_down_sample(0.1)
        N_vis = len(pcd_vis.points)
        self.get_logger().info(f"Visible map: {N_vis} points")
        self.map_xyz = np.asarray(pcd_vis.points, dtype=np.float32)
        if pcd_vis.has_colors():
            self.map_rgb_orig_ = (np.asarray(pcd_vis.colors) * 255).astype(np.uint8)
        else:
            self.map_rgb_orig_ = np.full((N_vis, 3), 160, dtype=np.uint8)

        self.map_rgb_orig = np.clip(
            self.map_rgb_orig_.astype(np.int32) * 120 // 100, 0, 255
        ).astype(np.uint8)

        # 배경 맵 — /map_cloud 용
        if bg_ply_ok:
            self.get_logger().info(f"Loading bg PLY: {bg_ply}")
            pcd_bg = o3d.io.read_point_cloud(bg_ply)
        else:
            pcd_bg = pcd_full.voxel_down_sample(map_voxel) if map_voxel > 0 else pcd_full
        N_bg = len(pcd_bg.points)
        self.get_logger().info(f"Background map: {N_bg} points (voxel={map_voxel}m)")

        bg_xyz = np.asarray(pcd_bg.points, dtype=np.float32)
        if pcd_bg.has_colors():
            bg_rgb_orig = (np.asarray(pcd_bg.colors) * 255).astype(np.uint8)
        else:
            bg_rgb_orig = np.full((N_bg, 3), 160, dtype=np.uint8)

        # 배경 맵 중간 밝기 (원본의 85%)
        bg_rgb_mid = np.clip(
            bg_rgb_orig.astype(np.int32) * 80 // 100, 40, 220
        ).astype(np.uint8)

        # ── Trajectory 로드 ────────────────────────────────────────────
        with open(traj_path) as f:
            traj_raw = json.load(f)

        self.frames = []
        for fname, mat in sorted(traj_raw.items()):
            T = np.array(mat, dtype=np.float64)
            img_path = os.path.join(self.img_dir, fname)
            self.frames.append({
                'fname': fname,
                'T': T,
                'img': img_path if os.path.exists(img_path) else None,
            })
        self.get_logger().info(f"Loaded {len(self.frames)} frames from {traj_path}")

        # ── TF broadcaster (map → cam) — RViz FPS 뷰가 이 프레임을 추적 ──
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ── step6 매칭 3D 포인트 per-frame pre-load ────────────────────
        # 구조: {step6_results_dir}/{fname_stem}/step6_data.pkl
        self.step6_per_frame = {}   # fname → np.ndarray (N,3) world coords or None
        if step6_results_dir and os.path.isdir(step6_results_dir):
            import pickle
            dmin, dmax = 0.3, self.depth_max
            loaded = 0
            for frame in self.frames:
                stem = os.path.splitext(frame['fname'])[0]
                pkl_path = os.path.join(step6_results_dir, stem, 'step6_data.pkl')
                if not os.path.exists(pkl_path):
                    continue
                s6 = pickle.load(open(pkl_path, 'rb'))
                matched_r = s6.get('matched_r_kps', np.zeros((0, 2)))
                ref_entry = s6.get('best_cand', {})
                dep_path  = ref_entry.get('depth_path', '')
                pose_raw  = ref_entry.get('pose', None)
                # depth_path가 상대 경로인 경우 render_loc base 기준으로 절대 경로 변환
                if dep_path and not os.path.isabs(dep_path):
                    dep_path = os.path.join(base, dep_path)
                if len(matched_r) == 0 or not dep_path or not os.path.exists(dep_path) or pose_raw is None:
                    continue
                dep  = np.load(dep_path)
                T_WR = np.array(pose_raw, dtype=np.float64)
                R_WR = T_WR[:3, :3];  t_WR = T_WR[:3, 3]
                pts3d = []
                for ru, rv in matched_r:
                    ri, rj = int(round(rv)), int(round(ru))
                    if not (0 <= ri < dep.shape[0] and 0 <= rj < dep.shape[1]):
                        continue
                    pz = float(dep[ri, rj])
                    if not np.isfinite(pz) or pz < dmin or pz > dmax:
                        continue
                    pt_cam = np.array([(ru - self.cx) * pz / self.fx,
                                       (rv - self.cy) * pz / self.fy, pz])
                    pts3d.append(R_WR @ pt_cam + t_WR)
                if pts3d:
                    self.step6_per_frame[frame['fname']] = np.array(pts3d, dtype=np.float32)
                    loaded += 1
            self.get_logger().info(
                f"step6 pre-loaded: {loaded}/{len(self.frames)} frames  "
                f"(dir={step6_results_dir})")
        elif step6_results_dir:
            self.get_logger().warn(f"step6_results_dir 없음: {step6_results_dir}")

        # ── Publishers ────────────────────────────────────────────────
        self.pub_map        = self.create_publisher(PointCloud2, '/map_cloud',        LATCHING)
        self.pub_visible    = self.create_publisher(PointCloud2, '/visible_region',  VOLATILE)
        self.pub_visited    = self.create_publisher(PointCloud2, '/visited_region',  LATCHING)
        self.pub_matching   = self.create_publisher(PointCloud2, '/matching_points', VOLATILE)
        self.pub_match_lines= self.create_publisher(MarkerArray, '/matching_lines',  VOLATILE)
        self.pub_bbox       = self.create_publisher(MarkerArray, '/visible_bbox',    VOLATILE)
        self.pub_footprint  = self.create_publisher(MarkerArray, '/view_footprint',  VOLATILE)
        # frustum wireframe + image → 하나의 MarkerArray로 동시 발행
        self.pub_frustum    = self.create_publisher(MarkerArray, '/camera_frustum',  VOLATILE)
        self.pub_pose       = self.create_publisher(PoseStamped, '/camera_pose',     VOLATILE)
        self.pub_traj       = self.create_publisher(MarkerArray, '/trajectory_line', VOLATILE)
        self.pub_image      = self.create_publisher(Image,       '/cam/image',       VOLATILE)

        # ── 누적 방문 영역 마스크 (vis_ply 인덱스 기준) ──────────────
        self.visited_mask = np.zeros(len(self.map_xyz), dtype=bool)

        # ── 전체 맵 1회 발행 (latching, 85% 밝기, super downsample) ──
        stamp0 = self.get_clock().now().to_msg()
        self.pub_map.publish(
            make_pointcloud2(bg_xyz, bg_rgb_mid, self.frame_id, stamp0))
        self.get_logger().info(
            f"Published /map_cloud (latching, 85% brightness, {N_bg} pts)")


        self.traj_positions = []
        self.frame_idx = 0

        if self._live_mode:
            # ── Subscriber 동기화 모드 ─────────────────────────────────
            # image + pose가 같은 프레임 타임스탬프로 도착했을 때만 publish
            sensor_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=10)
            img_sub  = message_filters.Subscriber(
                self, Image, live_image_topic, qos_profile=sensor_qos)
            pose_sub = message_filters.Subscriber(
                self, PoseStamped, live_pose_topic)
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [img_sub, pose_sub], queue_size=10, slop=self._sync_slop)
            self._sync.registerCallback(self._synced_callback)
            self.get_logger().info(
                f"Live sync mode: {live_image_topic} + {live_pose_topic}  "
                f"slop={self._sync_slop}s  "
                f"(frustum_depth={self.frust_d}m, "
                f"frustum_grid={self.frust_grid_w}x{self.frust_grid_h})")
        else:
            # ── Timer 기반 파일 replay 모드 ────────────────────────────
            self.timer = self.create_timer(1.0 / max(self.fps, 0.1), self._tick)
            self.get_logger().info(
                f"Replay started at {self.fps} fps  "
                f"(frustum_depth={self.frust_d}m, "
                f"frustum_grid={self.frust_grid_w}x{self.frust_grid_h})")

    # ─────────────────────────────────────────────────────────────────
    # 공통 compute + publish 헬퍼
    # ─────────────────────────────────────────────────────────────────
    def _publish_frame(self, T_WC, img_rgb, fname, stamp):
        """모든 데이터 계산 완료 후 동일 stamp로 일괄 publish."""
        R = T_WC[:3, :3]
        t = T_WC[:3, 3]
        fid = self.frame_id

        # ── 1단계: 모든 데이터 계산 ───────────────────────────────────

        # 가시 영역
        vis_idx = frustum_cull(
            self.map_xyz, T_WC,
            self.fx, self.fy, self.cx, self.cy,
            self.img_W, self.img_H, self.depth_max)
        if len(vis_idx) > 0:
            vis_xyz = self.map_xyz[vis_idx]
            vis_rgb = self.map_rgb_orig[vis_idx]
        else:
            vis_xyz = np.zeros((0, 3), np.float32)
            vis_rgb = np.zeros((0, 3), np.uint8)

        # 누적 방문 영역 — 새 점이 생길 때만 latching 재발행
        if len(vis_idx) > 0:
            newly_visible = ~self.visited_mask[vis_idx]
            if np.any(newly_visible):
                self.visited_mask[vis_idx] = True
                acc_xyz = self.map_xyz[self.visited_mask]
                acc_rgb = self.map_rgb_orig[self.visited_mask]
                self.pub_visited.publish(
                    make_pointcloud2(acc_xyz, acc_rgb, fid, stamp))

        # 쿼터니언
        qx, qy, qz, qw = rot2quat(R)

        # cam_view 쿼터니언 (광학→ThirdPersonFollower 방향 보정)
        CX, CY, CZ, CW = 0.5, -0.5, 0.5, 0.5
        vx = qw*CX + qx*CW + qy*CZ - qz*CY
        vy = qw*CY - qx*CZ + qy*CW + qz*CX
        vz = qw*CZ + qx*CY - qy*CX + qz*CW
        vw = qw*CW - qx*CX - qy*CY - qz*CZ

        # cam_fp 위치 (frustum 끝 중심)
        fp_world = R @ np.array([0.0, 0.0, self.frust_d]) + t

        # step6 매칭 포인트 — 1540×900 중심 영역 내 + 50% 랜덤 샘플
        pts3d_match = self.step6_per_frame.get(fname) if fname else None
        if pts3d_match is not None and len(pts3d_match) > 0:
            # 카메라 좌표로 투영해서 1540×900 crop 영역 필터링
            crop_w, crop_h = 1540, 900
            u_min = (self.img_W - crop_w) / 2   # 190
            v_min = (self.img_H - crop_h) / 2   # 150
            u_max = u_min + crop_w               # 1730
            v_max = v_min + crop_h               # 1050
            T_CW  = np.linalg.inv(T_WC)
            pts_c = (T_CW[:3, :3] @ pts3d_match.T).T + T_CW[:3, 3]
            front = pts_c[:, 2] > 0.1
            u_p   = np.where(front, pts_c[:, 0] / np.maximum(pts_c[:, 2], 1e-6) * self.fx + self.cx, -1)
            v_p   = np.where(front, pts_c[:, 1] / np.maximum(pts_c[:, 2], 1e-6) * self.fy + self.cy, -1)
            in_crop = front & (u_p >= u_min) & (u_p < u_max) & (v_p >= v_min) & (v_p < v_max)
            pts_crop = pts3d_match[in_crop]
            # 50% 랜덤 샘플
            n = len(pts_crop)
            if n > 0:
                rng_idx  = np.random.choice(n, size=max(1, n // 2), replace=False)
                match_xyz = pts_crop[rng_idx]
            else:
                match_xyz = np.zeros((0, 3), np.float32)
            match_rgb = np.tile(np.array([255, 0, 0], dtype=np.uint8), (len(match_xyz), 1))
        else:
            match_xyz = np.zeros((0, 3), np.float32)
            match_rgb = np.zeros((0, 3), np.uint8)

        # trajectory 위치 누적
        self.traj_positions.append(t.copy())

        # frustum image TRIANGLE_LIST (무거운 연산)
        frust_img_data = _build_frustum_image_data(
            T_WC, self.fx, self.fy, self.cx, self.cy,
            self.img_W, self.img_H, self.frust_d, img_rgb,
            grid_w=self.frust_grid_w, grid_h=self.frust_grid_h)

        # ── 2단계: 모든 계산 완료 후 동일 stamp로 일괄 publish ─────────

        def _tf(child, tx, ty, tz, rx, ry, rz, rw):
            msg = TransformStamped()
            msg.header.stamp = stamp;  msg.header.frame_id = fid
            msg.child_frame_id = child
            msg.transform.translation.x = float(tx)
            msg.transform.translation.y = float(ty)
            msg.transform.translation.z = float(tz)
            msg.transform.rotation.x = float(rx)
            msg.transform.rotation.y = float(ry)
            msg.transform.rotation.z = float(rz)
            msg.transform.rotation.w = float(rw)
            return msg

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = fid;  pose_msg.header.stamp = stamp
        pose_msg.pose.position.x    = float(t[0])
        pose_msg.pose.position.y    = float(t[1])
        pose_msg.pose.position.z    = float(t[2])
        pose_msg.pose.orientation.x = float(qx)
        pose_msg.pose.orientation.y = float(qy)
        pose_msg.pose.orientation.z = float(qz)
        pose_msg.pose.orientation.w = float(qw)

        # frustum 원점 → 빨간 점 연두색 라인 마커
        ml = Marker()
        ml.header.frame_id = fid;  ml.header.stamp = stamp
        ml.ns = "matching_lines";  ml.id = 0
        ml.type   = Marker.LINE_LIST
        ml.action = Marker.ADD
        ml.scale.x = 0.015          # 얇은 선
        ml.color.r = 0.4;  ml.color.g = 1.0;  ml.color.b = 0.2;  ml.color.a = self._match_line_alpha
        ml.pose.orientation.w = 1.0
        if len(match_xyz) > 0:
            origin_pt = Point(x=float(t[0]), y=float(t[1]), z=float(t[2]))
            for mp in match_xyz:
                ml.points.append(origin_pt)
                ml.points.append(Point(x=float(mp[0]), y=float(mp[1]), z=float(mp[2])))
        match_lines_ma = MarkerArray(); match_lines_ma.markers.append(ml)

        # frustum wireframe + image를 하나의 MarkerArray로 묶어 동시 발행 (동기화 보장)
        frustum_combined = MarkerArray()
        wireframe_ma = make_frustum_marker(
            T_WC, self.fx, self.fy, self.cx, self.cy,
            self.img_W, self.img_H, self.frust_d, fid, stamp)
        frustum_combined.markers.extend(wireframe_ma.markers)
        frust_img_ma = _make_frustum_image_marker(frust_img_data, fid, stamp)
        frustum_combined.markers.extend(frust_img_ma.markers)

        self.pub_frustum.publish(frustum_combined)
        self.pub_matching.publish(make_pointcloud2(match_xyz, match_rgb, fid, stamp))
        self.pub_match_lines.publish(match_lines_ma)
        self.pub_visible.publish(make_pointcloud2(vis_xyz,   vis_rgb,   fid, stamp))
        self.pub_bbox.publish(make_bbox_marker(vis_xyz, fid, stamp))
        self.pub_footprint.publish(
            make_view_footprint_marker(
                T_WC, self.fx, self.fy, self.cx, self.cy,
                self.img_W, self.img_H, self.frust_d, fid, stamp))
        self.pub_pose.publish(pose_msg)
        self.pub_traj.publish(make_trajectory_marker(self.traj_positions, fid, stamp))
        if img_rgb is not None:
            self.pub_image.publish(make_image_msg(img_rgb, fid, stamp))
        self.tf_broadcaster.sendTransform([
            _tf('cam',      t[0], t[1], t[2], qx, qy, qz, qw),
            _tf('cam_view', t[0], t[1], t[2], vx, vy, vz, vw),
            _tf('cam_fp',   fp_world[0], fp_world[1], fp_world[2], vx, vy, vz, vw),
        ])
        self.get_logger().info(f"published frame  visible={len(vis_idx)}  match={len(match_xyz)}")

    # ─────────────────────────────────────────────────────────────────
    # Live 모드: image + pose 동기화 callback
    # ─────────────────────────────────────────────────────────────────
    def _synced_callback(self, img_msg: Image, pose_msg: PoseStamped):
        """image + pose가 같은 프레임으로 도착했을 때만 호출됨."""
        # pose → T_WC
        p = pose_msg.pose
        from scipy.spatial.transform import Rotation
        q = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        R = Rotation.from_quat(q).as_matrix()
        t = np.array([p.position.x, p.position.y, p.position.z])
        T_WC = np.eye(4, dtype=np.float64)
        T_WC[:3, :3] = R
        T_WC[:3, 3]  = t

        # image msg → RGB numpy
        arr = np.frombuffer(img_msg.data, dtype=np.uint8)
        if img_msg.encoding == 'bgr8':
            img_rgb = arr.reshape(img_msg.height, img_msg.width, 3)[:, :, ::-1].copy()
        elif img_msg.encoding == 'rgb8':
            img_rgb = arr.reshape(img_msg.height, img_msg.width, 3).copy()
        else:
            img_rgb = None

        # 동기화 완료 시점의 현재 시각 사용 → 모든 메시지 동일 stamp로 RViz에 동시 표시
        stamp = self.get_clock().now().to_msg()
        self._publish_frame(T_WC, img_rgb, fname=None, stamp=stamp)

    # ─────────────────────────────────────────────────────────────────
    # Replay 모드: timer callback
    # ─────────────────────────────────────────────────────────────────
    def _tick(self):
        if self.frame_idx >= len(self.frames):
            self.get_logger().info("Replay finished.")
            self.timer.cancel()
            return

        frame = self.frames[self.frame_idx]
        self.frame_idx += 1

        # RGB 이미지 로드
        img_rgb = None
        if frame['img'] is not None:
            bgr = cv2.imread(frame['img'])
            if bgr is not None:
                img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        stamp = self.get_clock().now().to_msg()
        self._publish_frame(frame['T'], img_rgb, frame['fname'], stamp)
        self.get_logger().info(
            f"[{self.frame_idx}/{len(self.frames)}] {frame['fname']}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = MapLocalizerViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
