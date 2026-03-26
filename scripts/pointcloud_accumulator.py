#!/usr/bin/env python3
"""
PointCloud Accumulator + Frustum Visualizer
============================================
localization 결과 trajectory를 이용해 RGB-D 프레임을 누적하여 포인트클라우드를 생성하고
RViz2에서 시각화합니다.

Usage:
  python3 scripts/pointcloud_accumulator.py --ros-args \
      -p trajectory_json:=/path/to/trajectory_poses.json \
      -p images_dir:=/path/to/cam_3/images \
      -p depths_dir:=/path/to/cam_3/depths \
      -p fps:=5.0

또는 launch 파일:
  ros2 launch render_loc pointcloud_accumulator.launch.py
"""

import os, sys, json, time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import PointCloud2, PointField, CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
import struct


# ─────────────────────────────────────────────────────────────────────────────
# Helper: numpy array → PointCloud2 (XYZ + RGB)
# ─────────────────────────────────────────────────────────────────────────────
def make_pointcloud2(xyz, rgb, frame_id, stamp):
    """
    xyz : (N, 3) float32
    rgb : (N, 3) uint8
    """
    N = len(xyz)
    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 16
    data = bytearray(N * point_step)
    for i in range(N):
        offset = i * point_step
        struct.pack_into('fff', data, offset,
                         float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]))
        r, g, b = int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])
        rgb_int = (r << 16) | (g << 8) | b
        struct.pack_into('I', data, offset + 12, rgb_int)

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = N
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * N
    msg.data = bytes(data)
    msg.is_dense = True
    return msg


def make_pointcloud2_numpy(xyz, rgb, frame_id, stamp):
    """numpy 벡터화 버전 (훨씬 빠름)"""
    N = len(xyz)
    if N == 0:
        msg = PointCloud2()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp
        return msg

    xyz_f = xyz.astype(np.float32)
    r = rgb[:, 0].astype(np.uint32)
    g = rgb[:, 1].astype(np.uint32)
    b = rgb[:, 2].astype(np.uint32)
    rgb_packed = ((r << 16) | (g << 8) | b).view(np.float32)

    data = np.zeros(N, dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('rgb', np.float32)
    ])
    data['x'] = xyz_f[:, 0]
    data['y'] = xyz_f[:, 1]
    data['z'] = xyz_f[:, 2]
    data['rgb'] = rgb_packed

    fields = [
        PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = N
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * N
    msg.data = data.tobytes()
    msg.is_dense = True
    return msg


def voxel_downsample(xyz, rgb, voxel_size):
    """간단한 voxel 다운샘플링"""
    if len(xyz) == 0:
        return xyz, rgb
    vox_idx = np.floor(xyz / voxel_size).astype(np.int32)
    keys = vox_idx[:, 0] * 1000003 + vox_idx[:, 1] * 1000033 + vox_idx[:, 2]
    _, uniq = np.unique(keys, return_index=True)
    return xyz[uniq], rgb[uniq]


def make_frustum_markers(T_WC, fx, fy, cx, cy, w, h,
                         depth, frame_id, stamp, ns="frustum",
                         color=(0.0, 1.0, 0.0), line_width=0.05):
    """
    카메라 포즈에서 frustum MarkerArray (LINE_LIST) 생성
    4 corner rays + image plane rectangle
    """
    # 이미지 4 코너 → 카메라 좌표계
    corners_cam = np.array([
        [(0   - cx) / fx * depth, (0   - cy) / fy * depth, depth],
        [(w-1 - cx) / fx * depth, (0   - cy) / fy * depth, depth],
        [(w-1 - cx) / fx * depth, (h-1 - cy) / fy * depth, depth],
        [(0   - cx) / fx * depth, (h-1 - cy) / fy * depth, depth],
    ], dtype=np.float64)

    # 카메라 원점 (world)
    origin = T_WC[:3, 3]
    R = T_WC[:3, :3]

    # 코너 → world
    corners_world = (R @ corners_cam.T).T + origin

    # LINE_LIST: origin→corner (4개) + rectangle (4개)
    lines = []
    for c in corners_world:
        lines += [origin, c]       # ray
    for i in range(4):
        lines += [corners_world[i], corners_world[(i+1) % 4]]  # rect edges

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = ns
    marker.id = 0
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = line_width
    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = 1.0

    from geometry_msgs.msg import Point
    for pt in lines:
        p = Point()
        p.x, p.y, p.z = float(pt[0]), float(pt[1]), float(pt[2])
        marker.points.append(p)

    ma = MarkerArray()
    ma.markers.append(marker)
    return ma


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────
class PointCloudAccumulator(Node):

    def __init__(self):
        super().__init__('pointcloud_accumulator')

        # ── 파라미터 ───────────────────────────────────────────────────
        self.declare_parameter('trajectory_json',
            '/home/park/loc_ws/src/render_loc/output/step_by_step/trajectory_poses.json')
        self.declare_parameter('images_dir',
            '/home/park/loc_ws/src/render_loc/test_data/records_data/cam_3/images')
        self.declare_parameter('depths_dir',
            '/home/park/loc_ws/src/render_loc/test_data/records_data/cam_3/depths')
        self.declare_parameter('fps',           10.0)
        self.declare_parameter('frame_id',      'map')
        self.declare_parameter('fx',            910.0)
        self.declare_parameter('fy',            910.0)
        self.declare_parameter('cx',            960.0)
        self.declare_parameter('cy',            540.0)
        self.declare_parameter('depth_max',     50.0)   # 유효 depth 상한 (m)
        self.declare_parameter('depth_scale',   255.0)  # 8-bit vis용 스케일 (float32 .depth는 무시)
        self.declare_parameter('frustum_depth', 3.0)    # frustum 시각화 깊이 (m)
        self.declare_parameter('voxel_size',    0.05)   # 다운샘플 voxel (m), 0=비활성
        self.declare_parameter('stride',        4)      # depth pixel 샘플링 stride
        self.declare_parameter('max_points',    5_000_000)  # 누적 최대 포인트 수

        traj_path   = self.get_parameter('trajectory_json').value
        self.img_dir    = self.get_parameter('images_dir').value
        self.dep_dir    = self.get_parameter('depths_dir').value
        self.fps        = float(self.get_parameter('fps').value)
        self.frame_id   = self.get_parameter('frame_id').value
        self.fx         = float(self.get_parameter('fx').value)
        self.fy         = float(self.get_parameter('fy').value)
        self.cx         = float(self.get_parameter('cx').value)
        self.cy         = float(self.get_parameter('cy').value)
        self.depth_max  = float(self.get_parameter('depth_max').value)
        self.depth_scale= float(self.get_parameter('depth_scale').value)
        self.frust_d    = float(self.get_parameter('frustum_depth').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.stride     = int(self.get_parameter('stride').value)
        self.max_pts    = int(self.get_parameter('max_points').value)

        # ── trajectory 로드 ────────────────────────────────────────────
        with open(traj_path) as f:
            traj_raw = json.load(f)

        # {filename: 4x4 matrix} → sorted by filename
        self.frames = []
        for fname, mat in sorted(traj_raw.items()):
            T = np.array(mat, dtype=np.float64)
            stem = os.path.splitext(fname)[0]
            img_path = os.path.join(self.img_dir, fname)

            # depth 우선순위: float32 .depth > 16-bit PNG > 8-bit vis PNG
            dep_path, dep_type = None, None
            for candidate, dtype in [
                (os.path.join(self.dep_dir, f"{stem}.depth"),    "float32"),
                (os.path.join(self.dep_dir, f"{stem}.png"),      "png16"),
                (os.path.join(self.dep_dir, f"{stem}_vis.png"),  "vis8"),
            ]:
                if os.path.exists(candidate):
                    dep_path, dep_type = candidate, dtype
                    break

            if os.path.exists(img_path) and dep_path is not None:
                self.frames.append({'fname': fname, 'T': T,
                                    'img': img_path, 'dep': dep_path,
                                    'dep_type': dep_type})

        dep_types = set(f['dep_type'] for f in self.frames)
        self.get_logger().info(
            f"Loaded {len(self.frames)} frames from {traj_path}  "
            f"depth format: {dep_types}")

        # ── 누적 포인트클라우드 버퍼 ────────────────────────────────────
        self.acc_xyz = np.zeros((0, 3), dtype=np.float32)
        self.acc_rgb = np.zeros((0, 3), dtype=np.uint8)

        # ── Publishers ────────────────────────────────────────────────
        latching = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        volatile = QoSProfile(depth=5)

        self.pub_accum   = self.create_publisher(PointCloud2,  '/accumulated_cloud', latching)
        self.pub_current = self.create_publisher(PointCloud2,  '/current_frame_cloud', volatile)
        self.pub_frustum = self.create_publisher(MarkerArray,  '/camera_frustum', volatile)
        self.pub_pose    = self.create_publisher(PoseStamped,  '/camera_pose', volatile)
        self.pub_image   = self.create_publisher(Image,        '/cam_3/image', volatile)

        # ── 재생 타이머 ───────────────────────────────────────────────
        self.frame_idx = 0
        interval = 1.0 / max(self.fps, 0.1)
        self.timer = self.create_timer(interval, self._tick)
        self.get_logger().info(
            f"Replaying at {self.fps} fps  (stride={self.stride}, "
            f"voxel={self.voxel_size}m)")

    # ─────────────────────────────────────────────────────────────────
    def _tick(self):
        if self.frame_idx >= len(self.frames):
            self.get_logger().info("Replay finished.")
            self.timer.cancel()
            return

        frame = self.frames[self.frame_idx]
        self.frame_idx += 1
        stamp = self.get_clock().now().to_msg()

        T_WC = frame['T']

        # ── RGB 로드 ─────────────────────────────────────────────────
        img_bgr = cv2.imread(frame['img'])
        if img_bgr is None:
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        # ── Depth 로드 ────────────────────────────────────────────────
        dep_type = frame['dep_type']
        if dep_type == "float32":
            # raw float32 binary (값 단위: 미터)
            dep_metric = np.fromfile(frame['dep'], dtype=np.float32).reshape(H, W)
        elif dep_type == "png16":
            # 16-bit PNG (값 / depth_scale = 미터)
            dep_raw = cv2.imread(frame['dep'], cv2.IMREAD_UNCHANGED)
            if dep_raw is None:
                return
            dep_metric = dep_raw.astype(np.float32) / self.depth_scale
        else:  # vis8: 8-bit 시각화 (0~255 → 0~depth_max)
            dep_raw = cv2.imread(frame['dep'], cv2.IMREAD_UNCHANGED)
            if dep_raw is None:
                return
            if dep_raw.ndim == 3:
                dep_raw = dep_raw[:, :, 0]
            dep_metric = dep_raw.astype(np.float32) / 255.0 * self.depth_max

        # ── 현재 프레임 포인트클라우드 생성 ──────────────────────────
        cur_xyz, cur_rgb = self._unproject(dep_metric, img_rgb, T_WC)

        # ── 누적 ─────────────────────────────────────────────────────
        if len(cur_xyz) > 0:
            self.acc_xyz = np.concatenate([self.acc_xyz, cur_xyz], axis=0)
            self.acc_rgb = np.concatenate([self.acc_rgb, cur_rgb], axis=0)

            # voxel 다운샘플
            if self.voxel_size > 0 and len(self.acc_xyz) > 50_000:
                self.acc_xyz, self.acc_rgb = voxel_downsample(
                    self.acc_xyz, self.acc_rgb, self.voxel_size)

            # 최대 포인트 수 제한 (오래된 것 제거)
            if len(self.acc_xyz) > self.max_pts:
                keep = len(self.acc_xyz) - self.max_pts
                self.acc_xyz = self.acc_xyz[keep:]
                self.acc_rgb = self.acc_rgb[keep:]

        # ── Publish: 누적 클라우드 ─────────────────────────────────
        self.pub_accum.publish(
            make_pointcloud2_numpy(self.acc_xyz, self.acc_rgb, self.frame_id, stamp))

        # ── Publish: 현재 프레임 클라우드 ─────────────────────────
        if len(cur_xyz) > 0:
            self.pub_current.publish(
                make_pointcloud2_numpy(cur_xyz, cur_rgb, self.frame_id, stamp))

        # ── Publish: 카메라 포즈 ───────────────────────────────────
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.frame_id
        pose_msg.header.stamp = stamp
        pos = T_WC[:3, 3]
        from scipy.spatial.transform import Rotation
        quat = Rotation.from_matrix(T_WC[:3, :3]).as_quat()
        pose_msg.pose.position.x = float(pos[0])
        pose_msg.pose.position.y = float(pos[1])
        pose_msg.pose.position.z = float(pos[2])
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])
        self.pub_pose.publish(pose_msg)

        # ── Publish: Frustum ───────────────────────────────────────
        ma = make_frustum_markers(T_WC, self.fx, self.fy, self.cx, self.cy,
                                   W, H, self.frust_d, self.frame_id, stamp)
        self.pub_frustum.publish(ma)

        # ── Publish: 이미지 ────────────────────────────────────────
        img_msg = Image()
        img_msg.header.frame_id = self.frame_id
        img_msg.header.stamp = stamp
        img_msg.height = H
        img_msg.width  = W
        img_msg.encoding = 'rgb8'
        img_msg.step = W * 3
        img_msg.data = img_rgb.tobytes()
        self.pub_image.publish(img_msg)

        self.get_logger().info(
            f"[{self.frame_idx}/{len(self.frames)}] {frame['fname']}  "
            f"cur={len(cur_xyz)} pts  acc={len(self.acc_xyz)} pts")

    # ─────────────────────────────────────────────────────────────────
    def _unproject(self, dep, rgb, T_WC):
        """depth map + RGB → world 좌표 포인트클라우드"""
        H, W = dep.shape
        s = self.stride

        # 픽셀 그리드
        v_idx, u_idx = np.meshgrid(
            np.arange(0, H, s), np.arange(0, W, s), indexing='ij')
        v_idx = v_idx.flatten()
        u_idx = u_idx.flatten()

        pz = dep[v_idx, u_idx]

        # 유효 depth 필터
        valid = (pz > 0.1) & (pz < self.depth_max * 0.99)
        v_idx = v_idx[valid]
        u_idx = u_idx[valid]
        pz    = pz[valid]

        if len(pz) == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

        # 카메라 좌표계 3D
        px = (u_idx - self.cx) * pz / self.fx
        py = (v_idx - self.cy) * pz / self.fy
        pts_cam = np.stack([px, py, pz], axis=1).astype(np.float32)  # (N,3)

        # world 좌표계 변환
        R = T_WC[:3, :3].astype(np.float32)
        t = T_WC[:3, 3].astype(np.float32)
        pts_world = (R @ pts_cam.T).T + t

        # RGB
        colors = rgb[v_idx, u_idx]  # (N, 3) uint8

        return pts_world, colors


def main():
    rclpy.init()
    node = PointCloudAccumulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
