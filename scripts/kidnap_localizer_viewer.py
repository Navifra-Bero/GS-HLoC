#!/usr/bin/env python3
"""
kidnap_localizer_viewer.py
==========================
pre-computed test_results 폴더의 결과를 인터렉티브하게 RViz2로 탐색.

사용법:
  python3 scripts/kidnap_localizer_viewer.py --ros-args \\
      -p aligned_ply:=output/MegaLoc/aligned_map.ply \\
      -p test_results_dir:=output/MegaLoc/test_results/cam_3

인덱스 입력 (터미널):
  ros2 topic pub --once /kidnap_idx std_msgs/msg/Int32 "data: 1"
  ros2 topic pub --once /kidnap_idx std_msgs/msg/Int32 "data: 42"

인덱스 목록 확인:
  ros2 topic pub --once /kidnap_idx std_msgs/msg/Int32 "data: 0"
  (0을 입력하면 콘솔에 전체 목록 출력)

퍼블리시 토픽:
  /map_cloud        (PointCloud2, latching)
  /visible_region   (PointCloud2)
  /visited_region   (PointCloud2, latching)
  /matching_points  (PointCloud2)
  /matching_lines   (MarkerArray)
  /camera_frustum   (MarkerArray)
  /camera_pose      (PoseStamped)
  /trajectory_line  (MarkerArray)
  /cam/image        (Image)
"""

import os, sys, pickle
import numpy as np
import cv2
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from std_msgs.msg import Int32
from sensor_msgs.msg import PointCloud2, PointField, Image
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

# map_localizer_viewer의 헬퍼 함수들을 import
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from map_localizer_viewer import (
    make_pointcloud2, make_image_msg,
    make_frustum_marker, make_bbox_marker,
    make_trajectory_marker, make_view_footprint_marker,
    _build_frustum_image_data, _make_frustum_image_marker,
    frustum_cull, rot2quat,
)

LATCHING = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
)
VOLATILE = QoSProfile(depth=5)


class KidnapLocalizerViewer(Node):

    def __init__(self):
        super().__init__('kidnap_localizer_viewer')

        base = '/home/park/loc_ws/src/render_loc'
        out  = os.path.join(base, 'output/MegaLoc')

        self.declare_parameter('aligned_ply',
            os.path.join(out, 'aligned_map.ply'))
        self.declare_parameter('vis_ply',  '')
        self.declare_parameter('bg_ply',   '')
        self.declare_parameter('test_results_dir',
            os.path.join(out, 'test_results/cam_3'))
        self.declare_parameter('frame_id',       'map')
        self.declare_parameter('map_voxel',      0.3)
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
        self.declare_parameter('match_line_alpha', 0.2)
        self.declare_parameter('top_down_ply', '')  # /top_down_cloud 용 (비워두면 bg_ply 재사용)
        self.declare_parameter('view_footprint_range', 15.0)
        self.declare_parameter('view_footprint_floor_z', 0.0)

        ply_path      = self.get_parameter('aligned_ply').value
        vis_ply       = self.get_parameter('vis_ply').value
        bg_ply        = self.get_parameter('bg_ply').value
        top_down_ply  = self.get_parameter('top_down_ply').value
        results_dir = self.get_parameter('test_results_dir').value
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
        self._match_line_alpha = float(self.get_parameter('match_line_alpha').value)
        self._fp_range  = float(self.get_parameter('view_footprint_range').value)
        self._fp_floor_z = float(self.get_parameter('view_footprint_floor_z').value)

        # ── PLY 로드 ──────────────────────────────────────────────────
        vis_ply_ok = bool(vis_ply and os.path.exists(vis_ply))
        bg_ply_ok  = bool(bg_ply  and os.path.exists(bg_ply))

        if vis_ply_ok and bg_ply_ok:
            pcd_full = None
        else:
            self.get_logger().info(f"Loading PLY: {ply_path}")
            pcd_full = o3d.io.read_point_cloud(ply_path)

        if vis_ply_ok:
            pcd_vis = o3d.io.read_point_cloud(vis_ply)
        else:
            pcd_vis = pcd_full.voxel_down_sample(0.1)
        N_vis = len(pcd_vis.points)
        self.map_xyz = np.asarray(pcd_vis.points, dtype=np.float32)
        raw_rgb = (np.asarray(pcd_vis.colors) * 255).astype(np.uint8) \
                  if pcd_vis.has_colors() else np.full((N_vis, 3), 160, dtype=np.uint8)
        self.map_rgb_orig = np.clip(
            raw_rgb.astype(np.int32) * 120 // 100, 0, 255).astype(np.uint8)
        self.get_logger().info(f"Visible map: {N_vis} pts")

        if bg_ply_ok:
            pcd_bg = o3d.io.read_point_cloud(bg_ply)
        else:
            pcd_bg = pcd_full.voxel_down_sample(map_voxel) if map_voxel > 0 else pcd_full
        bg_xyz = np.asarray(pcd_bg.points, dtype=np.float32)
        bg_raw = (np.asarray(pcd_bg.colors) * 255).astype(np.uint8) \
                 if pcd_bg.has_colors() else np.full((len(bg_xyz), 3), 160, dtype=np.uint8)
        bg_rgb_mid = np.clip(bg_raw.astype(np.int32) * 120 // 100, 0, 255).astype(np.uint8)
        self.get_logger().info(f"Background map: {len(bg_xyz)} pts")

        # top_down_ply: 비워두면 bg_ply 재사용
        if top_down_ply and os.path.exists(top_down_ply):
            self.get_logger().info(f"Loading top_down PLY: {top_down_ply}")
            pcd_td = o3d.io.read_point_cloud(top_down_ply)
            td_xyz = np.asarray(pcd_td.points, dtype=np.float32)
            td_raw = (np.asarray(pcd_td.colors) * 255).astype(np.uint8) \
                     if pcd_td.has_colors() else np.full((len(td_xyz), 3), 160, dtype=np.uint8)
            self.get_logger().info(f"Top-down map: {len(td_xyz)} pts")
        else:
            td_xyz = bg_xyz
            td_raw = bg_raw
            self.get_logger().info("top_down_ply 없음 → bg_ply 재사용")
        td_rgb = np.clip(td_raw.astype(np.int32) * 120 // 100, 0, 255).astype(np.uint8)

        # ── test_results 폴더 스캔 → index 매핑 ───────────────────────
        # 각 서브폴더 = timestamp, 정렬 후 1-based index
        if not os.path.isdir(results_dir):
            self.get_logger().error(f"test_results_dir 없음: {results_dir}")
            return

        subdirs = sorted([
            d for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
        ])
        self.frames = []   # list of {'idx': 1-based, 'ts': timestamp_str, 'dir': abs_path}
        for i, ts in enumerate(subdirs):
            self.frames.append({
                'idx': i + 1,
                'ts':  ts,
                'dir': os.path.join(results_dir, ts),
            })

        self.get_logger().info(
            f"Loaded {len(self.frames)} frames from {results_dir}")
        self._print_index_map()

        # ── Publishers ────────────────────────────────────────────────
        self.pub_map       = self.create_publisher(PointCloud2, '/map_cloud',       LATCHING)
        self.pub_top_down  = self.create_publisher(PointCloud2, '/top_down_cloud',  LATCHING)
        self.pub_visible   = self.create_publisher(PointCloud2, '/visible_region',  VOLATILE)
        self.pub_visited   = self.create_publisher(PointCloud2, '/visited_region',  LATCHING)
        self.pub_matching  = self.create_publisher(PointCloud2, '/matching_points', VOLATILE)
        self.pub_match_lines = self.create_publisher(MarkerArray, '/matching_lines', VOLATILE)
        self.pub_bbox      = self.create_publisher(MarkerArray, '/visible_bbox',    VOLATILE)
        self.pub_footprint = self.create_publisher(MarkerArray, '/view_footprint',  VOLATILE)
        self.pub_frustum   = self.create_publisher(MarkerArray, '/camera_frustum',  VOLATILE)
        self.pub_pose      = self.create_publisher(PoseStamped, '/camera_pose',     VOLATILE)
        self.pub_traj      = self.create_publisher(MarkerArray, '/trajectory_line', VOLATILE)
        self.pub_image     = self.create_publisher(Image,       '/cam/image',       VOLATILE)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 누적 방문 영역 / 쿼리 trajectory
        self.visited_mask   = np.zeros(len(self.map_xyz), dtype=bool)
        self.traj_positions = []   # 쿼리한 위치 순서대로 누적

        # 전체 맵 1회 발행
        stamp0 = self.get_clock().now().to_msg()
        self.pub_map.publish(
            make_pointcloud2(bg_xyz, bg_rgb_mid, self.frame_id, stamp0))
        self.get_logger().info("Published /map_cloud (latching)")

        self.pub_top_down.publish(
            make_pointcloud2(td_xyz, td_rgb, self.frame_id, stamp0))
        self.get_logger().info(f"Published /top_down_cloud (latching, {len(td_xyz)} pts)")

        # ── /kidnap_idx 구독 ──────────────────────────────────────────
        self.create_subscription(Int32, '/kidnap_idx', self._on_idx, 10)
        self.get_logger().info(
            "Ready. Send index:\n"
            "  ros2 topic pub --once /kidnap_idx std_msgs/msg/Int32 \"data: 1\"\n"
            "  (data: 0 → 인덱스 목록 출력)")

    # ─────────────────────────────────────────────────────────────────
    def _print_index_map(self):
        lines = ["", "=" * 55, "  idx  |  timestamp", "=" * 55]
        for f in self.frames:
            lines.append(f"  {f['idx']:4d}  |  {f['ts']}")
        lines.append("=" * 55)
        self.get_logger().info("\n".join(lines))

    # ─────────────────────────────────────────────────────────────────
    def _on_idx(self, msg: Int32):
        idx = msg.data

        if idx == 0:
            self._print_index_map()
            return

        if idx < 1 or idx > len(self.frames):
            self.get_logger().warn(
                f"Index {idx} out of range [1, {len(self.frames)}]")
            return

        frame = self.frames[idx - 1]
        self.get_logger().info(
            f"Loading idx={idx}  ts={frame['ts']}")

        # ── pkl 로드 ──────────────────────────────────────────────────
        s7_path = os.path.join(frame['dir'], 'step7_data.pkl')
        s6_path = os.path.join(frame['dir'], 'step6_data.pkl')

        if not os.path.exists(s6_path):
            self.get_logger().error(f"step6_data.pkl 없음: {s6_path}")
            return

        s6 = pickle.load(open(s6_path, 'rb'))

        # T_WC: step7(PnP 추정 포즈) 우선, 없으면 best_ref 포즈 사용
        if os.path.exists(s7_path):
            s7 = pickle.load(open(s7_path, 'rb'))
            T_WC = np.array(s7['estimated_pose'], dtype=np.float64)
            self.get_logger().info(
                f"  pose from step7 (inliers={s7.get('inlier_count','?')})")
        else:
            ref_pose = s6['best_cand'].get('pose')
            T_WC = np.array(ref_pose, dtype=np.float64) if ref_pose is not None else np.eye(4)
            self.get_logger().warn("  step7 없음 → best_ref pose 사용")

        # query 이미지
        query_rgb = s6.get('query_rgb')   # (H, W, 3) uint8 RGB

        # 매칭 3D 포인트 backproject (reference depth 기준)
        match_xyz = self._backproject_matches(s6)

        stamp = self.get_clock().now().to_msg()
        self._publish_frame(T_WC, query_rgb, match_xyz, stamp)

    # ─────────────────────────────────────────────────────────────────
    def _backproject_matches(self, s6):
        """step6_data에서 matched_r_kps + best_cand depth → world 3D pts"""
        matched_r = s6.get('matched_r_kps', np.zeros((0, 2)))
        ref_entry = s6.get('best_cand', {})
        dep_path  = ref_entry.get('depth_path', '')
        pose_raw  = ref_entry.get('pose')

        base = '/home/park/loc_ws/src/render_loc'
        if dep_path and not os.path.isabs(dep_path):
            dep_path = os.path.join(base, dep_path)

        if (len(matched_r) == 0 or not dep_path
                or not os.path.exists(dep_path) or pose_raw is None):
            return np.zeros((0, 3), np.float32)

        dep  = np.load(dep_path)
        T_WR = np.array(pose_raw, dtype=np.float64)
        R_WR = T_WR[:3, :3];  t_WR = T_WR[:3, 3]
        dmin, dmax = 0.3, self.depth_max
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
        return np.array(pts3d, dtype=np.float32) if pts3d else np.zeros((0, 3), np.float32)

    # ─────────────────────────────────────────────────────────────────
    def _publish_frame(self, T_WC, img_rgb, match_xyz_all, stamp):
        R = T_WC[:3, :3]
        t = T_WC[:3, 3]
        fid = self.frame_id

        # 가시 영역
        vis_idx = frustum_cull(
            self.map_xyz, T_WC,
            self.fx, self.fy, self.cx, self.cy,
            self.img_W, self.img_H, self.depth_max)
        vis_xyz = self.map_xyz[vis_idx] if len(vis_idx) > 0 else np.zeros((0, 3), np.float32)
        vis_rgb = self.map_rgb_orig[vis_idx] if len(vis_idx) > 0 else np.zeros((0, 3), np.uint8)

        # 누적 방문 영역
        if len(vis_idx) > 0:
            newly_visible = ~self.visited_mask[vis_idx]
            if np.any(newly_visible):
                self.visited_mask[vis_idx] = True
                acc_xyz = self.map_xyz[self.visited_mask]
                acc_rgb = self.map_rgb_orig[self.visited_mask]
                self.pub_visited.publish(
                    make_pointcloud2(acc_xyz, acc_rgb, fid, stamp))

        # 매칭 포인트 crop + 50% 샘플
        crop_w = int(self.img_W * 0.8)
        crop_h = int(self.img_H * 0.75)
        u_min = (self.img_W - crop_w) / 2;  u_max = u_min + crop_w
        v_min = (self.img_H - crop_h) / 2;  v_max = v_min + crop_h
        if len(match_xyz_all) > 0:
            T_CW  = np.linalg.inv(T_WC)
            pts_c = (T_CW[:3, :3] @ match_xyz_all.T).T + T_CW[:3, 3]
            front = (pts_c[:, 2] > 0.1) & (pts_c[:, 2] <= self.depth_max)
            u_p = np.where(front, pts_c[:, 0] / np.maximum(pts_c[:, 2], 1e-6) * self.fx + self.cx, -1)
            v_p = np.where(front, pts_c[:, 1] / np.maximum(pts_c[:, 2], 1e-6) * self.fy + self.cy, -1)
            in_crop = front & (u_p >= u_min) & (u_p < u_max) & (v_p >= v_min) & (v_p < v_max)
            pts_crop = match_xyz_all[in_crop]
            n = len(pts_crop)
            if n > 0:
                rng_idx = np.random.choice(n, size=max(1, n // 2), replace=False)
                match_xyz = pts_crop[rng_idx]
            else:
                match_xyz = np.zeros((0, 3), np.float32)
        else:
            match_xyz = np.zeros((0, 3), np.float32)
        match_rgb = np.tile(np.array([255, 0, 0], dtype=np.uint8), (len(match_xyz), 1))

        # trajectory 누적
        self.traj_positions.append(t.copy())

        # quaternion
        qx, qy, qz, qw = rot2quat(R)
        CX, CY, CZ, CW = 0.5, -0.5, 0.5, 0.5
        vx = qw*CX + qx*CW + qy*CZ - qz*CY
        vy = qw*CY - qx*CZ + qy*CW + qz*CX
        vz = qw*CZ + qx*CY - qy*CX + qz*CW
        vw = qw*CW - qx*CX - qy*CY - qz*CZ
        fp_world = R @ np.array([0.0, 0.0, self.frust_d]) + t

        # frustum image
        frust_img_data = _build_frustum_image_data(
            T_WC, self.fx, self.fy, self.cx, self.cy,
            self.img_W, self.img_H, self.frust_d, img_rgb,
            grid_w=self.frust_grid_w, grid_h=self.frust_grid_h)

        # matching lines marker
        ml = Marker()
        ml.header.frame_id = fid;  ml.header.stamp = stamp
        ml.ns = "matching_lines";  ml.id = 0
        ml.type = Marker.LINE_LIST;  ml.action = Marker.ADD
        ml.scale.x = 0.015
        ml.color.r = 0.4;  ml.color.g = 1.0;  ml.color.b = 0.2
        ml.color.a = self._match_line_alpha
        ml.pose.orientation.w = 1.0
        if len(match_xyz) > 0:
            origin_pt = Point(x=float(t[0]), y=float(t[1]), z=float(t[2]))
            for mp in match_xyz:
                ml.points.append(origin_pt)
                ml.points.append(Point(x=float(mp[0]), y=float(mp[1]), z=float(mp[2])))
        match_lines_ma = MarkerArray();  match_lines_ma.markers.append(ml)

        # frustum wireframe + image 합쳐서 발행
        frustum_combined = MarkerArray()
        wireframe_ma = make_frustum_marker(
            T_WC, self.fx, self.fy, self.cx, self.cy,
            self.img_W, self.img_H, self.frust_d, fid, stamp)
        frustum_combined.markers.extend(wireframe_ma.markers)
        frust_img_ma = _make_frustum_image_marker(frust_img_data, fid, stamp)
        frustum_combined.markers.extend(frust_img_ma.markers)

        # pose msg
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = fid;  pose_msg.header.stamp = stamp
        pose_msg.pose.position.x = float(t[0])
        pose_msg.pose.position.y = float(t[1])
        pose_msg.pose.position.z = float(t[2])
        pose_msg.pose.orientation.x = float(qx)
        pose_msg.pose.orientation.y = float(qy)
        pose_msg.pose.orientation.z = float(qz)
        pose_msg.pose.orientation.w = float(qw)

        # TF
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

        # 일괄 발행
        self.pub_frustum.publish(frustum_combined)
        self.pub_matching.publish(make_pointcloud2(match_xyz, match_rgb, fid, stamp))
        self.pub_match_lines.publish(match_lines_ma)
        self.pub_visible.publish(make_pointcloud2(vis_xyz, vis_rgb, fid, stamp))
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
        self.get_logger().info(
            f"  published  visible={len(vis_idx)}  match={len(match_xyz)}")


def main(args=None):
    rclpy.init(args=args)
    node = KidnapLocalizerViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
