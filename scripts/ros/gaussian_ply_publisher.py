#!/usr/bin/env python3
"""정렬(Z-up) 가우시안 PLY 지도를 rviz2 용 PointCloud2 로 publish.

가우시안 PLY 는 단순 포인트클라우드가 아니라 per-Gaussian 속성(f_dc, opacity,
scale, rot)을 가진다. 여기서는 시각화를 위해 각 Gaussian 의 center(x,y,z)와
SH DC 계수에서 복원한 RGB 색만 사용해 PointCloud2(xyz+rgb)로 변환한다.

  rgb = clip(0.5 + 0.2820948 * f_dc, 0, 1)

latched(transient_local) QoS 로 주기적으로 publish 하므로 rviz2 가 나중에 떠도
즉시 지도를 받는다. localization 노드의 추정 pose 와 동일한 map frame.
"""
import os

import numpy as np
from plyfile import PlyData

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

SH_C0 = 0.28209479177387814


class GaussianPlyPublisher(Node):
    def __init__(self):
        super().__init__("gaussian_ply_publisher")

        self.declare_parameter("ply_path", "output/gs_sdf_omni/aligned_map.ply")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("topic", "~/cloud")
        self.declare_parameter("stride", 1)          # N개마다 1개 (다운샘플)
        self.declare_parameter("opacity_min", 0.0)   # opacity 필터 (>= 이 값만)
        self.declare_parameter("publish_period", 2.0)

        gp = self.get_parameter
        ply_path = gp("ply_path").value
        if not os.path.isabs(ply_path):
            ply_path = os.path.abspath(ply_path)
        self.map_frame = gp("map_frame").value
        topic = gp("topic").value
        stride = max(int(gp("stride").value), 1)
        opacity_min = float(gp("opacity_min").value)
        period = float(gp("publish_period").value)

        self.cloud_msg = self._build_cloud(ply_path, stride, opacity_min)

        qos = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(PointCloud2, topic, qos)
        self.pub.publish(self.cloud_msg)
        self.timer = self.create_timer(period, self._on_timer)
        self.get_logger().info(
            f"Gaussian map publish: {self.cloud_msg.width} pts  "
            f"frame={self.map_frame}  topic={self.pub.topic_name}")

    def _build_cloud(self, ply_path, stride, opacity_min):
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY 없음: {ply_path}")
        self.get_logger().info(f"PLY 로드: {ply_path}")
        ply = PlyData.read(ply_path)
        v = ply["vertex"].data
        names = v.dtype.names

        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

        # 색: f_dc_0/1/2 (SH DC) → RGB. 없으면 red/green/blue 또는 회색.
        if all(f"f_dc_{i}" in names for i in range(3)):
            f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
            rgb = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)
        elif all(c in names for c in ("red", "green", "blue")):
            rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1) / 255.0
        else:
            rgb = np.full((len(xyz), 3), 0.7, dtype=np.float32)
        rgb = (rgb * 255.0).astype(np.uint32)

        # opacity 필터 (sigmoid 적용 전 logit 값이지만 단조라 임계 비교에 사용 가능)
        keep = np.ones(len(xyz), dtype=bool)
        if opacity_min > 0.0 and "opacity" in names:
            op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))
            keep &= op >= opacity_min
        if stride > 1:
            idx = np.zeros(len(xyz), dtype=bool)
            idx[::stride] = True
            keep &= idx

        xyz = xyz[keep]
        rgb = rgb[keep]

        # rgb 를 float32 로 packed (rviz PointCloud2 표준): uint32 비트를 float32로 재해석
        rgb_packed = ((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]).astype(np.uint32)
        rgb_float = rgb_packed.view(np.float32)

        cloud = np.zeros(len(xyz), dtype=[
            ("x", np.float32), ("y", np.float32), ("z", np.float32),
            ("rgb", np.float32)])
        cloud["x"] = xyz[:, 0]
        cloud["y"] = xyz[:, 1]
        cloud["z"] = xyz[:, 2]
        cloud["rgb"] = rgb_float

        msg = PointCloud2()
        msg.header = Header(frame_id=self.map_frame)
        msg.height = 1
        msg.width = len(xyz)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = cloud.tobytes()
        return msg

    def _on_timer(self):
        self.cloud_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.cloud_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GaussianPlyPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
