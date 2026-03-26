#!/usr/bin/env python3
"""
ROS2 Localization Visualizer
============================
1. PLY map → /map_cloud (PointCloud2, alpha 0.5 in RViz)
2. Subscribe /cam_3/image → retrieval → top-1 DB pose
3. Publish /camera_pose (PoseStamped)
4. Publish /camera_frustum (MarkerArray LINE_LIST)
5. Publish /visible_region (PointCloud2, points in frustum, alpha 1.0)
6. Forward image + /camera_info for RViz Camera display

Usage:
  ros2 run render_loc localization_visualizer \
      --ros-args \
      -p ply_path:=/path/to/map.ply \
      -p db_path:=/path/to/step4_database.pkl \
      -p config_path:=/path/to/config.yaml \
      -p image_topic:=/cam_3/image \
      -p frustum_depth:=5.0
"""

import os, sys, pickle, time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
import struct


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def numpy_to_pointcloud2(points_xyz, colors_rgb=None, frame_id="map", stamp=None):
    """
    points_xyz : (N, 3) float32
    colors_rgb : (N, 3) uint8  (optional)
    """
    header = Header()
    header.frame_id = frame_id
    if stamp is not None:
        header.stamp = stamp

    if colors_rgb is not None:
        # Pack RGB into a single float field (RViz standard)
        r = colors_rgb[:, 0].astype(np.uint32)
        g = colors_rgb[:, 1].astype(np.uint32)
        b = colors_rgb[:, 2].astype(np.uint32)
        rgb_packed = (r << 16) | (g << 8) | b
        rgb_float  = rgb_packed.view(np.float32)

        data_arr = np.zeros(len(points_xyz),
                            dtype=[('x','f4'),('y','f4'),('z','f4'),('rgb','f4')])
        data_arr['x']   = points_xyz[:, 0]
        data_arr['y']   = points_xyz[:, 1]
        data_arr['z']   = points_xyz[:, 2]
        data_arr['rgb'] = rgb_float
        fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        point_step = 16
    else:
        data_arr = np.zeros(len(points_xyz),
                            dtype=[('x','f4'),('y','f4'),('z','f4')])
        data_arr['x'] = points_xyz[:, 0]
        data_arr['y'] = points_xyz[:, 1]
        data_arr['z'] = points_xyz[:, 2]
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        point_step = 12

    msg = PointCloud2()
    msg.header    = header
    msg.height    = 1
    msg.width     = len(points_xyz)
    msg.fields    = fields
    msg.is_bigendian = False
    msg.point_step   = point_step
    msg.row_step     = point_step * len(points_xyz)
    msg.data         = data_arr.tobytes()
    msg.is_dense     = True
    return msg


def make_frustum_marker(T_WC, fx, fy, cx, cy, w, h, depth,
                         frame_id="map", stamp=None, marker_id=0):
    """
    T_WC  : (4,4) world-from-camera transform
    Returns MarkerArray with LINE_LIST edges of the image frustum.
    """
    # 4 image corners → 3-D rays at given depth
    corners_img = np.array([
        [0, 0], [w, 0], [w, h], [0, h]
    ], dtype=np.float32)

    corners_cam = np.array([
        [(u - cx) / fx * depth,
         (v - cy) / fy * depth,
         depth]
        for u, v in corners_img
    ], dtype=np.float32)   # (4, 3)

    # Transform to world
    R = T_WC[:3, :3]
    t = T_WC[:3, 3]
    origin_w   = t
    corners_w  = (R @ corners_cam.T).T + t   # (4, 3)

    # Build LINE_LIST: origin→each corner + rectangle edges
    def pt(xyz):
        from geometry_msgs.msg import Point
        p = Point(); p.x, p.y, p.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        return p

    lines = []
    for c in corners_w:
        lines += [pt(origin_w), pt(c)]
    for i in range(4):
        lines += [pt(corners_w[i]), pt(corners_w[(i+1)%4])]

    m = Marker()
    m.header.frame_id = frame_id
    if stamp: m.header.stamp = stamp
    m.ns     = "frustum"
    m.id     = marker_id
    m.type   = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = 0.03
    m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0
    m.points  = lines
    m.pose.orientation.w = 1.0
    return m


def extract_visible_points(pts_xyz, pts_rgb, T_WC, fx, fy, cx, cy, w, h,
                            depth_max=30.0, margin_px=10):
    """
    T_WC : world-from-camera  →  T_CW = inv(T_WC)
    Projects all map points into camera, keeps those inside image bounds.
    Returns (N', 3) xyz,  (N', 3) rgb
    """
    T_CW = np.linalg.inv(T_WC)
    R, t = T_CW[:3, :3], T_CW[:3, 3]

    pts_c = (R @ pts_xyz.T).T + t    # (N, 3) in camera frame

    # Keep points in front of camera and within depth
    valid = (pts_c[:, 2] > 0.1) & (pts_c[:, 2] < depth_max)
    pts_c = pts_c[valid]; pts_rgb_v = pts_rgb[valid]

    # Project
    u = pts_c[:, 0] / pts_c[:, 2] * fx + cx
    v = pts_c[:, 1] / pts_c[:, 2] * fy + cy

    in_view = (u >= -margin_px) & (u < w + margin_px) & \
              (v >= -margin_px) & (v < h + margin_px)

    # Return world-frame coords of visible points
    pts_xyz_orig = pts_xyz[valid][in_view]
    pts_rgb_vis  = pts_rgb_v[in_view]
    return pts_xyz_orig, pts_rgb_vis


def get_query_descriptor(rgb_img, config, dev):
    """Replicates step5 descriptor extraction (EigenPlaces > NetVLAD > HSV)."""
    import torch
    fc = config["features"]

    # 1) EigenPlaces
    try:
        import eigenplaces
        import torchvision.transforms as T
        ep = eigenplaces.EigenPlaces(backbone="ResNet50", descriptors_dimension=512)
        ep = ep.eval().to(dev)
        tf = T.Compose([
            T.ToPILImage(), T.Resize((480, 640)), T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        t = tf(rgb_img).unsqueeze(0).to(dev)
        with torch.no_grad():
            d = ep(t).cpu().numpy().flatten()
        return d
    except Exception:
        pass

    # 2) NetVLAD TorchScript
    try:
        gm = torch.jit.load(fc.get("global_model","models/netvlad.pt"), map_location=dev)
        gm.eval()
        t = cv2.resize(rgb_img,(224,224)).astype(np.float32)/255.0
        t_th = __import__("torch").from_numpy(t.transpose(2,0,1)).unsqueeze(0).to(dev)
        with __import__("torch").no_grad():
            d = gm(t_th).cpu().numpy().flatten()
        return d
    except Exception:
        pass

    # 3) HSV histogram
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    d = np.concatenate([
        cv2.calcHist([hsv],[0],None,[64],[0,180]).flatten(),
        cv2.calcHist([hsv],[1],None,[64],[0,256]).flatten(),
    ])
    return d / (np.linalg.norm(d) + 1e-8)


def retrieve_top1(rgb_img, db, config):
    """Query image (RGB HxWx3 uint8) → top-1 DB entry dict."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gd_dim = config["features"].get("global_desc_dim", 512)

    q_gd = get_query_descriptor(rgb_img, config, dev)
    if len(q_gd) < gd_dim:
        q_gd = np.pad(q_gd, (0, gd_dim - len(q_gd)))
    q_gd = q_gd[:gd_dim]
    q_gd_norm = (q_gd / (np.linalg.norm(q_gd) + 1e-8)).astype(np.float32)

    dists, idxs = db["kdtree"].query(q_gd_norm, k=1)
    return db["entries"][int(idxs)]


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────────────────────

class LocalizationVisualizer(Node):

    def __init__(self):
        super().__init__("localization_visualizer")

        # ── Parameters ──────────────────────────────────────────────────
        self.declare_parameter("ply_path",     "")
        self.declare_parameter("db_path",      "")
        self.declare_parameter("config_path",  "")
        self.declare_parameter("image_topic",  "/cam_3/image")
        self.declare_parameter("frustum_depth", 5.0)
        self.declare_parameter("map_frame",    "map")
        self.declare_parameter("retrieval_every_n", 1)  # process every N frames

        ply_path    = self.get_parameter("ply_path").value
        db_path     = self.get_parameter("db_path").value
        config_path = self.get_parameter("config_path").value
        self.image_topic    = self.get_parameter("image_topic").value
        self.frustum_depth  = self.get_parameter("frustum_depth").value
        self.map_frame      = self.get_parameter("map_frame").value
        self.every_n        = self.get_parameter("retrieval_every_n").value
        self._frame_count   = 0

        # ── Config ──────────────────────────────────────────────────────
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        cam = self.config["camera"]
        self.fx = cam["fx"]; self.fy = cam["fy"]
        self.cx = cam["cx"]; self.cy = cam["cy"]
        self.img_w = cam["width"]; self.img_h = cam["height"]

        # ── Load DB ─────────────────────────────────────────────────────
        if not db_path or not os.path.exists(db_path):
            self.get_logger().error(f"DB not found: {db_path}")
            raise RuntimeError(f"DB not found: {db_path}")
        with open(db_path, "rb") as f:
            self.db = pickle.load(f)
        self.get_logger().info(f"DB loaded: {len(self.db['entries'])} entries")

        # ── Load PLY map ─────────────────────────────────────────────────
        if not ply_path or not os.path.exists(ply_path):
            self.get_logger().error(f"PLY not found: {ply_path}")
            raise RuntimeError(f"PLY not found: {ply_path}")
        self._load_ply(ply_path)

        # ── QoS ─────────────────────────────────────────────────────────
        latching_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10,
        )

        # ── Publishers ──────────────────────────────────────────────────
        self.pub_map      = self.create_publisher(PointCloud2,   "/map_cloud",      latching_qos)
        self.pub_vis_reg  = self.create_publisher(PointCloud2,   "/visible_region", 10)
        self.pub_frustum  = self.create_publisher(MarkerArray,   "/camera_frustum", 10)
        self.pub_pose     = self.create_publisher(PoseStamped,   "/camera_pose",    10)
        self.pub_image    = self.create_publisher(Image,         self.image_topic + "/viz", 10)
        self.pub_caminfo  = self.create_publisher(CameraInfo,    "/cam_3/camera_info", 10)

        # ── Subscriber ──────────────────────────────────────────────────
        self.sub_image = self.create_subscription(
            Image, self.image_topic, self._image_callback, sensor_qos)

        # ── Publish static map once ──────────────────────────────────────
        self._publish_map()
        self.get_logger().info("LocalizationVisualizer ready.")

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _default_config(self):
        return {
            "camera": {
                "fx": 525, "fy": 525, "cx": 319.5, "cy": 239.5,
                "width": 640, "height": 480,
            },
            "features": {
                "global_desc_dim": 512,
                "global_model": "models/netvlad.pt",
            },
            "online": {"top_k": 5},
        }

    def _load_ply(self, ply_path):
        import open3d as o3d
        self.get_logger().info(f"Loading PLY: {ply_path}")
        pcd = o3d.io.read_point_cloud(ply_path)
        self.map_pts   = np.asarray(pcd.points,  dtype=np.float32)
        if pcd.has_colors():
            self.map_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        else:
            self.map_colors = np.full((len(self.map_pts), 3), 128, dtype=np.uint8)
        self.get_logger().info(f"PLY loaded: {len(self.map_pts)} points")

    def _publish_map(self):
        stamp = self.get_clock().now().to_msg()
        # Downsample for RViz performance (every 4th point)
        step  = 12
        pts   = self.map_pts[::step]
        cols  = self.map_colors[::step]
        msg   = numpy_to_pointcloud2(pts, cols, frame_id=self.map_frame, stamp=stamp)
        self.pub_map.publish(msg)
        self.get_logger().info(f"Published /map_cloud ({len(pts)} pts, downsampled x{step})")

    def _ros_image_to_rgb(self, msg: Image):
        """Convert sensor_msgs/Image to (H,W,3) uint8 RGB numpy array."""
        dtype = np.uint8
        if msg.encoding in ("bgr8", "rgb8"):
            arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, 3)
            if msg.encoding == "bgr8":
                return arr[:, :, ::-1].copy()
            return arr.copy()
        elif msg.encoding in ("mono8",):
            arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        else:
            # Try bgr8 fallback
            arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, -1)
            return arr[:, :, :3][:, :, ::-1].copy()

    def _pose_to_transform(self, pose_4x4):
        """pose_4x4 list/array (4×4) → T_WC numpy (4,4)."""
        return np.array(pose_4x4, dtype=np.float64).reshape(4, 4)

    def _make_pose_stamped(self, T_WC, stamp):
        from scipy.spatial.transform import Rotation
        ps = PoseStamped()
        ps.header.frame_id = self.map_frame
        ps.header.stamp    = stamp
        t = T_WC[:3, 3]
        R = Rotation.from_matrix(T_WC[:3, :3])
        q = R.as_quat()   # x y z w
        ps.pose.position.x = float(t[0])
        ps.pose.position.y = float(t[1])
        ps.pose.position.z = float(t[2])
        ps.pose.orientation.x = float(q[0])
        ps.pose.orientation.y = float(q[1])
        ps.pose.orientation.z = float(q[2])
        ps.pose.orientation.w = float(q[3])
        return ps

    def _make_camera_info(self, stamp):
        ci = CameraInfo()
        ci.header.frame_id = "cam_3"
        ci.header.stamp    = stamp
        ci.width  = self.img_w
        ci.height = self.img_h
        ci.k = [self.fx, 0.0, self.cx,
                0.0, self.fy, self.cy,
                0.0, 0.0, 1.0]
        ci.p = [self.fx, 0.0, self.cx, 0.0,
                0.0, self.fy, self.cy, 0.0,
                0.0, 0.0, 1.0, 0.0]
        ci.distortion_model = "plumb_bob"
        ci.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        ci.r = [1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0]
        return ci

    def _rgb_to_ros_image(self, rgb_arr, stamp):
        msg = Image()
        msg.header.frame_id = "cam_3"
        msg.header.stamp    = stamp
        msg.height   = rgb_arr.shape[0]
        msg.width    = rgb_arr.shape[1]
        msg.encoding = "rgb8"
        msg.step     = rgb_arr.shape[1] * 3
        msg.data     = rgb_arr.tobytes()
        return msg

    # ── Main callback ────────────────────────────────────────────────────────

    def _image_callback(self, msg: Image):
        self._frame_count += 1
        if self._frame_count % self.every_n != 0:
            return

        t0 = time.time()
        stamp = msg.header.stamp

        # 1) Decode image
        try:
            rgb = self._ros_image_to_rgb(msg)
        except Exception as e:
            self.get_logger().warn(f"Image decode failed: {e}")
            return

        # 2) Retrieval → top-1 pose
        try:
            top1 = retrieve_top1(rgb, self.db, self.config)
        except Exception as e:
            self.get_logger().warn(f"Retrieval failed: {e}")
            return

        T_WC = self._pose_to_transform(top1["pose"])
        dt   = time.time() - t0
        self.get_logger().info(
            f"Frame {self._frame_count}: top1=#{top1['id']}  "
            f"pos=({T_WC[0,3]:.2f},{T_WC[1,3]:.2f},{T_WC[2,3]:.2f})  "
            f"[{dt:.2f}s]"
        )

        # 3) Publish camera pose
        self.pub_pose.publish(self._make_pose_stamped(T_WC, stamp))

        # 4) Publish camera frustum
        frustum_marker = make_frustum_marker(
            T_WC, self.fx, self.fy, self.cx, self.cy,
            self.img_w, self.img_h, self.frustum_depth,
            frame_id=self.map_frame, stamp=stamp, marker_id=0,
        )
        ma = MarkerArray(); ma.markers.append(frustum_marker)
        self.pub_frustum.publish(ma)

        # 5) Visible region (map points in frustum)
        try:
            vis_pts, vis_cols = extract_visible_points(
                self.map_pts, self.map_colors,
                T_WC, self.fx, self.fy, self.cx, self.cy,
                self.img_w, self.img_h,
                depth_max=self.frustum_depth * 2.0,
            )
            if len(vis_pts) > 0:
                vis_msg = numpy_to_pointcloud2(vis_pts, vis_cols,
                                               frame_id=self.map_frame, stamp=stamp)
                self.pub_vis_reg.publish(vis_msg)
        except Exception as e:
            self.get_logger().warn(f"Visible region failed: {e}")

        # 6) Forward image + camera_info for RViz Camera display
        self.pub_image.publish(self._rgb_to_ros_image(rgb, stamp))
        self.pub_caminfo.publish(self._make_camera_info(stamp))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    try:
        node = LocalizationVisualizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
