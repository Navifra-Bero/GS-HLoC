#!/usr/bin/env python3
"""실시간 ROS2 Gaussian-Map Localization 노드.

로봇 PC가 보내는 이미지 토픽(main/sub 2대)을 받아 검증된 오프라인 파이프라인
step5(retrieval) → step6(match) → step7(pnp)를 그대로 호출하고, 추정 pose를
정렬(Z-up) 가우시안 지도 frame 위에 TF/Path/PoseStamped로 publish 한다.

핵심 재사용:
  pipeline.batch_test.localize_single(main_path, db, config, work_dir,
                                      save_images, query_images={cam_id: path})
  → retrieval_type=="type2"면 step5_type2→step6_type2→step7 자동 실행,
    estimated_pose(4×4 c2w, 정렬 map frame) 반환.

쿼리 카메라는 fisheye이므로 camera_info(equidistant)로 원본 K를 유지한 채
undistort 한 뒤 파이프라인에 넣는다(= test_data_rectified와 동일한 PINHOLE K).

실행 환경:
  conda activate render_loc && source /opt/ros/humble/setup.bash
  (torch+cuda / plyfile / rclpy 가 모두 같은 env 에서 동작)
"""
import os
import sys
import threading
import time

import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# 파이프라인 import 를 위해 repo root 를 sys.path / cwd 에 등록한다.
# config 의 상대 경로(data/, test_data_rectified/, third_party/...) 가 repo root
# 기준이므로 cwd 도 그곳으로 맞춘다.
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_repo_root(explicit=None):
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.append(os.environ.get("RENDER_LOC_ROOT", ""))
    # __file__ 기준 상위 탐색 (소스 트리에서 실행하는 경우)
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    for _ in range(6):
        if os.path.isdir(os.path.join(cur, "scripts", "pipeline")):
            candidates.append(cur)
            break
        cur = os.path.dirname(cur)
    candidates.append("/home/park/loc_ws/src/render_loc")
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "scripts", "pipeline")):
            return os.path.abspath(c)
    return None


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import message_filters
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
import tf2_ros
from scipy.spatial.transform import Rotation


class RosLocalizerNode(Node):
    def __init__(self):
        super().__init__("ros_localizer")

        # ── 파라미터 ────────────────────────────────────────────────────────
        self.declare_parameter("repo_root", "")
        self.declare_parameter("config_file",
                               "config/render_loc_multi_cam.yaml")
        self.declare_parameter("output_dir", "output/gs_sdf_omni")
        self.declare_parameter("cam_topics",
                               ["/cam0/image_raw/compressed",
                                "/cam1/image_raw/compressed"])
        self.declare_parameter("cam_info_topics",
                               ["/cam0/camera_info", "/cam1/camera_info"])
        self.declare_parameter("cam_ids", ["cam_0", "cam_1"])
        self.declare_parameter("main_cam", "cam_0")
        self.declare_parameter("sub_cams", ["cam_1"])
        self.declare_parameter("undistort", True)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("optical_frame", "base_optical")
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("temp_dir", "/tmp/ros_localizer")
        # 웹 뷰어(web_pose_bridge)가 구독하는 토픽명
        self.declare_parameter("pose_topic", "/vps/current_pose")
        self.declare_parameter("path_topic", "/vps/pred_path")

        gp = self.get_parameter
        self.cam_topics = list(gp("cam_topics").value)
        self.cam_info_topics = list(gp("cam_info_topics").value)
        self.cam_ids = list(gp("cam_ids").value)
        self.main_cam = gp("main_cam").value
        self.sub_cams = list(gp("sub_cams").value)
        self.do_undistort = bool(gp("undistort").value)
        self.map_frame = gp("map_frame").value
        self.base_frame = gp("base_frame").value
        self.optical_frame = gp("optical_frame").value
        self.rate_hz = float(gp("rate_hz").value)
        self.sync_slop = float(gp("sync_slop").value)
        self.publish_tf = bool(gp("publish_tf").value)
        self.temp_dir = gp("temp_dir").value

        if not (len(self.cam_topics) == len(self.cam_info_topics) == len(self.cam_ids)):
            raise ValueError(
                "cam_topics / cam_info_topics / cam_ids 길이가 일치해야 합니다: "
                f"{len(self.cam_topics)}/{len(self.cam_info_topics)}/{len(self.cam_ids)}")

        # topic index → cam_id 매핑
        self.topic_cam_id = {i: cid for i, cid in enumerate(self.cam_ids)}

        # ── repo root / 파이프라인 로드 ─────────────────────────────────────
        repo_root = _resolve_repo_root(gp("repo_root").value or None)
        if repo_root is None:
            raise RuntimeError(
                "repo_root 를 찾지 못했습니다. 파라미터 repo_root 또는 "
                "환경변수 RENDER_LOC_ROOT 로 render_loc 소스 경로를 지정하세요.")
        self.repo_root = repo_root
        os.chdir(repo_root)
        scripts_dir = os.path.join(repo_root, "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        self.get_logger().info(f"repo_root = {repo_root} (cwd 설정 완료)")

        from pipeline import load_config, load_pkl  # noqa: E402
        from pipeline.batch_test import localize_single  # noqa: E402
        self._localize_single = localize_single

        cfg_path = gp("config_file").value
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(repo_root, cfg_path)
        self.config = load_config(cfg_path)

        # multi_cam 설정 주입 (런타임 cam 선택을 config 보다 우선)
        mc = self.config.setdefault("multi_cam", {})
        mc["enabled"] = True
        mc["retrieval_type"] = "type2"
        mc["main_cam"] = self.main_cam
        mc["sub_cams"] = self.sub_cams
        mc["cam_ids"] = self.cam_ids
        mc["primary_cam"] = self.main_cam
        # kapture_dir(rig/intrinsic 소스)는 config 값 유지 (test_data_rectified)

        out_dir = gp("output_dir").value
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(repo_root, out_dir)
        self.output_dir = out_dir
        self.db = load_pkl(out_dir, "step4_database.pkl")
        if self.db is None:
            raise RuntimeError(
                f"step4_database.pkl 를 찾을 수 없습니다: {out_dir}")
        self.get_logger().info(
            f"DB loaded: {len(self.db['entries'])} entries  "
            f"main={self.main_cam} sub={self.sub_cams}")

        os.makedirs(self.temp_dir, exist_ok=True)

        # ── undistort 맵 캐시 (camera_info 수신 시 1회 빌드) ─────────────────
        self.undistort_maps = {}     # cam_id → (map1, map2)
        self.cam_K = {}              # cam_id → 3×3 (참고용)

        # ── 동기 프레임 슬롯 + 워커 스레드 ─────────────────────────────────
        self._lock = threading.Lock()
        self._latest = None          # {cam_id: (bgr, stamp_msg)} or None
        self._stop = False
        self._last_proc_t = 0.0
        self._frame_seq = 0

        # ── publishers ─────────────────────────────────────────────────────
        pose_topic = gp("pose_topic").value
        path_topic = gp("path_topic").value
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.path_pub = self.create_publisher(Path, path_topic, 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.map_frame

        # ── subscriptions ──────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        # camera_info: cam 별 latest 저장 + undistort 맵 빌드
        self._info_subs = []
        for i, info_topic in enumerate(self.cam_info_topics):
            cid = self.topic_cam_id[i]
            sub = self.create_subscription(
                CameraInfo, info_topic,
                lambda msg, c=cid: self._on_camera_info(msg, c), sensor_qos)
            self._info_subs.append(sub)

        # image: ApproximateTimeSynchronizer 로 동기화
        self._img_subs = []
        for i, img_topic in enumerate(self.cam_topics):
            msg_type = (CompressedImage if img_topic.endswith("compressed")
                        else Image)
            self._img_subs.append(
                message_filters.Subscriber(self, msg_type, img_topic,
                                           qos_profile=sensor_qos))
        self._sync = message_filters.ApproximateTimeSynchronizer(
            self._img_subs, queue_size=5, slop=self.sync_slop)
        self._sync.registerCallback(self._on_synced_images)

        # 워커 스레드 시작
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self.get_logger().info(
            "구독 토픽:\n  " + "\n  ".join(
                f"{t}  → {self.topic_cam_id[i]}"
                for i, t in enumerate(self.cam_topics)))
        self.get_logger().info(
            f"처리 상한 rate={self.rate_hz}Hz  undistort={self.do_undistort}  "
            f"sync_slop={self.sync_slop}s")

    # ── camera_info → undistort 맵 ──────────────────────────────────────────
    def _on_camera_info(self, msg: CameraInfo, cam_id: str):
        if cam_id in self.undistort_maps:
            return
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.cam_K[cam_id] = K
        if not self.do_undistort:
            self.undistort_maps[cam_id] = None
            self.get_logger().info(f"[{cam_id}] undistort 비활성화, K 저장만 함")
            return
        D = np.array(msg.d, dtype=np.float64).reshape(-1)[:4].reshape(4, 1)
        size = (int(msg.width), int(msg.height))
        # keep_original_k=True → new_K = K (test_data_rectified 와 동일 규약)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, size, cv2.CV_16SC2)
        self.undistort_maps[cam_id] = (map1, map2)
        self.get_logger().info(
            f"[{cam_id}] undistort 맵 빌드 완료  size={size}  "
            f"fx={K[0,0]:.1f} fy={K[1,1]:.1f}")

    # ── 동기 이미지 콜백: 최신 프레임만 슬롯에 보관 (드롭) ───────────────────
    def _on_synced_images(self, *msgs):
        frame = {}
        for i, m in enumerate(msgs):
            cid = self.topic_cam_id[i]
            bgr = self._decode(m)
            if bgr is None:
                return
            frame[cid] = (bgr, m.header.stamp)
        with self._lock:
            self._latest = frame

    @staticmethod
    def _decode(msg):
        if isinstance(msg, CompressedImage):
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        # raw Image (bgr8/rgb8)
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1)
        if msg.encoding.lower().startswith("rgb"):
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr.copy()

    # ── 워커: 슬롯의 최신 프레임을 rate 상한으로 처리 ───────────────────────
    def _worker_loop(self):
        period = 1.0 / max(self.rate_hz, 1e-3)
        while not self._stop and rclpy.ok():
            now = time.time()
            if now - self._last_proc_t < period:
                time.sleep(0.005)
                continue
            with self._lock:
                frame = self._latest
                self._latest = None
            if frame is None:
                time.sleep(0.005)
                continue
            self._last_proc_t = now
            try:
                self._process_frame(frame)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"process_frame 오류: {e}")

    def _process_frame(self, frame):
        # main_cam 필수
        if self.main_cam not in frame:
            return
        # undistort 맵이 아직 안 만들어졌으면(camera_info 미수신) skip
        if self.do_undistort and any(
                self.undistort_maps.get(c) is None for c in frame
                if c not in self.undistort_maps or self.undistort_maps.get(c) is None):
            # 아직 맵 없는 cam 존재
            missing = [c for c in frame if c not in self.undistort_maps]
            if missing:
                self.get_logger().warn(
                    f"camera_info 대기 중 (undistort 맵 없음): {missing}", once=False)
                return

        # 각 cam undistort → temp 파일 저장 → query_images dict
        query_images = {}
        for cid, (bgr, _stamp) in frame.items():
            img = bgr
            if self.do_undistort and self.undistort_maps.get(cid) is not None:
                map1, map2 = self.undistort_maps[cid]
                img = cv2.remap(bgr, map1, map2, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
            cam_dir = os.path.join(self.temp_dir, cid)
            os.makedirs(cam_dir, exist_ok=True)
            path = os.path.join(cam_dir, "frame.jpg")
            cv2.imwrite(path, img)
            query_images[cid] = path

        stamp = frame[self.main_cam][1]
        work_dir = os.path.join(self.temp_dir, "work")

        t0 = time.time()
        est_pose = self._localize_single(
            query_images[self.main_cam], self.db, self.config, work_dir,
            save_images=False, query_images=query_images)
        dt = time.time() - t0
        self._frame_seq += 1

        if est_pose is None:
            self.get_logger().info(f"[{self._frame_seq}] localize FAIL  ({dt:.2f}s)")
            return

        xyz = est_pose[:3, 3]
        self.get_logger().info(
            f"[{self._frame_seq}] OK  xyz=({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})  "
            f"({dt:.2f}s)")
        self._publish_pose(est_pose, stamp)

    # ── pose publish: TF map→optical, PoseStamped, Path ─────────────────────
    def _publish_pose(self, c2w, stamp):
        quat = Rotation.from_matrix(c2w[:3, :3]).as_quat()  # xyzw
        t = c2w[:3, 3]

        if self.publish_tf:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.map_frame
            tf.child_frame_id = self.optical_frame
            tf.transform.translation.x = float(t[0])
            tf.transform.translation.y = float(t[1])
            tf.transform.translation.z = float(t[2])
            tf.transform.rotation.x = float(quat[0])
            tf.transform.rotation.y = float(quat[1])
            tf.transform.rotation.z = float(quat[2])
            tf.transform.rotation.w = float(quat[3])
            self.tf_broadcaster.sendTransform(tf)

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.map_frame
        ps.pose.position.x = float(t[0])
        ps.pose.position.y = float(t[1])
        ps.pose.position.z = float(t[2])
        ps.pose.orientation.x = float(quat[0])
        ps.pose.orientation.y = float(quat[1])
        ps.pose.orientation.z = float(quat[2])
        ps.pose.orientation.w = float(quat[3])
        self.pose_pub.publish(ps)

        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(ps)
        if len(self.path_msg.poses) > 5000:
            self.path_msg.poses = self.path_msg.poses[-5000:]
        self.path_pub.publish(self.path_msg)

    def destroy_node(self):
        self._stop = True
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RosLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
