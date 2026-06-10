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
import json
from collections import deque

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
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import ExternalShutdownException

import message_filters
from sensor_msgs.msg import CompressedImage, Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool, Header, String
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
        self.declare_parameter("static_camera_infos", Parameter.Type.STRING_ARRAY)
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
        self.declare_parameter("lidar_topic", "/hrz/points")
        self.declare_parameter("control_topic", "/vps/localizer_enabled")
        self.declare_parameter("status_topic", "/vps/localizer_status")
        self.declare_parameter("start_enabled", False)
        self.declare_parameter("status_timeout_sec", 2.0)
        # 웹 뷰어(web_pose_bridge)가 구독하는 토픽명
        self.declare_parameter("pose_topic", "/vps/current_pose")
        self.declare_parameter("path_topic", "/vps/pred_path")

        gp = self.get_parameter
        self.cam_topics = list(gp("cam_topics").value)
        self.cam_info_topics = list(gp("cam_info_topics").value)
        self.cam_ids = list(gp("cam_ids").value)
        static_infos = gp("static_camera_infos").value
        self.static_camera_infos = list(static_infos) if static_infos else []
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
        self.lidar_topic = gp("lidar_topic").value
        self.control_topic = gp("control_topic").value
        self.status_topic = gp("status_topic").value
        self.localization_enabled = bool(gp("start_enabled").value)
        self.status_timeout_sec = float(gp("status_timeout_sec").value)

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
        self._load_static_camera_infos(self.static_camera_infos)

        # ── 동기 프레임 슬롯 + 워커 스레드 ─────────────────────────────────
        self._lock = threading.Lock()
        self._latest = None          # {cam_id: (bgr, stamp_msg)} or None
        self._stop = False
        self._last_proc_t = 0.0
        self._frame_seq = 0
        self._cam_last_seen = {cid: None for cid in self.cam_ids}
        self._cam_last_stamp = {cid: None for cid in self.cam_ids}
        self._cam_seen_times = {cid: deque(maxlen=200) for cid in self.cam_ids}
        self._lidar_last_seen = None
        self._lidar_last_stamp = None
        self._lidar_seen_times = deque(maxlen=500)
        self._localize_done_times = deque(maxlen=100)
        self._last_localize_sec = None
        self._last_localize_ok = None
        self._last_status_t = 0.0

        # ── publishers ─────────────────────────────────────────────────────
        pose_topic = gp("pose_topic").value
        path_topic = gp("path_topic").value
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.path_pub = self.create_publisher(Path, path_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
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
        self._image_status_subs = []
        for i, img_topic in enumerate(self.cam_topics):
            msg_type = (CompressedImage if img_topic.endswith("compressed")
                        else Image)
            self._img_subs.append(
                message_filters.Subscriber(self, msg_type, img_topic,
                                           qos_profile=sensor_qos))
            cid = self.topic_cam_id[i]
            self._image_status_subs.append(self.create_subscription(
                msg_type, img_topic,
                lambda msg, c=cid: self._on_image_status(msg, c), sensor_qos))
        self._sync = message_filters.ApproximateTimeSynchronizer(
            self._img_subs, queue_size=5, slop=self.sync_slop)
        self._sync.registerCallback(self._on_synced_images)

        self._lidar_sub = None
        if self.lidar_topic:
            self._lidar_sub = self.create_subscription(
                PointCloud2, self.lidar_topic, self._on_lidar_status, sensor_qos)
        self._control_sub = self.create_subscription(
            Bool, self.control_topic, self._on_control, 10)
        self._status_timer = self.create_timer(0.5, self._publish_status)

        # 워커 스레드 시작
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self.get_logger().info(
            "구독 토픽:\n  " + "\n  ".join(
                f"{t}  → {self.topic_cam_id[i]}"
                for i, t in enumerate(self.cam_topics)))
        rate_text = "unlimited" if self.rate_hz <= 0.0 else f"{self.rate_hz}Hz"
        self.get_logger().info(
            f"처리 상한 rate={rate_text}  undistort={self.do_undistort}  "
            f"sync_slop={self.sync_slop}s")
        self.get_logger().info(
            f"localizer control={self.control_topic} start_enabled={self.localization_enabled} "
            f"status={self.status_topic} lidar={self.lidar_topic or '-'}")

    # ── camera_info → undistort 맵 ──────────────────────────────────────────
    def _on_camera_info(self, msg: CameraInfo, cam_id: str):
        if cam_id in self.undistort_maps:
            return
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.d, dtype=np.float64).reshape(-1)[:4].reshape(4, 1)
        size = (int(msg.width), int(msg.height))
        self._set_camera_model(cam_id, size, K, D, msg.distortion_model or "fisheye",
                               source="camera_info")

    def _load_static_camera_infos(self, specs):
        """CameraInfo topic이 없는 bag 테스트용 정적 intrinsic 로더.

        형식:
          cam_id MODEL width height fx fy cx cy k1 k2 k3 k4
        구분자는 공백 또는 ':' 모두 허용한다.
        """
        for raw in specs:
            if not raw:
                continue
            parts = str(raw).replace(":", " ").replace(",", " ").split()
            if len(parts) != 12:
                self.get_logger().warn(
                    "static_camera_infos 항목 형식 오류: "
                    f"{raw!r} (필요: cam_id model w h fx fy cx cy k1 k2 k3 k4)")
                continue
            cam_id, model = parts[0], parts[1]
            try:
                width = int(float(parts[2]))
                height = int(float(parts[3]))
                fx, fy, cx, cy = map(float, parts[4:8])
                dist = np.array(list(map(float, parts[8:12])),
                                dtype=np.float64).reshape(4, 1)
            except ValueError as e:
                self.get_logger().warn(
                    f"static_camera_infos 파싱 실패: {raw!r}: {e}")
                continue
            K = np.array([[fx, 0.0, cx],
                          [0.0, fy, cy],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
            self._set_camera_model(cam_id, (width, height), K, dist, model,
                                   source="static")

    def _set_camera_model(self, cam_id, size, K, D, model, source):
        self.cam_K[cam_id] = K
        if not self.do_undistort:
            self.undistort_maps[cam_id] = None
            self.get_logger().info(
                f"[{cam_id}] {source} K 저장 완료, undistort 비활성화")
            return
        model_l = str(model).lower()
        # keep_original_k=True → new_K = K (test_data_rectified 와 동일 규약)
        if "fisheye" in model_l or "equidistant" in model_l:
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), K, size, cv2.CV_16SC2)
        else:
            map1, map2 = cv2.initUndistortRectifyMap(
                K, D.reshape(-1), None, K, size, cv2.CV_16SC2)
        self.undistort_maps[cam_id] = (map1, map2)
        self.get_logger().info(
            f"[{cam_id}] {source} undistort 맵 빌드 완료  size={size}  "
            f"fx={K[0,0]:.1f} fy={K[1,1]:.1f}")

    # ── 동기 이미지 콜백: 최신 프레임만 슬롯에 보관 (드롭) ───────────────────
    def _on_synced_images(self, *msgs):
        if not self.localization_enabled:
            return
        if not self._pose_ready_now():
            return
        frame = {}
        for i, m in enumerate(msgs):
            cid = self.topic_cam_id[i]
            bgr = self._decode(m)
            if bgr is None:
                return
            frame[cid] = (bgr, m.header.stamp)
        with self._lock:
            self._latest = frame

    def _on_image_status(self, msg, cam_id: str):
        now = time.time()
        self._cam_last_seen[cam_id] = now
        self._cam_last_stamp[cam_id] = msg.header.stamp
        self._cam_seen_times[cam_id].append(now)
        self._publish_status()

    def _on_lidar_status(self, msg: PointCloud2):
        now = time.time()
        self._lidar_last_seen = now
        self._lidar_last_stamp = msg.header.stamp
        self._lidar_seen_times.append(now)
        self._publish_status()

    def _on_control(self, msg: Bool):
        self.localization_enabled = bool(msg.data)
        if not self.localization_enabled:
            with self._lock:
                self._latest = None
        self.get_logger().info(
            "localization " + ("ENABLED" if self.localization_enabled else "DISABLED"))
        self._publish_status(force=True)

    def _is_recent(self, t):
        return t is not None and (time.time() - t) <= self.status_timeout_sec

    @staticmethod
    def _hz_from_times(times, now, window=2.0):
        recent = [t for t in times if now - t <= window]
        if len(recent) < 2:
            return 0.0
        span = max(recent[-1] - recent[0], 1e-6)
        return (len(recent) - 1) / span

    def _pose_ready_now(self):
        undistort_ready = (not self.do_undistort) or all(
            cid in self.undistort_maps for cid in self.cam_ids)
        data_ready = all(self._is_recent(self._cam_last_seen.get(cid))
                         for cid in self.cam_ids)
        if self.lidar_topic:
            data_ready = data_ready and self._is_recent(self._lidar_last_seen)
        return data_ready and undistort_ready and self.db is not None

    def _publish_status(self, force=False):
        now = time.time()
        if not force and (now - self._last_status_t) < 0.2:
            return
        self._last_status_t = now
        cams = {
            cid: {
                "seen": self._cam_last_seen[cid] is not None,
                "recent": self._is_recent(self._cam_last_seen[cid]),
                "age": None if self._cam_last_seen[cid] is None else now - self._cam_last_seen[cid],
                "hz": self._hz_from_times(self._cam_seen_times[cid], now),
                "topic": self.cam_topics[i],
            }
            for i, cid in enumerate(self.cam_ids)
        }
        lidar_required = bool(self.lidar_topic)
        lidar = {
            "required": lidar_required,
            "seen": self._lidar_last_seen is not None,
            "recent": (not lidar_required) or self._is_recent(self._lidar_last_seen),
            "age": None if self._lidar_last_seen is None else now - self._lidar_last_seen,
            "hz": self._hz_from_times(self._lidar_seen_times, now),
            "topic": self.lidar_topic or "",
        }
        undistort_ready = (not self.do_undistort) or all(
            cid in self.undistort_maps for cid in self.cam_ids)
        data_ready = all(v["recent"] for v in cams.values()) and lidar["recent"]
        pose_ready = data_ready and undistort_ready and self.db is not None
        payload = {
            "type": "localizer_status",
            "enabled": self.localization_enabled,
            "data_ready": data_ready,
            "pose_ready": pose_ready,
            "undistort_ready": undistort_ready,
            "cams": cams,
            "lidar": lidar,
            "processed": self._frame_seq,
            "localization": {
                "enabled": self.localization_enabled,
                "hz": self._hz_from_times(self._localize_done_times, now, window=10.0),
                "last_sec": self._last_localize_sec,
                "last_ok": self._last_localize_ok,
                "rate_limit_hz": None if self.rate_hz <= 0.0 else self.rate_hz,
                "mode": "latest_frame_drop_queue",
            },
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.status_pub.publish(msg)

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
        period = 1.0 / self.rate_hz if self.rate_hz > 0.0 else 0.0
        while not self._stop and rclpy.ok():
            now = time.time()
            if period > 0.0 and now - self._last_proc_t < period:
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
        self._last_localize_sec = dt
        self._last_localize_ok = est_pose is not None
        self._localize_done_times.append(time.time())
        self._publish_status(force=True)

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
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
