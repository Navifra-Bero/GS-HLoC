#!/usr/bin/env python3
"""ROS pose → 웹(SSE) 브리지 + 정적 파일 서버.

Foxglove/rosbridge 없이, ROS 가 계산한 pose 를 JSON 으로 변환해 브라우저로
실시간 push 한다. 표준 라이브러리만 사용(추가 의존성 0).

제공 엔드포인트 (한 포트):
  GET /              → web/index.html (가우시안 스플래팅 뷰어)
  GET /<file>        → web/ 아래 정적 파일 (viewer.js, lib/*)
  GET /gaussian_map.splat → splat_path 또는 web/gaussian_map.splat
  GET /events        → Server-Sent Events. pose 가 들어올 때마다
                       data: {"type":"pose","position":{x,y,z},
                              "quaternion":{x,y,z,w},"stamp":...}\\n\\n

구독: PoseStamped 또는 Odometry (기본 /vps/current_pose).
좌표는 그대로 map frame의 c2w(OpenCV optical)로 취급한다.

사용:
  ros2 run render_loc web_pose_bridge.py --ros-args \\
    -p web_dir:=/abs/path/to/render_loc/web -p port:=8080
  브라우저: http://localhost:8080
"""
import json
import os
import queue
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage

try:
    import cv2
    import numpy as np
    _CV_OK = True
except Exception:  # cv2/numpy 없으면 raw Image/png 변환은 비활성, jpeg 패스스루만.
    _CV_OK = False


def _default_web_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    for _ in range(6):
        cand = os.path.join(cur, "web")
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "index.html")):
            return cand
        cur = os.path.dirname(cur)
    # 설치 트리 fallback
    return os.path.join("/home/park/loc_ws/src/render_loc", "web")


class _Broadcaster:
    """SSE 클라이언트 큐 레지스트리 + 최신 pose 캐시."""

    def __init__(self):
        self._clients = set()
        self._lock = threading.Lock()
        self._latest = None

    def register(self):
        q = queue.Queue(maxsize=10)
        with self._lock:
            self._clients.add(q)
            latest = self._latest
        if latest is not None:
            try:
                q.put_nowait(latest)
            except queue.Full:
                pass
        return q

    def unregister(self, q):
        with self._lock:
            self._clients.discard(q)

    def publish(self, payload: str):
        with self._lock:
            self._latest = payload
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                # 느린 클라이언트: 가장 오래된 것 버리고 최신 넣기
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except queue.Empty:
                    pass


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, broadcaster=None, image_bc=None, directory=None,
                 splat_path=None, topdown_splat_path=None, **kwargs):
        self._bc = broadcaster
        self._image_bc = image_bc
        self._splat_path = splat_path
        self._topdown_splat_path = topdown_splat_path
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, *args):
        pass  # 조용히

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        # 개발 중 viewer.js/index.html 등이 브라우저에 캐시되어 옛 버전이
        # 뜨는 것을 막는다(새로고침만으로 최신 코드 반영).
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/events":
            self._serve_sse()
            return
        if path == "/gaussian_map.splat" and self._splat_path:
            self._serve_splat(self._splat_path)
            return
        if path == "/gaussian_map_topdown.splat" and self._topdown_splat_path:
            self._serve_splat(self._topdown_splat_path)
            return
        if path == "/camera.mjpg" and self._image_bc is not None:
            self._serve_mjpg()
            return
        if path == "/config":
            self._serve_config()
            return
        if path == "/clientlog":
            from urllib.parse import urlparse, parse_qs
            msg = parse_qs(urlparse(self.path).query).get("m", [""])[0]
            print(f"[web client] {msg}", flush=True)
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()

    def do_HEAD(self):
        path = self.path.split("?")[0]
        if path == "/gaussian_map.splat" and self._splat_path:
            self._serve_splat(self._splat_path, head_only=True)
            return
        if path == "/gaussian_map_topdown.splat" and self._topdown_splat_path:
            self._serve_splat(self._topdown_splat_path, head_only=True)
            return
        super().do_HEAD()

    def _serve_config(self):
        """뷰어에 사용 가능한 부가 스트림(top-down splat, 카메라)을 알려준다."""
        td_ok = bool(self._topdown_splat_path
                     and os.path.exists(self._topdown_splat_path))
        body = json.dumps({
            "topdown": "/gaussian_map_topdown.splat" if td_ok else None,
            "camera": "/camera.mjpg" if self._image_bc is not None else None,
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _serve_mjpg(self):
        """카메라 프레임을 multipart/x-mixed-replace(MJPEG)로 스트리밍.
        브라우저는 <img src="/camera.mjpg"> 만으로 실시간 갱신된다."""
        boundary = "rlframe"
        self.send_response(200)
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = self._image_bc.register()
        try:
            while True:
                try:
                    frame = q.get(timeout=5.0)
                except queue.Empty:
                    continue  # 아직 프레임 없음 → 연결 유지
                self.wfile.write(
                    f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self._image_bc.unregister(q)

    def _serve_splat(self, splat_path, head_only=False):
        if not os.path.exists(splat_path):
            self.send_error(404, f"splat not found: {splat_path}")
            return
        try:
            size = os.path.getsize(splat_path)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if head_only:
                return
            with open(splat_path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        q = self._bc.register()
        try:
            # 초기 연결 확인용 주석
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while True:
                try:
                    payload = q.get(timeout=10.0)
                except queue.Empty:
                    # heartbeat (죽은 연결 감지 + keep-alive)
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                self.wfile.write(b"data: " + payload.encode("utf-8") + b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self._bc.unregister(q)


class WebPoseBridge(Node):
    def __init__(self):
        super().__init__("web_pose_bridge")
        self.declare_parameter("pose_topic", "/vps/current_pose")
        self.declare_parameter("pose_topic_type", "auto")  # auto(=pose) | pose | odom
        self.declare_parameter("web_dir", "")
        self.declare_parameter("splat_path", "")
        self.declare_parameter("topdown_splat_path", "")  # 비우면 splat_path 에서 파생
        self.declare_parameter("image_topic", "")  # 비우면 카메라 패널 비활성
        self.declare_parameter("image_topic_type", "auto")  # auto | raw | compressed
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8080)

        pose_topic = self.get_parameter("pose_topic").value
        topic_type = str(self.get_parameter("pose_topic_type").value).lower()
        web_dir = self.get_parameter("web_dir").value or _default_web_dir()
        splat_path = self.get_parameter("splat_path").value
        host = self.get_parameter("host").value
        port = int(self.get_parameter("port").value)
        self.web_dir = os.path.abspath(web_dir)
        if splat_path:
            if not os.path.isabs(splat_path):
                splat_path = os.path.abspath(splat_path)
        else:
            splat_path = os.path.join(self.web_dir, "gaussian_map.splat")
        self.splat_path = splat_path

        topdown_path = self.get_parameter("topdown_splat_path").value
        if not topdown_path:
            # launch 의 bash 규칙과 동일: foo.splat → foo_topdown.splat
            base, ext = os.path.splitext(self.splat_path)
            topdown_path = base + "_topdown" + (ext or ".splat")
        elif not os.path.isabs(topdown_path):
            topdown_path = os.path.abspath(topdown_path)
        self.topdown_splat_path = topdown_path

        if not os.path.exists(self.splat_path):
            self.get_logger().warn(
                f"gaussian_map.splat 없음: {self.splat_path}\n"
                f"  먼저: python3 scripts/ros/ply_to_splat.py "
                f"<aligned_map.ply> {self.splat_path}")
        if not os.path.exists(self.topdown_splat_path):
            self.get_logger().warn(
                f"top-down splat 없음: {self.topdown_splat_path}\n"
                f"  top-down 뷰는 full splat 으로 대체됩니다. 생성하려면: "
                f"ply_to_splat.py <aligned_map.ply> {self.topdown_splat_path} "
                f"--z-min 0 --z-max 3")

        self._bc = _Broadcaster()
        self._subs = []
        if topic_type == "auto":
            self.get_logger().info(
                "pose_topic_type=auto → PoseStamped로 구독합니다. "
                "Odometry topic이면 pose_topic_type:=odom 을 지정하세요.")
            topic_type = "pose"
        if topic_type in ("pose", "posestamped", "pose_stamped"):
            self._subs.append(
                self.create_subscription(PoseStamped, pose_topic, self._on_pose, 10))
        if topic_type in ("odom", "odometry"):
            self._subs.append(
                self.create_subscription(Odometry, pose_topic, self._on_odom, 10))
        if not self._subs:
            raise ValueError(
                "pose_topic_type must be one of: auto, pose, odom "
                f"(got {topic_type!r})")

        # 카메라 image 토픽 → MJPEG. 토픽이 비어있으면 패널 자체를 비활성.
        self._image_bc = None
        image_topic = self.get_parameter("image_topic").value
        if image_topic:
            img_type = str(self.get_parameter("image_topic_type").value).lower()
            if img_type == "auto":
                img_type = ("compressed" if image_topic.rstrip("/").endswith(
                    "compressed") else "raw")
            self._image_bc = _Broadcaster()
            self._cv_bridge = None
            if img_type == "compressed":
                self._subs.append(self.create_subscription(
                    CompressedImage, image_topic, self._on_compressed_image, 5))
            else:
                from cv_bridge import CvBridge
                self._cv_bridge = CvBridge()
                self._subs.append(self.create_subscription(
                    Image, image_topic, self._on_raw_image, 5))
            self.get_logger().info(
                f"카메라 구독: {image_topic} type={img_type} → MJPEG /camera.mjpg")

        handler = partial(
            _Handler,
            broadcaster=self._bc,
            image_bc=self._image_bc,
            directory=self.web_dir,
            splat_path=self.splat_path,
            topdown_splat_path=self.topdown_splat_path,
        )
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._srv_thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True)
        self._srv_thread.start()

        self.get_logger().info(
            f"웹 뷰어 서빙: http://localhost:{port}  (web_dir={self.web_dir})")
        self.get_logger().info(f"splat 서빙: {self.splat_path}")
        td_state = ("있음" if os.path.exists(self.topdown_splat_path) else "없음")
        self.get_logger().info(
            f"top-down splat({td_state}): {self.topdown_splat_path}")
        self.get_logger().info(
            f"pose 구독: {pose_topic} type={topic_type} → SSE /events")

    def _on_compressed_image(self, msg: CompressedImage):
        fmt = (msg.format or "").lower()
        if "jpeg" in fmt or "jpg" in fmt:
            self._image_bc.publish(bytes(msg.data))  # 이미 JPEG → 그대로
            return
        # png 등 다른 압축 → cv2 로 디코드 후 JPEG 재인코딩
        if not _CV_OK:
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        self._encode_and_publish(img)

    def _on_raw_image(self, msg: Image):
        if not _CV_OK or self._cv_bridge is None:
            return
        try:
            img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"이미지 변환 실패: {e}", throttle_duration_sec=5.0)
            return
        self._encode_and_publish(img)

    def _encode_and_publish(self, img):
        if img is None:
            return
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            self._image_bc.publish(buf.tobytes())

    def _on_pose(self, msg: PoseStamped):
        self._publish_pose_payload(
            msg.pose, msg.header.stamp, msg.header.frame_id,
            self.get_parameter("pose_topic").value)

    def _on_odom(self, msg: Odometry):
        frame_id = msg.header.frame_id or msg.child_frame_id
        self._publish_pose_payload(
            msg.pose.pose, msg.header.stamp, frame_id,
            self.get_parameter("pose_topic").value)

    def _publish_pose_payload(self, pose, stamp, frame_id, topic):
        p, o = pose.position, pose.orientation
        payload = json.dumps({
            "type": "pose",
            "position": {"x": p.x, "y": p.y, "z": p.z},
            "quaternion": {"x": o.x, "y": o.y, "z": o.z, "w": o.w},
            "stamp": stamp.sec + stamp.nanosec * 1e-9,
            "frame_id": frame_id,
            "topic": topic,
        })
        self._bc.publish(payload)

    def destroy_node(self):
        try:
            self._httpd.shutdown()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebPoseBridge()
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
