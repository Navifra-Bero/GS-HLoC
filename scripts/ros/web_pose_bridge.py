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
import signal
import subprocess
import struct
import threading
import time
import zlib
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, String

try:
    import cv2
    import numpy as np
    _CV_OK = True
except Exception:  # cv2/numpy 없으면 raw Image/png 변환은 비활성, jpeg 패스스루만.
    _CV_OK = False

try:
    from plyfile import PlyData
    _PLY_OK = True
except Exception:
    _PLY_OK = False


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


def _coerce_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class _Broadcaster:
    """SSE 클라이언트 큐 레지스트리 + 최신 pose 캐시."""

    def __init__(self, queue_size=10):
        self._clients = set()
        self._lock = threading.Lock()
        self._latest = None
        self._queue_size = int(queue_size)

    def register(self):
        q = queue.Queue(maxsize=max(1, self._queue_size))
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

    def has_clients(self):
        with self._lock:
            return bool(self._clients)

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
    def __init__(self, *args, broadcaster=None, image_bc=None, image_bcs=None,
                 directory=None,
                 splat_path=None, topdown_splat_path=None,
                 topdown_map_path=None, topdown_map_meta_path=None,
                 trajectory_json_path=None, **kwargs):
        self._bc = broadcaster
        self._image_bc = image_bc
        self._image_bcs = image_bcs or []
        self._splat_path = splat_path
        self._topdown_splat_path = topdown_splat_path
        self._topdown_map_path = topdown_map_path
        self._topdown_map_meta_path = topdown_map_meta_path
        self._trajectory_json_path = trajectory_json_path
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
            self._serve_mjpg(self._image_bc)
            return
        if path.startswith("/camera_") and path.endswith(".mjpg"):
            try:
                idx = int(path[len("/camera_"):-len(".mjpg")])
            except ValueError:
                idx = -1
            if 0 <= idx < len(self._image_bcs):
                self._serve_mjpg(self._image_bcs[idx][0])
            return
        if path == "/topdown_map.png" and self._topdown_map_path:
            self._serve_file(self._topdown_map_path, "image/png")
            return
        if path == "/topdown_map.json" and self._topdown_map_meta_path:
            self._serve_file(self._topdown_map_meta_path, "application/json")
            return
        if path == "/trajectory_poses.json" and self._trajectory_json_path:
            self._serve_file(self._trajectory_json_path, "application/json")
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

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/test_bag/play":
            self._serve_test_bag_play()
            return
        if path == "/test_bag/stop":
            stopped = self._stop_test_bag()
            self._serve_json({"ok": True, "stopped": stopped})
            return
        if path == "/localizer/start":
            self._serve_localizer_control(True)
            return
        if path == "/localizer/stop":
            self._serve_localizer_control(False)
            return
        if path == "/localizer/debug/toggle":
            self._serve_localizer_debug_toggle()
            return
        self.send_error(404, f"unknown endpoint: {path}")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET,HEAD,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_HEAD(self):
        path = self.path.split("?")[0]
        if path == "/gaussian_map.splat" and self._splat_path:
            self._serve_splat(self._splat_path, head_only=True)
            return
        if path == "/gaussian_map_topdown.splat" and self._topdown_splat_path:
            self._serve_splat(self._topdown_splat_path, head_only=True)
            return
        if path == "/topdown_map.png" and self._topdown_map_path:
            self._serve_file(self._topdown_map_path, "image/png", head_only=True)
            return
        if path == "/topdown_map.json" and self._topdown_map_meta_path:
            self._serve_file(self._topdown_map_meta_path, "application/json",
                             head_only=True)
            return
        if path == "/trajectory_poses.json" and self._trajectory_json_path:
            self._serve_file(self._trajectory_json_path, "application/json",
                             head_only=True)
            return
        super().do_HEAD()

    def _serve_config(self):
        """뷰어에 사용 가능한 부가 스트림(top-down splat, 카메라)을 알려준다."""
        td_ok = bool(self._topdown_splat_path
                     and os.path.exists(self._topdown_splat_path))
        map_ok = bool(self._topdown_map_path and self._topdown_map_meta_path
                      and os.path.exists(self._topdown_map_path)
                      and os.path.exists(self._topdown_map_meta_path))
        traj_ok = bool(self._trajectory_json_path
                       and os.path.exists(self._trajectory_json_path))
        traj_autoplay = bool(getattr(self.server, "trajectory_autoplay", False))
        traj_image_stride = int(getattr(self.server, "trajectory_image_stride", 1))
        body = json.dumps({
            "topdown": "/gaussian_map_topdown.splat" if td_ok else None,
            "topdown_map": "/topdown_map.png" if map_ok else None,
            "topdown_map_meta": "/topdown_map.json" if map_ok else None,
            "trajectory": "/trajectory_poses.json" if traj_ok else None,
            "trajectory_autoplay": traj_autoplay,
            "trajectory_image_stride": traj_image_stride,
            "test_bag": self._valid_test_bag_path() is not None,
            "camera": "/camera.mjpg" if self._image_bc is not None else None,
            "cameras": [
                {"url": f"/camera_{i}.mjpg", "label": label}
                for i, (_bc, label) in enumerate(self._image_bcs)
            ],
            "localizer_control": True,
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _serve_json(self, body, status=200):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _serve_localizer_control(self, enabled):
        pub = getattr(self.server, "localizer_control_pub", None)
        if pub is None:
            self._serve_json({
                "ok": False,
                "error": "localizer control publisher unavailable",
            }, status=500)
            return
        msg = Bool()
        msg.data = bool(enabled)
        pub.publish(msg)
        self._serve_json({"ok": True, "enabled": bool(enabled)})

    def _serve_localizer_debug_toggle(self):
        pub = getattr(self.server, "localizer_debug_pub", None)
        if pub is None:
            self._serve_json({
                "ok": False,
                "error": "localizer debug publisher unavailable",
            }, status=500)
            return
        enabled = not bool(getattr(self.server, "localizer_debug_enabled", False))
        self.server.localizer_debug_enabled = enabled
        msg = Bool()
        msg.data = enabled
        pub.publish(msg)
        self._serve_json({"ok": True, "enabled": enabled})

    def _valid_test_bag_path(self):
        bag_path = getattr(self.server, "test_bag_path", "")
        if not bag_path:
            return None
        bag_path = os.path.abspath(bag_path)
        if not os.path.exists(bag_path):
            return None
        return bag_path

    def _serve_test_bag_play(self):
        bag_path = self._valid_test_bag_path()
        if bag_path is None:
            self._serve_json({
                "ok": False,
                "error": "test_bag_path is empty or missing",
            }, status=404)
            return

        lock = getattr(self.server, "test_bag_lock", None)
        if lock is None:
            self._serve_json({"ok": False, "error": "bag lock unavailable"}, status=500)
            return

        with lock:
            existing = getattr(self.server, "test_bag_proc", None)
            if existing is not None and existing.poll() is None:
                self._serve_json({
                    "ok": True,
                    "bag": bag_path,
                    "pid": existing.pid,
                    "already_running": True,
                })
                return
            self._stop_test_bag_locked()
            try:
                log_path = "/tmp/render_loc_bag_play.log"
                log_f = open(log_path, "ab", buffering=0)
                log_f.write(
                    f"\n--- ros2 bag play {bag_path} ---\n".encode("utf-8"))
                env = os.environ.copy()
                env.setdefault("ROS_LOG_DIR", "/tmp")
                proc = subprocess.Popen(
                    ["ros2", "bag", "play", bag_path],
                    stdout=log_f,
                    stderr=log_f,
                    env=env,
                    start_new_session=True,
                )
            except Exception as e:  # noqa: BLE001
                self._serve_json({"ok": False, "error": str(e)}, status=500)
                return
            time.sleep(0.5)
            if proc.poll() is not None:
                try:
                    log_f.close()
                except Exception:
                    pass
                err = self._tail_file(log_path)
                self._serve_json({
                    "ok": False,
                    "error": f"ros2 bag play exited immediately ({proc.returncode})",
                    "detail": err,
                }, status=500)
                return
            self.server.test_bag_proc = proc
            self.server.test_bag_log_file = log_f
        self._serve_json({"ok": True, "bag": bag_path, "pid": proc.pid})

    @staticmethod
    def _tail_file(path, max_bytes=4000):
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes), os.SEEK_SET)
                return f.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _stop_test_bag_locked(self):
        proc = getattr(self.server, "test_bag_proc", None)
        if proc is None or proc.poll() is not None:
            self.server.test_bag_proc = None
            log_f = getattr(self.server, "test_bag_log_file", None)
            if log_f is not None:
                try:
                    log_f.close()
                except Exception:
                    pass
                self.server.test_bag_log_file = None
            return False
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=2.0)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        self.server.test_bag_proc = None
        log_f = getattr(self.server, "test_bag_log_file", None)
        if log_f is not None:
            try:
                log_f.close()
            except Exception:
                pass
            self.server.test_bag_log_file = None
        return True

    def _stop_test_bag(self):
        lock = getattr(self.server, "test_bag_lock", None)
        if lock is None:
            return False
        with lock:
            return self._stop_test_bag_locked()

    def _serve_mjpg(self, image_bc):
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
        q = image_bc.register()
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
            image_bc.unregister(q)

    def _serve_file(self, path, content_type, head_only=False):
        if not os.path.exists(path):
            self.send_error(404, f"file not found: {path}")
            return
        try:
            size = os.path.getsize(path)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            if head_only:
                return
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

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


def _png_chunk(kind, data):
    chunk = kind + data
    return (struct.pack(">I", len(data)) + chunk
            + struct.pack(">I", zlib.crc32(chunk) & 0xffffffff))


def _encode_png_rgb(img):
    """uint8 RGB image -> PNG bytes. Small stdlib encoder for cached minimaps."""
    h, w, c = img.shape
    if c != 3:
        raise ValueError("PNG encoder expects RGB image")
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    return b"".join([
        b"\x89PNG\r\n\x1a\n",
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(raw, 6)),
        _png_chunk(b"IEND", b""),
    ])


def _read_ply_rgb(v, names):
    if all(f"f_dc_{i}" in names for i in range(3)):
        sh_c0 = 0.28209479177387814
        rgb = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
        return np.clip((0.5 + sh_c0 * rgb.astype(np.float32)) * 255.0, 0, 255)
    if all(c in names for c in ("red", "green", "blue")):
        return np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32)
    return None


def _build_topdown_map(ply_path, png_path, meta_path, z_min=None, z_max=None,
                       image_size=1024):
    """Rasterize aligned_map.ply to a north-up XY PNG plus world bounds JSON."""
    if not (_PLY_OK and _CV_OK):
        raise RuntimeError("top-down map needs numpy and plyfile")
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    names = v.dtype.names or ()
    x = np.asarray(v["x"], dtype=np.float32)
    y = np.asarray(v["y"], dtype=np.float32)
    z = np.asarray(v["z"], dtype=np.float32)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if z_min is not None:
        m &= z >= float(z_min)
    if z_max is not None:
        m &= z <= float(z_max)
    if not np.any(m):
        raise RuntimeError("top-down map has no points after filtering")

    x = x[m]
    y = y[m]
    rgb_all = _read_ply_rgb(v, names)
    rgb = rgb_all[m] if rgb_all is not None else None

    min_x, max_x = float(np.min(x)), float(np.max(x))
    min_y, max_y = float(np.min(y)), float(np.max(y))
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)
    margin = max(span_x, span_y) * 0.035 + 0.5
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin

    span_x = max_x - min_x
    span_y = max_y - min_y
    span = max(span_x, span_y)
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    min_x = cx - span * 0.5
    max_x = cx + span * 0.5
    min_y = cy - span * 0.5
    max_y = cy + span * 0.5

    size = int(image_size)
    scale = (size - 1) / span
    px = np.clip(((x - min_x) * scale).astype(np.int32), 0, size - 1)
    py = np.clip(((max_y - y) * scale).astype(np.int32), 0, size - 1)
    idx = py * size + px
    n_pix = size * size
    count = np.bincount(idx, minlength=n_pix).astype(np.float32)

    bg = np.array([12, 15, 20], dtype=np.uint8)
    img = np.empty((size, size, 3), dtype=np.uint8)
    img[:] = bg
    mask = count > 0

    if rgb is not None:
        sums = [
            np.bincount(idx, weights=rgb[:, i], minlength=n_pix)
            for i in range(3)
        ]
        avg = np.stack(sums, axis=1)
        avg[mask] /= count[mask, None]
        p95 = max(float(np.percentile(count[mask], 95)), 1.0)
        density = np.clip(np.log1p(count) / np.log1p(p95), 0.0, 1.0)
        shaded = avg * (0.55 + 0.65 * density[:, None])
        img.reshape(-1, 3)[mask] = np.clip(shaded[mask], 0, 255).astype(np.uint8)
    else:
        p98 = max(float(np.percentile(count[mask], 98)), 1.0)
        val = 50.0 + 190.0 * np.clip(np.log1p(count) / np.log1p(p98), 0.0, 1.0)
        gray = np.stack([val, val, val], axis=1)
        img.reshape(-1, 3)[mask] = gray[mask].astype(np.uint8)

    # Sparse Gaussian centers read better after a tiny dilation at minimap scale.
    if size >= 512:
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(img, kernel, iterations=1)
        alpha_mask = mask.reshape(size, size).astype(np.uint8) * 255
        alpha = cv2.dilate(alpha_mask, kernel, iterations=1)
        img[alpha > 0] = dilated[alpha > 0]
        img = cv2.GaussianBlur(img, (3, 3), 0)

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    with open(png_path, "wb") as f:
        f.write(_encode_png_rgb(img))

    meta = {
        "image": "/topdown_map.png",
        "width": size,
        "height": size,
        "bounds": {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
        },
        "z_min": z_min,
        "z_max": z_max,
        "point_count": int(len(x)),
        "meters_per_pixel": span / size,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta


class WebPoseBridge(Node):
    def __init__(self):
        super().__init__("web_pose_bridge")
        self.declare_parameter("pose_topic", "/vps/current_pose")
        self.declare_parameter("pose_topic_type", "auto")  # auto(=pose) | pose | odom
        self.declare_parameter("web_dir", "")
        self.declare_parameter("aligned_ply", "")
        self.declare_parameter("splat_path", "")
        self.declare_parameter("topdown_splat_path", "")  # 비우면 splat_path 에서 파생
        self.declare_parameter("topdown_map_path", "")  # 비우면 splat_path 에서 파생
        self.declare_parameter("topdown_map_size", 1024)
        self.declare_parameter("topdown_z_min", -1.0)
        self.declare_parameter("topdown_z_max", 3.0)
        self.declare_parameter("trajectory_json", "")
        self.declare_parameter("trajectory_autoplay", False)
        self.declare_parameter("trajectory_image_stride", 1)
        self.declare_parameter("test_bag_path", "")
        self.declare_parameter("image_topic", "")  # 비우면 카메라 패널 비활성
        self.declare_parameter("image_topics", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("image_topic_type", "auto")  # auto | raw | compressed
        self.declare_parameter("camera_stream_enabled", True)
        self.declare_parameter("camera_stream_hz", 0.0)  # 0 이하 = 제한 없음
        self.declare_parameter("localizer_control_topic", "/vps/localizer_enabled")
        self.declare_parameter("localizer_debug_topic", "/vps/localizer_debug_enabled")
        self.declare_parameter("localizer_status_topic", "/vps/localizer_status")
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

        aligned_ply = self.get_parameter("aligned_ply").value
        if aligned_ply and not os.path.isabs(aligned_ply):
            aligned_ply = os.path.abspath(aligned_ply)
        topdown_map_path = self.get_parameter("topdown_map_path").value
        if not topdown_map_path:
            base, _ = os.path.splitext(self.splat_path)
            topdown_map_path = base + "_topdown.png"
        elif not os.path.isabs(topdown_map_path):
            topdown_map_path = os.path.abspath(topdown_map_path)
        topdown_map_meta_path = os.path.splitext(topdown_map_path)[0] + ".json"
        self.topdown_map_path = topdown_map_path
        self.topdown_map_meta_path = topdown_map_meta_path

        if aligned_ply and os.path.exists(aligned_ply):
            try:
                z_min = float(self.get_parameter("topdown_z_min").value)
                z_max = float(self.get_parameter("topdown_z_max").value)
                size = int(self.get_parameter("topdown_map_size").value)
                stale = (
                    not os.path.exists(self.topdown_map_path)
                    or not os.path.exists(self.topdown_map_meta_path)
                    or os.path.getmtime(self.topdown_map_path) < os.path.getmtime(aligned_ply)
                )
                if not stale:
                    with open(self.topdown_map_meta_path, encoding="utf-8") as f:
                        old_meta = json.load(f)
                    stale = (
                        int(old_meta.get("width", 0)) != size
                        or float(old_meta.get("z_min", z_min)) != z_min
                        or float(old_meta.get("z_max", z_max)) != z_max
                    )
                if stale:
                    self.get_logger().info(
                        f"PLY top-down 맵 생성: {aligned_ply} → {self.topdown_map_path}")
                    meta = _build_topdown_map(
                        aligned_ply, self.topdown_map_path,
                        self.topdown_map_meta_path,
                        z_min=z_min, z_max=z_max, image_size=size)
                    self.get_logger().info(
                        "PLY top-down 맵 완료: "
                        f"{meta['point_count']} pts, "
                        f"{meta['meters_per_pixel']:.3f} m/px")
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"PLY top-down 맵 생성 실패: {e}")
        elif aligned_ply:
            self.get_logger().warn(f"aligned_ply 없음: {aligned_ply}")

        trajectory_json = self.get_parameter("trajectory_json").value
        if trajectory_json and not os.path.isabs(trajectory_json):
            trajectory_json = os.path.abspath(trajectory_json)
        if not trajectory_json:
            map_dir = os.path.dirname(self.splat_path)
            candidates = [
                os.path.join(map_dir, "test_results", "cam_0", "trajectory_poses.json"),
                os.path.join(map_dir, "test_results", "cam_3", "trajectory_poses.json"),
                os.path.join(map_dir, "trajectory_poses.json"),
            ]
            trajectory_json = next((p for p in candidates if os.path.exists(p)), "")
        self.trajectory_json_path = trajectory_json

        test_bag_path = self.get_parameter("test_bag_path").value
        if test_bag_path and not os.path.isabs(test_bag_path):
            test_bag_path = os.path.abspath(test_bag_path)
        self.test_bag_path = test_bag_path

        self._bc = _Broadcaster()
        self._image_seq = 0
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

        control_topic = self.get_parameter("localizer_control_topic").value
        debug_topic = self.get_parameter("localizer_debug_topic").value
        status_topic = self.get_parameter("localizer_status_topic").value
        self.localizer_control_pub = self.create_publisher(Bool, control_topic, 10)
        self.localizer_debug_pub = self.create_publisher(Bool, debug_topic, 10)
        if status_topic:
            self._subs.append(self.create_subscription(
                String, status_topic, self._on_localizer_status, 10))

        # 카메라 image 토픽 → MJPEG. image_topics가 있으면 다중 패널로 서빙.
        self._image_bc = None
        self._image_bcs = []
        stream_enabled = _coerce_bool(self.get_parameter("camera_stream_enabled").value)
        stream_hz = float(self.get_parameter("camera_stream_hz").value)
        self._camera_stream_period = 1.0 / stream_hz if stream_hz > 0.0 else 0.0
        self._camera_stream_last_t = {}
        image_topics_value = self.get_parameter("image_topics").value
        image_topics = []
        if stream_enabled:
            image_topics = list(image_topics_value) if image_topics_value else []
            fallback_image_topic = self.get_parameter("image_topic").value
            if not image_topics and fallback_image_topic:
                image_topics = [fallback_image_topic]
        else:
            self.get_logger().info(
                "카메라 MJPEG 구독 비활성화(camera_stream_enabled=false)")
        img_type_param = str(self.get_parameter("image_topic_type").value).lower()
        self._cv_bridge = None
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        for image_idx, image_topic in enumerate(image_topics):
            if not image_topic:
                continue
            img_type = img_type_param
            if img_type == "auto":
                img_type = ("compressed" if image_topic.rstrip("/").endswith(
                    "compressed") else "raw")
            image_bc = _Broadcaster(queue_size=1)
            if self._image_bc is None:
                self._image_bc = image_bc
            label = image_topic.strip("/").split("/")[0] or f"camera {image_idx}"
            self._image_bcs.append((image_bc, label))
            if img_type == "compressed":
                self._subs.append(self.create_subscription(
                    CompressedImage, image_topic,
                    lambda msg, bc=image_bc: self._on_compressed_image(msg, bc),
                    image_qos))
            else:
                from cv_bridge import CvBridge
                if self._cv_bridge is None:
                    self._cv_bridge = CvBridge()
                self._subs.append(self.create_subscription(
                    Image, image_topic,
                    lambda msg, bc=image_bc: self._on_raw_image(msg, bc),
                    image_qos))
            self.get_logger().info(
                f"카메라 구독: {image_topic} type={img_type} "
                f"→ MJPEG /camera_{len(self._image_bcs) - 1}.mjpg "
                "(best_effort depth=1)")
        if self._camera_stream_period > 0.0:
            self.get_logger().info(
                f"카메라 MJPEG 표시 제한: {stream_hz:.1f}Hz "
                "(ROS image topic 구독/로컬라이제이션 입력은 제한하지 않음)")

        handler = partial(
            _Handler,
            broadcaster=self._bc,
            image_bc=self._image_bc,
            image_bcs=self._image_bcs,
            directory=self.web_dir,
            splat_path=self.splat_path,
            topdown_splat_path=self.topdown_splat_path,
            topdown_map_path=self.topdown_map_path,
            topdown_map_meta_path=self.topdown_map_meta_path,
            trajectory_json_path=self.trajectory_json_path,
        )
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._httpd.trajectory_autoplay = _coerce_bool(
            self.get_parameter("trajectory_autoplay").value)
        self._httpd.trajectory_image_stride = max(
            1, int(self.get_parameter("trajectory_image_stride").value))
        self._httpd.test_bag_path = self.test_bag_path
        self._httpd.test_bag_lock = threading.Lock()
        self._httpd.test_bag_proc = None
        self._httpd.localizer_control_pub = self.localizer_control_pub
        self._httpd.localizer_debug_pub = self.localizer_debug_pub
        self._httpd.localizer_debug_enabled = False
        self._srv_thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True)
        self._srv_thread.start()

        self.get_logger().info(
            f"웹 뷰어 서빙: http://localhost:{port}  (web_dir={self.web_dir})")
        self.get_logger().info(f"splat 서빙: {self.splat_path}")
        td_state = ("있음" if os.path.exists(self.topdown_splat_path) else "없음")
        self.get_logger().info(
            f"top-down splat({td_state}): {self.topdown_splat_path}")
        map_state = ("있음" if os.path.exists(self.topdown_map_path) else "없음")
        self.get_logger().info(
            f"PLY top-down map({map_state}): {self.topdown_map_path}")
        if self.trajectory_json_path:
            traj_state = ("있음" if os.path.exists(self.trajectory_json_path) else "없음")
            self.get_logger().info(
                f"trajectory json({traj_state}): {self.trajectory_json_path}")
        if self.test_bag_path:
            bag_state = ("있음" if os.path.exists(self.test_bag_path) else "없음")
            self.get_logger().info(
                f"test bag({bag_state}): {self.test_bag_path}")
        self.get_logger().info(
            f"pose 구독: {pose_topic} type={topic_type} → SSE /events")
        self.get_logger().info(
            f"localizer 제어: {control_topic}  debug: {debug_topic}  "
            f"상태: {status_topic or '-'}")

    def _on_compressed_image(self, msg: CompressedImage, image_bc=None):
        image_bc = image_bc or self._image_bc
        if image_bc is None:
            return
        if not self._should_publish_camera_frame(image_bc):
            return
        fmt = (msg.format or "").lower()
        if "jpeg" in fmt or "jpg" in fmt:
            image_bc.publish(bytes(msg.data))  # 이미 JPEG → 그대로
            self._publish_camera_frame_payload(msg.header.stamp, msg.header.frame_id)
            return
        # png 등 다른 압축 → cv2 로 디코드 후 JPEG 재인코딩
        if not _CV_OK:
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        self._encode_and_publish(img, image_bc)
        self._publish_camera_frame_payload(msg.header.stamp, msg.header.frame_id)

    def _on_raw_image(self, msg: Image, image_bc=None):
        image_bc = image_bc or self._image_bc
        if image_bc is None:
            return
        if not self._should_publish_camera_frame(image_bc):
            return
        if not _CV_OK or self._cv_bridge is None:
            return
        try:
            img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"이미지 변환 실패: {e}", throttle_duration_sec=5.0)
            return
        self._encode_and_publish(img, image_bc)
        self._publish_camera_frame_payload(msg.header.stamp, msg.header.frame_id)

    def _on_localizer_status(self, msg: String):
        try:
            payload = json.loads(msg.data)
            payload["type"] = "localizer_status"
            loc = payload.get("localization") or {}
            if hasattr(self, "_httpd"):
                self._httpd.localizer_debug_enabled = bool(loc.get("debug_enabled", False))
            self._bc.publish(json.dumps(payload))
        except Exception:
            self._bc.publish(json.dumps({
                "type": "localizer_status",
                "message": msg.data,
            }))

    def _encode_and_publish(self, img, image_bc=None):
        image_bc = image_bc or self._image_bc
        if img is None:
            return
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok and image_bc is not None:
            image_bc.publish(buf.tobytes())

    def _should_publish_camera_frame(self, image_bc):
        if image_bc is None or not image_bc.has_clients():
            return False
        if self._camera_stream_period <= 0.0:
            return True
        now = time.time()
        key = id(image_bc)
        last_t = self._camera_stream_last_t.get(key)
        if last_t is not None and now - last_t < self._camera_stream_period:
            return False
        self._camera_stream_last_t[key] = now
        return True

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

    def _publish_camera_frame_payload(self, stamp, frame_id):
        self._image_seq += 1
        stamp_ns = int(stamp.sec) * 1000000000 + int(stamp.nanosec)
        payload = json.dumps({
            "type": "camera_frame",
            "seq": self._image_seq,
            "stamp": stamp.sec + stamp.nanosec * 1e-9,
            "stamp_ns": stamp_ns,
            "frame_id": frame_id,
        })
        self._bc.publish(payload)

    def destroy_node(self):
        try:
            lock = getattr(self._httpd, "test_bag_lock", None)
            if lock is not None:
                with lock:
                    proc = getattr(self._httpd, "test_bag_proc", None)
                    if proc is not None and proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            pass
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
