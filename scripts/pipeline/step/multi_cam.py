"""Multi-camera utility for kapture/COLMAP-style datasets.

kapture records_camera.txt 구조:
  timestamp, cam_id, cam_X/images/timestamp.jpg

같은 timestamp = 같은 rig 위치에서 동시 촬영 → 파일명(timestamp)이 공통 키.

COLMAP fallback:
  cameras.txt + images.txt만 있는 경우 cam_X/images/<timestamp>.jpg 경로와
  timestamp 근접값으로 같은 rig 촬영 프레임을 묶는다.
"""
import os
import re
import glob
import numpy as np


def parse_kapture_records(kapture_dir):
    """records_camera.txt 파싱. 없으면 COLMAP images.txt fallback.

    Returns:
        dict: {timestamp_str: {cam_id: abs_path}}
    """
    rec_path  = os.path.join(kapture_dir, "records_camera.txt")
    data_base = os.path.join(kapture_dir, "records_data")
    records   = {}
    if not os.path.exists(rec_path):
        colmap_images = os.path.join(kapture_dir, "images.txt")
        if os.path.exists(colmap_images):
            return parse_colmap_records(kapture_dir)
        # 정성평가용 데이터: pose record 파일이 없고 cam_X/images/ 디렉터리만 존재.
        # 디렉터리를 직접 스캔해 timestamp 기반 sister 인덱스를 만든다.
        scanned = _scan_image_dir_records(kapture_dir)
        if scanned is not None:
            return scanned
        raise FileNotFoundError(
            f"Neither records_camera.txt nor images.txt found under {kapture_dir}")

    with open(rec_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            ts, cam_id, rel_path = parts[0], parts[1], parts[2]
            abs_path = os.path.join(data_base, rel_path)
            records.setdefault(ts, {})[cam_id] = abs_path
    return records


def _scan_image_dir_records(base_dir):
    """records 파일이 없는 정성평가용 데이터를 디렉터리 스캔으로 인덱싱한다.

    구조: ``base_dir/cam_X/images/<timestamp>.<ext>`` (images/ 없으면 cam_X/ 직속도 허용)

    카메라마다 timestamp가 미세하게 달라 stem이 정확히 일치하지 않을 수 있으므로,
    COLMAP fallback과 동일하게 ``__colmap_by_cam__`` 최근접-timestamp 인덱스를 만든다.
    동기화 허용오차는 프레임 간격 중앙값의 절반으로 추정한다.

    Returns:
        dict 또는 None (cam_X 디렉터리를 못 찾으면 None)
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    records = {}
    by_cam = {}
    for cam_dir in sorted(glob.glob(os.path.join(base_dir, "cam_*"))):
        if not os.path.isdir(cam_dir):
            continue
        cam_id = os.path.basename(cam_dir)
        img_dir = os.path.join(cam_dir, "images")
        if not os.path.isdir(img_dir):
            img_dir = cam_dir
        for fname in sorted(os.listdir(img_dir)):
            if os.path.splitext(fname)[1].lower() not in exts:
                continue
            stem = os.path.splitext(fname)[0]
            path = os.path.abspath(os.path.join(img_dir, fname))
            records.setdefault(stem, {})[cam_id] = path
            try:
                ts = int(stem)
            except ValueError:
                ts = None
            if ts is not None:
                by_cam.setdefault(cam_id, []).append((ts, path))

    if not records:
        return None

    gaps = []
    for cam_id in by_cam:
        by_cam[cam_id].sort(key=lambda item: item[0])
        ts_seq = [t for t, _ in by_cam[cam_id]]
        gaps.extend(abs(b - a) for a, b in zip(ts_seq, ts_seq[1:]))
    tolerance = int(np.median(gaps) * 0.5) if gaps else 200_000_000

    records["__colmap_by_cam__"] = by_cam
    records["__colmap_sync_tolerance_ns__"] = max(tolerance, 1)
    return records


def _parse_colmap_images(colmap_dir):
    images_path = os.path.join(colmap_dir, "images.txt")
    rows = []
    if not os.path.exists(images_path):
        return rows
    with open(images_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            name = parts[9]
            if not re.search(r"\.(jpg|jpeg|png|bmp)$", name, re.IGNORECASE):
                continue
            cam_id = infer_cam_id_from_path(name)
            stem = os.path.splitext(os.path.basename(name))[0]
            try:
                ts = int(stem)
            except ValueError:
                ts = None
            try:
                camera_id = int(parts[8])
                qvec = np.array([float(v) for v in parts[1:5]], dtype=np.float64)
                tvec = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
            except ValueError:
                camera_id, qvec, tvec = None, None, None
            rows.append({
                "name": name,
                "path": os.path.abspath(os.path.join(colmap_dir, name)),
                "cam_id": cam_id,
                "timestamp": ts,
                "stem": stem,
                "camera_id": camera_id,
                "qvec": qvec,
                "tvec": tvec,
            })
    return rows


def parse_colmap_records(colmap_dir):
    """COLMAP images.txt → kapture-like records.

    Exact timestamp records도 만들고, timestamp가 카메라별로 조금 다를 때를 위해
    ``__colmap_by_cam__`` 인덱스를 함께 저장한다.
    """
    records = {}
    by_cam = {}
    for row in _parse_colmap_images(colmap_dir):
        cam_id = row["cam_id"]
        if not cam_id:
            continue
        records.setdefault(row["stem"], {})[cam_id] = row["path"]
        if row["timestamp"] is not None:
            by_cam.setdefault(cam_id, []).append((row["timestamp"], row["path"]))

    for cam_id in list(by_cam):
        by_cam[cam_id].sort(key=lambda x: x[0])
    records["__colmap_by_cam__"] = by_cam
    records["__colmap_sync_tolerance_ns__"] = 200_000_000
    return records


def parse_kapture_sensors(kapture_dir):
    """sensors.txt 파싱 → 카메라별 intrinsics. 없으면 COLMAP cameras.txt fallback.

    Returns:
        dict: {cam_id: {"fx","fy","cx","cy","width","height"}}
    """
    sensors_path = os.path.join(kapture_dir, "sensors.txt")
    sensors = {}
    if not os.path.exists(sensors_path):
        return parse_colmap_sensors(kapture_dir)
    with open(sensors_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            # cam_0, cam_0, camera, PINHOLE, W, H, fx, fy, cx, cy
            if len(parts) < 10 or parts[2] != "camera" or parts[3] != "PINHOLE":
                continue
            cam_id = parts[0]
            sensors[cam_id] = {
                "width":  int(parts[4]),
                "height": int(parts[5]),
                "fx":     float(parts[6]),
                "fy":     float(parts[7]),
                "cx":     float(parts[8]),
                "cy":     float(parts[9]),
            }
    return sensors


def parse_colmap_sensors(colmap_dir):
    cameras_path = os.path.join(colmap_dir, "cameras.txt")
    if not os.path.exists(cameras_path):
        return {}

    camera_to_cam = {}
    for row in _parse_colmap_images(colmap_dir):
        if row["camera_id"] is not None and row["cam_id"]:
            camera_to_cam.setdefault(row["camera_id"], row["cam_id"])

    sensors = {}
    with open(cameras_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                camera_id = int(parts[0])
                model = parts[1].upper()
                width = int(parts[2])
                height = int(parts[3])
                params = [float(v) for v in parts[4:]]
            except ValueError:
                continue

            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = params[0]; cx = params[1]; cy = params[2]
            else:
                fx, fy, cx, cy = params[:4]
            cam_id = camera_to_cam.get(camera_id, f"cam_{camera_id - 1}")
            sensors[cam_id] = {
                "width": width, "height": height,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            }
    return sensors


def infer_cam_id_from_path(path, cam_ids=None):
    """이미지 경로에서 cam_0/cam_1/... ID를 추론한다.

    kapture records_data/cam_X/images/... 형태를 우선 지원하고, 일반 경로에
    포함된 cam_X 패턴도 fallback으로 처리한다.
    """
    if not path:
        return None

    norm = os.path.normpath(str(path))
    parts = norm.split(os.sep)
    candidates = [p for p in parts if re.fullmatch(r"cam_\d+", p)]
    if not candidates:
        candidates = re.findall(r"cam_\d+", norm)
    if not candidates:
        return None

    if cam_ids:
        allowed = set(cam_ids)
        for cam_id in reversed(candidates):
            if cam_id in allowed:
                return cam_id
    return candidates[-1]


def find_sister_images(query_image_path, records, cam_ids):
    """query_image_path의 timestamp로 자매 카메라 이미지 경로 반환.

    Args:
        query_image_path: 현재 쿼리 이미지 경로 (파일명 stem = timestamp)
        records:          parse_kapture_records() 결과
        cam_ids:          사용할 카메라 ID 리스트

    Returns:
        dict: {cam_id: abs_path}  (존재하는 카메라만 포함)
    """
    stem = os.path.splitext(os.path.basename(query_image_path))[0]
    by_cam = records.get("__colmap_by_cam__") if isinstance(records, dict) else None
    if by_cam:
        try:
            query_ts = int(stem)
        except ValueError:
            query_ts = None
        tolerance = int(records.get("__colmap_sync_tolerance_ns__", 200_000_000))
        result = {}
        query_cam = infer_cam_id_from_path(query_image_path, cam_ids)
        for cam_id in cam_ids:
            seq = by_cam.get(cam_id, [])
            best = None
            if query_ts is not None and seq:
                # sorted list라 선형 scan도 현재 규모(수천장)에서는 충분히 작다.
                best = min(seq, key=lambda item: abs(item[0] - query_ts))
                if abs(best[0] - query_ts) > tolerance:
                    best = None
            path = best[1] if best else None
            if cam_id == query_cam:
                path = os.path.abspath(query_image_path)
            if path and os.path.exists(path):
                result[cam_id] = path
        return result

    frame = records.get(stem, {})
    result = {}
    for cam_id in cam_ids:
        path = frame.get(cam_id)
        if path and os.path.exists(path):
            result[cam_id] = path
    return result


def parse_kapture_rigs(kapture_dir):
    """rigs.txt 파싱 → 카메라별 T_rig_to_cam (4×4).

    rigs.txt가 없으면 COLMAP images.txt의 동시 프레임 pose로 상대 extrinsic을
    추정한다. 이때 cam_0을 rig frame으로 둔다.

    kapture 형식: rig_id, sensor_id, qw, qx, qy, qz, tx, ty, tz
    반환값: {cam_id: np.ndarray (4×4)}  (rig frame → cam frame 변환)
    """
    from scipy.spatial.transform import Rotation

    rigs_path = os.path.join(kapture_dir, "rigs.txt")
    result = {}
    if not os.path.exists(rigs_path):
        return parse_colmap_rigs(kapture_dir)

    with open(rigs_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue
            cam_id = parts[1]
            qw, qx, qy, qz = map(float, parts[2:6])
            tx, ty, tz      = map(float, parts[6:9])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3]  = [tx, ty, tz]
            result[cam_id] = T  # T_rig_to_cam

    return result


def normalize_rig_transforms(rigs, config=None, direction=None):
    """step7 내부에서 쓰는 T_rig_to_cam(T_C_R) 규약으로 rig extrinsic을 정규화.

    Kapture/내부 dataset에 따라 rigs.txt의 4x4가 rig→cam일 수도,
    cam→rig일 수도 있어서 config로 명시할 수 있게 둔다.
    """
    if direction is None:
        mc = (config or {}).get("multi_cam", {}) if config is not None else {}
        direction = mc.get("rig_transform_direction", "rig_to_cam")
    direction = str(direction or "rig_to_cam").strip().lower()
    if direction in ("rig_to_cam", "rig2cam", "t_rig_to_cam", "t_c_r"):
        return rigs
    if direction in ("cam_to_rig", "camera_to_rig", "sensor_to_rig",
                     "cam2rig", "t_cam_to_rig", "t_r_c"):
        return {cam_id: np.linalg.inv(T) for cam_id, T in rigs.items()}
    raise ValueError(
        "multi_cam.rig_transform_direction must be rig_to_cam or cam_to_rig "
        f"(got {direction!r})")


def _colmap_qt_to_c2w(qvec, tvec):
    from scipy.spatial.transform import Rotation
    R_cw = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
    T_cw = np.eye(4, dtype=np.float64)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = tvec
    return np.linalg.inv(T_cw)


def parse_colmap_rigs(colmap_dir, primary_cam="cam_0", tolerance_ns=200_000_000):
    rows = [r for r in _parse_colmap_images(colmap_dir)
            if r["cam_id"] and r["timestamp"] is not None
            and r["qvec"] is not None and r["tvec"] is not None]
    by_cam = {}
    for row in rows:
        by_cam.setdefault(row["cam_id"], []).append(row)
    for cam_id in by_cam:
        by_cam[cam_id].sort(key=lambda r: r["timestamp"])
    if primary_cam not in by_cam and by_cam:
        primary_cam = sorted(by_cam)[0]

    rigs = {primary_cam: np.eye(4, dtype=np.float64)}
    primary_rows = by_cam.get(primary_cam, [])
    if not primary_rows:
        return rigs

    for cam_id, seq in by_cam.items():
        if cam_id == primary_cam:
            continue
        rels = []
        for row in seq:
            ref = min(primary_rows, key=lambda r: abs(r["timestamp"] - row["timestamp"]))
            if abs(ref["timestamp"] - row["timestamp"]) > tolerance_ns:
                continue
            T_wc_cam = _colmap_qt_to_c2w(row["qvec"], row["tvec"])
            T_wc_rig = _colmap_qt_to_c2w(ref["qvec"], ref["tvec"])
            rels.append(np.linalg.inv(T_wc_cam) @ T_wc_rig)
        if not rels:
            continue
        Ts = np.stack(rels, axis=0)
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = np.median(Ts[:, :3, 3], axis=0)
        try:
            from scipy.spatial.transform import Rotation
            T[:3, :3] = Rotation.from_matrix(Ts[:, :3, :3]).mean().as_matrix()
        except Exception:
            T[:3, :3] = Ts[0, :3, :3]
        rigs[cam_id] = T
    return rigs


def load_multi_cam_config(config):
    """config에서 multi_cam 설정 파싱.

    Returns:
        enabled    (bool)
        cam_ids    (list[str])
        kapture_dir (str)
        primary_cam (str)
    """
    mc = config.get("multi_cam", {})
    enabled     = bool(mc.get("enabled", False))
    cam_ids     = list(mc.get("cam_ids", ["cam_3"]))
    kapture_dir = mc.get("kapture_dir", "kapture/sensors")
    primary_cam = mc.get("primary_cam", cam_ids[0] if cam_ids else "cam_3")
    return enabled, cam_ids, kapture_dir, primary_cam
