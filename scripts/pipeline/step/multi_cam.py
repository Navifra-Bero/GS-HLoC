"""Multi-camera utility for kapture-based datasets.

kapture records_camera.txt 구조:
  timestamp, cam_id, cam_X/images/timestamp.jpg

같은 timestamp = 같은 rig 위치에서 동시 촬영 → 파일명(timestamp)이 공통 키.
"""
import os
import numpy as np


def parse_kapture_records(kapture_dir):
    """records_camera.txt 파싱.

    Returns:
        dict: {timestamp_str: {cam_id: abs_path}}
    """
    rec_path  = os.path.join(kapture_dir, "records_camera.txt")
    data_base = os.path.join(kapture_dir, "records_data")
    records   = {}
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


def parse_kapture_sensors(kapture_dir):
    """sensors.txt 파싱 → 카메라별 intrinsics.

    Returns:
        dict: {cam_id: {"fx","fy","cx","cy","width","height"}}
    """
    sensors_path = os.path.join(kapture_dir, "sensors.txt")
    sensors = {}
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
    frame = records.get(stem, {})
    result = {}
    for cam_id in cam_ids:
        path = frame.get(cam_id)
        if path and os.path.exists(path):
            result[cam_id] = path
    return result


def parse_kapture_rigs(kapture_dir):
    """rigs.txt 파싱 → 카메라별 T_rig_to_cam (4×4).

    kapture 형식: rig_id, sensor_id, qw, qx, qy, qz, tx, ty, tz
    반환값: {cam_id: np.ndarray (4×4)}  (rig frame → cam frame 변환)
    """
    from scipy.spatial.transform import Rotation

    rigs_path = os.path.join(kapture_dir, "rigs.txt")
    result = {}
    if not os.path.exists(rigs_path):
        return result

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
