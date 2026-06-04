#!/usr/bin/env python3
"""Rectify COLMAP fisheye image folders into pinhole image folders.

The script expects a sparse COLMAP-style directory with cameras.txt/images.txt
and image paths such as cam_0/images/<timestamp>.jpg. It writes the same layout
to the output directory, undistorts fisheye images, and rewrites cameras.txt as
PINHOLE so the rest of this project can consume it without fisheye handling.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEPTH_EXTS = {".depth", ".pcd"}


def parse_cameras(path):
    cameras = {}
    header = []
    with open(path) as f:
        for line in f:
            raw = line.rstrip("\n")
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header.append(raw)
                continue
            parts = stripped.split()
            if len(parts) < 5:
                continue
            cam_id = int(parts[0])
            model = parts[1].upper()
            width = int(parts[2])
            height = int(parts[3])
            params = [float(v) for v in parts[4:]]
            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras, header


def parse_image_camera_ids(path):
    image_to_camera = {}
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 10:
                continue
            suffix = Path(parts[9]).suffix.lower()
            if suffix not in IMAGE_EXTS:
                continue
            image_to_camera[parts[9]] = int(parts[8])
    return image_to_camera


def camera_matrix(camera):
    params = camera["params"]
    model = camera["model"]
    if model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE"):
        fx, fy, cx, cy = params[:4]
    elif model in ("SIMPLE_RADIAL", "RADIAL"):
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        raise ValueError(f"Unsupported camera model for rectification: {model}")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def distortion(camera):
    params = camera["params"]
    model = camera["model"]
    if model == "OPENCV_FISHEYE":
        return np.array(params[4:8], dtype=np.float64).reshape(4, 1)
    if model == "OPENCV":
        return np.array(params[4:8], dtype=np.float64)
    if model == "SIMPLE_RADIAL":
        return np.array([params[3], 0.0, 0.0, 0.0], dtype=np.float64)
    if model == "RADIAL":
        return np.array([params[3], params[4], 0.0, 0.0], dtype=np.float64)
    return None


def build_rectify_maps(cameras, keep_original_k=True, balance=0.0):
    maps = {}
    out_cameras = {}
    metadata = {}
    for cam_id, camera in cameras.items():
        width, height = camera["width"], camera["height"]
        size = (width, height)
        model = camera["model"]
        k = camera_matrix(camera)
        d = distortion(camera)
        new_k = k.copy()

        if model == "OPENCV_FISHEYE":
            if not keep_original_k:
                new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    k, d, size, np.eye(3), balance=balance, new_size=size
                )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                k, d, np.eye(3), new_k, size, cv2.CV_16SC2
            )
        elif model in ("OPENCV", "SIMPLE_RADIAL", "RADIAL"):
            if not keep_original_k:
                new_k, _ = cv2.getOptimalNewCameraMatrix(k, d, size, alpha=balance, newImgSize=size)
            map1, map2 = cv2.initUndistortRectifyMap(
                k, d, None, new_k, size, cv2.CV_16SC2
            )
        else:
            map1 = map2 = None

        maps[cam_id] = (map1, map2)
        out_cameras[cam_id] = {
            "width": width,
            "height": height,
            "k": new_k,
        }
        metadata[str(cam_id)] = {
            "input_model": model,
            "input_params": camera["params"],
            "output_model": "PINHOLE",
            "output_params": [
                float(new_k[0, 0]),
                float(new_k[1, 1]),
                float(new_k[0, 2]),
                float(new_k[1, 2]),
            ],
        }
    return maps, out_cameras, metadata


def write_pinhole_cameras(path, cameras):
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cam_id in sorted(cameras):
            cam = cameras[cam_id]
            k = cam["k"]
            f.write(
                f"{cam_id} PINHOLE {cam['width']} {cam['height']} "
                f"{k[0, 0]:.17g} {k[1, 1]:.17g} {k[0, 2]:.17g} {k[1, 2]:.17g}\n"
            )


def copy_tree_non_images(src_dir, dst_dir):
    for root, _, files in os.walk(src_dir):
        root_path = Path(root)
        rel_root = root_path.relative_to(src_dir)
        for name in files:
            src = root_path / name
            if src.suffix.lower() in IMAGE_EXTS:
                continue
            if src.suffix.lower() in DEPTH_EXTS:
                continue
            if src.name in {"cameras.txt", "images.txt"}:
                continue
            dst = Path(dst_dir) / rel_root / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def rectify_images(src_dir, dst_dir, image_to_camera, maps, overwrite=False, verbose_missing=False):
    processed = 0
    skipped = 0
    for rel_name, cam_id in image_to_camera.items():
        src = Path(src_dir) / rel_name
        if not src.exists() and "/images/" in rel_name:
            src = Path(src_dir) / rel_name.replace("/images/", "/")
        dst = Path(dst_dir) / rel_name
        if not src.exists():
            skipped += 1
            if verbose_missing:
                print(f"[warn] missing image: {src}")
            continue
        if dst.exists() and not overwrite:
            skipped += 1
            continue

        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            skipped += 1
            print(f"[warn] failed to read image: {src}")
            continue

        map1, map2 = maps[cam_id]
        if map1 is not None:
            image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        dst.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(dst), image)
        if not ok:
            raise RuntimeError(f"Failed to write {dst}")
        processed += 1
        if processed % 250 == 0:
            print(f"processed {processed} images...")
    return processed, skipped


def image_to_depth_name(image_name):
    parts = Path(image_name).parts
    if "images" in parts:
        parts = tuple("depths" if p == "images" else p for p in parts)
        return str(Path(*parts).with_suffix(".depth"))
    return str(Path(image_name).with_suffix(".depth"))


def candidate_depth_paths(src_dir, rel_image):
    depth_name = image_to_depth_name(rel_image)
    base = Path(src_dir) / depth_name
    candidates = [base]
    candidates.append(base.with_suffix(".pcd"))
    if "/depths/" in depth_name:
        flat = Path(src_dir) / depth_name.replace("/depths/", "/")
        candidates.extend([flat, flat.with_suffix(".pcd")])
    # de-duplicate while preserving order
    out = []
    seen = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def pcd_to_depth(path, out_camera, splat_radius_px=0):
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    width = int(out_camera["width"])
    height = int(out_camera["height"])
    depth = np.zeros((height, width), dtype=np.float32)
    if pts.size == 0:
        return depth

    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > 0.0)
    pts = pts[valid]
    if len(pts) == 0:
        return depth

    k = out_camera["k"]
    u = np.rint(k[0, 0] * (pts[:, 0] / pts[:, 2]) + k[0, 2]).astype(np.int32)
    v = np.rint(k[1, 1] * (pts[:, 1] / pts[:, 2]) + k[1, 2]).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[inside]
    v = v[inside]
    z = pts[inside, 2]
    if len(z) == 0:
        return depth

    order = np.argsort(z)[::-1]
    depth[v[order], u[order]] = z[order]

    radius = int(splat_radius_px)
    if radius > 0:
        # Sparse PCDs are often lidar-like. A tiny nearest-depth splat makes the
        # descriptor less brittle while keeping foreground surfaces dominant.
        base_u, base_v, base_z = u, v, z
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                uu = base_u + dx
                vv = base_v + dy
                ok = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
                if not np.any(ok):
                    continue
                old = depth[vv[ok], uu[ok]]
                zz = base_z[ok]
                write = (old <= 0.0) | (zz < old)
                if np.any(write):
                    depth[vv[ok][write], uu[ok][write]] = zz[write]
    return depth


def rectify_depths(src_dir, dst_dir, image_to_camera, cameras, maps,
                   out_cameras=None, overwrite=False, verbose_missing=False,
                   interpolation="nearest", pcd_splat_radius_px=0):
    interp = cv2.INTER_NEAREST if interpolation == "nearest" else cv2.INTER_LINEAR
    processed = 0
    skipped = 0
    for rel_image, cam_id in image_to_camera.items():
        src = next((p for p in candidate_depth_paths(src_dir, rel_image) if p.exists()), None)
        depth_name = image_to_depth_name(rel_image)
        dst = Path(dst_dir) / depth_name
        if src is None:
            skipped += 1
            if verbose_missing:
                print(f"[warn] missing depth: {Path(src_dir) / depth_name}")
            continue
        if dst.exists() and not overwrite:
            skipped += 1
            continue

        width = int(cameras[cam_id]["width"])
        height = int(cameras[cam_id]["height"])
        if src.suffix.lower() == ".pcd":
            if out_cameras is None:
                raise ValueError("out_cameras required for PCD depth projection")
            depth = pcd_to_depth(src, out_cameras[cam_id], splat_radius_px=pcd_splat_radius_px)
        else:
            depth = np.fromfile(src, dtype=np.float32)
            if depth.size != width * height:
                skipped += 1
                print(f"[warn] unexpected depth size: {src} has {depth.size}, expected {width * height}")
                continue
            depth = depth.reshape(height, width)
            depth[~np.isfinite(depth)] = 0.0

            map1, map2 = maps[cam_id]
            if map1 is not None:
                depth = cv2.remap(
                    depth, map1, map2,
                    interpolation=interp,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
        depth[~np.isfinite(depth)] = 0.0

        dst.parent.mkdir(parents=True, exist_ok=True)
        depth.astype(np.float32).tofile(dst)
        processed += 1
        if processed % 250 == 0:
            print(f"processed {processed} depths...")
    return processed, skipped


def write_filtered_images(src_path, dst_path, existing_names):
    """Copy images.txt, keeping only image entries that were actually written."""
    kept = 0
    dropped = 0
    with open(src_path) as src, open(dst_path, "w") as dst:
        pending_image_line = None
        for line in src:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                dst.write(line)
                continue

            parts = stripped.split()
            is_image_line = len(parts) >= 10 and Path(parts[9]).suffix.lower() in IMAGE_EXTS
            if is_image_line:
                pending_image_line = line
                continue

            if pending_image_line is not None:
                name = pending_image_line.strip().split()[9]
                if name in existing_names:
                    dst.write(pending_image_line)
                    dst.write(line)
                    kept += 1
                else:
                    dropped += 1
                pending_image_line = None
            else:
                dst.write(line)

        if pending_image_line is not None:
            name = pending_image_line.strip().split()[9]
            if name in existing_names:
                dst.write(pending_image_line)
                kept += 1
            else:
                dropped += 1
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Input COLMAP-style directory")
    parser.add_argument("--output_dir", required=True, help="Output rectified directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images")
    parser.add_argument(
        "--estimate-new-k",
        action="store_true",
        help="Estimate a new output K instead of preserving the original fx/fy/cx/cy",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.0,
        help="OpenCV fisheye balance/standard alpha when --estimate-new-k is used",
    )
    parser.add_argument("--verbose-missing", action="store_true", help="Print every missing image listed in images.txt")
    parser.add_argument("--skip-depths", action="store_true", help="Do not rectify matching .depth files")
    parser.add_argument(
        "--depth-interpolation",
        choices=("nearest", "linear"),
        default="nearest",
        help="Depth remap interpolation. nearest avoids blending across depth edges",
    )
    parser.add_argument(
        "--pcd-splat-radius-px",
        type=int,
        default=1,
        help="Pixel radius for sparse PCD depth projection before saving .depth",
    )
    args = parser.parse_args()

    src_dir = Path(args.input_dir)
    dst_dir = Path(args.output_dir)
    cameras_path = src_dir / "cameras.txt"
    images_path = src_dir / "images.txt"
    if not cameras_path.exists() or not images_path.exists():
        raise FileNotFoundError("input_dir must contain cameras.txt and images.txt")

    cameras, _ = parse_cameras(cameras_path)
    image_to_camera = parse_image_camera_ids(images_path)
    maps, out_cameras, metadata = build_rectify_maps(
        cameras, keep_original_k=not args.estimate_new_k, balance=args.balance
    )

    dst_dir.mkdir(parents=True, exist_ok=True)
    copy_tree_non_images(src_dir, dst_dir)
    write_pinhole_cameras(dst_dir / "cameras.txt", out_cameras)
    processed, skipped = rectify_images(
        src_dir, dst_dir, image_to_camera, maps,
        overwrite=args.overwrite,
        verbose_missing=args.verbose_missing,
    )
    depth_processed = depth_skipped = 0
    if not args.skip_depths:
        depth_processed, depth_skipped = rectify_depths(
            src_dir, dst_dir, image_to_camera, cameras, maps,
            out_cameras=out_cameras,
            overwrite=args.overwrite,
            verbose_missing=args.verbose_missing,
            interpolation=args.depth_interpolation,
            pcd_splat_radius_px=args.pcd_splat_radius_px,
        )
    existing_names = {name for name in image_to_camera if (dst_dir / name).exists()}
    kept, dropped = write_filtered_images(images_path, dst_dir / "images.txt", existing_names)

    with open(dst_dir / "rectify_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"wrote: {dst_dir}")
    print(f"images processed: {processed}, skipped: {skipped}")
    if not args.skip_depths:
        print(f"depths processed: {depth_processed}, skipped: {depth_skipped}")
    print(f"images.txt entries kept: {kept}, dropped: {dropped}")
    print("cameras.txt rewritten as PINHOLE")


if __name__ == "__main__":
    main()
