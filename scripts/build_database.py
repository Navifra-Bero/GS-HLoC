#!/usr/bin/env python3
"""
RenderLoc Offline Pipeline
===========================
PLY Map → Viewpoint Sampling → Render RGB+Depth → Feature Extraction → DB Build

This script handles the entire offline phase using Python (Open3D + PyTorch).
The resulting database is saved as a binary file loadable by the C++ localizer.

Usage:
  python build_database.py \
    --ply_map /path/to/map.ply \
    --config config/render_loc.yaml \
    --output data/image_db.bin
"""

import argparse
import os
import struct
import yaml

import numpy as np
import open3d as o3d
import cv2
import torch


# =============================================================================
# Step 1: Viewpoint Sampling
# =============================================================================

def sample_viewpoints(pcd, config):
    """Generate camera viewpoints inside the map."""
    sampling = config["sampling"]
    spacing = sampling["grid_spacing"]
    height  = sampling["height_above_floor"]
    n_yaw   = sampling["num_yaw_angles"]
    pitch   = np.radians(sampling.get("pitch_deg", 0.0))

    points = np.asarray(pcd.points)
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    # Estimate floor height (lowest 10th percentile)
    floor_z = np.percentile(points[:, 2], 5)
    cam_z = floor_z + height
    print(f"[Sampling] Floor z={floor_z:.2f}, Camera z={cam_z:.2f}")

    # Grid over XY bounding box
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()

    viewpoints = []
    vid = 0

    xs = np.arange(x_min + spacing, x_max - spacing, spacing)
    ys = np.arange(y_min + spacing, y_max - spacing, spacing)

    # Build KDTree for inside-map check
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    for x in xs:
        for y in ys:
            # Check if position is inside the map (has nearby points)
            [k, _, _] = pcd_tree.search_radius_vector_3d([x, y, cam_z], 3.0)
            if k < 10:
                continue

            # Generate multiple yaw angles
            for yaw_idx in range(n_yaw):
                yaw = 2.0 * np.pi * yaw_idx / n_yaw

                # Rotation matrix: yaw around Z, pitch around X
                Rz = np.array([
                    [np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw),  np.cos(yaw), 0],
                    [0, 0, 1]
                ])
                Rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch),  np.cos(pitch)]
                ])
                R = Rz @ Rx

                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [x, y, cam_z]

                viewpoints.append({"id": vid, "pose": T})
                vid += 1

    print(f"[Sampling] Generated {len(viewpoints)} viewpoints")
    return viewpoints


# =============================================================================
# Step 2: Render RGB + Depth
# =============================================================================

def render_viewpoints(pcd, viewpoints, config, output_dir):
    """Render RGB and Depth images from PLY at each viewpoint."""
    cam = config["camera"]
    render_cfg = config.get("rendering", {})

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)

    width, height = cam["width"], cam["height"]
    fx, fy = cam["fx"], cam["fy"]
    cx, cy = cam["cx"], cam["cy"]

    # Create offscreen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.point_size = render_cfg.get("point_size", 3.0)
    mat.shader = "defaultUnlit"

    renderer.scene.add_geometry("map", pcd, mat)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    results = []
    for i, vp in enumerate(viewpoints):
        pose = vp["pose"]

        # Open3D uses look-at convention, convert from our pose matrix
        # Camera looks along -Z in camera frame
        eye = pose[:3, 3]
        forward = pose[:3, :3] @ np.array([0, 0, 1])  # Forward direction
        up = pose[:3, :3] @ np.array([0, -1, 0])       # Up direction

        renderer.setup_camera(intrinsic, np.linalg.inv(pose))

        # Render
        rgb_o3d = renderer.render_to_image()
        depth_o3d = renderer.render_to_depth_float_buffer()

        rgb_np = np.asarray(rgb_o3d)
        depth_np = np.asarray(depth_o3d)

        # Save
        rgb_path = os.path.join(output_dir, "rgb", f"{vp['id']:06d}.png")
        depth_path = os.path.join(output_dir, "depth", f"{vp['id']:06d}.npy")

        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
        np.save(depth_path, depth_np)

        results.append({
            "id": vp["id"],
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "pose": pose,
        })

        if (i + 1) % 100 == 0:
            print(f"[Render] {i+1}/{len(viewpoints)} done")

    print(f"[Render] Rendered {len(results)} images")
    return results


# =============================================================================
# Step 3: Feature Extraction (SuperPoint + NetVLAD)
# =============================================================================

def extract_features(rendered_images, config):
    """Extract SuperPoint keypoints and NetVLAD global descriptors."""
    feat_cfg = config["features"]
    device = torch.device("cuda" if feat_cfg.get("use_gpu", True)
                          and torch.cuda.is_available() else "cpu")
    print(f"[Features] Using device: {device}")

    # Load SuperPoint
    sp_model = torch.jit.load(feat_cfg["superpoint_model"], map_location=device)
    sp_model.eval()

    # Load global descriptor model
    global_model = torch.jit.load(feat_cfg["global_model"], map_location=device)
    global_model.eval()

    max_kps = feat_cfg.get("max_keypoints", 1024)
    kp_thresh = feat_cfg.get("keypoint_threshold", 0.005)

    for i, img_info in enumerate(rendered_images):
        rgb = cv2.imread(img_info["rgb_path"])
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # SuperPoint
        inp = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            sp_out = sp_model(inp)

        # Parse SuperPoint output (keypoints, scores, descriptors)
        kps = sp_out["keypoints"][0].cpu().numpy()       # [N, 2]
        scores = sp_out["scores"][0].cpu().numpy()        # [N]
        descs = sp_out["descriptors"][0].cpu().numpy()    # [256, N]

        # Filter by score and limit count
        mask = scores > kp_thresh
        kps = kps[mask]
        scores = scores[mask]
        descs = descs[:, mask]

        # Top-K by score
        if len(kps) > max_kps:
            topk_idx = np.argsort(scores)[::-1][:max_kps]
            kps = kps[topk_idx]
            scores = scores[topk_idx]
            descs = descs[:, topk_idx]

        img_info["keypoints"] = kps              # [N, 2]
        img_info["descriptors"] = descs.T         # [N, 256]
        img_info["scores"] = scores

        # NetVLAD global descriptor
        rgb_resized = cv2.resize(rgb, (224, 224))
        rgb_tensor = torch.from_numpy(
            rgb_resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            global_desc = global_model(rgb_tensor).cpu().numpy().flatten()

        img_info["global_descriptor"] = global_desc

        if (i + 1) % 100 == 0:
            print(f"[Features] {i+1}/{len(rendered_images)} done")

    print(f"[Features] Extracted features for {len(rendered_images)} images")
    return rendered_images


# =============================================================================
# Step 4: Backproject keypoints to 3D
# =============================================================================

def backproject_keypoints(rendered_images, config):
    """Assign 3D coordinates to each keypoint using depth + pose."""
    cam = config["camera"]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    depth_min = cam.get("depth_min", 0.3)
    depth_max = cam.get("depth_max", 10.0)

    for img_info in rendered_images:
        depth = np.load(img_info["depth_path"])
        pose = img_info["pose"]  # 4x4 camera-to-world
        kps = img_info["keypoints"]

        kps_3d = []
        valid_mask = []

        for kp in kps:
            u, v = kp[0], kp[1]
            ui, vi = int(round(u)), int(round(v))

            if 0 <= ui < depth.shape[1] and 0 <= vi < depth.shape[0]:
                d = depth[vi, ui]
            else:
                d = 0.0

            if d < depth_min or d > depth_max or not np.isfinite(d):
                kps_3d.append([0, 0, 0])
                valid_mask.append(False)
                continue

            # Back-project to camera frame
            x_cam = (u - cx) * d / fx
            y_cam = (v - cy) * d / fy
            z_cam = d

            pt_cam = np.array([x_cam, y_cam, z_cam, 1.0])
            pt_world = pose @ pt_cam

            kps_3d.append(pt_world[:3])
            valid_mask.append(True)

        img_info["keypoints_3d"] = np.array(kps_3d)
        img_info["valid_mask"] = np.array(valid_mask)

    valid_total = sum(m.sum() for img in rendered_images for m in [img["valid_mask"]])
    total_kps = sum(len(img["keypoints"]) for img in rendered_images)
    print(f"[Backproject] {valid_total}/{total_kps} keypoints have valid 3D")
    return rendered_images


# =============================================================================
# Step 5: Save Database (binary format for C++ loader)
# =============================================================================

def save_database(rendered_images, config, output_path):
    """Save database in binary format loadable by C++ ImageDatabase::load()."""
    cam = config["camera"]
    global_dim = config["features"].get("global_desc_dim", 4096)
    local_dim = 256  # SuperPoint

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as f:
        n = len(rendered_images)
        f.write(struct.pack("i", n))
        f.write(struct.pack("i", global_dim))
        f.write(struct.pack("i", local_dim))

        # CameraIntrinsics struct
        f.write(struct.pack("dddd", cam["fx"], cam["fy"], cam["cx"], cam["cy"]))
        f.write(struct.pack("ii", cam["width"], cam["height"]))

        for img in rendered_images:
            f.write(struct.pack("i", img["id"]))

            # Pose (4x4, column-major for Eigen)
            pose = img["pose"].T.flatten()  # Row to column major
            f.write(struct.pack("16d", *pose))

            # Global descriptor
            gd = img["global_descriptor"]
            if len(gd) < global_dim:
                gd = np.pad(gd, (0, global_dim - len(gd)))
            f.write(struct.pack(f"{global_dim}f", *gd[:global_dim]))

            # Keypoints with 3D
            nkp = len(img["keypoints"])
            f.write(struct.pack("i", nkp))

            for j in range(nkp):
                kp2d = img["keypoints"][j]
                kp3d = img["keypoints_3d"][j]
                valid = img["valid_mask"][j]
                desc = img["descriptors"][j]

                f.write(struct.pack("dd", float(kp2d[0]), float(kp2d[1])))
                f.write(struct.pack("ddd", float(kp3d[0]), float(kp3d[1]), float(kp3d[2])))
                f.write(struct.pack("?", bool(valid)))
                f.write(struct.pack("i", len(desc)))
                f.write(struct.pack(f"{len(desc)}f", *desc))

    print(f"[DB] Saved {n} images to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RenderLoc Offline DB Builder")
    parser.add_argument("--ply_map", required=True, help="Path to PLY map")
    parser.add_argument("--config", default="config/render_loc.yaml")
    parser.add_argument("--output", default="data/image_db.bin")
    parser.add_argument("--render_dir", default="data/rendered")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=== RenderLoc Offline Database Builder ===")
    print(f"PLY: {args.ply_map}")

    # Load PLY
    print("[1/5] Loading PLY map...")
    pcd = o3d.io.read_point_cloud(args.ply_map)
    print(f"  Points: {len(pcd.points)}")

    # Sample viewpoints
    print("[2/5] Sampling viewpoints...")
    viewpoints = sample_viewpoints(pcd, config)

    # Render
    print("[3/5] Rendering synthetic images...")
    rendered = render_viewpoints(pcd, viewpoints, config, args.render_dir)

    # Extract features
    print("[4/5] Extracting features...")
    rendered = extract_features(rendered, config)

    # Backproject keypoints to 3D
    print("[5/5] Backprojecting keypoints...")
    rendered = backproject_keypoints(rendered, config)

    # Save
    save_database(rendered, config, args.output)
    print("=== Done ===")


if __name__ == "__main__":
    main()
