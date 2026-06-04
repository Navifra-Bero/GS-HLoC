import os, pickle
import yaml


def load_config(p):
    with open(p) as f:
        return yaml.safe_load(f)


def default_config():
    return {
        #cam_0
        # "camera": {
        #     "fx": 1027.659153, "fy": 1030.215807, "cx": 966.827228, "cy": 585.037643,
        #     "width": 1920, "height": 1080,
        #     "depth_scale": 1000, "depth_min": 0.3, "depth_max": 20.0,
        # },
        # #cam_1
        # "camera": {
        #     "fx": 1028.487260, "fy": 1030.283620, "cx": 949.476264, "cy": 597.274302,
        #     "width": 1920, "height": 1080,
        #     "depth_scale": 1000, "depth_min": 0.3, "depth_max": 20.0,
        # },
        # #cam_2
        # "camera": {
        #     "fx": 1041.454577, "fy": 1044.072979, "cx": 945.363937, "cy": 610.455165,
        #     "width": 1920, "height": 1080,
        #     "depth_scale": 1000, "depth_min": 0.3, "depth_max": 20.0,
        # },
        # #cam_3
        "camera": {
            "fx": 1039.04598063, "fy": 1041.49694151, "cx": 937.04407689, "cy": 560.82673816,
            "width": 1920, "height": 1080,
            "depth_scale": 1000, "depth_min": 0.3, "depth_max": 20.0,
        },
        #femto
        # "camera": {
        #     "fx": 2256.627197, "fy": 2254.400635, "cx": 1891.352783, "cy": 1087.097656,
        #     "width": 3840, "height": 2160,
        #     "depth_scale": 1000, "depth_min": 0.3, "depth_max": 20.0,
        # },
        "alignment": {
            "normal_threshold": 0.8, "ransac_distance": 0.05,
            "ransac_n": 3, "ransac_iterations": 1000, "pre_flip_x": False,
            "apply_pre_rotation": False,
        },
        "sampling": {
            "grid_resolution": 0.05, "path_spacing": 0.3,
            "use_step0_floor": True,
            "occupancy_mode": "height_slice_obstacle",
            "height_above_floor": 1.0,
            "height_offsets_m": [0.0],
            "slice_min_height_m": 0.5,
            "slice_max_height_m": 1.5,
            "slice_splat_radius_m": 0.25,
            "slice_wall_close_m": 1.0,
            "slice_obstacle_inflate_m": 0.45,
            "slice_margin_m": 2.0,
            "num_yaw_angles": 4,
            "pitch_angles_deg": [0.0], "morph_kernel_size": 5,
            "distance_thresh_ratio": 0.3, "min_floor_points": 100,
            "max_floors": 1, "min_floor_gap": 2.5,
            "max_floor_height": 5.0, "floor_band": 0.3,
            "sample_mode": "grid",
            "min_wall_dist_m": 0.3,
            "skel_grid_spacing_m": 0.5,
        },
        "rendering": {
            "point_size": 1.0,
            "brightness_scale": 1.0,
            "ceiling_clip": True,
            "ceiling_margin": 0.3,
            "near_dist": 3.0,
            "near_fill_radius": 6.0,
            "fill_radius": 2.0,
        },
        "features": {
            "global_desc_method": "megaloc",
            "dino_model": "dinov2_vitb14",
            "dino_img_size": 322,
            "vlad_clusters": 64,
            "vlad_pca_dim": 4096,
            "megaloc_dim": 8448,
            "eloftr_ckpt": "",
            "eloftr_opt": False,
            "eloftr_max_dim": 840,
            "match_conf_thresh": 0.2,
        },
        "online": {
            "top_k": 5,
            "reprojection_error": 8.0,
            "pnp_iterations": 1000,
            "pnp_confidence": 0.99,
        },
    }


def load_pkl(output_dir, name):
    p = os.path.join(output_dir, name)
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None
