#pragma once

#include <string>
#include <vector>

namespace renderloc {

struct Config {
  // =========================================================================
  // Camera
  // =========================================================================
  struct Camera {
    double fx = 525.0, fy = 525.0;
    double cx = 319.5, cy = 239.5;
    int width = 640, height = 480;
    double depth_scale = 1000.0;
    double depth_min = 0.3;
    double depth_max = 10.0;
  } camera;

  // =========================================================================
  // Viewpoint Sampling
  // =========================================================================
  struct Sampling {
    double grid_spacing = 0.5;        // meters between viewpoints
    double height_above_floor = 1.2;  // camera height
    int num_yaw_angles = 8;           // directions per position (360/8=45°)
    double pitch_deg = 0.0;           // camera tilt (0 = horizontal)
    double floor_z_min = -0.5;        // floor detection range
    double floor_z_max = 0.5;
  } sampling;

  // =========================================================================
  // Rendering
  // =========================================================================
  struct Rendering {
    std::string output_dir = "data/rendered/";
    double point_size = 3.0;          // Point size for PLY rendering
    bool render_color = true;         // Use PLY color if available
  } rendering;

  // =========================================================================
  // Feature Extraction
  // =========================================================================
  struct Features {
    // SuperPoint
    std::string superpoint_model = "models/superpoint_v1.pt";
    int max_keypoints = 1024;
    double keypoint_threshold = 0.005;

    // Global descriptor (NetVLAD / EigenPlaces)
    std::string global_model = "models/netvlad.pt";
    int global_desc_dim = 4096;

    bool use_gpu = true;
  } features;

  // =========================================================================
  // Matching
  // =========================================================================
  struct Matching {
    std::string superglue_model = "models/superglue_indoor.pt";
    double match_threshold = 0.2;
    int top_k_retrieval = 5;          // Number of DB images to match against
    bool use_gpu = true;
  } matching;

  // =========================================================================
  // PnP
  // =========================================================================
  struct PnP {
    int ransac_iterations = 1000;
    double ransac_reproj_threshold = 8.0;  // pixels
    int min_inliers = 15;
    bool refine_with_ba = true;       // Bundle adjustment after RANSAC
  } pnp;

  // =========================================================================
  // ICP (optional refinement using query depth)
  // =========================================================================
  struct ICP {
    bool enable = false;
    int max_iterations = 30;
    double max_correspondence_dist = 1.0;
    double fitness_threshold = 0.3;
    bool use_gicp = true;
  } icp;

  // =========================================================================
  // Database
  // =========================================================================
  struct Database {
    std::string db_path = "data/image_db.bin";
    std::string ply_map_path = "";
  } database;

  static Config loadFromYaml(const std::string& yaml_path);
  void saveToYaml(const std::string& yaml_path) const;
};

}  // namespace renderloc
