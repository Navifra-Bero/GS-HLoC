#include "render_loc/core/config.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>

namespace renderloc {

Config Config::loadFromYaml(const std::string& yaml_path) {
  Config cfg;
  try {
    YAML::Node root = YAML::LoadFile(yaml_path);

    if (auto n = root["camera"]) {
      cfg.camera.fx = n["fx"].as<double>(cfg.camera.fx);
      cfg.camera.fy = n["fy"].as<double>(cfg.camera.fy);
      cfg.camera.cx = n["cx"].as<double>(cfg.camera.cx);
      cfg.camera.cy = n["cy"].as<double>(cfg.camera.cy);
      cfg.camera.width = n["width"].as<int>(cfg.camera.width);
      cfg.camera.height = n["height"].as<int>(cfg.camera.height);
      cfg.camera.depth_scale = n["depth_scale"].as<double>(cfg.camera.depth_scale);
      cfg.camera.depth_min = n["depth_min"].as<double>(cfg.camera.depth_min);
      cfg.camera.depth_max = n["depth_max"].as<double>(cfg.camera.depth_max);
    }
    if (auto n = root["sampling"]) {
      cfg.sampling.grid_spacing = n["grid_spacing"].as<double>(cfg.sampling.grid_spacing);
      cfg.sampling.height_above_floor = n["height_above_floor"].as<double>(cfg.sampling.height_above_floor);
      cfg.sampling.num_yaw_angles = n["num_yaw_angles"].as<int>(cfg.sampling.num_yaw_angles);
      cfg.sampling.pitch_deg = n["pitch_deg"].as<double>(cfg.sampling.pitch_deg);
    }
    if (auto n = root["rendering"]) {
      cfg.rendering.output_dir = n["output_dir"].as<std::string>(cfg.rendering.output_dir);
      cfg.rendering.point_size = n["point_size"].as<double>(cfg.rendering.point_size);
      cfg.rendering.render_color = n["render_color"].as<bool>(cfg.rendering.render_color);
    }
    if (auto n = root["features"]) {
      cfg.features.superpoint_model = n["superpoint_model"].as<std::string>(cfg.features.superpoint_model);
      cfg.features.max_keypoints = n["max_keypoints"].as<int>(cfg.features.max_keypoints);
      cfg.features.keypoint_threshold = n["keypoint_threshold"].as<double>(cfg.features.keypoint_threshold);
      cfg.features.global_model = n["global_model"].as<std::string>(cfg.features.global_model);
      cfg.features.global_desc_dim = n["global_desc_dim"].as<int>(cfg.features.global_desc_dim);
      cfg.features.use_gpu = n["use_gpu"].as<bool>(cfg.features.use_gpu);
    }
    if (auto n = root["matching"]) {
      cfg.matching.superglue_model = n["superglue_model"].as<std::string>(cfg.matching.superglue_model);
      cfg.matching.match_threshold = n["match_threshold"].as<double>(cfg.matching.match_threshold);
      cfg.matching.top_k_retrieval = n["top_k_retrieval"].as<int>(cfg.matching.top_k_retrieval);
      cfg.matching.use_gpu = n["use_gpu"].as<bool>(cfg.matching.use_gpu);
    }
    if (auto n = root["pnp"]) {
      cfg.pnp.ransac_iterations = n["ransac_iterations"].as<int>(cfg.pnp.ransac_iterations);
      cfg.pnp.ransac_reproj_threshold = n["ransac_reproj_threshold"].as<double>(cfg.pnp.ransac_reproj_threshold);
      cfg.pnp.min_inliers = n["min_inliers"].as<int>(cfg.pnp.min_inliers);
      cfg.pnp.refine_with_ba = n["refine_with_ba"].as<bool>(cfg.pnp.refine_with_ba);
    }
    if (auto n = root["icp"]) {
      cfg.icp.enable = n["enable"].as<bool>(cfg.icp.enable);
      cfg.icp.max_iterations = n["max_iterations"].as<int>(cfg.icp.max_iterations);
      cfg.icp.max_correspondence_dist = n["max_correspondence_dist"].as<double>(cfg.icp.max_correspondence_dist);
      cfg.icp.fitness_threshold = n["fitness_threshold"].as<double>(cfg.icp.fitness_threshold);
      cfg.icp.use_gicp = n["use_gicp"].as<bool>(cfg.icp.use_gicp);
    }
    if (auto n = root["database"]) {
      cfg.database.db_path = n["db_path"].as<std::string>(cfg.database.db_path);
      cfg.database.ply_map_path = n["ply_map_path"].as<std::string>(cfg.database.ply_map_path);
    }
    std::cout << "[Config] Loaded: " << yaml_path << std::endl;
  } catch (const YAML::Exception& e) {
    std::cerr << "[Config] Error: " << e.what() << std::endl;
  }
  return cfg;
}

void Config::saveToYaml(const std::string& yaml_path) const {
  // Minimal implementation
  std::ofstream f(yaml_path);
  f << "# RenderLoc config (auto-generated)\n";
  f << "camera:\n  fx: " << camera.fx << "\n  fy: " << camera.fy << "\n";
  f.close();
}

}  // namespace renderloc
