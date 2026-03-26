#pragma once

#include "render_loc/core/types.h"
#include "render_loc/core/config.h"

#include <opencv2/core.hpp>

namespace renderloc {

/// Renders synthetic RGB and Depth images from PLY point cloud map
/// Uses Open3D for offscreen rendering (called via Python bridge)
/// or direct OpenGL rendering for C++ native approach
class MapRenderer {
public:
  MapRenderer(const Config::Rendering& render_cfg,
              const CameraIntrinsics& intrinsics);

  /// Load PLY map
  bool loadMap(const std::string& ply_path);

  /// Render at a single viewpoint
  /// @return pair of (RGB, Depth) images
  std::pair<cv::Mat, cv::Mat> render(const Viewpoint& viewpoint) const;

  /// Render all viewpoints and save to disk
  /// @return paths to rendered images per viewpoint
  struct RenderResult {
    std::string rgb_path;
    std::string depth_path;
    Pose6DoF pose;
    int viewpoint_id;
  };

  std::vector<RenderResult> renderAll(
    const std::vector<Viewpoint>& viewpoints,
    const std::string& output_dir) const;

  bool isLoaded() const { return map_loaded_; }

private:
  Config::Rendering render_cfg_;
  CameraIntrinsics intrinsics_;
  bool map_loaded_ = false;
  std::string ply_path_;
};

}  // namespace renderloc
