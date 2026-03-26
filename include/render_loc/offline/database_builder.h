#pragma once

#include "render_loc/core/types.h"
#include "render_loc/core/config.h"
#include "render_loc/offline/viewpoint_sampler.h"
#include "render_loc/offline/map_renderer.h"
#include "render_loc/offline/feature_extractor.h"
#include "render_loc/offline/depth_backprojector.h"

namespace renderloc {

/// Orchestrates the full offline pipeline:
/// PLY → Sample Viewpoints → Render → Extract Features → Backproject → DB
class DatabaseBuilder {
public:
  explicit DatabaseBuilder(const Config& config);

  /// Build complete database from PLY map
  ImageDatabase build(const std::string& ply_path);

private:
  Config config_;
  CameraIntrinsics intrinsics_;
};

}  // namespace renderloc
