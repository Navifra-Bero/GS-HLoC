#pragma once

#include "render_loc/core/types.h"
#include "render_loc/core/config.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace renderloc {

/// Generates camera viewpoints inside the PLY map for rendering
class ViewpointSampler {
public:
  explicit ViewpointSampler(const Config::Sampling& config,
                             const CameraIntrinsics& intrinsics);

  /// Sample viewpoints from PLY map bounding box
  /// Detects floor plane and generates grid + multi-yaw poses
  std::vector<Viewpoint> sample(const std::string& ply_path) const;

  /// Sample from pre-loaded cloud
  std::vector<Viewpoint> sample(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) const;

private:
  Config::Sampling config_;
  CameraIntrinsics intrinsics_;

  /// Estimate floor z-height from point cloud
  double estimateFloorHeight(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) const;

  /// Check if a position is inside the map (has nearby points)
  bool isInsideMap(const Vec3d& pos,
                   const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                   double radius = 3.0) const;
};

}  // namespace renderloc
