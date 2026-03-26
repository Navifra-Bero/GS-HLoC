#pragma once

#include "render_loc/core/types.h"
#include "render_loc/core/config.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace renderloc {

/// Optional ICP refinement using query depth image against PLY map
class ICPRefiner {
public:
  explicit ICPRefiner(const Config::ICP& config,
                       const CameraIntrinsics& intrinsics);

  /// Load the PLY map for ICP target
  bool loadMap(const std::string& ply_path);

  struct ICPResult {
    Pose6DoF pose;
    double fitness_score;
    bool converged;
  };

  /// Refine PnP pose using query depth
  /// @param depth_image   Query depth image
  /// @param initial_pose  PnP result pose
  ICPResult refine(const cv::Mat& depth_image,
                    const Pose6DoF& initial_pose) const;

private:
  Config::ICP config_;
  CameraIntrinsics intrinsics_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_;
  bool map_loaded_ = false;
};

}  // namespace renderloc
