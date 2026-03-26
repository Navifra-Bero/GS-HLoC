#pragma once

#include "render_loc/core/types.h"
#include "render_loc/core/config.h"

#include <opencv2/core.hpp>

namespace renderloc {

/// Back-projects 2D keypoints to 3D using rendered depth images
/// This creates the 3D reference map: each SuperPoint keypoint
/// in a DB image gets a 3D coordinate in the map frame.
class DepthBackprojector {
public:
  explicit DepthBackprojector(const CameraIntrinsics& intrinsics,
                               double depth_scale = 1000.0,
                               double depth_min = 0.3,
                               double depth_max = 10.0);

  /// Assign 3D coordinates to keypoints using depth image and camera pose
  /// @param keypoints  2D keypoints from SuperPoint
  /// @param depth      Rendered depth image (CV_32FC1 or CV_16UC1)
  /// @param pose       Camera pose in map frame
  /// @return Keypoints with 3D coordinates
  std::vector<Keypoint3D> backprojectKeypoints(
    const std::vector<Keypoint2D>& keypoints,
    const cv::Mat& depth,
    const Pose6DoF& pose) const;

private:
  CameraIntrinsics intrinsics_;
  double depth_scale_;
  double depth_min_, depth_max_;

  /// Read depth at sub-pixel location (bilinear interpolation)
  double readDepth(const cv::Mat& depth, double u, double v) const;
};

}  // namespace renderloc
