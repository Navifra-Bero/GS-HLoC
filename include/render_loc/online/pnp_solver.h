#pragma once

#include "render_loc/core/types.h"
#include "render_loc/core/config.h"
#include "render_loc/online/feature_matcher.h"

#include <opencv2/core.hpp>

namespace renderloc {

/// Estimates 6DoF camera pose from 2D-3D correspondences using PnP+RANSAC
class PnPSolver {
public:
  explicit PnPSolver(const Config::PnP& config,
                      const CameraIntrinsics& intrinsics);

  struct PnPResult {
    bool success = false;
    Pose6DoF pose;
    int num_inliers = 0;
    std::vector<bool> inlier_mask;
  };

  /// Solve PnP from 2D-3D correspondences
  PnPResult solve(const FeatureMatcher::Correspondences& corr) const;

  /// Solve PnP from multiple DB image matches (merged correspondences)
  PnPResult solveMulti(
    const std::vector<FeatureMatcher::Correspondences>& corrs) const;

private:
  Config::PnP config_;
  CameraIntrinsics intrinsics_;

  /// Convert Rodrigues + tvec to Pose6DoF
  Pose6DoF rvecTvecToPose(const cv::Mat& rvec, const cv::Mat& tvec) const;
};

}  // namespace renderloc
