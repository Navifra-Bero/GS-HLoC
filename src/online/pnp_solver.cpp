#include "render_loc/online/pnp_solver.h"

#include <opencv2/calib3d.hpp>
#include <iostream>

namespace renderloc {

PnPSolver::PnPSolver(const Config::PnP& config,
                       const CameraIntrinsics& intrinsics)
  : config_(config), intrinsics_(intrinsics) {}

PnPSolver::PnPResult PnPSolver::solve(
    const FeatureMatcher::Correspondences& corr) const
{
  PnPResult result;
  result.success = false;

  if (corr.pts2d.size() < 4) {
    std::cerr << "[PnP] Need at least 4 correspondences, got "
              << corr.pts2d.size() << std::endl;
    return result;
  }

  cv::Mat K = intrinsics_.toCvMat();
  cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
  cv::Mat rvec, tvec;
  cv::Mat inliers;

  bool ok = cv::solvePnPRansac(
    corr.pts3d, corr.pts2d, K, dist_coeffs,
    rvec, tvec, false,
    config_.ransac_iterations,
    config_.ransac_reproj_threshold,
    0.99,
    inliers,
    cv::SOLVEPNP_EPNP
  );

  if (!ok || inliers.rows < config_.min_inliers) {
    return result;
  }

  // Optional: refine with BA using inliers only
  if (config_.refine_with_ba && inliers.rows >= 6) {
    std::vector<cv::Point2f> inlier_pts2d;
    std::vector<cv::Point3f> inlier_pts3d;
    for (int i = 0; i < inliers.rows; ++i) {
      int idx = inliers.at<int>(i);
      inlier_pts2d.push_back(corr.pts2d[idx]);
      inlier_pts3d.push_back(corr.pts3d[idx]);
    }
    cv::solvePnPRefineLM(inlier_pts3d, inlier_pts2d, K, dist_coeffs,
                          rvec, tvec);
  }

  result.pose = rvecTvecToPose(rvec, tvec);
  result.num_inliers = inliers.rows;
  result.success = true;

  // Build inlier mask
  result.inlier_mask.assign(corr.pts2d.size(), false);
  for (int i = 0; i < inliers.rows; ++i) {
    result.inlier_mask[inliers.at<int>(i)] = true;
  }

  return result;
}

PnPSolver::PnPResult PnPSolver::solveMulti(
    const std::vector<FeatureMatcher::Correspondences>& corrs) const
{
  // Merge all correspondences from multiple DB images
  FeatureMatcher::Correspondences merged;
  for (const auto& c : corrs) {
    merged.pts2d.insert(merged.pts2d.end(), c.pts2d.begin(), c.pts2d.end());
    merged.pts3d.insert(merged.pts3d.end(), c.pts3d.begin(), c.pts3d.end());
    merged.confidences.insert(merged.confidences.end(),
                               c.confidences.begin(), c.confidences.end());
  }

  return solve(merged);
}

Pose6DoF PnPSolver::rvecTvecToPose(const cv::Mat& rvec,
                                     const cv::Mat& tvec) const
{
  cv::Mat R_cv;
  cv::Rodrigues(rvec, R_cv);

  // PnP gives world-to-camera transform
  // We want camera-in-world (map frame) pose
  Mat3d R;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      R(i, j) = R_cv.at<double>(i, j);

  Vec3d t(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

  // Invert: camera_in_world = (R^T, -R^T * t)
  Pose6DoF pose = Pose6DoF::Identity();
  pose.linear() = R.transpose();
  pose.translation() = -R.transpose() * t;

  return pose;
}

}  // namespace renderloc
