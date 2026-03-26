#include "render_loc/offline/depth_backprojector.h"

#include <cmath>
#include <iostream>

namespace renderloc {

DepthBackprojector::DepthBackprojector(const CameraIntrinsics& intrinsics,
                                        double depth_scale,
                                        double depth_min,
                                        double depth_max)
  : intrinsics_(intrinsics),
    depth_scale_(depth_scale),
    depth_min_(depth_min),
    depth_max_(depth_max) {}

std::vector<Keypoint3D> DepthBackprojector::backprojectKeypoints(
    const std::vector<Keypoint2D>& keypoints,
    const cv::Mat& depth,
    const Pose6DoF& pose) const
{
  std::vector<Keypoint3D> kps_3d;
  kps_3d.reserve(keypoints.size());

  for (const auto& kp : keypoints) {
    Keypoint3D kp3d;
    kp3d.pt2d = kp.pt;
    kp3d.descriptor = kp.descriptor;
    kp3d.valid = false;

    double u = kp.pt.x();
    double v = kp.pt.y();

    // Read depth at keypoint location
    double d = readDepth(depth, u, v);
    if (!std::isfinite(d) || d < depth_min_ || d > depth_max_) {
      kps_3d.push_back(kp3d);
      continue;
    }

    // Back-project to camera frame
    Vec3d pt_cam = intrinsics_.backproject(u, v, d);

    // Transform to map frame
    kp3d.pt3d = pose * pt_cam;
    kp3d.valid = true;

    kps_3d.push_back(kp3d);
  }

  int valid_count = 0;
  for (const auto& kp : kps_3d) {
    if (kp.valid) valid_count++;
  }

  return kps_3d;
}

double DepthBackprojector::readDepth(const cv::Mat& depth,
                                      double u, double v) const
{
  int ui = static_cast<int>(std::round(u));
  int vi = static_cast<int>(std::round(v));

  if (ui < 0 || ui >= depth.cols || vi < 0 || vi >= depth.rows) {
    return -1.0;
  }

  double raw = 0.0;
  if (depth.type() == CV_16UC1) {
    raw = static_cast<double>(depth.at<uint16_t>(vi, ui));
    return raw / depth_scale_;
  } else if (depth.type() == CV_32FC1) {
    return static_cast<double>(depth.at<float>(vi, ui));
  } else if (depth.type() == CV_64FC1) {
    return depth.at<double>(vi, ui);
  }

  return -1.0;
}

}  // namespace renderloc
