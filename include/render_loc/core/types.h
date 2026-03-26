#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace renderloc {

// =============================================================================
// Pose & Transform
// =============================================================================
using Pose6DoF = Eigen::Isometry3d;
using Vec3d    = Eigen::Vector3d;
using Vec2d    = Eigen::Vector2d;
using Mat3d    = Eigen::Matrix3d;
using Mat4d    = Eigen::Matrix4d;

struct CameraIntrinsics {
  double fx, fy, cx, cy;
  int width, height;

  Vec3d backproject(double u, double v, double depth) const {
    return Vec3d((u - cx) * depth / fx,
                 (v - cy) * depth / fy,
                 depth);
  }

  Vec2d project(const Vec3d& pt3d) const {
    return Vec2d(fx * pt3d.x() / pt3d.z() + cx,
                 fy * pt3d.y() / pt3d.z() + cy);
  }

  cv::Mat toCvMat() const {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = fx;
    K.at<double>(1, 1) = fy;
    K.at<double>(0, 2) = cx;
    K.at<double>(1, 2) = cy;
    return K;
  }
};

// =============================================================================
// Keypoint with 3D correspondence
// =============================================================================
struct Keypoint2D {
  Vec2d pt;               // 2D pixel location
  std::vector<float> descriptor;  // Local feature descriptor (256-dim for SuperPoint)
};

struct Keypoint3D {
  Vec2d pt2d;             // 2D pixel location in DB image
  Vec3d pt3d;             // 3D coordinate in map frame
  std::vector<float> descriptor;
  bool valid = true;      // False if depth was invalid
};

// =============================================================================
// Database Image Entry
// =============================================================================
struct DBImage {
  int id;
  Pose6DoF pose;                         // Camera pose in map frame
  std::string rgb_path;                  // Path to rendered RGB
  std::string depth_path;                // Path to rendered Depth

  // Features (populated after extraction)
  std::vector<float> global_descriptor;  // NetVLAD / EigenPlaces (4096 or 512-dim)
  std::vector<Keypoint3D> keypoints_3d;  // SuperPoint keypoints + 3D from depth
};

// =============================================================================
// Viewpoint for rendering
// =============================================================================
struct Viewpoint {
  Pose6DoF pose;
  int id;
};

// =============================================================================
// Feature Match
// =============================================================================
struct FeatureMatch {
  int query_idx;     // Index in query keypoints
  int db_idx;        // Index in DB keypoints
  float confidence;  // SuperGlue confidence
};

// =============================================================================
// Localization Result
// =============================================================================
struct LocalizationResult {
  bool success = false;

  // Retrieval
  int matched_db_id = -1;
  double retrieval_score = 0.0;

  // PnP result
  Pose6DoF pose;               // 6DoF in map frame
  int num_inliers = 0;
  int num_matches = 0;

  // Optional ICP refinement
  Pose6DoF refined_pose;
  double icp_fitness = -1.0;
  bool icp_used = false;

  // Timing (ms)
  double retrieval_ms = 0.0;
  double matching_ms = 0.0;
  double pnp_ms = 0.0;
  double icp_ms = 0.0;
  double total_ms = 0.0;
};

// =============================================================================
// Database
// =============================================================================
struct ImageDatabase {
  std::vector<DBImage> images;
  CameraIntrinsics intrinsics;
  int global_desc_dim = 0;
  int local_desc_dim = 0;

  // Retrieval: find top-K by global descriptor distance
  std::vector<std::pair<int, double>> retrieveTopK(
    const std::vector<float>& query_global_desc, int k) const;

  bool save(const std::string& path) const;
  bool load(const std::string& path);
};

}  // namespace renderloc
