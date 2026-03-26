#pragma once

#include "render_loc/core/types.h"
#include "render_loc/core/config.h"

#include <torch/torch.h>
#include <torch/script.h>

namespace renderloc {

/// Matches query features against DB features using SuperGlue
class FeatureMatcher {
public:
  explicit FeatureMatcher(const Config::Matching& config);

  /// Load SuperGlue model
  bool loadModel();

  /// Match query keypoints against a single DB image
  /// @return Vector of matches with confidence scores
  std::vector<FeatureMatch> match(
    const std::vector<Keypoint2D>& query_kps,
    const std::vector<Keypoint3D>& db_kps,
    const cv::Size& image_size) const;

  /// Collect 2D-3D correspondences from matches
  /// @return (2D query points, 3D map points) for PnP
  struct Correspondences {
    std::vector<cv::Point2f> pts2d;
    std::vector<cv::Point3f> pts3d;
    std::vector<float> confidences;
  };

  Correspondences collectCorrespondences(
    const std::vector<Keypoint2D>& query_kps,
    const std::vector<Keypoint3D>& db_kps,
    const std::vector<FeatureMatch>& matches) const;

private:
  Config::Matching config_;
  torch::Device device_;
  mutable torch::jit::script::Module superglue_;
  bool loaded_ = false;
};

}  // namespace renderloc
