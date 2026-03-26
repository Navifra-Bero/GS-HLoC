#pragma once

#include "render_loc/core/types.h"
#include "render_loc/core/config.h"
#include "render_loc/offline/feature_extractor.h"
#include "render_loc/online/feature_matcher.h"
#include "render_loc/online/pnp_solver.h"
#include "render_loc/online/icp_refiner.h"

#include <opencv2/core.hpp>

namespace renderloc {

/// Main online localization: single RGB image → 6DoF pose in map
///
/// Pipeline per query:
///   1. Extract global descriptor → retrieve top-K DB images
///   2. Extract local features → SuperGlue match against each DB image
///   3. Collect 2D-3D correspondences → PnP+RANSAC → 6DoF
///   4. (Optional) Depth → ICP refinement
class Localizer {
public:
  explicit Localizer(const Config& config);

  /// Initialize models and load database
  bool initialize();

  /// Load pre-built database
  bool loadDatabase(const std::string& db_path);

  /// Localize from single RGB image (main use case)
  LocalizationResult localize(const cv::Mat& rgb_image);

  /// Localize from RGB-D (enables optional ICP refinement)
  LocalizationResult localize(const cv::Mat& rgb_image,
                               const cv::Mat& depth_image);

  const LocalizationResult& lastResult() const { return last_result_; }

private:
  Config config_;
  CameraIntrinsics intrinsics_;

  FeatureExtractor feature_extractor_;
  FeatureMatcher matcher_;
  PnPSolver pnp_solver_;
  ICPRefiner icp_refiner_;

  ImageDatabase database_;
  LocalizationResult last_result_;
  bool initialized_ = false;
};

}  // namespace renderloc
