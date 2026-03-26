#include "render_loc/online/localizer.h"
#include "render_loc/utils/timer.h"

#include <iostream>
#include <algorithm>

namespace renderloc {

Localizer::Localizer(const Config& config)
  : config_(config),
    feature_extractor_(config.features),
    matcher_(config.matching),
    pnp_solver_(config.pnp, {config.camera.fx, config.camera.fy,
                              config.camera.cx, config.camera.cy,
                              config.camera.width, config.camera.height}),
    icp_refiner_(config.icp, {config.camera.fx, config.camera.fy,
                               config.camera.cx, config.camera.cy,
                               config.camera.width, config.camera.height})
{
  intrinsics_ = {config.camera.fx, config.camera.fy,
                  config.camera.cx, config.camera.cy,
                  config.camera.width, config.camera.height};
}

bool Localizer::initialize() {
  std::cout << "=== RenderLoc Localizer Init ===" << std::endl;

  if (!feature_extractor_.loadModels()) {
    std::cerr << "[Localizer] Feature extractor init failed" << std::endl;
    return false;
  }

  if (!matcher_.loadModel()) {
    std::cerr << "[Localizer] Feature matcher init failed" << std::endl;
    return false;
  }

  if (!database_.load(config_.database.db_path)) {
    std::cerr << "[Localizer] Database load failed" << std::endl;
    return false;
  }

  if (config_.icp.enable && !config_.database.ply_map_path.empty()) {
    icp_refiner_.loadMap(config_.database.ply_map_path);
  }

  initialized_ = true;
  std::cout << "[Localizer] Ready. DB has " << database_.images.size()
            << " images" << std::endl;
  return true;
}

bool Localizer::loadDatabase(const std::string& db_path) {
  return database_.load(db_path);
}

LocalizationResult Localizer::localize(const cv::Mat& rgb_image) {
  return localize(rgb_image, cv::Mat());
}

LocalizationResult Localizer::localize(const cv::Mat& rgb_image,
                                        const cv::Mat& depth_image) {
  LocalizationResult result;
  result.success = false;
  Timer timer;

  if (!initialized_) {
    std::cerr << "[Localizer] Not initialized" << std::endl;
    return result;
  }

  // ===========================================================
  // Step 1: Extract features from query image
  // ===========================================================
  timer.start("feature_extract");
  auto query_features = feature_extractor_.extractAll(rgb_image);
  timer.stop("feature_extract");

  if (query_features.keypoints.empty()) {
    std::cerr << "[Localizer] No keypoints in query" << std::endl;
    return result;
  }

  // ===========================================================
  // Step 2: Retrieve top-K similar DB images
  // ===========================================================
  timer.start("retrieval");
  auto candidates = database_.retrieveTopK(
    query_features.global_descriptor, config_.matching.top_k_retrieval);
  result.retrieval_ms = timer.stop("retrieval");

  if (candidates.empty()) {
    std::cerr << "[Localizer] No retrieval candidates" << std::endl;
    return result;
  }

  result.matched_db_id = candidates[0].first;
  result.retrieval_score = candidates[0].second;

  // ===========================================================
  // Step 3: Match features and collect 2D-3D correspondences
  // ===========================================================
  timer.start("matching");
  std::vector<FeatureMatcher::Correspondences> all_corrs;

  for (const auto& [db_id, score] : candidates) {
    const auto& db_img = database_.images[db_id];

    auto matches = matcher_.match(
      query_features.keypoints, db_img.keypoints_3d,
      cv::Size(intrinsics_.width, intrinsics_.height));

    if (!matches.empty()) {
      auto corr = matcher_.collectCorrespondences(
        query_features.keypoints, db_img.keypoints_3d, matches);

      if (!corr.pts2d.empty()) {
        all_corrs.push_back(std::move(corr));
      }
    }
  }
  result.matching_ms = timer.stop("matching");

  if (all_corrs.empty()) {
    std::cerr << "[Localizer] No correspondences found" << std::endl;
    return result;
  }

  // ===========================================================
  // Step 4: PnP + RANSAC → 6DoF pose
  // ===========================================================
  timer.start("pnp");
  auto pnp_result = pnp_solver_.solveMulti(all_corrs);
  result.pnp_ms = timer.stop("pnp");

  if (!pnp_result.success) {
    std::cerr << "[Localizer] PnP failed" << std::endl;
    return result;
  }

  result.pose = pnp_result.pose;
  result.num_inliers = pnp_result.num_inliers;
  result.refined_pose = pnp_result.pose;

  // ===========================================================
  // Step 5 (Optional): ICP refinement with query depth
  // ===========================================================
  if (config_.icp.enable && !depth_image.empty()) {
    timer.start("icp");
    auto icp_result = icp_refiner_.refine(depth_image, pnp_result.pose);
    result.icp_ms = timer.stop("icp");

    if (icp_result.converged) {
      result.refined_pose = icp_result.pose;
      result.icp_fitness = icp_result.fitness_score;
      result.icp_used = true;
    }
  }

  result.total_ms = result.retrieval_ms + result.matching_ms +
                    result.pnp_ms + result.icp_ms +
                    timer.get("feature_extract");
  result.success = true;
  last_result_ = result;

  return result;
}

}  // namespace renderloc
