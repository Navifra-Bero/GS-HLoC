#include "render_loc/online/feature_matcher.h"
#include <iostream>

namespace renderloc {

FeatureMatcher::FeatureMatcher(const Config::Matching& config)
  : config_(config),
    device_(config.use_gpu && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{}

bool FeatureMatcher::loadModel() {
  try {
    superglue_ = torch::jit::load(config_.superglue_model, device_);
    superglue_.eval();
    loaded_ = true;
    std::cout << "[SuperGlue] Loaded" << std::endl;
    return true;
  } catch (const c10::Error& e) {
    std::cerr << "[SuperGlue] Load failed: " << e.what() << std::endl;
    return false;
  }
}

std::vector<FeatureMatch> FeatureMatcher::match(
    const std::vector<Keypoint2D>& query_kps,
    const std::vector<Keypoint3D>& db_kps,
    const cv::Size& image_size) const
{
  if (!loaded_ || query_kps.empty() || db_kps.empty()) return {};

  int nq = static_cast<int>(query_kps.size());
  int nd = static_cast<int>(db_kps.size());

  // Build tensors for SuperGlue input
  auto kps0 = torch::zeros({1, nq, 2}, torch::kFloat32);
  auto desc0 = torch::zeros({1, 256, nq}, torch::kFloat32);
  auto scores0 = torch::ones({1, nq}, torch::kFloat32);

  for (int i = 0; i < nq; ++i) {
    kps0[0][i][0] = static_cast<float>(query_kps[i].pt.x());
    kps0[0][i][1] = static_cast<float>(query_kps[i].pt.y());
    for (int d = 0; d < 256 && d < static_cast<int>(query_kps[i].descriptor.size()); ++d) {
      desc0[0][d][i] = query_kps[i].descriptor[d];
    }
  }

  auto kps1 = torch::zeros({1, nd, 2}, torch::kFloat32);
  auto desc1 = torch::zeros({1, 256, nd}, torch::kFloat32);
  auto scores1 = torch::ones({1, nd}, torch::kFloat32);

  for (int i = 0; i < nd; ++i) {
    kps1[0][i][0] = static_cast<float>(db_kps[i].pt2d.x());
    kps1[0][i][1] = static_cast<float>(db_kps[i].pt2d.y());
    for (int d = 0; d < 256 && d < static_cast<int>(db_kps[i].descriptor.size()); ++d) {
      desc1[0][d][i] = db_kps[i].descriptor[d];
    }
  }

  // Image size tensor
  auto img_size = torch::tensor(
    {image_size.height, image_size.width}, torch::kFloat32).unsqueeze(0);

  // Forward pass
  torch::NoGradGuard no_grad;
  auto inputs = torch::jit::Stack();
  // SuperGlue expects a dict; exact format depends on TorchScript export
  // Simplified: pass as positional args
  auto output = superglue_.forward({
    kps0.to(device_), kps1.to(device_),
    desc0.to(device_), desc1.to(device_),
    scores0.to(device_), scores1.to(device_)
  });

  // Parse matches
  auto matches_tensor = output.toGenericDict().at("matches0").toTensor().squeeze(0).to(torch::kCPU);
  auto conf_tensor = output.toGenericDict().at("matching_scores0").toTensor().squeeze(0).to(torch::kCPU);

  std::vector<FeatureMatch> matches;
  for (int i = 0; i < nq; ++i) {
    int match_idx = matches_tensor[i].item<int>();
    float conf = conf_tensor[i].item<float>();

    if (match_idx >= 0 && conf > config_.match_threshold) {
      FeatureMatch fm;
      fm.query_idx = i;
      fm.db_idx = match_idx;
      fm.confidence = conf;
      matches.push_back(fm);
    }
  }

  return matches;
}

FeatureMatcher::Correspondences FeatureMatcher::collectCorrespondences(
    const std::vector<Keypoint2D>& query_kps,
    const std::vector<Keypoint3D>& db_kps,
    const std::vector<FeatureMatch>& matches) const
{
  Correspondences corr;
  for (const auto& m : matches) {
    const auto& db_kp = db_kps[m.db_idx];
    if (!db_kp.valid) continue;  // Skip keypoints without valid 3D

    const auto& q_kp = query_kps[m.query_idx];
    corr.pts2d.emplace_back(
      static_cast<float>(q_kp.pt.x()),
      static_cast<float>(q_kp.pt.y()));
    corr.pts3d.emplace_back(
      static_cast<float>(db_kp.pt3d.x()),
      static_cast<float>(db_kp.pt3d.y()),
      static_cast<float>(db_kp.pt3d.z()));
    corr.confidences.push_back(m.confidence);
  }
  return corr;
}

}  // namespace renderloc
