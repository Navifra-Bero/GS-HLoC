#include "render_loc/offline/feature_extractor.h"
#include <iostream>
#include <opencv2/imgproc.hpp>

namespace renderloc {

FeatureExtractor::FeatureExtractor(const Config::Features& config)
  : config_(config),
    device_(config.use_gpu && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{}

bool FeatureExtractor::loadModels() {
  try {
    superpoint_ = torch::jit::load(config_.superpoint_model, device_);
    superpoint_.eval();
    sp_loaded_ = true;
    std::cout << "[SuperPoint] Loaded" << std::endl;
  } catch (const c10::Error& e) {
    std::cerr << "[SuperPoint] Load failed: " << e.what() << std::endl;
    return false;
  }

  try {
    global_model_ = torch::jit::load(config_.global_model, device_);
    global_model_.eval();
    global_loaded_ = true;
    std::cout << "[NetVLAD] Loaded" << std::endl;
  } catch (const c10::Error& e) {
    std::cerr << "[NetVLAD] Load failed: " << e.what() << std::endl;
    return false;
  }
  return true;
}

std::vector<Keypoint2D> FeatureExtractor::extractLocal(const cv::Mat& image) const {
  if (!sp_loaded_) return {};

  auto input = preprocessForSuperPoint(image);
  input = input.to(device_);

  torch::NoGradGuard no_grad;
  std::vector<torch::jit::IValue> inputs{input};
  auto output = superpoint_.forward(inputs).toGenericDict();

  auto kps_tensor = output.at("keypoints").toTensor().squeeze(0).to(torch::kCPU);
  auto desc_tensor = output.at("descriptors").toTensor().squeeze(0).to(torch::kCPU);
  auto scores_tensor = output.at("scores").toTensor().squeeze(0).to(torch::kCPU);

  int n = kps_tensor.size(0);
  std::vector<Keypoint2D> keypoints;

  // Filter and limit
  std::vector<std::pair<float, int>> score_idx;
  for (int i = 0; i < n; ++i) {
    float s = scores_tensor[i].item<float>();
    if (s > config_.keypoint_threshold) {
      score_idx.emplace_back(s, i);
    }
  }
  std::sort(score_idx.begin(), score_idx.end(),
    [](const auto& a, const auto& b) { return a.first > b.first; });

  int limit = std::min(static_cast<int>(score_idx.size()), config_.max_keypoints);
  keypoints.reserve(limit);

  for (int k = 0; k < limit; ++k) {
    int i = score_idx[k].second;
    Keypoint2D kp;
    kp.pt = Vec2d(kps_tensor[i][0].item<float>(), kps_tensor[i][1].item<float>());

    kp.descriptor.resize(256);
    for (int d = 0; d < 256; ++d) {
      kp.descriptor[d] = desc_tensor[d][i].item<float>();
    }
    keypoints.push_back(std::move(kp));
  }
  return keypoints;
}

std::vector<float> FeatureExtractor::extractGlobal(const cv::Mat& image) const {
  if (!global_loaded_) return {};

  auto input = preprocessForGlobal(image);
  input = input.to(device_);

  torch::NoGradGuard no_grad;
  auto output = global_model_.forward({input}).toTensor().to(torch::kCPU).contiguous();

  std::vector<float> desc(config_.global_desc_dim);
  auto acc = output.accessor<float, 2>();
  for (int i = 0; i < config_.global_desc_dim && i < output.size(1); ++i) {
    desc[i] = acc[0][i];
  }
  return desc;
}

FeatureExtractor::Features FeatureExtractor::extractAll(const cv::Mat& image) const {
  Features f;
  f.keypoints = extractLocal(image);
  f.global_descriptor = extractGlobal(image);
  return f;
}

torch::Tensor FeatureExtractor::preprocessForSuperPoint(const cv::Mat& image) const {
  cv::Mat gray;
  if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = image;
  }
  gray.convertTo(gray, CV_32F, 1.0 / 255.0);
  auto tensor = torch::from_blob(gray.data, {1, 1, gray.rows, gray.cols}, torch::kFloat32).clone();
  return tensor;
}

torch::Tensor FeatureExtractor::preprocessForGlobal(const cv::Mat& image) const {
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(224, 224));
  resized.convertTo(resized, CV_32F, 1.0 / 255.0);

  auto tensor = torch::from_blob(resized.data, {1, resized.rows, resized.cols, 3}, torch::kFloat32);
  tensor = tensor.permute({0, 3, 1, 2}).clone();  // NHWC → NCHW
  return tensor;
}

}  // namespace renderloc
