#pragma once

#include "render_loc/core/types.h"
#include "render_loc/core/config.h"

#include <opencv2/core.hpp>
#include <torch/torch.h>
#include <torch/script.h>

namespace renderloc {

/// Extracts local features (SuperPoint) and global descriptors (NetVLAD)
class FeatureExtractor {
public:
  explicit FeatureExtractor(const Config::Features& config);

  /// Load pretrained models
  bool loadModels();

  // ---- Local Features (SuperPoint) ----

  /// Extract keypoints + descriptors from a single image
  std::vector<Keypoint2D> extractLocal(const cv::Mat& image) const;

  // ---- Global Descriptor (NetVLAD / EigenPlaces) ----

  /// Extract global descriptor for image retrieval
  std::vector<float> extractGlobal(const cv::Mat& image) const;

  // ---- Combined ----

  /// Extract both local and global features
  struct Features {
    std::vector<Keypoint2D> keypoints;
    std::vector<float> global_descriptor;
  };

  Features extractAll(const cv::Mat& image) const;

  int localDescDim() const { return 256; }  // SuperPoint default
  int globalDescDim() const { return config_.global_desc_dim; }

private:
  Config::Features config_;
  torch::Device device_;

  mutable torch::jit::script::Module superpoint_;
  mutable torch::jit::script::Module global_model_;
  bool sp_loaded_ = false;
  bool global_loaded_ = false;

  /// Preprocess image for SuperPoint (grayscale, normalize, tensor)
  torch::Tensor preprocessForSuperPoint(const cv::Mat& image) const;

  /// Preprocess image for NetVLAD (resize, normalize, tensor)
  torch::Tensor preprocessForGlobal(const cv::Mat& image) const;
};

}  // namespace renderloc
