#include "render_loc/core/types.h"

#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace renderloc {

std::vector<std::pair<int, double>> ImageDatabase::retrieveTopK(
    const std::vector<float>& query_global_desc, int k) const
{
  std::vector<std::pair<int, double>> results;
  results.reserve(images.size());

  for (size_t i = 0; i < images.size(); ++i) {
    const auto& db_desc = images[i].global_descriptor;
    if (db_desc.empty()) continue;

    double dist = 0.0;
    for (size_t d = 0; d < db_desc.size() && d < query_global_desc.size(); ++d) {
      double diff = query_global_desc[d] - db_desc[d];
      dist += diff * diff;
    }
    dist = std::sqrt(dist);
    results.emplace_back(static_cast<int>(i), dist);
  }

  std::sort(results.begin(), results.end(),
    [](const auto& a, const auto& b) { return a.second < b.second; });

  if (static_cast<int>(results.size()) > k) {
    results.resize(k);
  }
  return results;
}

bool ImageDatabase::save(const std::string& path) const {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs.is_open()) return false;

  int n = static_cast<int>(images.size());
  ofs.write(reinterpret_cast<const char*>(&n), sizeof(int));
  ofs.write(reinterpret_cast<const char*>(&global_desc_dim), sizeof(int));
  ofs.write(reinterpret_cast<const char*>(&local_desc_dim), sizeof(int));

  // Intrinsics
  ofs.write(reinterpret_cast<const char*>(&intrinsics), sizeof(CameraIntrinsics));

  for (const auto& img : images) {
    ofs.write(reinterpret_cast<const char*>(&img.id), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(img.pose.matrix().data()),
              16 * sizeof(double));

    // Global descriptor
    ofs.write(reinterpret_cast<const char*>(img.global_descriptor.data()),
              global_desc_dim * sizeof(float));

    // Keypoints with 3D
    int nkp = static_cast<int>(img.keypoints_3d.size());
    ofs.write(reinterpret_cast<const char*>(&nkp), sizeof(int));
    for (const auto& kp : img.keypoints_3d) {
      ofs.write(reinterpret_cast<const char*>(kp.pt2d.data()), 2 * sizeof(double));
      ofs.write(reinterpret_cast<const char*>(kp.pt3d.data()), 3 * sizeof(double));
      ofs.write(reinterpret_cast<const char*>(&kp.valid), sizeof(bool));

      int desc_size = static_cast<int>(kp.descriptor.size());
      ofs.write(reinterpret_cast<const char*>(&desc_size), sizeof(int));
      if (desc_size > 0) {
        ofs.write(reinterpret_cast<const char*>(kp.descriptor.data()),
                  desc_size * sizeof(float));
      }
    }
  }

  ofs.close();
  std::cout << "[DB] Saved " << n << " images to " << path << std::endl;
  return true;
}

bool ImageDatabase::load(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) return false;

  int n;
  ifs.read(reinterpret_cast<char*>(&n), sizeof(int));
  ifs.read(reinterpret_cast<char*>(&global_desc_dim), sizeof(int));
  ifs.read(reinterpret_cast<char*>(&local_desc_dim), sizeof(int));
  ifs.read(reinterpret_cast<char*>(&intrinsics), sizeof(CameraIntrinsics));

  images.resize(n);
  for (auto& img : images) {
    ifs.read(reinterpret_cast<char*>(&img.id), sizeof(int));

    Mat4d mat;
    ifs.read(reinterpret_cast<char*>(mat.data()), 16 * sizeof(double));
    img.pose.matrix() = mat;

    img.global_descriptor.resize(global_desc_dim);
    ifs.read(reinterpret_cast<char*>(img.global_descriptor.data()),
             global_desc_dim * sizeof(float));

    int nkp;
    ifs.read(reinterpret_cast<char*>(&nkp), sizeof(int));
    img.keypoints_3d.resize(nkp);
    for (auto& kp : img.keypoints_3d) {
      ifs.read(reinterpret_cast<char*>(kp.pt2d.data()), 2 * sizeof(double));
      ifs.read(reinterpret_cast<char*>(kp.pt3d.data()), 3 * sizeof(double));
      ifs.read(reinterpret_cast<char*>(&kp.valid), sizeof(bool));

      int desc_size;
      ifs.read(reinterpret_cast<char*>(&desc_size), sizeof(int));
      kp.descriptor.resize(desc_size);
      if (desc_size > 0) {
        ifs.read(reinterpret_cast<char*>(kp.descriptor.data()),
                 desc_size * sizeof(float));
      }
    }
  }

  ifs.close();
  std::cout << "[DB] Loaded " << n << " images from " << path << std::endl;
  return true;
}

}  // namespace renderloc
