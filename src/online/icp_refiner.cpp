#include "render_loc/online/icp_refiner.h"
#include <pcl/io/ply_io.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <cmath>

namespace renderloc {

ICPRefiner::ICPRefiner(const Config::ICP& config,
                        const CameraIntrinsics& intrinsics)
  : config_(config), intrinsics_(intrinsics) {}

bool ICPRefiner::loadMap(const std::string& ply_path) {
  map_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  if (pcl::io::loadPLYFile<pcl::PointXYZ>(ply_path, *map_cloud_) == -1) {
    std::cerr << "[ICP] Failed to load PLY" << std::endl;
    return false;
  }

  // Downsample for speed
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(map_cloud_);
  vg.setLeafSize(0.1f, 0.1f, 0.1f);
  auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  vg.filter(*filtered);
  map_cloud_ = filtered;

  map_loaded_ = true;
  std::cout << "[ICP] Map loaded: " << map_cloud_->size() << " pts" << std::endl;
  return true;
}

ICPRefiner::ICPResult ICPRefiner::refine(const cv::Mat& depth_image,
                                          const Pose6DoF& initial_pose) const
{
  ICPResult result;
  result.pose = initial_pose;
  result.converged = false;

  if (!map_loaded_) return result;

  // Backproject query depth to point cloud
  auto source = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  int stride = 4;
  for (int v = 0; v < depth_image.rows; v += stride) {
    for (int u = 0; u < depth_image.cols; u += stride) {
      double d = 0;
      if (depth_image.type() == CV_16UC1)
        d = depth_image.at<uint16_t>(v, u) / 1000.0;
      else if (depth_image.type() == CV_32FC1)
        d = depth_image.at<float>(v, u);

      if (d < 0.3 || d > 10.0 || !std::isfinite(d)) continue;

      double x = (u - intrinsics_.cx) * d / intrinsics_.fx;
      double y = (v - intrinsics_.cy) * d / intrinsics_.fy;
      source->push_back(pcl::PointXYZ(x, y, d));
    }
  }

  if (source->empty()) return result;

  // Run GICP
  pcl::PointCloud<pcl::PointXYZ> aligned;
  if (config_.use_gicp) {
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
    gicp.setInputSource(source);
    gicp.setInputTarget(map_cloud_);
    gicp.setMaximumIterations(config_.max_iterations);
    gicp.setMaxCorrespondenceDistance(config_.max_correspondence_dist);
    gicp.align(aligned, initial_pose.matrix().cast<float>());
    result.converged = gicp.hasConverged();
    result.fitness_score = gicp.getFitnessScore();
    result.pose.matrix() = gicp.getFinalTransformation().cast<double>();
  } else {
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source);
    icp.setInputTarget(map_cloud_);
    icp.setMaximumIterations(config_.max_iterations);
    icp.setMaxCorrespondenceDistance(config_.max_correspondence_dist);
    icp.align(aligned, initial_pose.matrix().cast<float>());
    result.converged = icp.hasConverged();
    result.fitness_score = icp.getFitnessScore();
    result.pose.matrix() = icp.getFinalTransformation().cast<double>();
  }

  if (result.fitness_score > config_.fitness_threshold) {
    result.converged = false;
  }
  return result;
}

}  // namespace renderloc
