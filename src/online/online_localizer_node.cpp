#include "render_loc/online/localizer.h"
#include "render_loc/core/config.h"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_ros/transform_broadcaster.h>

namespace renderloc {

class LocalizerNode : public rclcpp::Node {
public:
  LocalizerNode() : Node("renderloc_localizer") {
    declare_parameter("config_file", "");
    declare_parameter("database_path", "");
    declare_parameter("rgb_topic", "/camera/color/image_raw");
    declare_parameter("depth_topic", "/camera/depth/image_raw");
    declare_parameter("use_depth", false);
    declare_parameter("publish_tf", true);
    declare_parameter("map_frame", "map");
    declare_parameter("camera_frame", "camera_link");
    declare_parameter("rate_hz", 5.0);

    auto config_file = get_parameter("config_file").as_string();
    auto db_path     = get_parameter("database_path").as_string();
    auto rgb_topic   = get_parameter("rgb_topic").as_string();
    auto depth_topic = get_parameter("depth_topic").as_string();
    use_depth_       = get_parameter("use_depth").as_bool();
    publish_tf_      = get_parameter("publish_tf").as_bool();
    map_frame_       = get_parameter("map_frame").as_string();
    cam_frame_       = get_parameter("camera_frame").as_string();
    double hz        = get_parameter("rate_hz").as_double();

    min_interval_ns_ = static_cast<int64_t>(1e9 / hz);

    // Load config
    Config config;
    if (!config_file.empty()) {
      config = Config::loadFromYaml(config_file);
    }
    if (!db_path.empty()) {
      config.database.db_path = db_path;
    }

    localizer_ = std::make_unique<Localizer>(config);
    if (!localizer_->initialize()) {
      RCLCPP_ERROR(get_logger(), "Localizer init failed!");
      return;
    }

    // Publishers
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/renderloc/pose", 10);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/renderloc/odom", 10);

    if (publish_tf_) {
      tf_br_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    // Subscribers
    if (use_depth_) {
      rgb_sub_.subscribe(this, rgb_topic);
      depth_sub_.subscribe(this, depth_topic);
      using SyncPol = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
      sync_ = std::make_shared<message_filters::Synchronizer<SyncPol>>(
        SyncPol(10), rgb_sub_, depth_sub_);
      sync_->registerCallback(std::bind(&LocalizerNode::rgbdCb, this,
        std::placeholders::_1, std::placeholders::_2));
    } else {
      rgb_only_sub_ = create_subscription<sensor_msgs::msg::Image>(
        rgb_topic, 10, std::bind(&LocalizerNode::rgbCb, this, std::placeholders::_1));
    }

    RCLCPP_INFO(get_logger(), "RenderLoc ready (depth=%s)", use_depth_ ? "on" : "off");
  }

private:
  void rgbCb(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    if (!checkRate()) return;
    auto cv_img = cv_bridge::toCvShare(msg, "bgr8");
    auto result = localizer_->localize(cv_img->image);
    publishResult(result, msg->header.stamp);
  }

  void rgbdCb(const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
               const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) {
    if (!checkRate()) return;
    auto rgb_cv = cv_bridge::toCvShare(rgb_msg, "bgr8");
    auto depth_cv = cv_bridge::toCvShare(depth_msg);
    auto result = localizer_->localize(rgb_cv->image, depth_cv->image);
    publishResult(result, rgb_msg->header.stamp);
  }

  bool checkRate() {
    auto now = this->now().nanoseconds();
    if (now - last_time_ < min_interval_ns_) return false;
    last_time_ = now;
    return true;
  }

  void publishResult(const LocalizationResult& r,
                      const builtin_interfaces::msg::Time& stamp) {
    if (!r.success) return;

    const auto& pose = r.icp_used ? r.refined_pose : r.pose;
    const auto& t = pose.translation();
    Eigen::Quaterniond q(pose.rotation());

    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp = stamp;
    ps.header.frame_id = map_frame_;
    ps.pose.position.x = t.x();
    ps.pose.position.y = t.y();
    ps.pose.position.z = t.z();
    ps.pose.orientation.x = q.x();
    ps.pose.orientation.y = q.y();
    ps.pose.orientation.z = q.z();
    ps.pose.orientation.w = q.w();
    pose_pub_->publish(ps);

    nav_msgs::msg::Odometry odom;
    odom.header = ps.header;
    odom.child_frame_id = cam_frame_;
    odom.pose.pose = ps.pose;
    odom_pub_->publish(odom);

    if (publish_tf_ && tf_br_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header = ps.header;
      tf.child_frame_id = cam_frame_;
      tf.transform.translation.x = t.x();
      tf.transform.translation.y = t.y();
      tf.transform.translation.z = t.z();
      tf.transform.rotation = ps.pose.orientation;
      tf_br_->sendTransform(tf);
    }

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
      "Loc: db=%d inliers=%d total=%.0fms [ret=%.0f match=%.0f pnp=%.0f]",
      r.matched_db_id, r.num_inliers, r.total_ms,
      r.retrieval_ms, r.matching_ms, r.pnp_ms);
  }

  std::unique_ptr<Localizer> localizer_;

  // RGB-only
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_only_sub_;

  // RGB-D sync
  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_, depth_sub_;
  std::shared_ptr<message_filters::Synchronizer<
    message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image>>> sync_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_br_;

  bool use_depth_, publish_tf_;
  std::string map_frame_, cam_frame_;
  int64_t min_interval_ns_ = 0, last_time_ = 0;
};

}  // namespace renderloc

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<renderloc::LocalizerNode>());
  rclcpp::shutdown();
}
