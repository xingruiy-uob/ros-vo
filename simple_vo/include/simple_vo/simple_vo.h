#ifndef SIMPLE_VO_H
#define SIMPLE_VO_H

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <cuda_icp/icp_tracker.h>
#include "simple_vo/frame.h"

using RGBDPolicy =
    message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image,
        sensor_msgs::CameraInfo,
        sensor_msgs::Image,
        sensor_msgs::CameraInfo>;

class SimpleVO
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SimpleVO(
        const int image_width,
        const int image_height,
        const bool enable_debug_output = false,
        const char *depth_image_topic = "/camera/depth/image_raw",
        const char *color_image_topic = "/camera/rgb/image_raw",
        const char *depth_camera_info = "/camera/depth/camera_info",
        const char *color_camera_info = "/camera/rgb/camera_info",
        const Eigen::Matrix3d &K = Eigen::Matrix3d(),
        const double icp_max_error = 10e-4,
        const double rgb_max_error = 10e-3,
        const double max_time_diff = 0.03,
        const size_t warm_up_frame = 0);
    void reset_vo();
    void handle_images(
        const sensor_msgs::Image::ConstPtr &rgb_raw,
        const sensor_msgs::CameraInfo::ConstPtr &rgb_info,
        const sensor_msgs::Image::ConstPtr &depth_registered,
        const sensor_msgs::CameraInfo::ConstPtr &depth_info);

    void publish_pose();
    bool check_keyframe();
    void create_new_keyframe();
    bool check_result(const ICPTracker::Result &result);

private:
    ros::NodeHandle ros_node;
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub;
    message_filters::Subscriber<sensor_msgs::CameraInfo> rgb_info_sub;
    message_filters::Subscriber<sensor_msgs::CameraInfo> depth_info_sub;
    std::shared_ptr<message_filters::Synchronizer<RGBDPolicy>> sync;
    std::shared_ptr<ICPTracker> tracker;

    ros::Publisher debug_depth_diff;
    ros::Publisher debug_gradient_x;
    ros::Publisher debug_intensity_diff;

    Eigen::Affine3d camera_pose;
    tf2_ros::TransformBroadcaster br;

    cv::Mat image_src_float;
    cv::Mat intensity_src_float;

    size_t num_frame;
    size_t warm_up_frame;
    Eigen::Matrix3d cam_param;

    const int image_width;
    const int image_height;
    const double max_time_diff;
    const bool enable_debug_output;
    const double rgb_max_error;
    const double icp_max_error;

    Frame last_frame;
    Frame current_frame;
    Frame current_keyframe;
};

#endif