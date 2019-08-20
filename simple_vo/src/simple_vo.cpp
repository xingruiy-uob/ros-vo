#include "simple_vo/simple_vo.h"
#include <tf2_eigen/tf2_eigen.h>
#include <cuda_ros/debug_image.h>

enum ResultType
{
    RES_OK,
    RES_LOST,
    RES_NOT_ACCURATE
};

SimpleVO::SimpleVO(
    const int image_width,
    const int image_height,
    const bool enable_debug_output,
    const char *depth_image_topic,
    const char *color_image_topic,
    const char *depth_camera_info,
    const char *color_camera_info,
    const Eigen::Matrix3d &K,
    const double icp_max_error,
    const double rgb_max_error,
    const double max_time_diff,
    const size_t warm_up_frame)
    : image_width(image_width),
      image_height(image_height),
      max_time_diff(max_time_diff),
      warm_up_frame(warm_up_frame),
      icp_max_error(icp_max_error),
      rgb_max_error(rgb_max_error),
      enable_debug_output(enable_debug_output),
      cam_param(K),
      tracker(NULL),
      num_frame(0)
{
    ROS_INFO("SimpleVO ctor...");
    camera_pose.setIdentity();

    rgb_sub.subscribe(ros_node, color_image_topic, 10);
    rgb_info_sub.subscribe(ros_node, color_camera_info, 10);
    depth_sub.subscribe(ros_node, depth_image_topic, 10);
    depth_info_sub.subscribe(ros_node, depth_camera_info, 10);

    debug_depth_diff = ros_node.advertise<sensor_msgs::Image>("/simplevo/debug/depth_diff", 1);
    debug_gradient_x = ros_node.advertise<sensor_msgs::Image>("/simplevo/debug/gradient_x", 1);
    debug_intensity_diff = ros_node.advertise<sensor_msgs::Image>("/simplevo/debug/intensity_diff", 1);

    sync = std::make_shared<message_filters::Synchronizer<RGBDPolicy>>(
        RGBDPolicy(100),
        rgb_sub,
        rgb_info_sub,
        depth_sub,
        depth_info_sub);

    sync->registerCallback(boost::bind(&SimpleVO::handle_images, this, _1, _2, _3, _4));

    tracker = std::make_shared<ICPTracker>(
        image_width,
        image_height,
        K,
        5,
        false, // robust estimator
        -1,    // debug on
        true,  // use icp
        true,  // use rgb
        3.0,   // depth cutoff
        0.03,  // max dist
        25     // max residual
    );

    last_frame.clear_pose();
    current_frame.clear_pose();
    current_keyframe.clear_pose();
}

bool SimpleVO::check_keyframe()
{
    return false;
}

void SimpleVO::create_new_keyframe()
{
    last_frame.copy_to(current_keyframe);
}

bool SimpleVO::check_result(const ICPTracker::Result &result)
{
    return true;
}

void SimpleVO::handle_images(
    const sensor_msgs::Image::ConstPtr &rgb_raw,
    const sensor_msgs::CameraInfo::ConstPtr &rgb_info,
    const sensor_msgs::Image::ConstPtr &depth_registered,
    const sensor_msgs::CameraInfo::ConstPtr &depth_info)
{
    double time_diff = std::abs(rgb_raw->header.stamp.toSec() - depth_registered->header.stamp.toSec());
    ROS_INFO_COND(time_diff > max_time_diff, "Images received with time difference: %f sec(s)", time_diff);

    if (rgb_info != NULL)
    {
        ROS_ERROR_COND(rgb_info->width != image_width || rgb_info->height != image_height, "Image size wrong!");
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K(rgb_info->K.data());
        if (abs((K.inverse() * cam_param).determinant() - 1.0) >= DBL_EPSILON)
        {
            ROS_INFO("Intrinsic Matrix K changed!");
            cam_param = K;
            tracker->set_intrinsics(K);
        }
    }

    if (warm_up_frame > 0)
    {
        warm_up_frame -= 1;
        return;
    }

    if (sensor_msgs::image_encodings::TYPE_32FC1 == depth_registered->encoding)
    {
        current_frame.depth = cv::Mat(
            depth_registered->height,
            depth_registered->width,
            CV_32FC1,
            (void *)depth_registered->data.data(),
            depth_registered->step);
    }
    else if (sensor_msgs::image_encodings::TYPE_16UC1 == depth_registered->encoding)
    {
        cv::Mat depth_raw(
            depth_registered->height,
            depth_registered->width,
            CV_16UC1,
            (void *)depth_registered->data.data(),
            depth_registered->step);

        depth_raw.convertTo(current_frame.depth, CV_32FC1, 1.0 / 1000);
    }
    else
    {
        ROS_ERROR("only support depth encoding CV_16UC1 and CV_32FC1!");
    }

    cv::Mat image_raw(
        rgb_raw->height,
        rgb_raw->width,
        CV_8UC3,
        (void *)rgb_raw->data.data(),
        rgb_raw->step);

    image_raw.convertTo(image_src_float, CV_32FC3);
    cv::cvtColor(image_src_float, current_frame.image, cv::COLOR_RGB2GRAY);

    if (num_frame != 0)
    {
        tracker->set_source_depth(cv::cuda::GpuMat(current_frame.depth));
        tracker->set_source_image(cv::cuda::GpuMat(current_frame.image));
        tracker->set_reference_depth(cv::cuda::GpuMat(last_frame.depth));
        tracker->set_reference_image(cv::cuda::GpuMat(last_frame.image));
        auto result = tracker->compute_transform(true, NULL, {10, 5, 3, 3, 3});

        if (check_result(result))
        {
            auto pose = last_frame.get_pose();
            pose = pose * result.update.inverse();
            current_frame.set_pose(pose);

            if (check_keyframe())
            {
                create_new_keyframe();
                ROS_INFO("New keyframe created...");
            }
        }

        if (enable_debug_output)
        {
            cv::Mat out;
            cv::subtract(current_frame.depth, last_frame.depth, out);
            sensor_msgs::Image msg;
            msg.header.stamp = ros::Time::now();
            msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            msg.width = out.cols;
            msg.height = out.rows;
            msg.step = out.step;
            msg.data = std::vector<uchar>(out.data, out.data + out.step * out.rows);
            debug_depth_diff.publish(msg);

            cv::subtract(current_frame.image, last_frame.image, out);
            msg.header.stamp = ros::Time::now();
            msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            msg.width = out.cols;
            msg.height = out.rows;
            msg.step = out.step;
            msg.data = std::vector<uchar>(out.data, out.data + out.step * out.rows);
            debug_intensity_diff.publish(msg);

            out = cv::Mat(tracker->get_gradient_x());
            msg.header.stamp = ros::Time::now();
            msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            msg.width = out.cols;
            msg.height = out.rows;
            msg.step = out.step;
            msg.data = std::vector<uchar>(out.data, out.data + out.step * out.rows);
            debug_gradient_x.publish(msg);
        }
    }

    current_frame.copy_to(last_frame);

    num_frame += 1;
    publish_pose();
}

void SimpleVO::reset_vo()
{
    num_frame = 0;
    last_frame.clear_pose();
    current_frame.clear_pose();
    current_keyframe.clear_pose();
}

void SimpleVO::publish_pose()
{
    auto zplus = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0)).toRotationMatrix();
    auto xplus = Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0)).toRotationMatrix();
    auto transformStamped = tf2::eigenToTransform(xplus * zplus * current_frame.get_pose().inverse());
    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = "camera_link";
    transformStamped.child_frame_id = "world";
    br.sendTransform(transformStamped);
}