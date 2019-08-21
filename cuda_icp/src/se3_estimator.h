#ifndef SE3_ESTIMATOR_H
#define SE3_ESTIMATOR_H

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

void compute_icp_step(
    const cv::cuda::GpuMat vmap_src,
    const cv::cuda::GpuMat nmap_src,
    const cv::cuda::GpuMat vmap_ref,
    const cv::cuda::GpuMat nmap_ref,
    cv::cuda::GpuMat sum,
    cv::cuda::GpuMat out,
    const Eigen::Affine3d &T,
    const Eigen::Matrix3d &K,
    const float dist_thresh,
    const float angle_thresh,
    float *const H,
    float *const b,
    float &residual_sum,
    size_t &num_residual,
    const bool enable_debug_info = false);

void compute_rgb_step(
    const cv::cuda::GpuMat &vmap_ref,
    const cv::cuda::GpuMat &depth_src,
    const cv::cuda::GpuMat &image_ref,
    const cv::cuda::GpuMat &image_src,
    const cv::cuda::GpuMat &gradient_x,
    const cv::cuda::GpuMat &gradient_y,
    cv::cuda::GpuMat &sum,
    cv::cuda::GpuMat &out,
    cv::cuda::GpuMat &sum_res,
    cv::cuda::GpuMat &out_res,
    cv::cuda::GpuMat &corr_map,
    const Eigen::Affine3d &T,
    const Eigen::Matrix3d &K,
    const float min_gradient,
    const float max_gradient,
    const float max_residual,
    const float max_dist,
    float *const H,
    float *const b,
    float &residual_sum,
    size_t &num_residual,
    bool enable_debug_info = false);

#endif