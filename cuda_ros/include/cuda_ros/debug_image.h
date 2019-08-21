#ifndef IMG_PROC_H
#define IMG_PROC_H

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

void imshow(
    const char *title,
    const cv::cuda::GpuMat in);

void warp_image(
    const cv::cuda::GpuMat &vmap,
    const cv::cuda::GpuMat &src,
    cv::cuda::GpuMat &out_dst,
    const Eigen::Affine3d &T,
    const Eigen::Matrix3d &K);

void compute_photometric_residual(
    const cv::cuda::GpuMat vmap_ref,
    const cv::cuda::GpuMat image_src,
    const cv::cuda::GpuMat image_ref,
    const Eigen::Affine3d &T,
    const Eigen::Matrix3d &K,
    cv::cuda::GpuMat &out_residual);

#endif