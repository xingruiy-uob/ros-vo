#ifndef IMG_PROC_H
#define IMG_PROC_H

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

void inline imshow(const char *title, cv::cuda::GpuMat in)
{
    cv::Mat img(in);
    cv::imshow(title, img);
};

void warp_image(
    const cv::cuda::GpuMat &vmap,
    const cv::cuda::GpuMat &src,
    cv::cuda::GpuMat &out_dst,
    const Eigen::Affine3d &T,
    const Eigen::Matrix3d &K);

#endif