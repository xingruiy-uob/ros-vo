#ifndef PYR_DOWN_H
#define PYR_DOWN_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

void compute_vmap(const cv::cuda::GpuMat depth, cv::cuda::GpuMat &vmap, const Eigen::Matrix3f &Kinv, const float depth_cutoff);
void compute_nmap(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat &nmap);
void compute_derivative(const cv::cuda::GpuMat image, cv::cuda::GpuMat &dx, cv::cuda::GpuMat &dy);
void pyr_down_depth(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst);
void pyr_down_image(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst);
void pyr_down_vmap(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst);
void compute_gradient_sobel(const cv::cuda::GpuMat intensity, cv::cuda::GpuMat &gradient_x, cv::cuda::GpuMat &gradient_y);

#endif