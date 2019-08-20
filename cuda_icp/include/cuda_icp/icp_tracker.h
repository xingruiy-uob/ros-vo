#ifndef ICP_TRACKER_H
#define ICP_TRACKER_H

#include <Eigen/Geometry>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

class ICPTracker
{
public:
  struct Result
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Affine3d update;
    bool icp_enabled;
    bool rgb_enabled;
    float icp_resdiual_sum;
    float rgb_residual_sum;
  };

  ICPTracker() = delete;
  ICPTracker(const ICPTracker &) = delete;

  ICPTracker(
      const int image_width,
      const int image_height,
      const Eigen::Matrix3d &K,
      const int num_octave,
      const bool enable_robust_method = false,
      const int display_debug_info = -1,
      const bool enable_icp = true,
      const bool enable_rgb = true,
      const float depth_cutoff = 10.0f,
      const float max_dist = 1,
      const float max_residual = 255,
      const float min_gradient = 0,
      const float max_gradient = 255,
      const float dist_thresh = 0.01,
      const float angle_thresh = cos(30 * 3.14 / 180));

  Result compute_transform(
      const bool early_stop = true,
      const cudaStream_t stream = NULL,
      const std::vector<int> &max_iterations = {10, 5, 5, 3, 3},
      const Eigen::Affine3d initial_estimate = Eigen::Affine3d::Identity());

  void set_intrinsics(const Eigen::Matrix3d &K);
  void set_source_depth(const cv::cuda::GpuMat depth);
  void set_source_image(const cv::cuda::GpuMat image);
  void set_reference_depth(const cv::cuda::GpuMat depth);
  void set_reference_image(const cv::cuda::GpuMat image);
  void set_reference_vmap(const cv::cuda::GpuMat vmap);

  cv::cuda::GpuMat inline get_vmap_src(const int level = 0)
  {
    return vmap_src_pyr[level];
  }

  cv::cuda::GpuMat inline get_vmap_ref(const int level = 0)
  {
    return vmap_ref_pyr[level];
  }

  cv::cuda::GpuMat inline get_nmap_src(const int level = 0)
  {
    return nmap_src_pyr[level];
  }

  cv::cuda::GpuMat inline get_nmap_ref(const int level = 0)
  {
    return nmap_ref_pyr[level];
  }

  cv::cuda::GpuMat inline get_depth_src(const int level = 0)
  {
    return depth_src_pyr[level];
  }

  cv::cuda::GpuMat inline get_depth_ref(const int level = 0)
  {
    return depth_ref_pyr[level];
  }

  cv::cuda::GpuMat inline get_image_src(const int level = 0)
  {
    return image_src_pyr[level];
  }

  cv::cuda::GpuMat inline get_image_ref(const int level = 0)
  {
    return image_ref_pyr[level];
  }

  cv::cuda::GpuMat inline get_gradient_x(const int level = 0)
  {
    return gradient_x_pyr[level];
  }

  cv::cuda::GpuMat inline get_gradient_y(const int level = 0)
  {
    return gradient_y_pyr[level];
  }

private:
  std::vector<Eigen::Matrix3d> cam_param_pyr;
  std::vector<cv::cuda::GpuMat> vmap_src_pyr;
  std::vector<cv::cuda::GpuMat> vmap_ref_pyr;
  std::vector<cv::cuda::GpuMat> nmap_src_pyr;
  std::vector<cv::cuda::GpuMat> nmap_ref_pyr;
  std::vector<cv::cuda::GpuMat> depth_src_pyr;
  std::vector<cv::cuda::GpuMat> depth_ref_pyr;
  std::vector<cv::cuda::GpuMat> image_src_pyr;
  std::vector<cv::cuda::GpuMat> image_ref_pyr;
  std::vector<cv::cuda::GpuMat> gradient_x_pyr;
  std::vector<cv::cuda::GpuMat> gradient_y_pyr;

  cv::cuda::GpuMat SUM_SE3;
  cv::cuda::GpuMat OUT_SE3;
  cv::cuda::GpuMat SUM_RES;
  cv::cuda::GpuMat OUT_RES;
  cv::cuda::GpuMat CORRES_MAP;

  const bool enable_icp;
  const bool enable_rgb;
  const bool enable_robust_method;
  const int display_debug_info;

  const int num_octave;
  const float dist_thresh;
  const float angle_thresh;
  const float max_residual;
  const float depth_cutoff;
  const float min_gradient;
  const float max_gradient;
  const float max_dist;

  int early_stop_count;
  float last_error;
};

#endif