#include "cuda_icp/icp_tracker.h"
#include "pyr_down.h"
#include "se3_estimator.h"
#include "se3_exp_map.h"
#include <Eigen/SparseCholesky>
#include <cuda_ros/debug_image.h>

ICPTracker::ICPTracker(
    const int image_width,
    const int image_height,
    const Eigen::Matrix3d &K,
    const int num_octave,
    const bool enable_robust_method,
    const int display_debug_info,
    const bool enable_icp,
    const bool enable_rgb,
    const float depth_cutoff,
    const float max_dist,
    const float max_residual,
    const float min_gradient,
    const float max_gradient,
    const float dist_thresh,
    const float angle_thresh)
    : dist_thresh(dist_thresh),
      angle_thresh(angle_thresh),
      enable_icp(enable_icp),
      enable_rgb(enable_rgb),
      max_residual(max_residual),
      depth_cutoff(depth_cutoff),
      min_gradient(min_gradient),
      max_gradient(max_gradient),
      max_dist(max_dist),
      enable_robust_method(enable_robust_method),
      display_debug_info(display_debug_info),
      num_octave(num_octave)
{
  cam_param_pyr.resize(num_octave);
  vmap_src_pyr.resize(num_octave);
  vmap_ref_pyr.resize(num_octave);
  nmap_src_pyr.resize(num_octave);
  nmap_ref_pyr.resize(num_octave);
  depth_src_pyr.resize(num_octave);
  depth_ref_pyr.resize(num_octave);
  image_src_pyr.resize(num_octave);
  image_ref_pyr.resize(num_octave);
  gradient_x_pyr.resize(num_octave);
  gradient_y_pyr.resize(num_octave);

  for (int i = 0; i < num_octave; ++i)
  {
    int cols = image_width / (1 << i);
    int rows = image_height / (1 << i);

    cam_param_pyr[i] = K / (1 << i);
    cam_param_pyr[i](2, 2) = 1.0f;
    vmap_src_pyr[i].create(rows, cols, CV_32FC4);
    vmap_ref_pyr[i].create(rows, cols, CV_32FC4);
    nmap_src_pyr[i].create(rows, cols, CV_32FC4);
    nmap_ref_pyr[i].create(rows, cols, CV_32FC4);
    depth_src_pyr[i].create(rows, cols, CV_32FC1);
    depth_ref_pyr[i].create(rows, cols, CV_32FC1);
    image_src_pyr[i].create(rows, cols, CV_32FC1);
    image_ref_pyr[i].create(rows, cols, CV_32FC1);
    gradient_x_pyr[i].create(rows, cols, CV_32FC1);
    gradient_y_pyr[i].create(rows, cols, CV_32FC1);
  }

  SUM_SE3.create(96, 29, CV_32FC1);
  OUT_SE3.create(1, 29, CV_32FC1);
  SUM_RES.create(96, 2, CV_32FC1);
  OUT_RES.create(1, 2, CV_32FC1);
  CORRES_MAP.create(image_height, image_width, CV_32FC4);
}

void ICPTracker::set_intrinsics(const Eigen::Matrix3d &K)
{
  for (int i = 0; i < num_octave; ++i)
  {
    cam_param_pyr[i] = K / (1 << i);
    cam_param_pyr[i](2, 2) = 1.0f;
  }
}

void ICPTracker::set_source_depth(const cv::cuda::GpuMat depth)
{
  depth.copyTo(depth_src_pyr[0]);
  for (int i = 1; i < num_octave; ++i)
  {
    pyr_down_depth(depth_src_pyr[i - 1], depth_src_pyr[i]);
  }

  for (int i = 0; i < num_octave; ++i)
  {
    compute_vmap(depth_src_pyr[i], vmap_src_pyr[i], cam_param_pyr[i].inverse().cast<float>(), depth_cutoff);
    compute_nmap(vmap_src_pyr[i], nmap_src_pyr[i]);
  }
}

void ICPTracker::set_source_image(const cv::cuda::GpuMat image)
{
  image.copyTo(image_src_pyr[0]);

  for (int i = 1; i < num_octave; ++i)
  {
    pyr_down_image(image_src_pyr[i - 1], image_src_pyr[i]);
  }

  for (int i = 0; i < num_octave; ++i)
  {
    compute_derivative(image_src_pyr[i], gradient_x_pyr[i], gradient_y_pyr[i]);
    // double min_val, max_val;
    // auto sum = cv::cuda::sum(gradient_x_pyr[i]);
    // std::cout << sum / (gradient_x_pyr[i].cols * gradient_x_pyr[i].rows) << std::endl;
    // compute_gradient_sobel(image_src_pyr[i], gradient_x_pyr[i], gradient_y_pyr[i]);
  }
}

void ICPTracker::set_reference_depth(const cv::cuda::GpuMat depth)
{
  depth.copyTo(depth_ref_pyr[0]);
  for (int i = 1; i < num_octave; ++i)
  {
    pyr_down_depth(depth_ref_pyr[i - 1], depth_ref_pyr[i]);
  }

  for (int i = 0; i < num_octave; ++i)
  {
    compute_vmap(depth_ref_pyr[i], vmap_ref_pyr[i], cam_param_pyr[i].inverse().cast<float>(), depth_cutoff);
    compute_nmap(vmap_ref_pyr[i], nmap_ref_pyr[i]);
  }
}

void ICPTracker::set_reference_image(const cv::cuda::GpuMat image)
{
  image.copyTo(image_ref_pyr[0]);

  for (int i = 1; i < num_octave; ++i)
  {
    pyr_down_image(image_ref_pyr[i - 1], image_ref_pyr[i]);
  }
}

void ICPTracker::set_reference_vmap(const cv::cuda::GpuMat vmap)
{
  vmap.copyTo(vmap_ref_pyr[0]);
  for (int i = 1; i < num_octave; ++i)
  {
    pyr_down_vmap(vmap_ref_pyr[i - 1], vmap_ref_pyr[i]);
  }
}

ICPTracker::Result ICPTracker::compute_transform(
    const bool early_stop,
    const cudaStream_t stream,
    const std::vector<int> &max_iterations,
    const Eigen::Affine3d initial_estimate)
{
  auto estimate = initial_estimate;
  Eigen::Affine3d last_success_estimate;

  for (int level = num_octave - 1; level >= 0; --level)
  {
    const auto K = cam_param_pyr[level];
    const auto vmap_src = vmap_src_pyr[level];
    const auto vmap_ref = vmap_ref_pyr[level];
    const auto nmap_src = nmap_src_pyr[level];
    const auto nmap_ref = nmap_ref_pyr[level];

    if (early_stop)
    {
      early_stop_count = 0;
      last_error = std::numeric_limits<float>::max();
    }

    if (enable_icp && display_debug_info >= 0 & level == 0)
    {
      imshow("vmap_src", vmap_src);
      imshow("vmap_ref", vmap_ref);
      imshow("nmap_src", nmap_src);
      imshow("nmap_ref", nmap_ref);
      cv::waitKey(display_debug_info);
    }

    const auto image_src = image_src_pyr[level];
    const auto image_ref = image_ref_pyr[level];
    const auto gradient_x = gradient_x_pyr[level];
    const auto gradient_y = gradient_y_pyr[level];
    const auto depth_src = depth_src_pyr[level];

    if (enable_rgb && display_debug_info >= 0 & level == 0)
    {
      imshow("image_src", image_src);
      imshow("image_ref", image_ref);
      imshow("gradient_x", gradient_x);
      imshow("gradient_y", gradient_y);
      cv::waitKey(display_debug_info);
    }

    for (int iter = 0; iter < max_iterations[level]; ++iter)
    {
      Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
      Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
      Eigen::Matrix<float, 6, 6> H_TEMP;
      Eigen::Matrix<float, 6, 1> b_TEMP;

      float residual_sum = 0;
      size_t num_residual = 0;
      float error = 0;

      if (enable_icp)
      {
        H_TEMP.setZero();
        b_TEMP.setZero();

        compute_icp_step(
            vmap_src,
            nmap_src,
            vmap_ref,
            nmap_ref,
            SUM_SE3,
            OUT_SE3,
            estimate,
            K,
            dist_thresh,
            angle_thresh,
            H_TEMP.data(),
            b_TEMP.data(),
            residual_sum,
            num_residual,
            display_debug_info >= 0);

        error += sqrt(residual_sum) / (num_residual + 1);

        H += H_TEMP;
        b += b_TEMP;
      }

      if (enable_rgb)
      {
        H_TEMP.setZero();
        b_TEMP.setZero();

        compute_rgb_step(
            vmap_ref,
            depth_src,
            image_ref,
            image_src,
            gradient_x,
            gradient_y,
            SUM_SE3,
            OUT_SE3,
            SUM_RES,
            OUT_RES,
            CORRES_MAP,
            estimate,
            K,
            min_gradient,
            max_gradient,
            max_residual,
            max_dist,
            H_TEMP.data(),
            b_TEMP.data(),
            residual_sum,
            num_residual,
            display_debug_info >= 0);

        error += sqrt(residual_sum) / (num_residual + 1);

        if (!enable_icp)
        {
          H += H_TEMP;
          b += b_TEMP;
        }
        else
        {
          H += 0.1 * H_TEMP;
          b += 0.1 * b_TEMP;
        }
      }

      const bool error_increased = (error - last_error) > FLT_EPSILON;

      if (!enable_icp && !enable_rgb)
      {
        std::cout << "Please at least enable one tracking method!" << std::endl;
      }
      else
      {
        if (early_stop && error_increased)
        {
          early_stop_count++;

          if (early_stop_count >= 2)
          {
            if (display_debug_info >= 0)
            {
              std::cout << "early stop condition reached..." << std::endl;
              break;
            }
          }
        }

        Eigen::Matrix<double, 6, 1> xi = H.cast<double>().ldlt().solve(b.cast<double>());
        Eigen::Affine3d dT = se3_exp_map(1.5 * xi);
        estimate = dT * estimate;

        if (!early_stop || !error_increased)
        {
          last_error = error;
          last_success_estimate = estimate;
        }
      }
    }
  }

  Result result;
  result.update = last_success_estimate;
  return result;
}
