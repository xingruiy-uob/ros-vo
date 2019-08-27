#include "reduction.h"
#include "se3_experiment.h"
#include <cuda_runtime_api.h>
#include <cuda_ros/vector_math.h>
#include <cuda_ros/device_util.h>
#include <cuda_ros/interpolation.h>

struct ComputeResidualFunctor
{
    Vector3f Kt;
    Matrix3x3f KRKinv;
    int cols, rows, Np;
    float max_residual, max_dist;
    float min_gradient, max_gradient;
    cv::cuda::PtrStep<float> image_ref;
    cv::cuda::PtrStep<float> image_src;
    cv::cuda::PtrStep<float> gradient_x;
    cv::cuda::PtrStep<float> gradient_y;
    cv::cuda::PtrStep<float> depth_ref;

    mutable cv::cuda::PtrStep<Vector4f> corr_map;
    mutable cv::cuda::PtrStep<float> out;

    __device__ __forceinline__ bool find_corresp(
        const int &x,
        const int &y,
        float &residual,
        Vector2f &gradient) const
    {
        auto z = depth_ref.ptr(y)[x];
        if (z != z)
            return false;

        Vector3f pt2d = KRKinv(Vector3f(x, y, z)) + Kt;
        const float u = pt2d.x / pt2d.z;
        const float v = pt2d.y / pt2d.z;
        if (u >= 2 && u < cols - 2 && v >= 2 && v < rows - 2)
        {
            const float val_ref = image_ref.ptr(y)[x];
            const float val_src = interpolate(image_src, u, v);
            gradient.x = interpolate(gradient_x, u, v);
            gradient.y = interpolate(gradient_y, u, v);

            residual = val_src - val_ref;
            return (gradient.norm() > min_gradient &&
                    gradient.norm() < max_gradient &&
                    abs(residual) < max_residual);
        }
        else
        {
            return false;
        }
    }

    __device__ __forceinline__ void compute_residual(
        const int &k, float *const val) const
    {
        const int y = k / cols;
        const int x = k - y * cols;

        float residual = 0;
        Vector2f gradient;

        bool found = find_corresp(x, y, residual, gradient);

        if (found)
        {
            val[0] = residual * residual;
            val[1] = 1;
            corr_map.ptr(y)[x] = Vector4f(gradient.x, gradient.y, residual, 1.0);
        }
        else
        {
            val[0] = val[1] = 0;
            corr_map.ptr(y)[x] = Vector4f(0, 0, 0, -1.0);
        }
    }

    __device__ __forceinline__ void operator()() const
    {
        float sum[2] = {0, 0};
        float val[2];
        for (int k = blockIdx.x * blockDim.x + threadIdx.x;
             k < Np; k += blockDim.x * gridDim.x)
        {
            compute_residual(k, val);
#pragma unroll
            for (int i = 0; i < 2; ++i)
                sum[i] += val[i];
        }

        block_reduce_sum<float, 2>(sum);

        if (threadIdx.x == 0)
#pragma unroll
            for (int i = 0; i < 2; ++i)
                out.ptr(blockIdx.x)[i] = sum[i];
    }
};

void compute_se3_step(
    const cv::cuda::GpuMat &depth_ref,
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
    bool enable_debug_info)
{
}

void compute_depth_step()
{
}