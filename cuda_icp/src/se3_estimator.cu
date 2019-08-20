#include "se3_estimator.h"
#include "reduction.h"
#include <cuda_runtime_api.h>
#include <cuda_ros/vector_math.h>
#include <cuda_ros/device_util.h>
#include <cuda_ros/interpolation.h>

template <int rows, int cols>
void inline create_matrix(
    cv::Mat &host,
    float *const H,
    float *const b)
{
    int shift = 0;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = i; j < cols; ++j)
        {
            const float value = host.ptr<float>(0)[shift++];
            if (j == rows)
                b[i] = value;
            else
                H[j * rows + i] = H[i * rows + j] = value;
        }
    }
}

struct ICPFunctor
{
    Matrix3x4f T;
    cv::cuda::PtrStep<Vector4f> vmap_src;
    cv::cuda::PtrStep<Vector4f> vmap_ref;
    cv::cuda::PtrStep<Vector4f> nmap_src;
    cv::cuda::PtrStep<Vector4f> nmap_ref;
    int cols, rows, Np;
    float fx, fy, cx, cy;
    float angle_thresh, dist_thresh;
    mutable cv::cuda::PtrStepSz<float> out;

    __device__ __forceinline__ bool search_corresp(
        const int &x,
        const int &y,
        Vector3f &vcurr_g,
        Vector3f &vlast_g,
        Vector3f &nlast_g) const
    {
        Vector3f vlast_c = ToVector3(vmap_ref.ptr(y)[x]);
        if (isnan(vlast_c.x))
            return false;

        vlast_g = T(vlast_c);

        const float invz = 1.0 / vlast_g.z;
        const int u = __float2int_rd(vlast_g.x * invz * fx + cx + 0.5);
        const int v = __float2int_rd(vlast_g.y * invz * fy + cy + 0.5);
        if (u < 0 || v < 0 || u >= cols || v >= rows)
            return false;

        vcurr_g = ToVector3(vmap_src.ptr(v)[u]);

        Vector3f nlast_c = ToVector3(nmap_ref.ptr(y)[x]);
        nlast_g = T.rotate(nlast_c);

        Vector3f ncurr_g = ToVector3(nmap_src.ptr(v)[u]);

        float dist = (vlast_g - vcurr_g).norm();
        float sine = ncurr_g.cross(nlast_g).norm();

        return (sine < angle_thresh &&
                dist <= dist_thresh &&
                !isnan(ncurr_g.x) &&
                !isnan(nlast_g.x));
    }

    __device__ __forceinline__ void compute_jacobian(
        const int &i,
        float *const sum) const
    {
        const int y = i / cols;
        const int x = i - y * cols;

        Vector3f v_src, v_ref, n_ref;
        bool found = search_corresp(x, y, v_src, v_ref, n_ref);
        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        if (found)
        {
            *(Vector3f *)&row[0] = n_ref;
            *(Vector3f *)&row[3] = v_ref.cross(n_ref);
            row[6] = n_ref * (v_src - v_ref);
        }

        int count = 0;
#pragma unroll
        for (int i = 0; i < 7; ++i)
        {
#pragma unroll
            for (int j = i; j < 7; ++j)
                sum[count++] = row[i] * row[j];
        }

        sum[count] = (float)found;
    }

    __device__ __forceinline__ void operator()() const
    {
        float sum[29];
        memset(&sum[0], 0, sizeof(float) * 29);

        float val[29];
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < Np; i += blockDim.x * gridDim.x)
        {
            compute_jacobian(i, val);
#pragma unroll
            for (int j = 0; j < 29; ++j)
                sum[j] += val[j];
        }

        block_reduce_sum<float, 29>(sum);

        if (threadIdx.x == 0)
        {
#pragma unroll
            for (int i = 0; i < 29; ++i)
                out.ptr(blockIdx.x)[i] = sum[i];
        }
    }
};

void compute_icp_step(
    const cv::cuda::GpuMat vmap_src,
    const cv::cuda::GpuMat nmap_src,
    const cv::cuda::GpuMat vmap_ref,
    const cv::cuda::GpuMat nmap_ref,
    cv::cuda::GpuMat sum,
    cv::cuda::GpuMat out,
    const Eigen::Affine3d &T,
    const Eigen::Matrix3f &K,
    const float dist_thresh,
    const float angle_thresh,
    float *const H,
    float *const b,
    float &residual_sum,
    size_t &num_residual,
    const bool enable_debug_info)
{
    const int cols = vmap_src.cols;
    const int rows = vmap_src.rows;

    ICPFunctor functor;

    functor.out = sum;
    functor.vmap_src = vmap_src;
    functor.nmap_src = nmap_src;
    functor.vmap_ref = vmap_ref;
    functor.nmap_ref = nmap_ref;
    functor.cols = cols;
    functor.rows = rows;
    functor.Np = cols * rows;
    functor.T = T.matrix();
    functor.dist_thresh = dist_thresh;
    functor.angle_thresh = angle_thresh;
    functor.fx = K(0, 0);
    functor.fy = K(1, 1);
    functor.cx = K(0, 2);
    functor.cy = K(1, 2);

    call_device_functor<<<96, 224>>>(functor);

    if (enable_debug_info)
        cuda_check_error("icp_reduce_sum");

    cv::cuda::reduce(sum, out, 0, cv::REDUCE_SUM);

    cv::Mat host_data;
    out.download(host_data);
    create_matrix<6, 7>(host_data, H, b);

    residual_sum = sqrt(host_data.ptr<float>()[27]);
    num_residual = (size_t)host_data.ptr<float>()[28];
}

/*
=====================================
RGB STEP
=====================================
*/

// __device__ __forceinline__ float interpolate(
//     const cv::cuda::PtrStep<float> &map,
//     const float &x, const float &y)
// {
//     int u = static_cast<int>(std::floor(x));
//     int v = static_cast<int>(std::floor(y));
//     float cox = x - u, coy = y - v;
//     return (map.ptr(v)[u] * (1 - cox) + map.ptr(v)[u + 1] * cox) * (1 - coy) +
//            (map.ptr(v + 1)[u] * (1 - cox) + map.ptr(v + 1)[u + 1] * cox) * coy;
// }

struct ComputeResidualFunctor
{
    int cols, rows, Np;
    Matrix3x3f KR;
    Vector3f Kt;
    float max_residual, max_dist;
    float min_gradient, max_gradient;
    cv::cuda::PtrStep<float> image_ref;
    cv::cuda::PtrStep<float> image_src;
    cv::cuda::PtrStep<float> gradient_x;
    cv::cuda::PtrStep<float> gradient_y;
    cv::cuda::PtrStep<float> depth_src;
    cv::cuda::PtrStep<Vector4f> vmap_ref;
    mutable cv::cuda::PtrStep<Vector4f> corr_map;
    mutable cv::cuda::PtrStep<float> out;

    __device__ __forceinline__ bool find_corresp(
        const int &x,
        const int &y,
        float &residual,
        Vector2f &gradient) const
    {
        Vector3f pt = ToVector3(vmap_ref.ptr(y)[x]);
        if (isnan(pt.x))
            return false;

        Vector3f proj = KR(pt) + Kt;
        const float u = proj.x / proj.z;
        const float v = proj.y / proj.z;
        if (u >= 2 && u < cols - 2 && v >= 2 && v < rows - 2)
        {
            const float val_ref = image_ref.ptr(y)[x];
            const float val_src = interpolate(image_src, u, v);
            gradient.x = interpolate(gradient_x, u, v);
            gradient.y = interpolate(gradient_y, u, v);
            const float z_src = depth_src.ptr((int)v)[(int)u];

            residual = val_src - val_ref;
            return (gradient.norm() > min_gradient &&
                    gradient.norm() < max_gradient &&
                    abs(residual) < max_residual &&
                    abs(z_src - pt.z) < max_dist);
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

struct RGBStepFunctor
{
    float fx, fy, sigma;
    int cols, rows, Np;
    Matrix3x4f T;
    cv::cuda::PtrStep<Vector4f> vmap_ref;
    cv::cuda::PtrStep<Vector4f> corr_map;
    mutable cv::cuda::PtrStep<float> out;

    __device__ __forceinline__ void compute_jacobian(
        const int &k,
        float *const val) const
    {
        const int y = k / cols;
        const int x = k - y * cols;

        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        const Vector4f &res = corr_map.ptr(y)[x];

        if (res.w > 0)
        {
            Vector3f left;
            Vector3f pt = T(ToVector3(vmap_ref.ptr(y)[x]));

            float w = sigma + abs(res.z);
            w = w > FLT_EPSILON ? 1.0f / w : 1.0f;

            float z_inv = 1.0 / pt.z;
            left.x = w * res.x * fx * z_inv;
            left.y = w * res.y * fy * z_inv;
            left.z = -(left.x * pt.x + left.y * pt.y) * z_inv;

            row[6] = -w * res.z;
            *(Vector3f *)&row[0] = left;
            *(Vector3f *)&row[3] = pt.cross(left);
        }

        int count = 0;
#pragma unroll
        for (int i = 0; i < 7; ++i)
        {
#pragma unroll
            for (int j = i; j < 7; ++j)
            {
                val[count++] = row[i] * row[j];
            }
        }

        val[count] = res.z;
    }

    __device__ __forceinline__ void operator()() const
    {
        float sum[29];
        memset(&sum[0], 0, sizeof(float) * 29);

        float val[29];
        for (int k = blockIdx.x * blockDim.x + threadIdx.x;
             k < Np; k += blockDim.x * gridDim.x)
        {
            compute_jacobian(k, val);
#pragma unroll
            for (int i = 0; i < 29; ++i)
                sum[i] += val[i];
        }

        block_reduce_sum<float, 29>(sum);

        if (threadIdx.x == 0)
#pragma unroll
            for (int i = 0; i < 29; ++i)
                out.ptr(blockIdx.x)[i] = sum[i];
    }
};

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
    const Eigen::Matrix3f &K,
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
    const int cols = vmap_ref.cols;
    const int rows = vmap_ref.rows;

    ComputeResidualFunctor res_functor;
    res_functor.cols = cols;
    res_functor.rows = rows;
    res_functor.Np = cols * rows;
    res_functor.vmap_ref = vmap_ref;
    res_functor.depth_src = depth_src;
    res_functor.image_src = image_src;
    res_functor.image_ref = image_ref;
    res_functor.gradient_x = gradient_x;
    res_functor.gradient_y = gradient_y;
    res_functor.KR = K * T.matrix().topLeftCorner(3, 3).cast<float>();
    res_functor.Kt = K * T.matrix().topRightCorner(3, 1).cast<float>();
    res_functor.max_residual = max_residual;
    res_functor.max_dist = max_dist;
    res_functor.min_gradient = min_gradient;
    res_functor.max_gradient = max_gradient;
    res_functor.out = sum_res;
    res_functor.corr_map = corr_map;

    call_device_functor<<<96, 224>>>(res_functor);

    if (enable_debug_info)
        cuda_check_error("rgb_compute_res");

    cv::cuda::reduce(sum_res, out_res, 0, cv::REDUCE_SUM);

    cv::Mat host_res(out_res);
    residual_sum = host_res.ptr<float>()[0];
    num_residual = (size_t)host_res.ptr<float>()[1];

    float sigma = sqrt(residual_sum / (num_residual + 1));

    RGBStepFunctor functor;
    functor.fx = K(0, 0);
    functor.fy = K(1, 1);
    functor.cols = cols;
    functor.rows = rows;
    functor.Np = cols * rows;
    functor.T = T.matrix();
    functor.vmap_ref = vmap_ref;
    functor.corr_map = corr_map;
    functor.out = sum;
    functor.sigma = sigma;

    call_device_functor<<<96, 224>>>(functor);

    if (enable_debug_info)
        cuda_check_error("rgb_reduce_sum");

    cv::cuda::reduce(sum, out, 0, cv::REDUCE_SUM);

    cv::Mat host_data(out);
    create_matrix<6, 7>(host_data, H, b);

    float res_sum = sqrt(host_data.ptr<float>()[27]);
    float num_res = (size_t)host_data.ptr<float>()[28];
}