#include <cuda_ros/debug_image.h>
#include <cuda_ros/vector_math.h>
#include <cuda_ros/device_util.h>
#include <cuda_ros/interpolation.h>

void imshow(const char *title, const cv::cuda::GpuMat in)
{
    cv::Mat img(in);
    cv::imshow(title, img);
};

__global__ void warp_image_kernel(
    const cv::cuda::PtrStepSz<Vector4f> vmap,
    const cv::cuda::PtrStep<float> src,
    cv::cuda::PtrStep<float> dst,
    const Matrix3x3f KR,
    const Vector3f Kt)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= vmap.cols || y >= vmap.rows)
        return;

    Vector3f pt = KR(ToVector3(vmap.ptr(y)[x])) + Kt;
    const float u = pt.x / pt.z;
    const float v = pt.y / pt.z;

    if (u > 0 && v > 0 && u < vmap.cols - 1 && v < vmap.rows - 1)
    {
        dst.ptr(y)[x] = interpolate(src, u, v);
    }
    else
    {
        dst.ptr(y)[x] = 0;
    }
}

void warp_image(
    const cv::cuda::GpuMat &vmap,
    const cv::cuda::GpuMat &src,
    cv::cuda::GpuMat &out_dst,
    const Eigen::Affine3d &T,
    const Eigen::Matrix3d &K)
{
    const int cols = vmap.cols;
    const int rows = vmap.rows;

    if (out_dst.empty())
        out_dst.create(src.size(), src.type());

    dim3 block(8, 8);
    dim3 grid(div_up(cols, block.x), div_up(rows, block.y));

    Matrix3x3f KR = K * T.matrix().topLeftCorner(3, 3);
    Vector3f Kt = K * T.matrix().topRightCorner(3, 1);

    warp_image_kernel<<<grid, block>>>(vmap, src, out_dst, KR, Kt);
}

__global__ void compute_photometric_residual_kernel(
    const cv::cuda::PtrStepSz<Vector4f> vmap_ref,
    const cv::cuda::PtrStep<float> image_src,
    const cv::cuda::PtrStep<float> image_ref,
    cv::cuda::PtrStep<float> out_residual,
    const Matrix3x4f T,
    const float fx, const float fy,
    const float cx, const float cy)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= vmap_ref.cols || y >= vmap_ref.rows)
        return;

    Vector3f pt = T(ToVector3(vmap_ref.ptr(y)[x]));
    const float u = fx * pt.x / pt.z + cx;
    const float v = fy * pt.y / pt.z + cy;

    if (u > 0 && v > 0 && u < vmap_ref.cols - 1 && v < vmap_ref.rows - 1)
        out_residual.ptr(y)[x] = interpolate(image_src, u, v) - image_ref.ptr(y)[x];
    else
        out_residual.ptr(y)[x] = std::nan("a");
}

void compute_photometric_residual(
    const cv::cuda::GpuMat vmap_ref,
    const cv::cuda::GpuMat image_src,
    const cv::cuda::GpuMat image_ref,
    const Eigen::Affine3d &T,
    const Eigen::Matrix3d &K,
    cv::cuda::GpuMat &out_residual)
{
    const int cols = vmap_ref.cols;
    const int rows = vmap_ref.rows;

    if (out_residual.empty())
        out_residual.create(image_src.size(), image_src.type());

    dim3 block(8, 8);
    dim3 grid(div_up(cols, block.x), div_up(rows, block.y));

    compute_photometric_residual_kernel<<<grid, block>>>(
        vmap_ref,
        image_src,
        image_ref,
        out_residual,
        T.matrix(),
        K(0, 0), K(1, 1),
        K(0, 2), K(1, 2));
}