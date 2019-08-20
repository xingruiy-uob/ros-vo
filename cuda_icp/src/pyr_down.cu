#include "pyr_down.h"
#include <cuda_ros/vector_math.h>
#include <cuda_ros/device_util.h>

__global__ void compute_vmap_kernel(
    const cv::cuda::PtrStepSz<float> depth,
    cv::cuda::PtrStep<Vector4f> vmap,
    const Matrix3x3f Kinv,
    const float depth_cutoff)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x > depth.cols - 1 || y > depth.rows - 1)
        return;

    const float z = depth.ptr(y)[x];
    if (z > 0.3f && z < depth_cutoff)
    {
        Vector3f pt = Kinv(Vector3f(x, y, 1)) * z;
        vmap.ptr(y)[x] = Vector4f(pt, 1.0f);
    }
    else
    {
        vmap.ptr(y)[x] = Vector4f(nan("x"));
    }
}

void compute_vmap(const cv::cuda::GpuMat depth, cv::cuda::GpuMat &vmap, const Eigen::Matrix3f &Kinv, const float depth_cutoff)
{
    if (vmap.empty())
        vmap.create(depth.size(), CV_32FC4);

    dim3 block(8, 8);
    dim3 grid(div_up(depth.cols, block.x), div_up(depth.rows, block.y));

    compute_vmap_kernel<<<grid, block>>>(depth, vmap, Kinv, depth_cutoff);
}

__global__ void compute_nmap_kernel(
    const cv::cuda::PtrStep<Vector4f> vmap,
    cv::cuda::PtrStepSz<Vector4f> nmap)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= nmap.cols - 1 || y >= nmap.rows - 1)
        return;

    int x10 = max(x - 1, 0);
    int x01 = min(x + 1, nmap.cols);
    int y10 = max(y - 1, 0);
    int y01 = min(y + 1, nmap.rows);

    Vector3f v00 = ToVector3(vmap.ptr(y)[x10]);
    Vector3f v01 = ToVector3(vmap.ptr(y)[x01]);
    Vector3f v10 = ToVector3(vmap.ptr(y10)[x]);
    Vector3f v11 = ToVector3(vmap.ptr(y01)[x]);

    nmap.ptr(y)[x] = Vector4f(normalised((v01 - v00).cross(v11 - v10)), 1.f);
}

void compute_nmap(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat &nmap)
{
    if (nmap.empty())
        nmap.create(vmap.size(), vmap.type());

    dim3 block(8, 8);
    dim3 grid(div_up(nmap.cols, block.x), div_up(nmap.rows, block.y));

    compute_nmap_kernel<<<grid, block>>>(vmap, nmap);
}

__global__ void compute_derivative_kernel(
    const cv::cuda::PtrStepSz<float> image,
    cv::cuda::PtrStep<float> gradient_x,
    cv::cuda::PtrStep<float> gradient_y)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= image.cols - 1 || y >= image.rows - 1)
        return;

    int x10 = max(x - 1, 0);
    int x01 = min(x + 1, image.cols);
    int y10 = max(y - 1, 0);
    int y01 = min(y + 1, image.rows);

    gradient_x.ptr(y)[x] = (image.ptr(y)[x01] - image.ptr(y)[x10]) * 0.5;
    gradient_y.ptr(y)[x] = (image.ptr(y01)[x] - image.ptr(y10)[x]) * 0.5;
}

void compute_derivative(const cv::cuda::GpuMat image, cv::cuda::GpuMat &dx, cv::cuda::GpuMat &dy)
{
    if (dx.empty())
        dx.create(image.size(), image.type());
    if (dy.empty())
        dy.create(image.size(), image.type());

    dim3 block(8, 8);
    dim3 grid(div_up(image.cols, block.x), div_up(image.rows, block.y));

    compute_derivative_kernel<<<grid, block>>>(image, dx, dy);
}

void pyr_down_depth(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst)
{
    cv::cuda::resize(src, dst, cv::Size(0, 0), 0.5, 0.5);
}

void pyr_down_image(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst)
{
    cv::cuda::pyrDown(src, dst);
}

void pyr_down_vmap(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst)
{
    cv::cuda::resize(src, dst, cv::Size(0, 0), 0.5, 0.5);
}

void compute_gradient_sobel(
    const cv::cuda::GpuMat intensity,
    cv::cuda::GpuMat &gradient_x,
    cv::cuda::GpuMat &gradient_y)
{
    auto filter_x = cv::cuda::createSobelFilter(CV_32FC1, CV_32FC1, 1, 0, 3, 1 / 8.0f);
    auto filter_y = cv::cuda::createSobelFilter(CV_32FC1, CV_32FC1, 0, 1, 3, 1 / 8.0f);

    filter_x->apply(intensity, gradient_x);
    filter_y->apply(intensity, gradient_y);
}