#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

template <typename NumType>
__device__ __forceinline__ NumType interpolate(
    const cv::cuda::PtrStep<NumType> &map,
    const float &x, const float &y)
{
    int u = static_cast<int>(std::floor(x));
    int v = static_cast<int>(std::floor(y));
    float cox = x - u, coy = y - v;
    return (map.ptr(v)[u] * (1 - cox) + map.ptr(v)[u + 1] * cox) * (1 - coy) +
           (map.ptr(v + 1)[u] * (1 - cox) + map.ptr(v + 1)[u + 1] * cox) * coy;
}

#endif