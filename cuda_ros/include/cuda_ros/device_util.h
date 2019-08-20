#ifndef DEVICE_UTIL_ROS
#define DEVICE_UTIL_ROS

#include <cuda_runtime_api.h>

template <typename NumType1, typename NumType2>
static inline int div_up(NumType1 dividend, NumType2 divisor)
{
    return (int)((dividend + divisor - 1) / divisor);
}

template <class T>
__global__ void call_device_functor(const T functor)
{
    functor();
}

void inline cuda_check_error(const char *func_name)
{
    cudaDeviceSynchronize();
    const auto result = cudaGetLastError();
    if (cudaSuccess != result)
    {
        std::cout << "something's wrong! the location is: "
                  << func_name
                  << " with reason: " << cudaGetErrorString(result)
                  << std::endl;
    }
}

#endif