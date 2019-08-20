#ifndef REDUCTION_H
#define REDUCTION_H

#include <cuda_runtime_api.h>

#define WARP_SIZE 32

template <typename T, int size>
__device__ __forceinline__ void warp_reduce_sum(T *val)
{
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
#pragma unroll
        for (int i = 0; i < size; ++i)
        {
            val[i] += __shfl_down_sync(0xffffffff, val[i], offset);
        }
    }
}

template <typename T, int size>
__device__ __forceinline__ void block_reduce_sum(T *val)
{
    static __shared__ T shared[32 * size];
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    warp_reduce_sum<T, size>(val);

    if (lane_id == 0)
        memcpy(&shared[warp_id * size], val, sizeof(T) * size);

    __syncthreads();

    if (threadIdx.x < blockDim.x / WARP_SIZE)
        memcpy(val, &shared[lane_id * size], sizeof(T) * size);
    else
        memset(val, 0, sizeof(T) * size);

    if (warp_id == 0)
        warp_reduce_sum<T, size>(val);
}

#endif