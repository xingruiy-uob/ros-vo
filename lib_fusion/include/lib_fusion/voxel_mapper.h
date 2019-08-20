#ifndef VOXEL_MAPPER_H
#define VOXEL_MAPPER_H

#include <vector>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include "lib_fusion/voxel_map.h"

template <class VoxelType>
class VoxelMapper
{
    void update_voxel_map(
        const cv::cuda::GpuMat depth,
        const Eigen::Affine3d &T);

    std::vector<VoxelMap<VoxelType>> maps;
};

#endif