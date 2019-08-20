#ifndef VOXEL_MAP_H
#define VOXEL_MAP_H

template <class VoxelType>
class VoxelMap
{

public:
    VoxelMap();
    VoxelMap(const VoxelMap &);
    VoxelMap &operator=(VoxelMap);

    template <class T>
    friend void swap(VoxelMap<T> &, VoxelMap<T> &);
};

template <class T>
void swap(VoxelMap<T> &, VoxelMap<T> &)
{
}

#endif