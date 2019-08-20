#ifndef VOXEL_MAP_H
#define VOXEL_MAP_H

#include "lib_fusion/hash_entry.h"
#include "lib_fusion/voxel_types.h"

template <class VoxelType>
class VoxelMap
{
    VoxelType *voxel_list;
    HashEntry *hash_table;

public:
    VoxelMap();
    VoxelMap(const VoxelMap &);
    VoxelMap &operator=(VoxelMap);

    template <class T>
    friend void swap(VoxelMap<T> &, VoxelMap<T> &);

    void create();
    void release();
    void copy_to(VoxelMap<VoxelType> &);
    void write_to_distk();
    void read_from_disk();

    inline VoxelType *get_voxel_list() const;
    inline HashEntry *get_hash_table() const;
};

template <class T>
void swap(VoxelMap<T> &, VoxelMap<T> &)
{
}

template <class VoxelType>
inline VoxelType *VoxelMap<VoxelType>::get_voxel_list() const
{
    return voxel_list;
}

template <class VoxelType>
inline HashEntry *VoxelMap<VoxelType>::get_hash_table() const
{
    return hash_table;
}

#endif