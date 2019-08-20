#include <cuda_ros/vector_math.h>

class VoxelW
{
    short weight;
    short sdf;

public:
    VoxelW();
    float get_sdf() const;
    void set_sdf(const float val);
    short get_weight() const;
    void set_weight(const short val);
};

class VoxelWRGB
{
    short weight;
    short sdf;
    Vector3c rgb;

public:
    VoxelWRGB();
    float get_sdf() const;
    void set_sdf(const float val);
    short get_weight() const;
    void set_weight(const short val);
    Vector3f get_rgb() const;
    void set_rgb(const Vector3f val);
};