#include <cuda_ros/vector_math.h>

class HashEntry
{
    Vector3i pos;
    int ptr;
    int offset;

public:
    HashEntry();
};