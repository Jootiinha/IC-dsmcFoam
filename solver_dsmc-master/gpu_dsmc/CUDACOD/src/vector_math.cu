#include "vector_math.cuh"

__host__ __device__ double dot(double3 a, double3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ double3 unit(double3 vec) {
    return vec / norm(vec);
}

__host__ __device__ double norm(double3 vec) {
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}
