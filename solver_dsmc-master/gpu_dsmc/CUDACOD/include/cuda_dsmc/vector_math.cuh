#ifndef VECTOR_MATH_CUH_INCLUDED
#define VECTOR_MATH_CUH_INCLUDED 1

// #ifdef CUDA
#include <cuda_runtime.h>
// #endif

#include <math.h>

static __inline__ __host__ __device__ double3 operator*(double3 a, 
                                                        double b) {
    return make_double3(a.x*b, a.y*b, a.z*b);
}

static __inline__ __host__ __device__ double3 operator*(double a, 
                                                        double3 b) {
    return make_double3(a*b.x, a*b.y, a*b.z);
}

static __inline__ __host__ __device__ double3 operator/(double3 a, 
                                                        double b) {
    return make_double3(a.x/b, a.y/b, a.z/b);
}

static __inline__ __host__ __device__ double3 operator+(double3 a, 
                                                        double3 b) {
    return make_double3(a.x+b.x, a.y+b.y, a.z+b.z);
}

static __inline__ __host__ __device__ double3 operator+(double a, 
                                                        double3 b) {
    return make_double3(a+b.x, a+b.y, a+b.z);
}

static __inline__ __host__ __device__ double3 operator+(double3 a, 
                                                        double b) {
    return make_double3(a.x+b, a.y+b, a.z+b);
}

__host__ __device__ double dot(double3 a, double3 b);

__host__ __device__ double3 unit(double3 vec);

__host__ __device__ double norm(double3 vec);

#endif  // VECTOR_MATH_CUH_INCLUDED