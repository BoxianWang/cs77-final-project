//
// Created by smooth_operator on 5/17/22.
//

#include "vec3.cuh"

#ifndef CUDA_RAY_CUH
#define CUDA_RAY_CUH


class ray {
  public:
    __device__ ray() {}
    __device__ ray(const point3& origin, const vec3& direction)
        : orig(origin), dir(direction)
    {}
    __device__ vec3 origin() const       { return orig; }
    __device__ vec3 direction() const    { return dir; }
    __device__ vec3 at(float t) const { return orig + t*dir; }

    vec3 orig;
    vec3 dir;
};

#endif //CUDA_RAY_CUH
