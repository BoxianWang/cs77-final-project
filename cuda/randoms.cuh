//
// Created by smooth_operator on 5/17/22.
//
#ifndef CUDA_RANDOMS_CUH
#define CUDA_RANDOMS_CUH

#include <curand_kernel.h>
#include "vec3.cuh"

__device__ inline float random_float(curandState* rand_state) {
  return curand_uniform(rand_state);
}

__device__ inline float random_float(curandState* rand_state, float min, float max) {
  return min + (max - min)* random_float(rand_state);
}

__device__ inline static vec3 vec3_random(curandState* rand_state) {
  return vec3(random_float(rand_state), random_float(rand_state), random_float(rand_state));
}

__device__ inline static vec3 vec3_random(curandState* rand_state, float min, float max) {
  return vec3(random_float(rand_state, min,max), random_float(rand_state, min,max), random_float(rand_state, min,max));
}

__device__ vec3 random_in_unit_sphere(curandState* rand_state) {
  while (true) {
    auto p = vec3_random(rand_state, -1,1);
    if (p.length_squared() >= 1) continue;
    return p;
  }
}

__device__ vec3 random_in_unit_disk(curandState* rand_state) {
  while (true) {
    auto p = vec3(random_float(rand_state, -1,1), random_float(rand_state, -1,1), 0);
    if (p.length_squared() >= 1) continue;
    return p;
  }
}

#endif //CUDA_RANDOMS_CUH
