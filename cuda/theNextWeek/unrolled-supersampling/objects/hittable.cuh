//
// Created by smooth_operator on 5/17/22.
//

#ifndef CUDA_HITTABLE_CUH
#define CUDA_HITTABLE_CUH

#include "../ray.cuh"
#include "../materials/material.cuh"
#include "aabb.cuh"

class hittable {
  public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    __device__ virtual int getObjectNumber() {
      return 0;
    }
    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const = 0;
    __device__ virtual bool isNode() {
      return false;
    }

    __device__ virtual void print(int depth) {
      printf("0\n");
    }
};


#endif //CUDA_HITTABLE_CUH
