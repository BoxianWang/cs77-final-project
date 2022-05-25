//
// Created by smooth_operator on 5/17/22.
//

#ifndef CUDA_HITTABLE_CUH
#define CUDA_HITTABLE_CUH

#include "../ray.cuh"
#include "../materials/material.cuh"

class hittable {
  public:
    __device__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
    __device__ virtual int getObjectNumber() {
      return 0;
    }
};


#endif //CUDA_HITTABLE_CUH
