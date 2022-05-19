//
// Created by smooth_operator on 5/19/22.
//

#ifndef CUDA_MATERIAL_CUH
#define CUDA_MATERIAL_CUH

#include <curand_kernel.h>
#include "../hittable.cuh"

struct hit_record;

class material {
  public:
    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state
    ) const = 0;
};


#endif //CUDA_MATERIAL_CUH
