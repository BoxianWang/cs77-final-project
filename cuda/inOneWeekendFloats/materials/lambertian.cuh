//
// Created by smooth_operator on 5/19/22.
//

#ifndef CUDA_LAMBERTIAN_CUH
#define CUDA_LAMBERTIAN_CUH
#include "material.cuh"
#include "../vec3.cuh"
#include "../randoms.cuh"

class lambertian : public material {
public:
  __device__ lambertian(const color& a) : albedo(a) {}

  __device__ bool scatter(
      const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state
  ) const override {
    auto scatter_direction = rec.normal + vec3_random(rand_state);

    if (scatter_direction.near_zero()) {
      scatter_direction = rec.normal;
    }

    scattered = ray(rec.p, scatter_direction);
    attenuation = albedo;
    return true;
  }

public:
  color albedo;
};

#endif //CUDA_LAMBERTIAN_CUH
