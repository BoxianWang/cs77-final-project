//==============================================================================================
// Adapted  from P. Shirley's 'the next week'
//==============================================================================================

#ifndef CUDA_LAMBERTIAN_CUH
#define CUDA_LAMBERTIAN_CUH
#include "material.cuh"
#include "../vec3.cuh"
#include "../randoms.cuh"
#include "../texture/texture.cuh"
#include "../texture/solid_color.cuh"

class lambertian : public material {
public:
  __device__ lambertian(const color& a) : albedo(new solid_color(a)) {}
  __device__ lambertian(Texture *a) : albedo(a) {}

  __device__ bool scatter(
      const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state
  ) const override {
    auto scatter_direction = rec.normal + 
    unit_vector(random_in_unit_sphere(rand_state));

    if (scatter_direction.near_zero()) {
      scatter_direction = rec.normal;
    }

    scattered = ray(rec.p, scatter_direction, r_in.time());
    attenuation = albedo->value(rec.u, rec.v, rec.p);
    return true;
  }

public:
  Texture *albedo;
};

#endif //CUDA_LAMBERTIAN_CUH
