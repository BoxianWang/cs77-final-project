//
// Created by smooth_operator on 5/19/22.
//

#ifndef CUDA_DIELECTRIC_CUH
#define CUDA_DIELECTRIC_CUH

#include "material.cuh"
#include "../randoms.cuh"

__device__ vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
  auto cos_theta = fminf(dot(-uv, n), 1.0);
  vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
  vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
  vec3 refracted = r_out_perp + r_out_parallel;

  return refracted;
}

class dielectric : public material {
  public:
    __device__ dielectric(float index_of_refraction) : ir(index_of_refraction) {}

    __device__ bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state
    ) const override {
      attenuation = color(1.0, 1.0, 1.0);
      float refraction_ratio = rec.front_face ? (1.0f/ir) : ir;

            vec3 unit_direction = unit_vector(r_in.direction());
      float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0);
      float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);

      bool cannot_refract = refraction_ratio * sin_theta > 1.0;
      vec3 direction;

      if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(rand_state))
        direction = reflect(unit_direction, rec.normal);
      else
        direction = refract(unit_direction, rec.normal, refraction_ratio);

      scattered = ray(rec.p, direction);

      return true;
    }

  public:
    float ir; // Index of Refraction

  private:
    __device__ static float reflectance(float cosine, float ref_idx) {
      // Use Schlick's approximation for reflectance.
      auto r0 = (1-ref_idx) / (1+ref_idx);
      r0 = r0*r0;
      return r0 + (1-r0)*pow((1 - cosine),5);
    }
};


#endif //CUDA_DIELECTRIC_CUH
