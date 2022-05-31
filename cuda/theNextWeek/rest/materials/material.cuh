//
// Created by smooth_operator on 5/19/22.
//

#ifndef CUDA_MATERIAL_CUH
#define CUDA_MATERIAL_CUH

#include <curand_kernel.h>
#include "../objects/hittable.cuh"

struct hit_record;

class material {
  public:
    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state
    ) const = 0;

     __device__  virtual color emitted(double u, double v, const point3& p) const {
      return color(0,0,0);
    }
};

struct hit_record {
  point3 p;
  vec3 normal;
  material* mat_ptr;
  float t;
  float u;
  float v;
  bool front_face;

  __device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
    front_face = dot(unit_vector(r.direction()), unit_vector(outward_normal)) < 0;
    normal = front_face ? outward_normal :-outward_normal;
  }
};


#endif //CUDA_MATERIAL_CUH
