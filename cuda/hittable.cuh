//
// Created by smooth_operator on 5/17/22.
//

#ifndef CUDA_HITTABLE_CUH
#define CUDA_HITTABLE_CUH

#include "ray.cuh"

struct hit_record {
  point3 p;
  vec3 normal;
  double t;
  bool front_face;

  __device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
    front_face = dot(r.direction(), outward_normal) < 0;
    normal = front_face ? outward_normal :-outward_normal;
  }
};

class hittable {
  public:
    __device__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};


#endif //CUDA_HITTABLE_CUH
