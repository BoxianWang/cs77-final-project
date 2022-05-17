//
// Created by smooth_operator on 5/17/22.
//

#ifndef CUDA_SPHERE_CUH
#define CUDA_SPHERE_CUH

#include "hittable.cuh"
#include "vec3.cuh"

class sphere : public hittable {
  public:
    __host__ __device__ sphere() {}
    __host__ __device__ sphere(point3 cen, double r) : center(cen), radius(r) {};

    __device__ virtual bool hit(
        const ray& r, double t_min, double t_max, hit_record& rec) const override;

  public:
    point3 center;
    double radius;
};

__device__ bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
  vec3 oc = r.origin() - center;
  auto a = r.direction().length_squared();
  auto half_b = dot(oc, r.direction());
  auto c = oc.length_squared() - radius*radius;

  auto discriminant = half_b*half_b - a*c;
  if (discriminant < 0) return false;
  auto sqrtd = sqrt(discriminant);

  // Find the nearest root that lies in the acceptable range.
  auto root = (-half_b - sqrtd) / a;
  if (root < t_min || t_max < root) {
    root = (-half_b + sqrtd) / a;
    if (root < t_min || t_max < root)
      return false;
  }

  rec.t = root;
  rec.p = r.at(rec.t);
  rec.normal = (rec.p - center) / radius;
  vec3 outward_normal = (rec.p - center) / radius;
  rec.set_face_normal(r, outward_normal);

  return true;
}


#endif //CUDA_SPHERE_CUH
