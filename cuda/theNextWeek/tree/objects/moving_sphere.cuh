//
// Created by smooth_operator on 5/25/22.
//

#ifndef CUDA_MOVING_SPHERE_CUH
#define CUDA_MOVING_SPHERE_CUH

#include "hittable.cuh"
#include "../vec3.cuh"

class moving_sphere : public hittable {
public:
  moving_sphere() {}
  __device__ moving_sphere(
      point3 cen0, point3 cen1, float _time0, float _time1, float r, material* m)
      : center0(cen0), center1(cen1), time0(_time0), time1(_time1), radius(r), mat_ptr(m)
  {};

  __device__ bool bounding_box(float time0, float time1, aabb& output_box) const override;

  __device__ bool hit(
      const ray& r, float t_min, float t_max, hit_record& rec) const override;

  __device__ point3 center(float time) const;

public:
  point3 center0, center1;
  float time0, time1;
  float radius;
  material* mat_ptr;
};

__device__ bool moving_sphere::bounding_box(float _time0, float _time1, aabb& output_box) const {
  aabb box0 = aabb(
      center(_time0) - vec3(radius, radius, radius),
      center(_time0) + vec3(radius, radius, radius));
  aabb box1 = aabb(
      center(_time1) - vec3(radius, radius, radius),
      center(_time1) + vec3(radius, radius, radius));
  output_box = surrounding_box(box0, box1);
  return true;
}

__device__ point3 moving_sphere::center(float time) const {
  return center0 + ((time - time0) / (time1 - time0))*(center1 - center0);
}

__device__ bool moving_sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
  vec3 oc = r.origin() - center(r.time());
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
  auto outward_normal = (rec.p - center(r.time())) / radius;
  rec.set_face_normal(r, outward_normal);
  rec.mat_ptr = mat_ptr;

  return true;
}

#endif //CUDA_MOVING_SPHERE_CUH
