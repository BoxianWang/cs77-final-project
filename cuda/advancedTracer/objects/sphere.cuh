//==============================================================================================
// Adapted  from P. Shirley's 'the next week'
//==============================================================================================

#ifndef CUDA_BUMPYSPHERE_CUH
#define CUDA_BUMPYSPHERE_CUH

#include "hittable.cuh"
#include "../vec3.cuh"
#include "../materials/material.cuh"

class sphere : public hittable {
  public:
    __host__ __device__ sphere(point3 cen, float r, material* mat) : center(cen), radius(r), mat_ptr(mat) {};

    __device__ bool bounding_box(float time0, float time1, aabb& output_box) const override;

    __device__ bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ void print(int depth) override {
        printf("{center: (%f,%f,%f), radius: %f}\n", center.x(), center.y(), center.z(), radius);
    }
  private:
    __device__ static void get_sphere_uv(const point3& p, float& u, float& v) {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>
        auto pi = 3.1415926;

        auto theta = acos(-p.y());
        auto phi = atan2(-p.z(), p.x()) + pi;

        u = phi / (2*pi);
        v = theta / pi;
    }

    public:
    point3 center;
    float radius;
    material* mat_ptr;
};

// bounding box is bounded by the most extreme corners
__device__ bool sphere::bounding_box(float time0, float time1, aabb& output_box) const {
  output_box = aabb(
      center - vec3(radius, radius, radius),
      center + vec3(radius, radius, radius));
  return true;
}

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
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
  rec.mat_ptr = mat_ptr;
  vec3 outward_normal = (rec.p - center) / radius;
  rec.set_face_normal(r, outward_normal);
  get_sphere_uv(outward_normal, rec.u, rec.v);
  return true;
}

#endif //CUDA_SPHERE_CUH
