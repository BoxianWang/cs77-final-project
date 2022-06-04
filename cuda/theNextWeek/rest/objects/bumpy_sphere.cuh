//
// Created by smooth_operator on 5/17/22.
//

#ifndef CUDA_SPHERE_CUH
#define CUDA_SPHERE_CUH

#include "hittable.cuh"
#include "sphere.cuh"
#include "../randoms.cuh"
#include "../perlin.cuh"
#include "../vec3.cuh"
#include "../materials/material.cuh"

class bumpy_sphere : public sphere {

  public:
    __device__ bumpy_sphere(point3 cen, float r, material* mat, curandState* rand_state) : sphere(cen, r, mat), noise(perlin(rand_state)) {};

     __device__ bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;

  private:

    __device__ static void get_uv(point3 p, float &u, float &v, float r) {
        p = p / r;
        v = acos(p.z());
        u = atan2(p.y(), p.x());
    }

    __device__ static point3 get_p(float u, float v, float r) {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>
        return point3(r*cos(u)*sin(v), r*sin(u)*sin(v), r*cos(v));
    }

    __device__ static vec3 get_du(float u, float v, float r) {
        return vec3(-r*sin(u)*sin(v), r*cos(u)*sin(v), 0);
    }

    __device__ static vec3 get_dv(float u, float v, float r) {
        return vec3(r*cos(u)*cos(v), r*sin(u)*cos(v), -r*sin(v));
    }

    __device__ vec3 get_displaced_normal(point3 p, vec3 normal) const{
      auto eps = 0.0001;
      float u, v;
      get_uv(p, u, v, radius);
      auto bu = 0.1*radius*(noise.turb(10*get_p(u+eps, v, radius)) - noise.turb(10*get_p(u, v, radius)))/eps;
      auto bv = 0.1*radius*(noise.turb(10*get_p(u, v+eps, radius)) - noise.turb(10*get_p(u, v, radius)))/eps;

      return unit_vector(normal + bu*cross(normal, get_dv(u, v, radius)) - bv*cross(normal, get_du(u, v, radius)));

    }

  public:
      perlin noise;

};


__device__ bool bumpy_sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
  bool h = sphere::hit(r, t_min, t_max, rec);
  if (!h) return false;
  rec.normal = get_displaced_normal(rec.p, rec.normal);
  return true;
}

#endif //CUDA_SPHERE_CUH
