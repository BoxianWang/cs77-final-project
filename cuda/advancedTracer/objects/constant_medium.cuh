#ifndef CONSTANT_MEDIUM_CUH
#define CONSTANT_MEDIUM_CUH
//==============================================================================================
// Adapted  from P. Shirley's 'the next week'
//==============================================================================================

#include "hittable.cuh"
#include "../materials/material.cuh"
#include "../texture/texture.cuh"
#include "../materials/iso.cuh"
#include "../vec3.cuh"
#include "../randoms.cuh"


class constant_medium : public hittable  {
    public:
        __device__ constant_medium(curandState *rand_state, hittable  * b, float d, Texture * a)
            : boundary(b),
              neg_inv_density(-1/d),
              phase_function(new isotropic(a)),
              rand(rand_state)
            {}

        __device__ constant_medium(curandState *rand_state, hittable  * b, float d, color c)
            : boundary(b),
              neg_inv_density(-1/d),
              phase_function(new isotropic(c)),
              rand(rand_state)
            {}

        __device__ virtual bool hit(
            const ray& r, float t_min, float t_max, hit_record& rec) const override;

        __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
            return boundary->bounding_box(time0, time1, output_box);
        }

    public:
        hittable  * boundary;
        material * phase_function;
        float neg_inv_density;
        curandState *rand;
};


__device__ bool constant_medium::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    // Print occasional samples when debugging. To enable, set enableDebug true.


    hit_record rec1, rec2;

    if (!boundary->hit(r, -infinity, infinity, rec1))
        return false;

    if (!boundary->hit(r, rec1.t+0.0001, infinity, rec2))
        return false;


    if (rec1.t < t_min) rec1.t = t_min;
    if (rec2.t > t_max) rec2.t = t_max;

    if (rec1.t >= rec2.t)
        return false;

    if (rec1.t < 0)
        rec1.t = 0;

    const auto ray_length = r.direction().length();
    const auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
    const auto hit_distance = neg_inv_density * logf(random_float(rand));

    if (hit_distance > distance_inside_boundary)
        return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.at(rec.t);

    rec.normal = vec3(1,0,0);  // arbitrary
    rec.front_face = true;     // also arbitrary
    rec.mat_ptr = phase_function;

    return true;
}

#endif
