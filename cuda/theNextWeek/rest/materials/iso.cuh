#ifndef ISO_CUH
#define ISO_CUH

#include "material.cuh"
#include "../randoms.cuh"
#include "../texture/solid_color.cuh"
#include "../texture/texture.cuh"


class isotropic : public material {
    public:
        __device__ isotropic(color c) : albedo(new solid_color(c)) {}
        __device__ isotropic(Texture  * a) : albedo(a) {}

        __device__ virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state
        ) const override {
            scattered = ray(rec.p, random_in_unit_sphere(rand_state), r_in.time());
            attenuation = albedo->value(rec.u, rec.v, rec.p);
            return true;
        }

    public:
        Texture  * albedo;
};

#endif