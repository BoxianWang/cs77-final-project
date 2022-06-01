//
// Boxian Wang
//

#ifndef DIFFUSE_CUH
#define DIFFUSE_CUH

#include "../vec3.cuh"
#include "../randoms.cuh"
#include "../texture/texture.cuh"
#include "../texture/solid_color.cuh"

class diffuse_light : public material {
  public:
        __device__ diffuse_light(Texture * a) : emit(a) {}
        __device__ diffuse_light(color c) : emit(new solid_color(c)) {}

        __device__ virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state
        ) const override {
            return false;
        }

        __device__ virtual color emitted(double u, double v, const point3& p) const override {
            return emit->value(u, v, p);
        }

    public:
        Texture * emit;
};


#endif //CUDA_METAL_CUH
