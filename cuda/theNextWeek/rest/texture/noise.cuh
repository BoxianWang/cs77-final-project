#ifndef NOISE_CUH
#define NOISE_CUH

#include "../vec3.cuh"
#include "../perlin.cuh"
#include "texture.cuh"

class noise_texture : public Texture {
    public:
        __device__ noise_texture(curandState* rand_state) : noise(perlin(rand_state)) {}
        __device__ noise_texture(float sc, curandState* rand_state) : noise(perlin(rand_state)), scale(sc) {}

        __device__ virtual color value(float u, float v, const vec3& p) const override {
            // return color(1,1,1)*0.5*(1 + noise.noise(scale * p));
            // return color(1,1,1)*noise.turb(scale * p);
            return color(1,1,1)*0.5*(1 + sin(scale*p.z() + 10*noise.turb(p)));
            // return color(1,1,1) * noise.noise(p);
        }

    public:
        perlin noise;
        float scale;
};

#endif