#ifndef NOISE_CUH
#define NOISE_CUH
//==============================================================================================
// Adapted  from P. Shirley's 'the next week'
//==============================================================================================
#include "../vec3.cuh"
#include "../perlin.cuh"
#include "texture.cuh"

class noise_texture : public Texture {
    public:
        __device__ noise_texture(curandState* rand_state, int ca=0) : noise(perlin(rand_state)),  c(ca) {}
        __device__ noise_texture(float sc, curandState* rand_state, int ca=0) : noise(perlin(rand_state)), scale(sc), c(ca) {}

        __device__ virtual color value(float u, float v, const vec3& p) const override {
            switch (c)
            {
            case 0:
                return color(1,1,1)*0.5*(1 + sin(scale*p.z() + 10*noise.turb(p)));
                break;
            
            default:
            case 1:
                return color(1,0.8,0.2)*0.5*(1 + sin(scale*sqrt(p.y()*p.y()  + p.x()*p.x())+ 5*noise.turb(p, 4) ));
                break;
            }

        }

    public:
        perlin noise;
        float scale;
        int c;
};


#endif