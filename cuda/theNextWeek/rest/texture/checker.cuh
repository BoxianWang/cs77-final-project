#ifndef CHECKER_CUH
#define CHECKER_CUH
//==============================================================================================
//   Adapted by Boxian Wang, based on P. Shirley's 'The next week'
//==============================================================================================

#include "../vec3.cuh"
#include "texture.cuh"
#include "solid_color.cuh"




class checker_texture : public Texture {
    public:
        __device__ checker_texture() {}

        __device__ checker_texture(Texture *_even, Texture *_odd)
            : even(_even), odd(_odd) {}

        __device__ checker_texture(color c1, color c2)
            : even(new solid_color(c1)) , odd(new solid_color(c2)) {}

        __device__ virtual color value(float u, float v, const vec3& p) const override {
            auto sines = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());
            if (sines < 0)
                return odd->value(u, v, p);
            else
                return even->value(u, v, p);
        }

    public:
        Texture *odd;
        Texture *even;
};





#endif
