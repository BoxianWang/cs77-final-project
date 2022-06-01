#ifndef SOLID_CUH
#define SOLID_CUH
//==============================================================================================
//   Adapted by Boxian Wang, based on P. Shirley's 'The next week'
//==============================================================================================

#include "../vec3.cuh"
#include "texture.cuh"


class solid_color : public Texture {
    public:
        __device__ solid_color() {}
        __device__ solid_color(color c) : color_value(c) {}

        __device__ solid_color(float red, float green, float blue)
          : solid_color(color(red,green,blue)) {}

        __device__ virtual color value(float u, float v, const vec3& p) const override {
            return color_value;
        }

    private:
        color color_value;
};







#endif
