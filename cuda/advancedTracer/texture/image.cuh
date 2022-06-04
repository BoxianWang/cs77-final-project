#ifndef IMAGE_CUH
#define IMAGE_CUH
//==============================================================================================
// Adapted  from P. Shirley's 'the next week'
//==============================================================================================
#include "texture.cuh"
#include <cuda_runtime.h>

texture<unsigned char, 1, cudaReadModeElementType> tex;


class image_texture : public Texture {
    public:
        const static int bytes_per_pixel = 3;

        __device__ image_texture()
          :  width(0), height(0), bytes_per_scanline(0) {}

        __device__ image_texture(int _width, int _height)
            : width(_width), height(_height), bytes_per_scanline(bytes_per_pixel*_width)
        {}


        __device__ virtual color value(float u, float v, const vec3& p) const override {
            // If we have no texture tex, then return solid cyan as a debugging aid.

            // Clamp input texture coordinates to [0,1] x [1,0]
            u = __saturatef(u);
            v = 1.0 - __saturatef(v);  // Flip V to image coordinates

            auto i = __float2int_rd(u * width);
            auto j = __float2int_rd(v * height);

            // Clamp integer mapping, since actual coordinates should be less than 1.0
            if (i >= width)  i = width-1;
            if (j >= height) j = height-1;

            const auto color_scale = 1.0 / 255.0;
            auto pixel = j*bytes_per_scanline + i*bytes_per_pixel;
            auto r = color_scale*tex1Dfetch(tex, pixel), g = color_scale*tex1Dfetch(tex, pixel+1),
            b = color_scale*tex1Dfetch(tex, pixel+2);

            return color(r, g, b);
        }

    private:
        
        int width, height;
        int bytes_per_scanline;
};

#endif