#ifndef TEXTURE_CUH
#define TEXTURE_CUH
//==============================================================================================
//   Adapted by Boxian Wang, based on P. Shirley's 'The next week'
//==============================================================================================

#include "../vec3.cuh"


class Texture  {
    public:
        __device__ virtual color value(float u, float v, const vec3& p) const = 0;
};



#endif
