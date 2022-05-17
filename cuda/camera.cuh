//
// Created by smooth_operator on 5/17/22.
//

#ifndef CUDA_CAMERA_CUH
#define CUDA_CAMERA_CUH

#include <cmath>

__device__ float degrees_to_radians(float degrees) {
  return degrees / 180 * float(M_PI);
}

class camera {
  public:

  __device__ camera(
        point3 lookfrom,
        point3 lookat,
        vec3   vup,
        float vfov, // vertical field of view
        double aspect_ratio
    ) {
      auto theta = degrees_to_radians(vfov);
      auto h = tan(theta/2);
      auto viewport_height = 2.0 * h;
      auto viewport_width = aspect_ratio * viewport_height;

      auto w = unit_vector(lookfrom - lookat);
      auto u = unit_vector(cross(vup, w));
      auto v = cross(w, u);

      origin = lookfrom;
      horizontal = viewport_width * u;
      vertical = viewport_height * v;
      lower_left_corner = origin - horizontal/2 - vertical/2 - w;
    }

    __device__ ray get_ray(double u, double v) const {
      return ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);
    }

  private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
};


#endif //CUDA_CAMERA_CUH
