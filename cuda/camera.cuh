//
// Created by smooth_operator on 5/17/22.
//

#ifndef CUDA_CAMERA_CUH
#define CUDA_CAMERA_CUH

#include <cmath>
#include "randoms.cuh"

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
        double aspect_ratio,
        double aperture,
        double focus_dist
    ) {
      auto theta = degrees_to_radians(vfov);
      auto h = tan(theta/2);
      auto viewport_height = 2.0 * h;
      auto viewport_width = aspect_ratio * viewport_height;

      w = unit_vector(lookfrom - lookat);
      u = unit_vector(cross(vup, w));
      v = cross(w, u);

      origin = lookfrom;
      horizontal = focus_dist * viewport_width * u;
      vertical = focus_dist * viewport_height * v;
      lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

      lens_radius = aperture / 2;
    }

    __device__ ray get_ray(curandState* rand_state, double u, double v) const {
      vec3 rd = lens_radius * random_in_unit_disk(rand_state);
      vec3 offset = u*rd.x()*vec3(1,0,0) + v*rd.y()*vec3(0,1,0);

      return ray(origin+offset, lower_left_corner + u*horizontal + v*vertical - origin - offset);
    }

  private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    double lens_radius;
};


#endif //CUDA_CAMERA_CUH
