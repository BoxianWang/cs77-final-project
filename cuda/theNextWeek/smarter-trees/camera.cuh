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
        point3 lookFrom,
        point3 lookAt,
        vec3   vup,
        float vFov, // vertical field of view
        float aspect_ratio,
        float aperture,
        float focus_dist,
        float _time0 = 0,
        float _time1 = 0
    ) {
      auto theta = degrees_to_radians(vFov);
      auto h = tan(theta/2.f);
      auto viewport_height = 2.0f * h;
      auto viewport_width = aspect_ratio * viewport_height;

      w = unit_vector(lookFrom - lookAt);
      u = unit_vector(cross(vup, w));
      v = cross(w, u);

      origin = lookFrom;
      horizontal = focus_dist * viewport_width * u;
      vertical = focus_dist * viewport_height * v;
      lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

      lens_radius = aperture / 2;
      time0 = _time0;
      time1 = _time1;
    }

    __device__ ray get_ray(curandState* rand_state, float u_r, float v_r) const {
      vec3 rd = lens_radius * random_in_unit_disk(rand_state);
      vec3 offset = u_r*rd.x()*vec3(1,0,0) + v_r*rd.y()*vec3(0,1,0);

      return ray(
          origin+offset,
          lower_left_corner + u_r*horizontal + v_r*vertical - origin - offset,
          time0 == time1 ? time0 : random_float(rand_state, time0, time1)
          );
    }

  private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
    float time0, time1;
};


#endif //CUDA_CAMERA_CUH
