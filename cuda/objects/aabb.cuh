//
// Created by smooth_operator on 5/25/22.
//

#ifndef CUDA_AABB_CUH
#define CUDA_AABB_CUH

class aabb {
  public:
    __device__ aabb() {}
    __device__ aabb(const point3& a, const point3& b) { minimum = a; maximum = b;}

    __device__ point3 min() const {return minimum; }
    __device__ point3 max() const {return maximum; }

    // https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    __device__ bool hit(const ray& r, float t_min, float t_max) const {
      // this code runs the same as above, but without the for loop
      vec3 invD = vec3(1.0f / r.direction()[0], 1.0f / r.direction()[1], 1.0f / r.direction()[2]);
      vec3 t0s = (min() - r.origin()) * invD;
      vec3 t1s = (max() - r.origin()) * invD;

      vec3 tsmaller = vec3(fmin(t0s[0], t1s[0]),
                           fmin(t0s[1], t1s[1]),
                           fmin(t0s[2], t1s[2]));
      vec3 tbigger = vec3(fmax(t0s[0], t1s[0]),
                          fmax(t0s[1], t1s[1]),
                          fmax(t0s[2], t1s[2]));

      float tmin = fmax(t_min, fmax(tsmaller[0], fmax(tsmaller[1], tsmaller[2])));
      float tmax = fmin(t_max, fmin(tbigger[0], fmin(tbigger[1], tbigger[2])));

      return (tmin < tmax);
    }

    __device__ float volume() {
      vec3 diagonal = maximum - minimum;
      return fabsf(diagonal.x() * diagonal.y() * diagonal.z());
    }

  public:
    point3 minimum;
    point3 maximum;
};

__device__ aabb surrounding_box(aabb box0, aabb box1) {
  point3 small(fmin(box0.min().x(), box1.min().x()),
               fmin(box0.min().y(), box1.min().y()),
               fmin(box0.min().z(), box1.min().z()));

  point3 big(fmax(box0.max().x(), box1.max().x()),
             fmax(box0.max().y(), box1.max().y()),
             fmax(box0.max().z(), box1.max().z()));

  return aabb(small,big);
}

#endif //CUDA_AABB_CUH
