//==============================================================================================
// Adapted  from P. Shirley's 'the next week'
//==============================================================================================

#ifndef CUDA_METAL_CUH
#define CUDA_METAL_CUH

#include "../vec3.cuh"
#include "../randoms.cuh"

class metal : public material {
  public:
    __device__ metal(const color& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    __device__ bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state
    ) const override {
      vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
      scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(rand_state), r_in.time());
      attenuation = albedo;
      return (dot(scattered.direction(), rec.normal) > 0);
    }

  public:
    color albedo;
    float fuzz;
};


#endif //CUDA_METAL_CUH
