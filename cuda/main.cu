#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#include <iostream>
#include "ray.cuh"
#include "objects/sphere.cuh"
#include "objects/moving_sphere.cuh"
#include "objects/hittable.cuh"
#include "objects/hittable_list.cuh"
#include <stdexcept>
#include <limits>
#include <curand_kernel.h>
#include "camera.cuh"
#include "randoms.cuh"
#include "materials/material.cuh"
#include "materials/lambertian.cuh"
#include "materials/metal.cuh"
#include "materials/dielectric.cuh"

const float infinity = std::numeric_limits<float>::infinity();

// prints out any cuda errors that occur
void check_cuda(cudaError_t result, const char *const func, const char *const file, const int line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
              file << ":" << line << " '" << func << "'\n" <<
              cudaGetErrorString(result)
              << "\n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

// creates the world -- both the materials and the spheres
__global__ void create_world(hittable **d_list, hittable **d_world, curandState *rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState* local_rand_state = rand_state;
    auto material_ground = new lambertian(vec3(0.5, 0.5, 0.5));
    auto material_1   = new dielectric(1.5);
    auto material_2 = new lambertian(vec3(0.4, 0.2, 0.1));
    auto material_3  = new metal(vec3(0.7, 0.6, 0.5), 0.);

    int sphereNum = 0;
    d_list[sphereNum++] = new sphere(vec3(0,-1000,0), 1000, material_ground);
    d_list[sphereNum++] = new sphere(vec3(0, 1, 0), 1, material_1);
    d_list[sphereNum++] = new sphere(vec3(-4, 1, 0), 1, material_2);
    d_list[sphereNum++] = new sphere(vec3(4, 1, 0), 1, material_3);

    for (int a = -11; a < 11; a++) {
      for (int b = -11; b < 11; b++) {
        float choose_mat = random_float(local_rand_state);
        point3 center(float(a) + 0.f*random_float(local_rand_state), 0.2f, float(b) + 0.9f*random_float(local_rand_state));

        if ((center - point3(4, 0.2, 0)).length() > 0.9) {
          material* sphere_material;

          if (choose_mat < 0.8) {
            // diffuse
            auto albedo = vec3_random(local_rand_state) * vec3_random(local_rand_state);
            sphere_material = new lambertian(albedo);
//            vec3 center2 = center + vec3(0.f, random_float(local_rand_state, 0.f, .5f), 0.f);
//            d_list[sphereNum++] = new moving_sphere(center, center2, 0., 1., 0.2, sphere_material);
            d_list[sphereNum++] = new sphere(center, 0.2, sphere_material);
          } else if (choose_mat < 0.95) {
            // metal
            auto albedo = vec3_random(local_rand_state, 0.5, 1);
            auto fuzz = random_float(local_rand_state, 0, 0.5);
            sphere_material = new metal(albedo, fuzz);
            d_list[sphereNum++] = new sphere(center, 0.2, sphere_material);
          } else {
            // glass
            sphere_material = new dielectric(1.5);
            d_list[sphereNum++] = new sphere(center, 0.2, sphere_material);
          }
        }
      }
    }

    *d_world = new hittable_list(d_list, sphereNum, local_rand_state);
  }
}

__global__ void create_bigger_world(hittable **d_list, hittable **d_world, curandState *rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState* local_rand_state = rand_state;
    auto material_ground = new lambertian(vec3(0.5, 0.5, 0.5));
    auto material_1   = new dielectric(1.5);
    auto material_2 = new lambertian(vec3(0.4, 0.2, 0.1));
    auto material_3  = new metal(vec3(0.7, 0.6, 0.5), 0.);

    int sphereNum = 0;
    d_list[sphereNum++] = new sphere(vec3(0,-1000,0), 1000, material_ground);
    d_list[sphereNum++] = new sphere(vec3(0, 1, 0), 1, material_1);
    d_list[sphereNum++] = new sphere(vec3(-4, 1, 0), 1, material_2);
    d_list[sphereNum++] = new sphere(vec3(4, 1, 0), 1, material_3);

    for (int a = -22; a < 22; a++) {
      for (int b = -22; b < 22; b++) {
        float choose_mat = random_float(local_rand_state);
        point3 center(float(a)/1.5 + 0.f*random_float(local_rand_state), 0.2f, float(b)/1.5 + 0.45f*random_float(local_rand_state));

        if ((center - point3(4, 0.2, 0)).length() > 0.9) {
          material* sphere_material;

          if (choose_mat < 0.8) {
            // diffuse
            auto albedo = vec3_random(local_rand_state) * vec3_random(local_rand_state);
            sphere_material = new lambertian(albedo);
//            vec3 center2 = center + vec3(0.f, random_float(local_rand_state, 0.f, .5f), 0.f);
//            d_list[sphereNum++] = new moving_sphere(center, center2, 0., 1., 0.2, sphere_material);
            d_list[sphereNum++] = new sphere(center, 0.15, sphere_material);
          } else if (choose_mat < 0.95) {
            // metal
            auto albedo = vec3_random(local_rand_state, 0.5, 1);
            auto fuzz = random_float(local_rand_state, 0, 0.5);
            sphere_material = new metal(albedo, fuzz);
            d_list[sphereNum++] = new sphere(center, 0.15, sphere_material);
          } else {
            // glass
            sphere_material = new dielectric(1.5);
            d_list[sphereNum++] = new sphere(center, 0.15, sphere_material);
          }
        }
      }
    }

    *d_world = new hittable_list(d_list, sphereNum, local_rand_state);
  }
}

// cleans up the world
__global__ void free_world(hittable **d_list, hittable **d_world) {
  int objectNumber = (*d_world)->getObjectNumber();
  for (int i = 0; i < objectNumber; i++) {
    delete *(d_list+i);
  }
  delete *d_world;
}

// creates the camera
__global__ void create_camera(camera **d_cam) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto origin = point3(13, 2, 3);
    auto lookAt = point3(0, 0, 0);
    auto vup = point3(0, 1.0, 0);
    float vFov = 20;
    float aspectRatio = 3./2.;
    float aperture = .1;
    float dist_to_focus = 10;
    *d_cam = new camera(origin, lookAt, vup, vFov, aspectRatio, aperture, dist_to_focus, 0., 1.);
  }
}

// deletes the camera
__global__ void free_camera(camera **d_cam) {
  delete *(d_cam);
}

// initializes the rand state for memory
__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = int(j)*max_x + int(i);
  //Each thread gets same seed, a different sequence number, no offset
  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

// colors the ray
__device__ vec3 ray_color(const ray& r, hittable **world, curandState *local_rand_state) {
  ray cur_ray = r;
  vec3 cur_attenuation = vec3(1,1,1);
  for(int i = 0; i < 50; i++) {
    hit_record rec;
    if ((*world)->hit(cur_ray, 0.001f, infinity, rec)) {
      ray scattered;
      color new_attenuation;
      if (rec.mat_ptr->scatter(cur_ray, rec, new_attenuation, scattered, local_rand_state)) {
        cur_attenuation = new_attenuation * cur_attenuation;
        cur_ray = scattered;
      }
    }
    else {
      // light stops bouncing
      vec3 unit_direction = unit_vector(cur_ray.direction());
      float t = 0.5f*(unit_direction.y() + 1.0f);
      vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
      return cur_attenuation * c;
    }
  }
  return vec3(0.0,0.0,0.0); // exceeded recursion
}

// __global__ void superSample(vec3 *fb, unsigned int i, unsigned int j, 
//     int max_x, int max_y, int idx, hittable** world,
//     curandState local_rand_state, camera** cam, int number_samples)
// {
//   if (blockIdx.x >= number_samples) return;

//   float u = (float(i) + curand_uniform(&local_rand_state)) / float(max_x);
//   float v = (float(j) + curand_uniform(&local_rand_state)) / float(max_y);
//   ray r = (*cam)->get_ray(&local_rand_state, u, v);
//   fb[idx] = ray_color(r, world, &local_rand_state) / float(number_samples);
// }

// actually renders a pixel
__global__ void render(
    vec3 *fb, int max_x, int max_y, hittable** world, curandState *rand_state,
    camera** cam,
    int number_samples
) {
  unsigned int numX = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int i = numX / number_samples;
  unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = int(j)*max_x*number_samples + int(numX);
  curandState local_rand_state = rand_state[pixel_index];
  float u = (float(i) + curand_uniform(&local_rand_state)) / float(max_x);
  float v = (float(j) + curand_uniform(&local_rand_state)) / float(max_y);

  // get the ray and ray color
  ray r = (*cam)->get_ray(&local_rand_state, u, v);
  fb[pixel_index] = ray_color(r, world, &local_rand_state);

  // fb[pixel_index] = vec3(0,0,0);

  
  // this loop handles super sampling for antialiasing
  // for (int it = 0; it < number_samples; it++) {
  //   float u = (float(i) + curand_uniform(&local_rand_state)) / float(max_x);
  //   float v = (float(j) + curand_uniform(&local_rand_state)) / float(max_y);
  //   ray r = (*cam)->get_ray(&local_rand_state, u, v);
  //   fb[pixel_index] += ray_color(r, world, &local_rand_state) / float(number_samples);
  // }

  
  
  //superSample<<<1,number_samples>>>(fb, i, j, max_x, max_y, pixel_index, world, local_rand_state, cam, number_samples);
  //cudaDeviceSynchronize();

  // TODO: gamma correct at the end
  // fb[pixel_index] = vec3(sqrt(fb[pixel_index].x()),
  //                        sqrt(fb[pixel_index].y()),
  //                        sqrt(fb[pixel_index].z())
  //                        );
}

int main() {
  // Image
  const auto aspect_ratio = 3.0 / 2.0;
  const int nx = 1200;
  const int ny = static_cast<int>(nx / aspect_ratio);
  std::cerr << nx << " by " << ny << " image\n";

  size_t size = 0;
  // we need to increase the stack size, so we don't blow it while generating the world.
  cudaThreadSetLimit(cudaLimitStackSize, 4096);
  cudaThreadGetLimit(&size, cudaLimitStackSize);
  std::cerr << "STACK SIZE: " << size << std::endl;
  // Camera

  int num_pixels = nx*ny;
  int num_samples = 10;
  size_t fb_size = num_pixels*num_samples*sizeof(vec3);

  // allocate FB
  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  // allocate cuda rand state
  curandState *d_rand_state;
  checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*num_samples*sizeof(curandState)));

  // one block is 64 threads
  int tx=8;
  int ty=8;

  // set up the random state
  dim3 blocks(num_samples*nx/tx + 1, ny/ty + 1);
  // total threads are tx*ty
  dim3 threads(tx,ty);
  // initialize the rand state
  render_init<<<blocks, threads>>>(nx*num_samples, ny, d_rand_state);

  // generate the world
  hittable **d_list;
  hittable **d_world;
  bool biggerWorld = false;

  if (biggerWorld) {
    checkCudaErrors(cudaMalloc((void **)&d_list, 1956*sizeof(hittable *)));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    checkCudaErrors(cudaDeviceSynchronize());
    create_bigger_world<<<1,1>>>(d_list, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }

  else {
    checkCudaErrors(cudaMalloc((void **)&d_list, 488*sizeof(hittable *)));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    checkCudaErrors(cudaDeviceSynchronize());
    create_world<<<1,1>>>(d_list, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }

  // generate the camera
  camera **d_cam;
  checkCudaErrors(cudaMalloc((void **)&d_cam, sizeof(camera*)));
  create_camera<<<1,1>>>(d_cam);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // reset the stack size
  cudaThreadSetLimit(cudaLimitStackSize, 1024);

  std::cerr << "Rendering...";

  clock_t start, stop;
  start = clock();

  std::cerr << "\nInit fin...\n";
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  render<<<blocks, threads>>>(fb, nx, ny,
                              d_world,
                              d_rand_state,
                              d_cam,
                              num_samples       // number_samples
                              );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  stop = clock();
  float timer_seconds = ((float)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  // Output FB as Image
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny-1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_index = j*nx + i;
      float r = 0.f, g = 0.f, b = 0.f;

      for (int it = 0; it < num_samples; it++)
      {
        r += fb[pixel_index*num_samples + it].x();
        g += fb[pixel_index*num_samples + it].y();
        b += fb[pixel_index*num_samples + it].z();
      }

      // gamma correct
      r = sqrt(r / num_samples); g = sqrt(g / num_samples); b = sqrt(b / num_samples);

      // float r = fb[pixel_index].x();
      // float g = fb[pixel_index].y();
      // float b = fb[pixel_index].z();
      int ir = int(255.99*r);
      int ig = int(255.99*g);
      int ib = int(255.99*b);
      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }
  // clean up
  free_world<<<1,1>>>(d_list,d_world);
  free_camera<<<1,1>>>(d_cam);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_cam));
  checkCudaErrors(cudaFree(fb));

  return 0;
}
