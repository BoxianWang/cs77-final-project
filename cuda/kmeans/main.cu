//
// Test file for kmeans by SpencerWarezak -- 5/29/22
//

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#include <iostream>
#include "../objects/sphere.cuh"
#include "../objects/moving_sphere.cuh"
#include "../objects/hittable.cuh"
#include "../objects/hittable_list.cuh"
#include <stdexcept>
#include <limits>
#include <curand_kernel.h>
#include "../randoms.cuh"
#include "../materials/material.cuh"
#include "../materials/lambertian.cuh"
#include "../materials/metal.cuh"
#include "../materials/dielectric.cuh"
#include "kmeans.cuh"

int sphereNum = 0;

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


// initializes the rand state for memory
__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = int(j)*max_x + int(i);
  //Each thread gets same seed, a different sequence number, no offset
  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
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
            vec3 center2 = center + vec3(0.f, random_float(local_rand_state, 0.f, .5f), 0.f);
            d_list[sphereNum++] = new moving_sphere(center, center2, 0., 1., 0.2, sphere_material);
            //d_list[sphereNum++] = new sphere(center, 0.2, sphere_material);
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

// cleans up the world
__global__ void free_world(hittable **d_list, hittable **d_world) {
  int objectNumber = (*d_world)->getObjectNumber();
  for (int i = 0; i < objectNumber; i++) {
    delete *(d_list+i);
  }
  delete *d_world;
}

int main() 
{
    const auto aspect_ratio = 3.0 / 2.0;
    const int nx = 1200;
    const int ny = static_cast<int>(nx / aspect_ratio);
    std::cerr << nx << " by " << ny << " image\n";

    size_t size = 0;
    cudaThreadSetLimit(cudaLimitStackSize, 4096);
    cudaThreadGetLimit(&size, cudaLimitStackSize);
    std::cerr << "STACK SIZE: " << size << std::endl;
    // Camera

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate cuda rand state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    // one block is 64 threads
    int tx=8;
    int ty=8;

    // set up the random state
    dim3 blocks(nx/tx + 1, ny/ty + 1);
    // total threads are tx*ty
    dim3 threads(tx,ty);
    // initialize the rand state
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);

    // generate the world
    hittable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 488*sizeof(hittable *)));
    checkCudaErrors(cudaDeviceSynchronize());
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    checkCudaErrors(cudaDeviceSynchronize());
    create_world<<<1,1>>>(d_list, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "length is " << sphereNum << "\n";

    // kmeans testing
    std::cout << "Testing Kmeans Clustering\n";

    // clean up
    free_world<<<1,1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));

    return 0;
}