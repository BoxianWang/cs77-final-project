#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#include <iostream>
#include "ray.cuh"
#include "objects/sphere.cuh"
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

const double infinity = std::numeric_limits<double>::infinity();

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
__global__ void create_world(hittable **d_list, hittable **d_world) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto material_ground = new lambertian(vec3(0.8, 0.8, 0.0));
    auto material_center = new lambertian(vec3(0.7, 0.3, 0.3));
    auto material_left   = new dielectric(1.5);
    auto material_right  = new metal(vec3(0.8, 0.6, 0.2), 1.);

    *(d_list) = new sphere(vec3(0,-100.5,-1), 100, material_ground);
    *(d_list+1) = new sphere(vec3(0,0,-1), 0.5, material_center);
    *(d_list+2) = new sphere(vec3(-1,0,-1), 0.5, material_left);
//    *(d_list+3) = new sphere(vec3(-1,0,-1), -0.4, material_left);
    *(d_list+3) = new sphere(vec3(1,0,-1), 0.5, material_right);
    *d_world    = new hittable_list(d_list,4);
  }
}

// cleans up the world
__global__ void free_world(hittable **d_list, hittable **d_world) {
  delete *(d_list);
  delete *(d_list+1);
  delete *(d_list+2);
  delete *(d_list+3);
//  delete *(d_list+4);
  delete *d_world;
}

// creates the camera
__global__ void create_camera(camera **d_cam) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto origin = point3(0, 0, 0);
    auto lookat = point3(0, 0, 2.0);
    auto vup = point3(0, 1.0, 0);
    float vfov = 100;
    float aspectRatio = 16./9.;
    *d_cam = new camera(origin, lookat, vup, vfov, aspectRatio);
  }
}

// deletes the camera
__global__ void free_camera(camera **d_cam) {
  delete *(d_cam);
}

// initializes the rand state for memory
__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j*max_x + i;
  //Each thread gets same seed, a different sequence number, no offset
  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

// colors the ray
__device__ vec3 ray_color(const ray& r, hittable **world, curandState *local_rand_state, int ival, int j, int it) {
  ray cur_ray = r;
  vec3 cur_attenuation = vec3(1,1,1);
  for(int i = 0; i < 50; i++) {
    hit_record rec;
    if ((*world)->hit(cur_ray, 0.001f, infinity, rec)) {
      if (ival == 90 && j == 128 && it == 0) {
        printf("Hit: %f, %f, %f\n", rec.p.x(), rec.p.y(), rec.p.z());
      }

      ray scattered;
      color new_attenuation;
      if (rec.mat_ptr->scatter(r, rec, new_attenuation, scattered, local_rand_state)) {
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

// actually renders a pixel
__global__ void render(
    vec3 *fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal,
    vec3 vertical, vec3 origin, hittable** world, curandState *rand_state,
    int number_samples
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j*max_x + i;
  curandState local_rand_state = rand_state[pixel_index];

  fb[pixel_index] = vec3(0,0,0);

  // this loop handles supersampling for antialiasing
  for (int it = 0; it < number_samples; it++) {
    float u = (float(i) + curand_uniform(&local_rand_state)) / float(max_x);
    float v = (float(j) + curand_uniform(&local_rand_state)) / float(max_y);
    ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    fb[pixel_index] += ray_color(r, world, &local_rand_state, i, j, it)/number_samples;
  }

  // gamma correct
  fb[pixel_index] = vec3(sqrt(fb[pixel_index].x()),
                         sqrt(fb[pixel_index].y()),
                         sqrt(fb[pixel_index].z())
                         );
}

int main() {
  // Image
  const auto aspect_ratio = 16.0 / 9.0;
  const int nx = 400;
  const int ny = static_cast<int>(nx / aspect_ratio);
  std::cerr << nx << " by " << ny << " image\n";

  // Camera
  auto viewport_height = 2.0;
  auto viewport_width = aspect_ratio * viewport_height;
  auto focal_length = 1.0;

  auto origin = point3(0, 0, 0);
  auto horizontal = vec3(viewport_width, 0, 0);
  auto vertical = vec3(0, viewport_height, 0);
  auto lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);

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

  // generate the world
  hittable **d_list;
  checkCudaErrors(cudaMalloc((void **)&d_list, 5*sizeof(hittable *)));
  hittable **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
  create_world<<<1,1>>>(d_list,d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // generate the camera
  camera **d_cam;
  checkCudaErrors(cudaMalloc((void **)&d_cam, sizeof(camera*)));
  create_camera<<<1,1>>>(d_cam);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  std::cerr << "Rendering...";

  clock_t start, stop;
  start = clock();

  // setup the render
  dim3 blocks(nx/tx + 1, ny/ty + 1);
  // total threads are tx*ty
  dim3 threads(tx,ty);
  // initialize the rand state
  render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
  std::cerr << "\nInit fin...\n";
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  render<<<blocks, threads>>>(fb, nx, ny,
                              lower_left_corner,
                              horizontal,
                              vertical,
                              origin,
                              d_world,
                              d_rand_state,
                              100       // number_samples
                              );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  // Output FB as Image
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny-1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_index = j*nx + i;
      float r = fb[pixel_index].x();
      float g = fb[pixel_index].y();
      float b = fb[pixel_index].z();
      int ir = int(255.99*r);
      int ig = int(255.99*g);
      int ib = int(255.99*b);
//      std::cout << ir << " " << ig << " " << ib << "\n";
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
