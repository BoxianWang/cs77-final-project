#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#include <iostream>
#include "ray.cuh"
#include "objects/sphere.cuh"
#include "objects/moving_sphere.cuh"
#include "objects/hittable.cuh"
#include "objects/box.cuh"
#include "objects/hittable_list.cuh"
#include "objects/aarect.cuh"
#include <stdexcept>
#include <limits>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "camera.cuh"
#include "randoms.cuh"
#include "rtw_stb_image.cuh"
#include "texture/texture.cuh"
#include "texture/image.cuh"
#include "texture/checker.cuh"
#include "texture/noise.cuh"
#include "materials/material.cuh"
#include "materials/lambertian.cuh"
#include "materials/metal.cuh"
#include "materials/dielectric.cuh"
#include "materials/diffuse.cuh"
#include "objects/constant_medium.cuh"




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

__global__ void two_perlin_spheres(hittable **d_list, hittable **d_world, curandState *rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto pertext = new noise_texture(4, rand_state);
    auto mat = new lambertian(pertext);

    int sphereNum = 0;
    d_list[sphereNum++] = new sphere(point3(0,-1000,0), 1000, mat);
    d_list[sphereNum++] = new sphere(point3(0, 2, 0), 2, mat);

    *d_world = new hittable_list(d_list, sphereNum, rand_state);
  }
}

__global__ void earth(hittable **d_list, hittable **d_world, curandState *rand_state, int height, int width) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {


    // build sphere
    auto imtext = new image_texture(width, height);
    auto mat = new lambertian(imtext);

    int sphereNum = 0;
    d_list[sphereNum++] = new sphere(point3(0,0,0), 2, mat);

    *d_world = new hittable_list(d_list, sphereNum, rand_state);
  }
}

__global__ void cornell_box(hittable **d_list, hittable **d_world, curandState *rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {


    // build sphere

    int sphereNum = 0;
    auto red   = new  lambertian(color(.65, .05, .05));
    auto white = new  lambertian(color(.73, .73, .73));
    auto green = new  lambertian(color(.12, .45, .15));
    auto light = new diffuse_light(color(15, 15, 15));

    d_list[sphereNum++] = new yz_rect(0, 555, 0, 555, 555, green);
    d_list[sphereNum++] = new yz_rect(0, 555, 0, 555, 0, red);
    d_list[sphereNum++] = new xz_rect(213, 343, 227, 332, 554, light);
    d_list[sphereNum++] = new xz_rect(0, 555, 0, 555, 0, white);
    d_list[sphereNum++] = new xz_rect(0, 555, 0, 555, 555, white);
    d_list[sphereNum++] = new xy_rect(0, 555, 0, 555, 555, white);

    hittable **side1 = (hittable **)malloc(sizeof(hittable*)*6);
    hittable **side2 = (hittable **)malloc(sizeof(hittable*)*6);
    hittable *box1 = new box(point3(0, 0, 0), point3(165, 330, 165), white, side1, rand_state);
    hittable *box2 = new box(point3(0,0,0), point3(165,165,165), white, side2, rand_state);
    d_list[sphereNum++] = new translate(new rotate_y(box1, 15), vec3(265,0,295));
    d_list[sphereNum++] = new translate(new rotate_y(box2, -18), vec3(130,0,65));

    *d_world = new hittable_list(d_list, sphereNum, rand_state);
  }
}

__global__ void cornell_smoke(hittable **d_list, hittable **d_world, curandState *rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {


    // build sphere

    int sphereNum = 0;
    auto red   = new  lambertian(color(.65, .05, .05));
    auto white = new  lambertian(color(.73, .73, .73));
    auto green = new  lambertian(color(.12, .45, .15));
    auto light = new diffuse_light(color(7, 7, 7));

    d_list[sphereNum++] = new yz_rect(0, 555, 0, 555, 555, green);
    d_list[sphereNum++] = new yz_rect(0, 555, 0, 555, 0, red);
    d_list[sphereNum++] = new xz_rect(113, 443, 127, 432, 554, light);
    d_list[sphereNum++] = new xz_rect(0, 555, 0, 555, 0, white);
    d_list[sphereNum++] = new xz_rect(0, 555, 0, 555, 555, white);
    d_list[sphereNum++] = new xy_rect(0, 555, 0, 555, 555, white);

    hittable **side1 = (hittable **)malloc(sizeof(hittable*)*6);
    hittable **side2 = (hittable **)malloc(sizeof(hittable*)*6);
    hittable *box1 = new box(point3(0, 0, 0), point3(165, 330, 165), white, side1, rand_state);
    hittable *box2 = new box(point3(0,0,0), point3(165,165,165), white, side2, rand_state);
    box1 = new translate(new rotate_y(box1, 15), vec3(265,0,295));
    box2 = new translate(new rotate_y(box2, -18), vec3(130,0,65));
    d_list[sphereNum++] = new constant_medium(rand_state, box1, 0.01, color(0,0,0));
    d_list[sphereNum++] = new constant_medium(rand_state, box2, 0.01, color(1,1,1));

    *d_world = new hittable_list(d_list, sphereNum, rand_state);
  }
}

__global__ void simple_light(hittable **d_list, hittable **d_world, curandState *rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {


    // build sphere
    auto pertext = new noise_texture(4, rand_state);
    auto mat = new lambertian(pertext);

    int sphereNum = 0;
    d_list[sphereNum++] = new sphere(point3(0,-1000,0), 1000, mat);
    d_list[sphereNum++] = new sphere(point3(0, 2, 0), 2, mat);

    auto diff = new diffuse_light(color(4,4,4));
    // auto diff  = new metal(vec3(0.7, 0.6, 0.5), 0.);
    d_list[sphereNum++] = new xy_rect(3, 5, 1, 3, -2, diff);

    d_list[sphereNum++] = new sphere(point3(0, 7, 0), 2, diff);

    *d_world = new hittable_list(d_list, sphereNum, rand_state);
  }
}

// creates the world -- both the materials and the spheres
__global__ void random_world(hittable **d_list, hittable **d_world, curandState *rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState* local_rand_state = rand_state;
    auto checker = new checker_texture(color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));
    auto material_ground = new lambertian(checker);

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
            //d_list[sphereNum++] = new moving_sphere(center, center2, 0., 1., 0.2, sphere_material);
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
    vec3 origin, lookAt, vup;
    float vFov, aspectRatio, aperture, dist_to_focus;
    switch (4)
    {
    case 0:
    // perlin
      origin = point3(13, 2, 3);
      lookAt = point3(0, 0, 0);
      vup = point3(0, 1.0, 0);
      vFov = 20;
      aspectRatio = 3./2.;
      aperture = .0;
      dist_to_focus = 10;
      break;
    
    default:
    case 1:
    // random
      origin = point3(13, 2, 3);
      lookAt = point3(0, 0, 0);
      vup = point3(0, 1.0, 0);
      vFov = 20;
      aspectRatio = 3./2.;
      aperture = .1;
      dist_to_focus = 10;
      
      break;
    case 3:
      origin = point3(26,3,6);
      lookAt = point3(0,2,0);
      vup = point3(0, 1.0, 0);
      vFov = 20;
      aspectRatio = 3./2.;
      aperture = .0;
      dist_to_focus = 10;
    case 4:
      origin = point3(278, 278, -800);
      lookAt = point3(278, 278, 0);
      vup = point3(0, 1.0, 0);
      vFov = 40;
      aspectRatio = 1.0;
      dist_to_focus = 10;

    }
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
  curand_init(clock64(), pixel_index, 0, &rand_state[pixel_index]);
}

// colors the ray
__device__ vec3 ray_color(const ray& r, const color& background, hittable **world, curandState *local_rand_state) {
  ray cur_ray = r;
  vec3 cur_prod = vec3(1,1,1);
  vec3 cur_sum = vec3(0,0,0);
  int max_depth = 200;
  for(int i = 0; i < max_depth; i++) {
    hit_record rec;
    if ((*world)->hit(cur_ray, 0.001f, infinity, rec)) {
      ray scattered;
      color new_attenuation;
      color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
      if (rec.mat_ptr->scatter(cur_ray, rec, new_attenuation, scattered, local_rand_state)) {
        cur_sum = cur_sum + emitted * cur_prod;
        cur_prod = cur_prod * new_attenuation;
        cur_ray = scattered;
      } else return cur_sum + cur_prod * emitted;
    }
    else {
      // light stops bouncing
      return cur_sum + cur_prod * background;
    }
  }
  return cur_sum; // exceeded recursion
}



__host__ void prepare_texture(int *height, int *width, char * filename) {
    unsigned char *data;

    int components_per_pixel = 3;
    data = stbi_load(
            filename, width, height, &components_per_pixel, components_per_pixel);
    int length = *width * *height* 3;

    // build 1D texture 
    size_t offset = 0;
    tex.addressMode[0] = cudaAddressModeBorder;
    tex.addressMode[1] = cudaAddressModeBorder;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = false;

    unsigned int* ddata;

    checkCudaErrors(cudaMalloc((void**)&ddata, sizeof(unsigned char)*length));
    cudaMemcpy(ddata, data, sizeof(unsigned char)*length, cudaMemcpyHostToDevice);

    cudaBindTexture(&offset, tex, ddata, sizeof(unsigned char)*length);
    checkCudaErrors(cudaGetLastError());
}

// actually renders a pixel
__global__ void render(
    vec3 *fb, int max_x, int max_y, hittable** world, curandState *rand_state,
    camera** cam,
    int number_samples, color background
) {
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = int(j)*max_x + int(i);
  curandState local_rand_state = rand_state[pixel_index];

  fb[pixel_index] = vec3(0,0,0);

  // this loop handles super sampling for antialiasing
  for (int it = 0; it < number_samples; it++) {
    float u = (float(i) + curand_uniform(&local_rand_state)) / float(max_x);
    float v = (float(j) + curand_uniform(&local_rand_state)) / float(max_y);
    ray r = (*cam)->get_ray(&local_rand_state, u, v);
    fb[pixel_index] += ray_color(r, background, world, &local_rand_state);
  }

  fb[pixel_index] /= float(number_samples);

  // gamma correct
  fb[pixel_index] = vec3(__saturatef(sqrt(fb[pixel_index].x())),
                         __saturatef(sqrt(fb[pixel_index].y())),
                         __saturatef(sqrt(fb[pixel_index].z()))
                         );
}

int main() {
  // Image
  float aspect_ratio;
  switch (1)
  {
  case 0:
    aspect_ratio = 3.0/2.0;
    break;
  
  default:
  // cornell
  case 1:
    aspect_ratio = 1.0;
    break;
  }
  const int nx = 600;
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

  // bg color
  color background(0, 0, 0);
  int num_samples = 10;
  int width, height;  //img texture

  hittable **d_list;
  hittable **d_world;
  camera **d_cam;

  checkCudaErrors(cudaMalloc((void **)&d_list, 488*sizeof(hittable *)));
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
  checkCudaErrors(cudaDeviceSynchronize());


  // choose a scene
  switch (5)
  {
  case 0:
    two_perlin_spheres<<<1,1>>>(d_list, d_world, d_rand_state);
    background = color(0.70, 0.80, 1.00);
    break;
  
  default:
  case 1:
      // generate the world
    random_world<<<1,1>>>(d_list, d_world, d_rand_state);
    background = color(0.70, 0.80, 1.00);

    break;
  case 2:
      // read file
    prepare_texture(&height, &width, "../earthmap.jpg");

    earth<<<1,1>>>(d_list, d_world, d_rand_state, height, width);
    background = color(0.70, 0.80, 1.00);

    break;
  case 3:
    simple_light<<<1,1>>>(d_list, d_world, d_rand_state);
    background = color(0,0,0);
    //  background = color(0.70, 0.80, 1.00);
    num_samples = 400;
    break;
  
  case 4:
    cornell_box<<<1,1>>>(d_list, d_world, d_rand_state);
    background = color(0,0,0);
    num_samples = 200;
    break;
  case 5:
    cornell_smoke<<<1,1>>>(d_list, d_world, d_rand_state);
    background = color(0,0,0);
    num_samples = 200;
    break;
  }
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

      // generate the camera
    
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
                              num_samples, background      // number_samples
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
      float r = fb[pixel_index].x();
      float g = fb[pixel_index].y();
      float b = fb[pixel_index].z();
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