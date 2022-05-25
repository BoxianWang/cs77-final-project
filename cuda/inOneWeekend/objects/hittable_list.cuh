//
// Created by smooth_operator on 5/17/22.
//

#ifndef CUDA_HITTABLE_LIST_CUH
#define CUDA_HITTABLE_LIST_CUH

#include "hittable.cuh"

class hittable_list : public hittable {
  public:
    hittable_list() {}

    // hittable_list is intended to be called from the gpu in order to create a hittable object
    __device__ hittable_list(hittable** objs, int numObjects) {
      objects = objs;
      objectNumber = numObjects;
      objectCapacity = numObjects;
    }
    // alternative if capacity > numObjects
    __device__ hittable_list(hittable** objs, int numObjects, int capacity) {
      objects = objs;
      objectNumber = numObjects;
      objectCapacity = capacity;
    }

    // does a ray hit the object?
    __device__ virtual bool hit(
        const ray& r, double t_min, double t_max, hit_record& rec
    ) const override;

    __device__ int getObjectNumber() override {
      return objectNumber;
    }

  public:
    // the actual objects
    hittable** objects;
    // the number of objects currently stored
    int objectNumber;
    // the capacity of objects we can store
    int objectCapacity;
};


__device__ bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
  hit_record temp_rec;
  bool hit_anything = false;
  auto closest_so_far = t_max;

  for (int i = 0; i < objectNumber; i++) {
    if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }

  return hit_anything;
}

#endif //CUDA_HITTABLE_LIST_CUH
