//==============================================================================================
// Adapted  from P. Shirley's 'the next week'
//==============================================================================================

#ifndef CUDA_HITTABLE_LIST_CUH
#define CUDA_HITTABLE_LIST_CUH

#include "hittable.cuh"
#include "aabb.cuh"
#include "bvh_node.cuh"

class hittable_list : public hittable {
  public:
    // hittable_list is intended to be called from the gpu in order to create a hittable object
    __device__ hittable_list(hittable** objs, int numObjects, curandState *rand_state,  float time0=0, float time1=0) {
      objects = objs;
      objectNumber = numObjects;
//      node = new bvh_node(objs, 0, numObjects, time0, time1, rand_state);
//      node->print(0);
    }

    // does a ray hit the object?
    __device__ bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec
    ) const override;

    __device__ bool bounding_box(
        float time0, float time1, aabb& output_box) const override;

    __device__ int getObjectNumber() override {
      return objectNumber;
    }

    // deletes the hittable list, deleting the trees
    __device__ ~hittable_list() {
      delete node;
    }

  public:
    // the actual objects
    hittable** objects;
    // the number of objects currently stored
    int objectNumber;
    // the underlying tree that backs the list
    bvh_node* node;
};

// redirect the hit fn to the backing tree
// __device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
//   return node->hit(r, t_min, t_max, rec);
// }

__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
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

__device__ bool hittable_list::bounding_box(float time0, float time1, aabb& output_box) const {
  if (objectNumber == 0) return false;

  aabb temp_box;
  bool first_box = true;

  for (int i = 0; i < objectNumber; i++) {
    if (!objects[i]->bounding_box(time0, time1, temp_box)) {
      return false;
    }
    // if it's the first box, just store that; otherwise take the union
    output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
    first_box = false;
  }

  return true;
}

#endif //CUDA_HITTABLE_LIST_CUH
