//
// Created by smooth_operator on 5/17/22.
//

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

      // this gets us to one less than the next square above numObjects --> i.e, numObjects=48, 64-1
      int backingArraySize = 1;
      while (backingArraySize < numObjects) {
        backingArraySize *= 2;
      }
      backingArraySize--;

      treeBackingArray = new bvh_node[backingArraySize];

      // we store the top node in the middle of the tree backing array
      treeBackingArray[backingArraySize/2] = bvh_node(objs, 0, numObjects, time0, time1, rand_state, treeBackingArray, backingArraySize);
      // we store the pointer to the middle of the tree backing array
      node = treeBackingArray+(backingArraySize/2);
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
    bvh_node* treeBackingArray;
    // the actual objects
    hittable** objects;
    // the number of objects currently stored
    int objectNumber;
    // the underlying tree that backs the list
    bvh_node* node;
};

// redirect the hit fn to the backing tree
__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
  return node->hit(r, t_min, t_max, rec);
}

__device__ bool hittable_list::bounding_box(float time0, float time1, aabb& output_box) const {
  if (objectNumber == 0) return false;

  return node->bounding_box(time0, time1, output_box);
}

#endif //CUDA_HITTABLE_LIST_CUH
