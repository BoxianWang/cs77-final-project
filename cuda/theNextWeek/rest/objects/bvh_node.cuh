//==============================================================================================
// Adapted  from P. Shirley's 'the next week'
//==============================================================================================

#ifndef CUDA_BVH_NODE_CUH
#define CUDA_BVH_NODE_CUH

#include "hittable.cuh"
#include "../randoms.cuh"

class bvh_node : public hittable {
  public:
    __device__ bvh_node();

    __device__ bvh_node(
        hittable** src_objects,
        size_t start, size_t end, float time0, float time1, curandState *rand_state);

    __device__ virtual bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;

    __device__ int getObjectNumber() override {return int(size);}

    __device__ void print(int depth) override;

//    __device__ ~bvh_node() {
//      delete left;
//      delete right;
//    }

  public:
    hittable* left;
    hittable* right;
    unsigned long size;
    aabb box;
};

__device__ void bvh_node::print(int depth) {
  printf("{\n");
  for (int i = 0; i < depth+1; i++) {
    printf(" ");
  }
  printf("left: ");
  left->print(depth+1);

  for (int i = 0; i < depth+1; i++) {
    printf(" ");
  }
  printf("right: ");
  right->print(depth+1);

  for (int i = 0; i < depth; i++) {
    printf(" ");
  }
  printf("},\n");
}

__device__ bool bvh_node::bounding_box(float time0, float time1, aabb& output_box) const {
  output_box = box;
  return true;
}

__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
  // exit early if we miss the box
  if (!box.hit(r, t_min, t_max)) {
    return false;
  }

  bool hit_left = left->hit(r, t_min, t_max, rec);
  // right is constrained to impact closer than left
  bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

  return hit_left || hit_right;
}

__device__ inline bool box_compare(const hittable* a, const hittable* b, int axis) {
  aabb box_a;
  aabb box_b;

  bool aExists = a->bounding_box(0,0, box_a);
  bool bExists = b->bounding_box(0,0, box_b);
  if (!aExists) {
    box_a = box_b;
  }
  else if (!bExists) {
    box_b = box_a;
  }

  return box_a.min().e[axis] < box_b.min().e[axis];
}


__device__ bool box_x_compare (const hittable* a, const hittable* b) {
  return box_compare(a, b, 0);
}

__device__ bool box_y_compare (const hittable* a, const hittable* b) {
  return box_compare(a, b, 1);
}

__device__ bool box_z_compare (const hittable* a, const hittable* b) {
  return box_compare(a, b, 2);
}

__device__ void sort (hittable** objects, size_t start, size_t end, bool (*comparator)(const hittable* a, const hittable* b)) {
  int i, j;
  for (i = (int)start+1; i < (int)end; i++)
  {
    hittable* carried = objects[i];
    j = i - 1;

    // Move elements of arr[0..i-1],
    // that are greater than key, to one
    // position ahead of their
    // current position
    while (j >= start && comparator(carried, objects[j]))
    {
      objects[j + 1] = objects[j];
      j = j - 1;
    }
    objects[j + 1] = carried;
  }
}

// recursively subdivides the src_objects array
// between "start" and "end"
// start is inclusive, end is exclusive
__device__ bvh_node::bvh_node(
    hittable** objects,
    size_t start, size_t end, float time0, float time1, curandState *rand_state
) {
  // choose an axis to compare over randomly
  int axis = random_int(rand_state, 0,2);
  auto comparator = (axis == 0) ? box_x_compare
                                : (axis == 1) ? box_y_compare
                                              : box_z_compare;

  // span --> 1 more than sub because it's inclusive
  size_t object_span = end - start;

  if (object_span == 1) {
    left = right = objects[start];
  } else if (object_span == 2) {
    if (comparator(objects[start], objects[start+1])) {
      left = objects[start];
      right = objects[start+1];
    } else {
      left = objects[start+1];
      right = objects[start];
    }
  // more than 2 objects
  } else {
    // sort over the array
    sort(objects, start, end, comparator);
    // cut the array in half
    auto mid = start + object_span/2;
    left = new bvh_node(objects, start, mid, time0, time1, rand_state);
    right = new bvh_node(objects, mid, end, time0, time1, rand_state);
  }

  aabb box_left, box_right;
  bool isLeft = left->bounding_box (time0, time1, box_left);
  bool isRight = right->bounding_box(time0, time1, box_right);
  if (!isLeft) {
    box_left = box_right;
  }
  if (!isRight) {
    box_right = box_left;
  }
  box = surrounding_box(box_left, box_right);
}

#endif //CUDA_BVH_NODE_CUH
