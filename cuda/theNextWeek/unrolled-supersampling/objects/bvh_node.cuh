//
// Created by smooth_operator on 5/25/22.
//

#ifndef CUDA_BVH_NODE_CUH
#define CUDA_BVH_NODE_CUH

#include "hittable.cuh"
#include "../randoms.cuh"

class bvh_node : public hittable {
  public:
    __device__ bvh_node() {
      left= nullptr;
      right= nullptr;
      size=0;
    }

    __device__ bvh_node(
        hittable** src_objects,
        size_t start, size_t end,
        float time0, float time1,
        curandState *rand_state
      );

    __device__ virtual bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;

    __device__ int getObjectNumber() override {return int(size);}

    __device__ void print(int depth) override;

    __device__ bool isNode() override {
      return true;
    };

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
  printf("volume: %f\n", box.volume());

  for (int i = 0; i < depth+1; i++) {
    printf(" ");
  }
  printf("number objects: %ld\n", size);

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

// Avoids actual recursion and manages an explicit cache instead
// Uses https://www.techiedelight.com/inorder-tree-traversal-iterative-recursive/
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
  // create stack of pointers to hittable

//  bvh_node* stack[64];
//  int stackLoc = -1;
//  bvh_node* currNode = (bvh_node *)(this);
//  bool didHitAny = false;
//  bool didHitTemp;
//
//  // if the current node is null and stack is empty, we are done
//  while (stackLoc >= 0 || currNode != nullptr) {
//    // if we don't hit the box, pop the stack
//    if (!box.hit(r, t_min, t_max)) {
//      // break if we get to the bottom of the stack
//      if (stackLoc < 0) {
//        break;
//      }
//
//      currNode = stack[stackLoc--];
//    }
//
//    // if is a node
//    if (currNode->left->isNode()) {
//      // push to the stack
//      stack[++stackLoc] = currNode;
//      currNode = (bvh_node*)currNode->left;
//    }
//    // hit it directly, then recurse right a level up
//    else {
//      didHitTemp = currNode->left->hit(r, t_min, t_max, rec);
//      // decrease t_max if needed
//      t_max = (didHitTemp && (rec.t < t_max)) ? rec.t : t_max;
//      didHitAny = didHitAny || didHitTemp;
//
//      // only break the loop once we've found a valid node
//      while (true) {
//        // if we've gotten to the bottom of the stack, then we break both loops
//        if (stackLoc < 0) {
//          currNode = nullptr;
//          break;
//        }
//        // pop the stack, recurse right if right is a node
//        currNode = stack[stackLoc--];
//        if (currNode->right->isNode()) {
//          currNode = (bvh_node*)currNode->right;
//          break;
//        }
//          // if right is not a node, hit it directly
//        else {
//          didHitTemp = currNode->right->hit(r, t_min, t_max, rec);
//          // decrease t_max if needed
//          t_max = (didHitTemp && (rec.t < t_max)) ? rec.t : t_max;
//          didHitAny = didHitAny || didHitTemp;
//          // loop back up, popping the stack again!
//        }
//      }
//    }
//  }
//
//  return didHitAny;
//}

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

__device__ float getBoundingVolume(hittable** objects, size_t start, size_t end) {
  aabb box;
  objects[start]->bounding_box(0,0, box);
  vec3 highestPoint = vec3(box.max().x(), box.max().y(), box.max().z());
  vec3 lowestPoint = vec3(box.min().x(), box.min().y(), box.min().z());

//  objects[start]->print(0);

  for (int i = start+1; i < end; i++) {
    objects[i]->bounding_box(0,0, box);

    highestPoint = vec3(
        (highestPoint.x() > box.max().x()) ? highestPoint.x() : box.max().x(),
        (highestPoint.y() > box.max().y()) ? highestPoint.y() : box.max().y(),
        (highestPoint.z() > box.max().z()) ? highestPoint.z() : box.max().z()
        );

    lowestPoint = vec3(
        (lowestPoint.x() < box.min().x()) ? lowestPoint.x() : box.min().x(),
        (lowestPoint.y() < box.min().y()) ? lowestPoint.y() : box.min().y(),
        (lowestPoint.z() < box.min().z()) ? lowestPoint.z() : box.min().z()
    );

//    objects[i]->print(0);
  }

  return (highestPoint.x()-lowestPoint.x()) * (highestPoint.y()-lowestPoint.y()) * (highestPoint.z()-lowestPoint.z());
}

// recursively subdivides the src_objects array
// between "start" and "end"
// start is inclusive, end is exclusive
__device__ bvh_node::bvh_node(
    hittable** objects,
    size_t start, size_t end,
    float time0, float time1,
    curandState *rand_state
) {
  float bestVolumeSum;
  int bestAxis;

  size_t object_span = end - start;
  size = object_span;

  // check over all axes
  for (int axis = 0; axis < 3; axis++ ) {
    auto comparator = (axis == 0) ? box_x_compare
                                  : (axis == 1) ? box_y_compare
                                                : box_z_compare;

    sort(objects, start, end, comparator);
    auto mid = start + object_span/2;
//    printf("Sorting...\n");
    float volumeSum = getBoundingVolume(objects, start, mid) + getBoundingVolume(objects, mid, end);
//    printf("Sorted...\n\n");

    if (axis == 0 || volumeSum < bestVolumeSum) {
      bestAxis = axis;
      bestVolumeSum = volumeSum;
    }
  }
  auto comparator = (bestAxis == 0) ? box_x_compare
                                : (bestAxis == 1) ? box_y_compare
                                              : box_z_compare;
  sort(objects, start, end, comparator);

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
    if (mid-start == 1) {
      left = objects[start];
    }
    else {
      // goes from index 0 to index size/2 exclusive (i.e., first size/2-1 elements)
      left = new bvh_node(
          objects, start, mid, time0, time1, rand_state
      );
    }

    if (end-mid == 1) {
      right = objects[mid];
    }
    else {
      // goes from index size/2+1 to the end (exclusive) (i.e., starts at size/2+2 element to the end)
      right = new bvh_node(
          objects, mid, end, time0, time1, rand_state);
    }
  }

  aabb box_left, box_right;
  // call bounding_box to retrieve box_left, box_right
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
