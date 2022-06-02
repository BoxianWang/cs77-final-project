//
// Created by smooth_operator on 5/25/22.
//

#ifndef CUDA_BVH_NODE_CUH
#define CUDA_BVH_NODE_CUH

#include "hittable.cuh"
#include "../randoms.cuh"

class bvh_node : public hittable {
  public:
    __device__ bvh_node();

    __device__ bvh_node(
        hittable** src_objects,
        size_t start, size_t end,
        float time0, float time1,
        curandState *rand_state,
        bvh_node* nodeParent,
        bool hasParent
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
    bvh_node* parent;
    bool hasParent = false;
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

// only recurses on the left
// avoids recursion to be friendlier to the stack and cache
// TODO -- not working yet -- use https://developer.nvidia.com/thrust and https://www.techiedelight.com/inorder-tree-traversal-iterative-recursive/
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
//  bvh_node* currNode = const_cast<bvh_node *>(this);
//  aabb currBox;
//  bool didHitAny = false;
//  int depth = 0;
//
//  while (true) {
//    currNode->bounding_box(t_min, t_max, currBox);
//
//    // exit early if we miss the box
//    if (!box.hit(r, t_min, t_max)) {
//      break;
//    }
//
////    printf("%d\n", depth);
//
//    // actually recurse left
//    bool didHitTemp = currNode->left->hit(r, t_min, t_max, rec);
//    didHitAny = didHitAny || didHitTemp;
//    // decrease t_max if needed
//    t_max = (didHitTemp && (rec.t < t_max)) ? rec.t : t_max;
//
//    // don't bother with the right if it's a duplicate
//    if (currNode->left != currNode->right) {
//      // if is a node, loop down a level
//      if (currNode->right->isNode()) {
//        depth++;
//        currNode = (bvh_node *)(currNode->right);
//        continue;
//      }
//      // if not, hit the object, return if we hit anything
//      else {
//        didHitTemp = currNode->right->hit(r, t_min, t_max, rec);
//        didHitAny = didHitAny || didHitTemp;
//        break;
//      }
//    }
//    else {
//      break;
//    }
//  }
//  return didHitAny;
//}
//  auto currNode = const_cast<bvh_node *>(this);
//  aabb currBox;
//  bool didHitAny = false;
//  bool didHitTemp;
//
//  while (true) {
//    currNode->bounding_box(t_min, t_max, currBox);
//    // go up a level if we miss the box -- if we can't, then end the program
//    // we don't "continue" because we already checked box intersection of the top
//    if (!box.hit(r, t_min, t_max)) {
//      if (currNode->hasParent) {
//        currNode = currNode->parent;
//      } else {
//        break;
//      }
//    }
//
//    // if the left is a node, recurse left
//    if (currNode->left->isNode()) {
//      currNode = (bvh_node*)currNode->left;
//      continue;
//    }
//    // otherwise, try to hit the object with the ray
//    else {
//      didHitTemp = currNode->left->hit(r, t_min, t_max, rec);
//      didHitAny = didHitAny || didHitTemp;
//      // decrease t_max if needed
//      t_max = (didHitTemp && (rec.t < t_max)) ? rec.t : t_max;
//    }
//
//    // if right and left pointers are the same, skip calculating duplicates!
//    if (currNode->right != currNode->left) {
//      // if the right is a node, recurse right
//      if (currNode->right->isNode()) {
//        currNode = (bvh_node*)currNode->right;
//        continue;
//      }
//        // otherwise, try to hit the object with the ray
//      else {
//        didHitTemp = currNode->right->hit(r, t_min, t_max, rec);
//        didHitAny = didHitAny || didHitTemp;
//        // decrease t_max if needed
//        t_max = (didHitTemp && (rec.t < t_max)) ? rec.t : t_max;
//      }
//    }
//
//    // go up a level if we still can
//    if (currNode->hasParent) {
//      currNode = currNode->parent;
//    } else {
//      break;
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
    curandState *rand_state,
    bvh_node* nodeParent,
    bool hasNodeParent
) {
  float bestVolumeSum;
  int bestAxis;

  size_t object_span = end - start;
  size = object_span;

  if (hasNodeParent) {
    parent = nodeParent;
    hasParent = hasNodeParent;
  }

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
      left = new bvh_node(objects, start, mid, time0, time1, rand_state, this, true);
    }

    if (end-mid == 1) {
      right = objects[mid];
    }
    else {
      right = new bvh_node(objects, mid, end, time0, time1, rand_state, this, true);
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
