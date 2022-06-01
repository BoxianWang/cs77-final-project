//
// Created by SpencerWarezak on 5/27/22
//

#ifndef CUDA_KMEANS_CUH
#define CUDA_KMEANS_CUH

#include "../objects/hittable.cuh"
#include "../randoms.cuh"

__global__ __device__ float distance(hittable *x, hittable *centroid)
{
    point3 p_x = x->rec->p;
    point3 p_centroid = centroid->rec->p;

    return sqrt(dot(p_x, p_centroid));
}

__global__ __device__ int length(hittable **src_objects)
{
    int count = 0;
    hittable *curr = src_objects[0];

    while (curr != NULL)
        curr = src_objects[count++];

    return count;
}

// random split initialization
__global__ __device__ void init_buckets(hittable **src_objects, hittable ***buckets, int num_buckets curandState *rand_state)
{
    int counts[num_buckets];
    for (int i=0; i < num_buckets; i++)
        counts[i] = 0;

    int num_objects = length(src_objects);
    for (int i = 0; i < num_objects; i++)
    {
        int idx = random_int(rand_state, 0, num_buckets);
        if (buckets[idx] == NULL)
        {
            hittable** bucket;
            buckets[idx] = bucket;
        }

        hittable** bucket = buckets[idx];
        bucket[counts[idx]] = src_objects[i];
        (counts[idx])++;
    }
}

__global__ int get_closest(hittable *obj, hittable **centroids, int num_buckets)
{
    float closest = float(0x7f800000);
    int closest_idx = 0;
    for (int i = 0; i < num_buckets; i++)
    {
        float curr = distance(obj, centroids[i]);
        if (curr < closest)
        {
            closest = curr;
            closest_idx = i;
        }
    }

    return i;
}

__global__ hittable_list** k_cluster(hittable **src_objects, int num_buckets, int max_iters, curandState *rand_state)
{
    // get index
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;

    hittable **buckets[num_buckets];
    init_buckets(src_objects, buckets, num_buckets, rand_state);
    point3 centroids[num_buckets];

    int iters = 0;
    while(iters < max_iters)
    {
        // new buckets to fill
        hittable **new_buckets[num_buckets];

        // initialize centroids array
        for (int i = 0; i < num_buckets; i++)
        {
            vec3 sum = new vec3(0.,0.,0.);
            hittable** curr = buckets[i];
            for (int j = 0; j < length(curr); j++)
                sum += curr[i];
            
            centroids[i] = sum / length(curr);
        }

        print("Clustering on %d...\n", iters);
        for (int i = 0; i < length(src_objects); i++)
        {
            int new_idx = get_closest(src_objects[i], centroids, num_buckets);
            new_buckets[idx] = src_objects[i];
        }

        buckets = new_buckets;
        iters++;
    }

    printf("Clustering complete...\n");
    return buckets;
}

