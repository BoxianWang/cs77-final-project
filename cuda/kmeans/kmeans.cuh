//
// Created by SpencerWarezak on 5/27/22
//

#ifndef CUDA_KMEANS_CUH
#define CUDA_KMEANS_CUH

#include <iostream>
#include "../objects/hittable.cuh"
#include "../randoms.cuh"

__device__ float distance(point3 x, point3 centroid)
{
    float dx = x.x() - centroid.x();
    float dy = x.y() - centroid.y();
    float dz = x.z() - centroid.z();

    return sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
}

__device__ int get_closest(hittable *src_object, 
                            point3 *centroids,
                            point3 *new_sums,
                            int *counts,
                            int num_buckets) 
{
    const hittable *curr = src_object;
    float closest = INFINITY;
    int closest_idx = 0;
    point3 p = curr->rec().p();

    for (int i = 0; i < num_buckets; i++)
    {
        float d = distance(p, centroids[i]);
        if (d < closest)
        {
            closest = d;
            closest_idx = i;
        }
    }

    new_sums[closest_idx] += p;
    counts[closest_idx]++;
    return closest_idx;
}

__device__ void compute_means(point3 *centroids,
                                point3 *new_sums, 
                                int *counts,
                                int k)
{
    for (int i = 0; i < k; i++)
    {
        int count = counts[i] < 1 ? 1 : counts[k];
        centroids[i] = new_sums[i] / count;
    }
}

__device__ void init_buckets(hittable **src_objects, 
                                hittable ***buckets, 
                                int num_buckets, 
                                const int num_objects, 
                                curandState *rand_state)
{
    int *seen;
    cudaMalloc(&seen, sizeof(int) * num_objects);
    for (int i = 0; i < num_buckets; i++)
    {
        int count = 0;
        for (int j = 0; j < num_objects / num_buckets; j++)
        {
            int idx = random_int(rand_state, 0, num_objects);
            while (seen[idx])
            {
                idx = random_int(rand_state, 0, num_objects);
                seen[idx] = 1;
            }

            hittable *curr = src_objects[idx];
            if (count == 0)
            {
                hittable **bucket;
                cudaMalloc(&bucket, (num_objects / num_buckets) * sizeof(hittable*));
                buckets[i] = bucket;
            }

            buckets[i][count] = curr;
            count++;
        }
    }

    cudaFree(seen);
}

__device__ hittable*** k_cluser(hittable **src_objects, 
                                    const int num_buckets, 
                                    int num_objects, 
                                    int max_iters, 
                                    curandState *rand_state)
{
    hittable ***buckets;
    cudaMalloc(&buckets, sizeof(hittable**) * num_buckets);
    init_buckets(src_objects, buckets, num_buckets, num_objects, rand_state);

    point3 *centroids;
    cudaMalloc(&centroids, sizeof(point3) * num_buckets);

    int iters = 0;
    while (iters < max_iters)
    {
        // new buckets to fill and new sums to count
        hittable ***new_buckets;
        cudaMalloc(&new_buckets, sizeof(hittable**) * num_buckets);

        // initialize centroids array with a new sums array and a counts array
        point3 *new_sums;
        int *counts;
        cudaMalloc(&new_sums, sizeof(point3) * num_buckets);
        cudaMalloc(&counts, sizeof(int) * num_buckets);

        for (int i = 0; i < num_buckets; i++)
        {
            counts[i] = 0;
            new_sums[i] = new point3(0,0,0);
        }

        printf("Clustering %d buckets on %d points\n", num_buckets, num_objects);
        for (int i = 0; i < num_objects; i++)
        {
            int new_idx = get_closest(src_objects[i], centroids, new_sums, counts, num_buckets);

            new_buckets[new_idx][counts[new_idx]] = src_objects[i];
            counts[new_idx]++;
            new_sums[new_idx] += src_objects[i]->rec().p();
        }

        // compute the means for the centroids
        compute_means(centroids, new_sums, counts, num_buckets);

        cudaMemcpy(buckets, new_buckets, sizeof(hittable**) * num_buckets, cudaMemcpyHostToDevice);

        // freeeeee stuff
        cudaFree(new_sums);
        cudaFree(counts);
        cudaFree(new_buckets);

        // increment iterations
        iters++;
    }

    printf("Clustering complete...\n");
    return buckets;
}

#endif //CUDA_KMEANS_CUH

