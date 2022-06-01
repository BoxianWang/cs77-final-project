#ifndef BOX_CUH
#define BOX_CUH
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "hittable.cuh"
#include "../ray.cuh"
#include "../vec3.cuh"
#include "../materials/material.cuh"

#include "aarect.cuh"
#include "hittable_list.cuh"
#include "../randoms.cuh"


class box : public hittable  {
    public:
        __device__ box() {}
        __device__ box(const point3& p0, const point3& p1, material * ptr, hittable** d_list, curandState *rand_state);

        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

        __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
            output_box = aabb(box_min, box_max);
            return true;
        }

    public:
        point3 box_min;
        point3 box_max;
        hittable_list *sides;
};

// d_list must be malloc'd w/ size 6
__device__ box::box(const point3& p0, const point3& p1, material * ptr, hittable** d_list, curandState *rand_state) {
    box_min = p0;
    box_max = p1;

    int sphereNum = 0;
    d_list[sphereNum++] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr);
    d_list[sphereNum++] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr);

    d_list[sphereNum++] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr);
    d_list[sphereNum++] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr);

    d_list[sphereNum++] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr);
    d_list[sphereNum++] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr);

    sides = new hittable_list(d_list, sphereNum, rand_state);
}

__device__ bool box::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    return sides->hit(r, t_min, t_max, rec);
}


#endif
