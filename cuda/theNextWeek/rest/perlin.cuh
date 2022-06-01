#ifndef PERLIN_H
#define PERLIN_H
//==============================================================================================
// Adapted by Boxian Wang from P. Shirley's 'the next week'
//==============================================================================================

#include "vec3.cuh"
#include "randoms.cuh"



class perlin {
    public:
        __device__ perlin(curandState* rand_state) {
            ranvec = new vec3[point_count];
            for (int i = 0; i < point_count; ++i) {
                ranvec[i] = unit_vector(vec3_random(rand_state, -1,1));
            }
            // ranfloat = new float[point_count];
            // for (int i = 0; i < point_count; ++i) {
            //     ranfloat[i] = random_float(rand_state);
            // }

            perm_x = perlin_generate_perm(rand_state);
            perm_y = perlin_generate_perm(rand_state);
            perm_z = perlin_generate_perm(rand_state);
        }

        __device__ float noise(const point3& p) const {
            auto u = p.x() - floor(p.x());
            auto v = p.y() - floor(p.y());
            auto w = p.z() - floor(p.z());
            auto i = __float2int_rd(floor(p.x()));
            auto j = __float2int_rd(floor(p.y()));
            auto k = __float2int_rd(floor(p.z()));
            vec3 c[2][2][2];
            // float c[2][2][2];

            for (int di=0; di < 2; di++)
                for (int dj=0; dj < 2; dj++)
                    for (int dk=0; dk < 2; dk++)
                        c[di][dj][dk] = ranvec[
                            perm_x[(i+di) & 255] ^
                            perm_y[(j+dj) & 255] ^
                            perm_z[(k+dk) & 255]
                        ];
            // return trilinear_interp(c, u, v, w);

            return perlin_interp(c, u, v, w);
            // auto i = __float2int_rd(4*p.x()) & 255;
            // auto j = __float2int_rd(4*p.y()) & 255;
            // auto k = __float2int_rd(4*p.z()) & 255;

            // return ranfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
        }

        __device__ float turb(const point3& p, int depth=7) const {
            auto accum = 0.0;
            auto temp_p = p;
            auto weight = 1.0;

            for (int i = 0; i < depth; i++) {
                accum += weight * noise(temp_p);
                weight *= 0.5;
                temp_p *= 2;
            }

            return fabs(accum);
        }

    private:
        static const int point_count = 256;
        vec3* ranvec;
        float* ranfloat;
        int* perm_x;
        int* perm_y;
        int* perm_z;


        __device__ static int* perlin_generate_perm(curandState* rand_state) {
            auto p = new int[point_count];

            for (int i = 0; i < point_count; i++)
                p[i] = i;

            permute(rand_state, p, point_count);

            return p;
        }

        __device__ static void permute(curandState* rand_state, int* p, int n) {
            for (int i = n-1; i > 0; i--) {
                int target = random_int(rand_state, 0,i);
                int tmp = p[i];
                p[i] = p[target];
                p[target] = tmp;
            }
        }

        __device__ static float perlin_interp(vec3 c[2][2][2], float u, float v, float w) {
            auto uu = u*u*(3-2*u);
            auto vv = v*v*(3-2*v);
            auto ww = w*w*(3-2*w);
            auto accum = 0.0;

            for (int i=0; i < 2; i++)
                for (int j=0; j < 2; j++)
                    for (int k=0; k < 2; k++) {
                        vec3 weight_v(u-i, v-j, w-k);
                        accum += (i*uu + (1-i)*(1-uu))*
                            (j*vv + (1-j)*(1-vv))*
                            (k*ww + (1-k)*(1-ww))*dot(c[i][j][k], weight_v);
                    }

            return accum;
        }

        __device__ static float trilinear_interp(float c[2][2][2], float u, float v, float w) {
            auto accum = 0.0;
            for (int i=0; i < 2; i++)
                for (int j=0; j < 2; j++)
                    for (int k=0; k < 2; k++)
                        accum += (i*u + (1-i)*(1-u))*
                                (j*v + (1-j)*(1-v))*
                                (k*w + (1-k)*(1-w))*c[i][j][k];

            return accum;
        }
};


#endif
