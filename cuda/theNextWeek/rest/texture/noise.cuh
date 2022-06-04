#ifndef NOISE_CUH
#define NOISE_CUH
//==============================================================================================
// Adapted  from P. Shirley's 'the next week'
//==============================================================================================
#include "../vec3.cuh"
#include "../perlin.cuh"
#include "texture.cuh"
#include "fBm.cuh"

#define OCTAVES 4
#define PERLIN_1 0
#define PERLIN_2 1
#define PERLIN_3 2
#define PERLIN_4 3
#define PERLIN_5 4
#define PERLIN_6 5
#define PERLIN_7 6
#define PERLIN_8 7
#define PERLIN_9 8
#define PERLIN_10 9
#define PERLIN_11 10
#define PERLIN_12 11

class noise_texture : public Texture {
    public:
        __device__ noise_texture(curandState* rand_state) : noise(perlin(rand_state)) {}
        __device__ noise_texture(float sc, curandState* rand_state, int ca=PERLIN_1) : noise(perlin(rand_state)), scale(sc), c(ca) {}

        __device__ virtual color value(float u, float v, const vec3& p) const override {
            color col;

            switch (c)
            {
                case PERLIN_1:
                    col = color(1,1,1)*0.5*(1 + noise.noise(scale * p));
                    break;

                case PERLIN_2:
                    col = color(1,1,1)*noise.turb(scale * p);
                    break;

                case PERLIN_3:
                    col = color(1,1,1)*0.5*(1 + sin(scale*p.z() + 10*noise.turb(p)));
                    break;

                case PERLIN_4:
                    col = color(1,1,1) * noise.noise(p);
                    break;
                
                case PERLIN_5:
                    col = color(1,1,1) * fabs(noise.noise(p));
                    break;

                case PERLIN_6:
                    col = color(1,1,1) * fabs(noise.noise(point3(4*p.x(), p.y(),p.z())));
                    break;

                case PERLIN_7:
                    col = color(1,1,1) * fabs(noise.noise(point3(p.x(),4*p.y(),p.z())));
                    break;

                case PERLIN_8:
                    col = color(1,1,1) * 0.5 * (1 + sin(scale * p.x() + noise.turb(scale * p)));
                    break;

                case PERLIN_9:
                    col =  color(1,1,1) * 0.5 * (1 + sin(scale * p.x()*p.y() + 3*noise.turb(3.1415 * p)));
                    break;

                case PERLIN_10:
                    col = color(1,1,1) * (1 + sin(sqrt(pow(p.x(), 2) + pow(p.y(), 2) + fBm.fBm_value(p, noise, OCTAVES, scale)))) / 2;
                    break;
                    
                case PERLIN_11:
                    col = color(1,0.8,0.2)*0.5*(1 + sin(scale*sqrt(p.z()*p.z()  + p.x()*p.x())+ 3*noise.turb(p, 4) ));
                    break;
                    
                case PERLIN_12:
                    col = color(1,1,1)*0.5*(1 + sin(scale*p.z() + 10*noise.turb(p)));
                    break;

            }
            
            return col;
            
        }

    public:
        perlin noise;
        float scale;
        fBm fBm;
        int c;
};


#endif
