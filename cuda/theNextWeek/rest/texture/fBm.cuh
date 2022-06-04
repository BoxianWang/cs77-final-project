class fBm {
    public:
    __device__ static float fBm_value(const vec3& p, const perlin& noise, int octave, float scale)
    {
        float value = 0.f;

        for (int i = 0; i < octave; i++)
        {
            value += scale * noise.noise(pow(2,i) * p);
            scale *= pow(2, -i);
        }

        return value;
    }
};
