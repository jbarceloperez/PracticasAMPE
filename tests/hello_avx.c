#include <immintrin.h>
#include <stdio.h>

int main() {

    /* Initialize the two argument vectors */
    __m256 evens = _mm256_set_ps(8.0, 5.0, 3.0, 6.0, 1.0, 8.0, 2.0, 3.0);
    __m256 odds = _mm256_set_ps(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);

    float *e = (float *)&evens;
    float *o = (float *)&odds;
    printf("\t%f %f %f %f %f %f %f %f\n  -\t",
        e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7]);
    printf("%f %f %f %f %f %f %f %f\n\t________________________________________________________________________________\n\t",
        o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]);

    /* Compute the difference between the two vectors */
    __m256 result = _mm256_sub_ps(evens, odds);

    /* Display the elements of the result vector */
    float* f = (float*)&result;
    printf("%f %f %f %f %f %f %f %f\n",
        f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);

    return 0;
}