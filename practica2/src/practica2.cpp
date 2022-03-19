#include "../include/practica2.h"
#include <iostream>
#include <immintrin.h>

#define VECTOR_SIZE 8

void practica2Linear(float* MK_matrix, float* KN_matrix, float* output_matrix, int M, int K, int N) {
    std::cout << "Running the code for optimized AVX matrix multiplication" << std::endl;
    __m256 v1, v2, v3, aux; // vectores para calcular la multiplicación de matrices

    for(int i=0; i<M; i++) {    // la última iteración se trata de forma diferente
        // v1 = _mm256_set_ps(MK_matrix[0*i],MK_matrix[1*i],MK_matrix[2*i],MK_matrix[3*i],MK_matrix[4*i],MK_matrix[5*i],MK_matrix[6*i],MK_matrix[7*i]);
        v1 = _mm256_load_ps(&MK_matrix[i*VECTOR_SIZE]);
        for(int j=0; j<N/VECTOR_SIZE; j++) {    // la última iteración se trata de forma diferente
            v3 = _mm256_setzero_ps();
	        // v2 = _mm256_set_ps(KN_matrix[0*j],KN_matrix[1*j],KN_matrix[2*j],KN_matrix[3*j],KN_matrix[4*j],KN_matrix[5*j],KN_matrix[6*j],KN_matrix[7*j]);
            v2 = _mm256_load_ps(&KN_matrix[j*VECTOR_SIZE]);
            for(int k=0; k<K; k++) {
                aux = _mm256_mul_ps(v1,v2);
                v3 = _mm256_add_ps(v3,aux);
	        }
        }
        if (K%VECTOR_SIZE == 0)   // se itera normal
        {
            v3 = _mm256_setzero_ps();
	        // v2 = _mm256_set_ps(KN_matrix[0*j],KN_matrix[1*j],KN_matrix[2*j],KN_matrix[3*j],KN_matrix[4*j],KN_matrix[5*j],KN_matrix[6*j],KN_matrix[7*j]);
            v2 = _mm256_load_ps(&KN_matrix[(N-1)*VECTOR_SIZE]);
            for(int k=0; k<K; k++) {
                aux = _mm256_mul_ps(v1,v2);
                v3 = _mm256_add_ps(v3,aux);
	        }
        }
        else 
        {
            int r = K%VECTOR_SIZE;
            float *resto = (float *) malloc(VECTOR_SIZE*sizeof(float));
            int j;
            for (j=0;j<VECTOR_SIZE; j++)
            {
                if
            }
            free(resto);
        }
    }
}

void practica2LinearSeq(float* MK_matrix, float* KN_matrix, float* output_matrix, int M, int K, int N) {
     std::cout << "Running the code for sequencial matrix multiplication" << std::endl;
     for(int i=0; i<M; i++) {
         for(int j=0; j<N; j++) {
             float suma=0.0;
             for(int k=0; k<K; k++) {
                 suma+=MK_matrix[i*K+k]*KN_matrix[j*K+k];
             }

             output_matrix[i*N+j]=suma;
         }
     }

}


