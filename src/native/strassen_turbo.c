/*
 * Strassen TURBO - Ultra-Optimized Implementation
 * Author: grisun0
 * 
 * Features:
 * - AVX2 SIMD vectorization (256-bit)
 * - OpenMP multithreading
 * - Cache-friendly memory access
 * - Loop unrolling
 * 
 * Compile: gcc -O3 -ffast-math -march=native -fopenmp -mavx2 -shared -fPIC -o libstrassen_turbo.so strassen_turbo.c
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include <immintrin.h>

#define THRESHOLD 128
#define BLOCK_SIZE 32
#define ALIGN 32

/* Aligned allocation */
static inline float* alloc_matrix(int n) {
    return (float*)aligned_alloc(ALIGN, n * n * sizeof(float));
}

/* AVX2 vectorized matrix addition: C = A + B */
static void mat_add_avx(float* __restrict C, const float* __restrict A, 
                        const float* __restrict B, int n) {
    int nn = n * n;
    int i = 0;
    
    /* Process 8 floats at a time with AVX */
    for (; i <= nn - 8; i += 8) {
        __m256 va = _mm256_load_ps(&A[i]);
        __m256 vb = _mm256_load_ps(&B[i]);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_store_ps(&C[i], vc);
    }
    
    /* Handle remainder */
    for (; i < nn; i++) {
        C[i] = A[i] + B[i];
    }
}

/* AVX2 vectorized matrix subtraction: C = A - B */
static void mat_sub_avx(float* __restrict C, const float* __restrict A, 
                        const float* __restrict B, int n) {
    int nn = n * n;
    int i = 0;
    
    for (; i <= nn - 8; i += 8) {
        __m256 va = _mm256_load_ps(&A[i]);
        __m256 vb = _mm256_load_ps(&B[i]);
        __m256 vc = _mm256_sub_ps(va, vb);
        _mm256_store_ps(&C[i], vc);
    }
    
    for (; i < nn; i++) {
        C[i] = A[i] - B[i];
    }
}

/* Cache-blocked matrix multiplication with AVX2 */
static void matmul_blocked_avx(float* __restrict C, const float* __restrict A, 
                               const float* __restrict B, int n) {
    memset(C, 0, n * n * sizeof(float));
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
                int i_end = (ii + BLOCK_SIZE < n) ? ii + BLOCK_SIZE : n;
                int j_end = (jj + BLOCK_SIZE < n) ? jj + BLOCK_SIZE : n;
                int k_end = (kk + BLOCK_SIZE < n) ? kk + BLOCK_SIZE : n;
                
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        float a_ik = A[i * n + k];
                        __m256 va = _mm256_set1_ps(a_ik);
                        
                        int j = jj;
                        for (; j <= j_end - 8; j += 8) {
                            __m256 vb = _mm256_loadu_ps(&B[k * n + j]);
                            __m256 vc = _mm256_loadu_ps(&C[i * n + j]);
                            vc = _mm256_fmadd_ps(va, vb, vc);
                            _mm256_storeu_ps(&C[i * n + j], vc);
                        }
                        
                        for (; j < j_end; j++) {
                            C[i * n + j] += a_ik * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

/* Extract quadrant */
static void extract_quadrant(float* __restrict Q, const float* __restrict M, 
                            int n, int row, int col) {
    int h = n / 2;
    #pragma omp parallel for if(h > 64)
    for (int i = 0; i < h; i++) {
        memcpy(&Q[i * h], &M[(row + i) * n + col], h * sizeof(float));
    }
}

/* Insert quadrant */
static void insert_quadrant(float* __restrict M, const float* __restrict Q, 
                           int n, int row, int col) {
    int h = n / 2;
    #pragma omp parallel for if(h > 64)
    for (int i = 0; i < h; i++) {
        memcpy(&M[(row + i) * n + col], &Q[i * h], h * sizeof(float));
    }
}

/* Strassen recursive with parallelism */
void strassen_turbo_recursive(float* C, float* A, float* B, int n, int depth) {
    if (n <= THRESHOLD) {
        matmul_blocked_avx(C, A, B, n);
        return;
    }
    
    int h = n / 2;
    
    /* Allocate all matrices */
    float *A11 = alloc_matrix(h), *A12 = alloc_matrix(h);
    float *A21 = alloc_matrix(h), *A22 = alloc_matrix(h);
    float *B11 = alloc_matrix(h), *B12 = alloc_matrix(h);
    float *B21 = alloc_matrix(h), *B22 = alloc_matrix(h);
    
    float *M1 = alloc_matrix(h), *M2 = alloc_matrix(h);
    float *M3 = alloc_matrix(h), *M4 = alloc_matrix(h);
    float *M5 = alloc_matrix(h), *M6 = alloc_matrix(h);
    float *M7 = alloc_matrix(h);
    
    float *T1 = alloc_matrix(h), *T2 = alloc_matrix(h);
    float *C11 = alloc_matrix(h), *C12 = alloc_matrix(h);
    float *C21 = alloc_matrix(h), *C22 = alloc_matrix(h);
    
    /* Extract quadrants */
    extract_quadrant(A11, A, n, 0, 0);
    extract_quadrant(A12, A, n, 0, h);
    extract_quadrant(A21, A, n, h, 0);
    extract_quadrant(A22, A, n, h, h);
    
    extract_quadrant(B11, B, n, 0, 0);
    extract_quadrant(B12, B, n, 0, h);
    extract_quadrant(B21, B, n, h, 0);
    extract_quadrant(B22, B, n, h, h);
    
    /* Compute 7 products - parallelize at top level */
    if (depth == 0 && n >= 512) {
        float *T1_1 = alloc_matrix(h), *T2_1 = alloc_matrix(h);
        float *T1_2 = alloc_matrix(h), *T2_2 = alloc_matrix(h);
        float *T1_3 = alloc_matrix(h);
        float *T1_4 = alloc_matrix(h);
        float *T1_5 = alloc_matrix(h);
        float *T1_6 = alloc_matrix(h), *T2_6 = alloc_matrix(h);
        float *T1_7 = alloc_matrix(h), *T2_7 = alloc_matrix(h);
        
        /* Prepare all temporaries first */
        mat_add_avx(T1_1, A11, A22, h);  /* M1 */
        mat_add_avx(T2_1, B11, B22, h);
        mat_add_avx(T1_2, A21, A22, h);  /* M2 */
        mat_sub_avx(T1_3, B12, B22, h);  /* M3 */
        mat_sub_avx(T1_4, B21, B11, h);  /* M4 */
        mat_add_avx(T1_5, A11, A12, h);  /* M5 */
        mat_sub_avx(T1_6, A21, A11, h);  /* M6 */
        mat_add_avx(T2_6, B11, B12, h);
        mat_sub_avx(T1_7, A12, A22, h);  /* M7 */
        mat_add_avx(T2_7, B21, B22, h);
        
        #pragma omp parallel sections
        {
            #pragma omp section
            strassen_turbo_recursive(M1, T1_1, T2_1, h, depth + 1);
            
            #pragma omp section
            strassen_turbo_recursive(M2, T1_2, B11, h, depth + 1);
            
            #pragma omp section
            strassen_turbo_recursive(M3, A11, T1_3, h, depth + 1);
            
            #pragma omp section
            strassen_turbo_recursive(M4, A22, T1_4, h, depth + 1);
            
            #pragma omp section
            strassen_turbo_recursive(M5, T1_5, B22, h, depth + 1);
            
            #pragma omp section
            strassen_turbo_recursive(M6, T1_6, T2_6, h, depth + 1);
            
            #pragma omp section
            strassen_turbo_recursive(M7, T1_7, T2_7, h, depth + 1);
        }
        
        free(T1_1); free(T2_1); free(T1_2); free(T2_2);
        free(T1_3); free(T1_4); free(T1_5);
        free(T1_6); free(T2_6); free(T1_7); free(T2_7);
    } else {
        /* Sequential for smaller matrices */
        mat_add_avx(T1, A11, A22, h);
        mat_add_avx(T2, B11, B22, h);
        strassen_turbo_recursive(M1, T1, T2, h, depth + 1);
        
        mat_add_avx(T1, A21, A22, h);
        strassen_turbo_recursive(M2, T1, B11, h, depth + 1);
        
        mat_sub_avx(T1, B12, B22, h);
        strassen_turbo_recursive(M3, A11, T1, h, depth + 1);
        
        mat_sub_avx(T1, B21, B11, h);
        strassen_turbo_recursive(M4, A22, T1, h, depth + 1);
        
        mat_add_avx(T1, A11, A12, h);
        strassen_turbo_recursive(M5, T1, B22, h, depth + 1);
        
        mat_sub_avx(T1, A21, A11, h);
        mat_add_avx(T2, B11, B12, h);
        strassen_turbo_recursive(M6, T1, T2, h, depth + 1);
        
        mat_sub_avx(T1, A12, A22, h);
        mat_add_avx(T2, B21, B22, h);
        strassen_turbo_recursive(M7, T1, T2, h, depth + 1);
    }
    
    /* Compute C quadrants */
    mat_add_avx(T1, M1, M4, h);
    mat_sub_avx(T2, T1, M5, h);
    mat_add_avx(C11, T2, M7, h);
    
    mat_add_avx(C12, M3, M5, h);
    mat_add_avx(C21, M2, M4, h);
    
    mat_sub_avx(T1, M1, M2, h);
    mat_add_avx(T2, T1, M3, h);
    mat_add_avx(C22, T2, M6, h);
    
    /* Assemble result */
    insert_quadrant(C, C11, n, 0, 0);
    insert_quadrant(C, C12, n, 0, h);
    insert_quadrant(C, C21, n, h, 0);
    insert_quadrant(C, C22, n, h, h);
    
    /* Free memory */
    free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
    free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
    free(T1); free(T2);
    free(C11); free(C12); free(C21); free(C22);
}

/* Public API */
void strassen_turbo(float* C, float* A, float* B, int n) {
    omp_set_num_threads(omp_get_max_threads());
    strassen_turbo_recursive(C, A, B, n, 0);
}

/* Get number of threads */
int get_num_threads(void) {
    return omp_get_max_threads();
}
