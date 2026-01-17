/*
 * Strassen Matrix Multiplication - C Implementation
 * Author: grisun0
 * 
 * Compila: gcc -O3 -ffast-math -march=native -shared -fPIC -o libstrassen.so strassen_c.c
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define THRESHOLD 64

/* Allocate matrix */
static float* alloc_matrix(int n) {
    return (float*)aligned_alloc(32, n * n * sizeof(float));
}

/* Standard matrix multiplication for small matrices */
static void matmul_standard(float* C, float* A, float* B, int n) {
    memset(C, 0, n * n * sizeof(float));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float a_ik = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

/* Add matrices: C = A + B */
static void mat_add(float* C, float* A, float* B, int n) {
    int nn = n * n;
    for (int i = 0; i < nn; i++) {
        C[i] = A[i] + B[i];
    }
}

/* Subtract matrices: C = A - B */
static void mat_sub(float* C, float* A, float* B, int n) {
    int nn = n * n;
    for (int i = 0; i < nn; i++) {
        C[i] = A[i] - B[i];
    }
}

/* Extract quadrant from matrix */
static void extract_quadrant(float* Q, float* M, int n, int row, int col) {
    int h = n / 2;
    for (int i = 0; i < h; i++) {
        memcpy(&Q[i * h], &M[(row + i) * n + col], h * sizeof(float));
    }
}

/* Insert quadrant into matrix */
static void insert_quadrant(float* M, float* Q, int n, int row, int col) {
    int h = n / 2;
    for (int i = 0; i < h; i++) {
        memcpy(&M[(row + i) * n + col], &Q[i * h], h * sizeof(float));
    }
}

/* Strassen recursive */
void strassen_recursive(float* C, float* A, float* B, int n) {
    if (n <= THRESHOLD) {
        matmul_standard(C, A, B, n);
        return;
    }
    
    int h = n / 2;
    int hh = h * h;
    
    /* Allocate submatrices */
    float *A11 = alloc_matrix(h), *A12 = alloc_matrix(h);
    float *A21 = alloc_matrix(h), *A22 = alloc_matrix(h);
    float *B11 = alloc_matrix(h), *B12 = alloc_matrix(h);
    float *B21 = alloc_matrix(h), *B22 = alloc_matrix(h);
    
    float *M1 = alloc_matrix(h), *M2 = alloc_matrix(h);
    float *M3 = alloc_matrix(h), *M4 = alloc_matrix(h);
    float *M5 = alloc_matrix(h), *M6 = alloc_matrix(h);
    float *M7 = alloc_matrix(h);
    
    float *T1 = alloc_matrix(h), *T2 = alloc_matrix(h);
    
    /* Extract quadrants */
    extract_quadrant(A11, A, n, 0, 0);
    extract_quadrant(A12, A, n, 0, h);
    extract_quadrant(A21, A, n, h, 0);
    extract_quadrant(A22, A, n, h, h);
    
    extract_quadrant(B11, B, n, 0, 0);
    extract_quadrant(B12, B, n, 0, h);
    extract_quadrant(B21, B, n, h, 0);
    extract_quadrant(B22, B, n, h, h);
    
    /* M1 = (A11 + A22)(B11 + B22) */
    mat_add(T1, A11, A22, h);
    mat_add(T2, B11, B22, h);
    strassen_recursive(M1, T1, T2, h);
    
    /* M2 = (A21 + A22) * B11 */
    mat_add(T1, A21, A22, h);
    strassen_recursive(M2, T1, B11, h);
    
    /* M3 = A11 * (B12 - B22) */
    mat_sub(T1, B12, B22, h);
    strassen_recursive(M3, A11, T1, h);
    
    /* M4 = A22 * (B21 - B11) */
    mat_sub(T1, B21, B11, h);
    strassen_recursive(M4, A22, T1, h);
    
    /* M5 = (A11 + A12) * B22 */
    mat_add(T1, A11, A12, h);
    strassen_recursive(M5, T1, B22, h);
    
    /* M6 = (A21 - A11)(B11 + B12) */
    mat_sub(T1, A21, A11, h);
    mat_add(T2, B11, B12, h);
    strassen_recursive(M6, T1, T2, h);
    
    /* M7 = (A12 - A22)(B21 + B22) */
    mat_sub(T1, A12, A22, h);
    mat_add(T2, B21, B22, h);
    strassen_recursive(M7, T1, T2, h);
    
    /* C11 = M1 + M4 - M5 + M7 */
    float *C11 = alloc_matrix(h);
    mat_add(T1, M1, M4, h);
    mat_sub(T2, T1, M5, h);
    mat_add(C11, T2, M7, h);
    
    /* C12 = M3 + M5 */
    float *C12 = alloc_matrix(h);
    mat_add(C12, M3, M5, h);
    
    /* C21 = M2 + M4 */
    float *C21 = alloc_matrix(h);
    mat_add(C21, M2, M4, h);
    
    /* C22 = M1 - M2 + M3 + M6 */
    float *C22 = alloc_matrix(h);
    mat_sub(T1, M1, M2, h);
    mat_add(T2, T1, M3, h);
    mat_add(C22, T2, M6, h);
    
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
void strassen_multiply(float* C, float* A, float* B, int n) {
    strassen_recursive(C, A, B, n);
}

/* Standard multiply for comparison */
void standard_multiply(float* C, float* A, float* B, int n) {
    matmul_standard(C, A, B, n);
}
