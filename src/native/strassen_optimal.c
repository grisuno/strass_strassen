/*
 * Strassen OPTIMAL - Minimal overhead implementation
 * Author: grisun0
 * 
 * Uses in-place operations where possible and only applies Strassen
 * for very large matrices where the asymptotic advantage overcomes overhead.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cblas.h>

/* Only use Strassen for huge matrices where O(n^2.807) wins */
#define STRASSEN_THRESHOLD 4096

/* Strassen for matrices >= threshold */
static void strassen_level(float* C, float* A, float* B, int n, 
                           float* workspace, int ws_offset) {
    if (n < STRASSEN_THRESHOLD) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, 1.0f, A, n, B, n, 0.0f, C, n);
        return;
    }
    
    int h = n / 2;
    int hh = h * h;
    
    /* Use pre-allocated workspace */
    float* T1 = workspace + ws_offset;
    float* T2 = T1 + hh;
    float* M = T2 + hh;
    int next_offset = ws_offset + 3 * hh;
    
    /* Pointers to quadrants (no copy needed for input) */
    float* A11 = A;
    float* A12 = A + h;
    float* A21 = A + h * n;
    float* A22 = A + h * n + h;
    
    float* B11 = B;
    float* B12 = B + h;
    float* B21 = B + h * n;
    float* B22 = B + h * n + h;
    
    float* C11 = C;
    float* C12 = C + h;
    float* C21 = C + h * n;
    float* C22 = C + h * n + h;
    
    /* We need to extract quadrants for recursive calls */
    /* This is the overhead that makes Strassen slower for small n */
    float* qA11 = workspace + next_offset; next_offset += hh;
    float* qA12 = workspace + next_offset; next_offset += hh;
    float* qA21 = workspace + next_offset; next_offset += hh;
    float* qA22 = workspace + next_offset; next_offset += hh;
    float* qB11 = workspace + next_offset; next_offset += hh;
    float* qB12 = workspace + next_offset; next_offset += hh;
    float* qB21 = workspace + next_offset; next_offset += hh;
    float* qB22 = workspace + next_offset; next_offset += hh;
    
    /* Extract quadrants */
    for (int i = 0; i < h; i++) {
        memcpy(qA11 + i*h, A + i*n, h*sizeof(float));
        memcpy(qA12 + i*h, A + i*n + h, h*sizeof(float));
        memcpy(qA21 + i*h, A + (i+h)*n, h*sizeof(float));
        memcpy(qA22 + i*h, A + (i+h)*n + h, h*sizeof(float));
        memcpy(qB11 + i*h, B + i*n, h*sizeof(float));
        memcpy(qB12 + i*h, B + i*n + h, h*sizeof(float));
        memcpy(qB21 + i*h, B + (i+h)*n, h*sizeof(float));
        memcpy(qB22 + i*h, B + (i+h)*n + h, h*sizeof(float));
    }
    
    float* M1 = workspace + next_offset; next_offset += hh;
    float* M2 = workspace + next_offset; next_offset += hh;
    float* M3 = workspace + next_offset; next_offset += hh;
    float* M4 = workspace + next_offset; next_offset += hh;
    float* M5 = workspace + next_offset; next_offset += hh;
    float* M6 = workspace + next_offset; next_offset += hh;
    float* M7 = workspace + next_offset; next_offset += hh;
    float* TMP = workspace + next_offset; next_offset += hh;
    
    /* M1 = (A11 + A22)(B11 + B22) */
    for (int i = 0; i < hh; i++) T1[i] = qA11[i] + qA22[i];
    for (int i = 0; i < hh; i++) T2[i] = qB11[i] + qB22[i];
    strassen_level(M1, T1, T2, h, workspace, next_offset);
    
    /* M2 = (A21 + A22) * B11 */
    for (int i = 0; i < hh; i++) T1[i] = qA21[i] + qA22[i];
    strassen_level(M2, T1, qB11, h, workspace, next_offset);
    
    /* M3 = A11 * (B12 - B22) */
    for (int i = 0; i < hh; i++) T1[i] = qB12[i] - qB22[i];
    strassen_level(M3, qA11, T1, h, workspace, next_offset);
    
    /* M4 = A22 * (B21 - B11) */
    for (int i = 0; i < hh; i++) T1[i] = qB21[i] - qB11[i];
    strassen_level(M4, qA22, T1, h, workspace, next_offset);
    
    /* M5 = (A11 + A12) * B22 */
    for (int i = 0; i < hh; i++) T1[i] = qA11[i] + qA12[i];
    strassen_level(M5, T1, qB22, h, workspace, next_offset);
    
    /* M6 = (A21 - A11)(B11 + B12) */
    for (int i = 0; i < hh; i++) T1[i] = qA21[i] - qA11[i];
    for (int i = 0; i < hh; i++) T2[i] = qB11[i] + qB12[i];
    strassen_level(M6, T1, T2, h, workspace, next_offset);
    
    /* M7 = (A12 - A22)(B21 + B22) */
    for (int i = 0; i < hh; i++) T1[i] = qA12[i] - qA22[i];
    for (int i = 0; i < hh; i++) T2[i] = qB21[i] + qB22[i];
    strassen_level(M7, T1, T2, h, workspace, next_offset);
    
    /* Assemble results */
    /* C11 = M1 + M4 - M5 + M7 */
    /* C12 = M3 + M5 */
    /* C21 = M2 + M4 */
    /* C22 = M1 - M2 + M3 + M6 */
    
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < h; j++) {
            int q = i * h + j;
            C[i*n + j] = M1[q] + M4[q] - M5[q] + M7[q];
            C[i*n + h + j] = M3[q] + M5[q];
            C[(i+h)*n + j] = M2[q] + M4[q];
            C[(i+h)*n + h + j] = M1[q] - M2[q] + M3[q] + M6[q];
        }
    }
}

void strassen_optimal(float* C, float* A, float* B, int n) {
    if (n < STRASSEN_THRESHOLD) {
        /* Just use BLAS directly for small matrices */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, 1.0f, A, n, B, n, 0.0f, C, n);
        return;
    }
    
    /* Allocate workspace (enough for all levels) */
    size_t ws_size = 20 * n * n * sizeof(float);
    float* workspace = (float*)aligned_alloc(64, ws_size);
    
    strassen_level(C, A, B, n, workspace, 0);
    
    free(workspace);
}
