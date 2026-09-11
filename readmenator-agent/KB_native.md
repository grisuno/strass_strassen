# Subsystem: native

## src/native/strassen_c.c
- Layer: utility
- Language: c
- Symbols:
  - `alloc_matrix` (function, line 15) `static float* alloc_matrix(int n)`
  - `matmul_standard` (function, line 20) `static void matmul_standard(float* C, float* A, float* B, int n)`
  - `mat_add` (function, line 33) `static void mat_add(float* C, float* A, float* B, int n)`
  - `mat_sub` (function, line 41) `static void mat_sub(float* C, float* A, float* B, int n)`
  - `extract_quadrant` (function, line 49) `static void extract_quadrant(float* Q, float* M, int n, int row, int col)`
  - `insert_quadrant` (function, line 57) `static void insert_quadrant(float* M, float* Q, int n, int row, int col)`
  - `strassen_recursive` (function, line 65) `void strassen_recursive(float* C, float* A, float* B, int n)`
  - `strassen_multiply` (function, line 164) `void strassen_multiply(float* C, float* A, float* B, int n)`
  - `standard_multiply` (function, line 169) `void standard_multiply(float* C, float* A, float* B, int n)`
  - `memset` (function, line 21) `memset(C, 0, n * n * sizeof(float));`
  - `memcpy` (function, line 52) `memcpy(&Q[i * h], &M[(row + i) * n + col], h * sizeof(float));`
  - `free` (function, line 156) `free(A11);`
  - `THRESHOLD` (macro, line 11) `#define THRESHOLD`

## src/native/strassen_optimal.c
- Layer: utility
- Language: c
- Symbols:
  - `strassen_level` (function, line 18) `static void strassen_level(float* C, float* A, float* B, int n, 
                           float...`
  - `strassen_optimal` (function, line 130) `void strassen_optimal(float* C, float* A, float* B, int n)`
  - `cblas_sgemm` (function, line 21) `cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A, n, B, n, 0.0f, C, n);`
  - `memcpy` (function, line 64) `memcpy(qA11 + i*h, A + i*n, h*sizeof(float));`
  - `free` (function, line 144) `free(workspace);`
  - `STRASSEN_THRESHOLD` (macro, line 15) `#define STRASSEN_THRESHOLD`

## src/native/strassen_turbo.c
- Layer: utility
- Language: c
- Symbols:
  - `alloc_matrix` (function, line 25) `static inline float* alloc_matrix(int n)`
  - `mat_add_avx` (function, line 30) `static void mat_add_avx(float* __restrict C, const float* __restrict A, 
                        ...`
  - `mat_sub_avx` (function, line 50) `static void mat_sub_avx(float* __restrict C, const float* __restrict A, 
                        ...`
  - `matmul_blocked_avx` (function, line 68) `static void matmul_blocked_avx(float* __restrict C, const float* __restrict A, 
                 ...`
  - `extract_quadrant` (function, line 104) `static void extract_quadrant(float* __restrict Q, const float* __restrict M, 
                   ...`
  - `insert_quadrant` (function, line 114) `static void insert_quadrant(float* __restrict M, const float* __restrict Q, 
                    ...`
  - `strassen_turbo_recursive` (function, line 124) `void strassen_turbo_recursive(float* C, float* A, float* B, int n, int depth)`
  - `strassen_turbo` (function, line 261) `void strassen_turbo(float* C, float* A, float* B, int n)`
  - `get_num_threads` (function, line 267) `int get_num_threads(void)`
  - `_mm256_store_ps` (function, line 40) `_mm256_store_ps(&C[i], vc);`
  - `memset` (function, line 70) `memset(C, 0, n * n * sizeof(float));`
  - `_mm256_storeu_ps` (function, line 90) `_mm256_storeu_ps(&C[i * n + j], vc);`
  - `memcpy` (function, line 109) `memcpy(&Q[i * h], &M[(row + i) * n + col], h * sizeof(float));`
  - `free` (function, line 203) `free(T1_1);`
  - `omp_set_num_threads` (function, line 262) `omp_set_num_threads(omp_get_max_threads());`
  - `omp_get_max_threads` (function, line 268) `return omp_get_max_threads();`
  - `THRESHOLD` (macro, line 19) `#define THRESHOLD`
  - `BLOCK_SIZE` (macro, line 21) `#define BLOCK_SIZE`
  - `ALIGN` (macro, line 22) `#define ALIGN`
