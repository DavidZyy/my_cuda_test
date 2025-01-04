/**
 * @file gemm.cpp
 * @author Yangyang Zhu (1929772352@qq.com)
 * @version 0.1
 * @date 2024-07-31
 * 
 * @copyright Copyright (c) 2024
 * cpu acceleration gemm
 * reference: https://dlsyscourse.org/slides/11-hardware-acceleration.pdf, seems some mistake in ppt, no need to use B.T, just can use B.
 */


#include <cblas.h>  // Include OpenBLAS
#include <cstring>

void gemm_cpu_cblas(float* A, float* B, float* C, int M, int N, int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}


/**
 * naive inner prod to calculate gemm 
 * A: M x K
 * B: K x N
 */
void gemm_cpu_naive(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float value = 0.0f;
            for (int e = 0; e < K; ++e) {
                value += A[i * K + e] * B[e * N + j];
            }
            C[i * N + j] = value;
        }
    }
}

/**
 * use out product to calculate gemm
 * A: M x K
 * B: K x N
 */
void gemm_cpu_out_prod(float* A, float* B, float* C, int M, int N, int K) {
    // select the i column of A, and the i row of B
    // the C is added by i M x N matrices produced by the outer product of above
    for (int i = 0; i < K; ++i) {

        for (int j = 0; j < M; ++j) {
            for (int e = 0; e < N; ++e) {
                if (i == 0) {
                    C[j * N + e] = 0;
                }
                C[j * N + e] += A[j * K + i] * B[i * N + e];
            }
        }

    }
}

/**
 * do not know why this method is called tile ... 
 * it is just block matrix multiplication, reference: https://en.wikipedia.org/wiki/Block_matrix#Multiplication
 * in reference ppt above, the blocked matrix multiplication is calcuted by inner prod.
 * the A and B is partitioned into TILE_SIZE * TILE_SIZE tiles, and then calculate the inner prod of each tile.
 * A: M x K
 * B: K x N
 */
#define TILE_SIZE 4
void dot(float* A, float* B, float* C) {

}

/**
 * NOTE!!: the memory layout should be compacted if use submatrix. or the 
 * submatrix's memory is not contiguous.
 */
void gemm_cpu_register_tiled(float* A, float* B, float* C, int M, int N, int K) {
    int new_M = M / TILE_SIZE;
    int new_N = N / TILE_SIZE;
    int new_K = K / TILE_SIZE;
    int submat_size = TILE_SIZE * TILE_SIZE; // the size of each submatrix

    for (int i = 0; i < new_M; ++i) {
        for (int j = 0; j < new_N; ++j) {

            for (int e = 0; e < new_K; ++e) {
                if(e == 0) {
                    memset(&C[i * new_N * submat_size + j * submat_size], 0, submat_size * sizeof(float));
                }
                dot(&A[i * new_K * submat_size + e * submat_size],
                    &B[e * new_N * submat_size + j * submat_size],
                    &C[i * new_N * submat_size + j * submat_size]);
            }

        }
    }
}