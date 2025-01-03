/**
 * @file a.cu
 * @author Yangyang Zhu (yangyangzhu12@qq.com)
 * @version 0.1
 * @date 2025-01-03
 * use tensor core to calculate
 * mma.m8n8k16
 */

#include <mma.h>
using namespace nvcuda;

// GEMM kernel using mma.m8n8k16
__global__ void gemm_m8n8k16_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    // Thread identifiers
    int warpM = threadIdx.y + blockIdx.y * blockDim.y; // Warp row
    int warpN = threadIdx.x + blockIdx.x * blockDim.x; // Warp col
    
    // Tile indices
    int row = warpM * 8;
    int col = warpN * 8;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, 8, 8, 16, float, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, 8, 8, 16, float, wmma::col_major> fragB;
    wmma::fragment<wmma::accumulator, 8, 8, 16, float> fragC;

    // Initialize accumulator fragment
    wmma::fill_fragment(fragC, 0.0f);

    // Iterate over K dimension
    for (int k = 0; k < K; k += 16) {
        // Load fragments from global memory
        wmma::load_matrix_sync(fragA, A + row * K + k, K);
        wmma::load_matrix_sync(fragB, B + k * N + col, N);

        // Perform MMA operation
        wmma::mma_sync(fragC, fragA, fragB, fragC);
    }

    // Store result
    wmma::store_matrix_sync(C + row * N + col, fragC, N, wmma::mem_row_major);
}

// Host code
void gemm_m8n8k16(const float *A, const float *B, float *C, int M, int N, int K) {
    // Define CUDA grid/block sizes
    dim3 threadsPerBlock(32, 8); // 32 threads per warp, 8 warps per block
    dim3 blocksPerGrid((N + 7) / 8, (M + 7) / 8);

    // Launch kernel
    gemm_m8n8k16_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
