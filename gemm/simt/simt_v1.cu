#include "common.hpp"
#include <cassert>

#define TM  4 
#define TN  4
template <typename dtype>
__global__ void simt_v1_kernel(const dtype* lhs, const dtype* rhs, dtype* result, size_t M, size_t N, size_t K) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;  // Row index
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;  // Column index

    float a[TM], b[TN];
    float c[TM][TN] = {0};

    for (int k = 0; k < K; k++) {

        // fetch data from global memory
        for (int i = 0; i < TM; i++) {
            a[i] = lhs[(row * TM + i) * K + k];
        }
        for (int j = 0; j < TN; j++) {
            b[j] = rhs[k * N + (col * TN + j)];
        }

        // compute
        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < TN; j++) {
                c[i][j] += a[i] * b[j];
            }
        }
    }

    // write back to global memory
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            result[(row * TM + i) * N + col * TN + j] = c[i][j];
        }
    }
}

/**
 * block matrix multiplication
 * @tparam dtype 
 */
template<typename dtype>
void simt_v1(const dtype* lhs, const dtype* rhs, dtype* result, size_t M, size_t N, size_t K) {
    assert (M % (TM * 16) == 0);
    assert (N % (TN * 16) == 0);
    dim3 threadsPerBlock(16, 16);  // Define block size (16x16 is a typical choice, can be adjusted)
    dim3 numBlocks((M + (threadsPerBlock.x * TM) - 1) / (threadsPerBlock.x * TM),
                   (N + (threadsPerBlock.y * TN) - 1) / (threadsPerBlock.y * TN));  // Number of blocks

    simt_v1_kernel<dtype><<<numBlocks, threadsPerBlock>>>(lhs, rhs, result, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


template void simt_v1<float>(const float* lhs, const float* rhs, float* result, size_t M, size_t N, size_t K);
