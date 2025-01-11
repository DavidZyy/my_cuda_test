#include "common.hpp"

template <typename dtype>
__global__ void simt_v0_kernel(const dtype* lhs, const dtype* rhs, dtype* result, 
                               size_t M, size_t N, size_t K)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;  // Row index
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;  // Column index

    if (i < M && j < N) {
        dtype sum = 0;
        #pragma unroll
        for (size_t k = 0; k < K; ++k) {
            sum += lhs[i * K + k] * rhs[k * N + j];
        }
        result[i * N + j] = sum;
    }
}

/**
 * naive matrix multiplication
 * @tparam dtype 
 */
template<typename dtype>
void simt_v0(const dtype* lhs, const dtype* rhs, dtype* result, size_t M, size_t N, size_t K) {
    dim3 threadsPerBlock(16, 16);  // Define block size (16x16 is a typical choice, can be adjusted)
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);  // Number of blocks

    simt_v0_kernel<dtype><<<numBlocks, threadsPerBlock>>>(lhs, rhs, result, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// instantiate the templates
template void simt_v0<float>(const float* lhs, const float* rhs, float* result, size_t M, size_t N, size_t K);
template void simt_v0<half>(const half* lhs, const half* rhs, half* result, size_t M, size_t N, size_t K);
