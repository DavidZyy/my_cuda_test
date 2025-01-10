// reference: https://github.com/nicolaswilde/cuda-sgemm
#include "common.hpp"

#include <cassert>
#include <ctime>
#include <iostream>
#include "omp.h"

/***********************************************************************************************************************************************************/
template <typename dtype>
void cpu_matmul2d(const dtype* A, const dtype* B, dtype* C, size_t M, size_t N, size_t K) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            dtype sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/***********************************************************************************************************************************************************/
void matmul2d_Cublas(const float* lhs, const float* rhs, float* result, size_t M, size_t N, size_t K) {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        float alpha = 1.0f;
        float beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, rhs, N, lhs, K, &beta, result, N));
        CUBLAS_CHECK(cublasDestroy(handle));
}

/***********************************************************************************************************************************************************/
template <typename dtype>
__global__ void matmul2dKernelV0(const dtype* lhs, const dtype* rhs, dtype* result, 
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
void matmul2dV0(const dtype* lhs, const dtype* rhs, dtype* result, size_t M, size_t N, size_t K) {
    dim3 threadsPerBlock(16, 16);  // Define block size (16x16 is a typical choice, can be adjusted)
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);  // Number of blocks

    matmul2dKernelV0<dtype><<<numBlocks, threadsPerBlock>>>(lhs, rhs, result, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/***********************************************************************************************************************************************************/
#define TM  4 
#define TN  4
template <typename dtype>
__global__ void matmulKernelV1(const dtype* lhs, const dtype* rhs, dtype* result, size_t M, size_t N, size_t K) {
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
void matmulV1(const dtype* lhs, const dtype* rhs, dtype* result, size_t M, size_t N, size_t K) {
    assert (M % (TM * 16) == 0);
    assert (N % (TN * 16) == 0);
    dim3 threadsPerBlock(16, 16);  // Define block size (16x16 is a typical choice, can be adjusted)
    dim3 numBlocks((M + (threadsPerBlock.x * TM) - 1) / (threadsPerBlock.x * TM),
                   (N + (threadsPerBlock.y * TN) - 1) / (threadsPerBlock.y * TN));  // Number of blocks

    matmulKernelV1<dtype><<<numBlocks, threadsPerBlock>>>(lhs, rhs, result, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * test the max error with cpu and cuda
 */
// template <>
float testMaxError(int M, int N, int K) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    CUDA_CHECK(cudaMalloc(&d_a, size_a));
    CUDA_CHECK(cudaMalloc(&d_b, size_b));
    CUDA_CHECK(cudaMalloc(&d_c, size_c));
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    cpu_matmul2d<float>(h_a, h_b, h_c, M, N, K);
    // matmul2d_Cublas(d_a, d_b, d_c, M, N, K);
    // matmul2dV0<float>(d_a, d_b, d_c, M, N, K);
    matmulV1<float>(d_a, d_b, d_c, M, N, K);

    CUDA_CHECK(cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost));

    float max_error = 0;
    for (int i = 0; i < M * N; i++) {
        float this_error = std::abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = std::max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_d_c);

    return max_error;
}

float testPerformance(int repeat, int M, int N, int K) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size_a));
    CUDA_CHECK(cudaMalloc(&d_b, size_b));
    CUDA_CHECK(cudaMalloc(&d_c, size_c));

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++)
        // cuda.matmul2d(d_a, d_b, d_c, M, N, K);
        matmul2d_Cublas(d_a, d_b, d_c, M, N, K);
        // matmul2dV0<float>(d_a, d_b, d_c, M, N, K);
        // matmulV1<float>(d_a, d_b, d_c, M, N, K);
        // assert(0); // should call kernel directly
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float msec, sec;
    CUDA_CHECK(cudaEventElapsedTime(&msec, start, end));
    sec = msec / 1000.0 / repeat;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return sec;
}

int main() {
    int M0 = 1024;
    int N0 = 1024;
    int K0 = 1024;

//     int M0 = 64;
//     int N0 = 64;
//     int K0 = 64;

    int M1 = 1024 * 4;
    int N1 = 1024 * 4;
    int K1 = 1024 * 4;

    // float max_error = testMaxError(M0, N0, K0);
    // std::cout << "max error: " << max_error << std::endl;

    int repeat = 1;
    float total_sec = testPerformance(repeat, M1, N1, K1);
    double avg_sec = total_sec / repeat;
    double avg_Gflops = ((double)M1) * N1 * K1 * 2 / 1024 / 1024 / 1024 / avg_sec;
    std::cout << "average time: " << avg_sec << "s" << std::endl;
    std::cout << "average Gflops: " << avg_Gflops << std::endl;
    return 0;
}
