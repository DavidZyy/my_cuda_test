/**
 * @file sgemm.cu
 * @author Yangyang Zhu (1929772352@qq.com)
 * @version 0.1
 * @date 2024-07-31
 * 
 * @copyright Copyright (c) 2024
 * gpu accelerate gemm
 * reference: https://dlsyscourse.org/slides/12-gpu-acceleration.pdf
 * use cuda core do single gemm ??
 */

#include <cassert>
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* threads num in x, y direction */
// #define TILE_SIZE 2
#define TILE_SIZE 16

#define CUDA_CHECK(call)                                                    \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess)                                               \
    {                                                                       \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", ";      \
        std::cerr << "code: " << error << ", reason: " << cudaGetErrorString(error) << std::endl; \
        exit(1);                                                            \
    }                                                                       \
}

#define CUBLAS_CHECK(call)                                                  \
{                                                                           \
    const cublasStatus_t status = call;                                     \
    if (status != CUBLAS_STATUS_SUCCESS)                                    \
    {                                                                       \
        std::cerr << "CUBLAS Error: " << __FILE__ << ":" << __LINE__ << ", "; \
        std::cerr << "status: " << status << std::endl;                     \
        exit(1);                                                            \
    }                                                                       \
}

/************************************************************************************************************ */

/**
 * v0 methods, naive methods
 */
__global__ void gemm_kernel_v0(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float value = 0.0f;
        #pragma unroll
        for (int e = 0; e < K; ++e) {
            value += A[row * K + e] * B[e * N + col];
        }
        C[row * N + col] = value;
    }
}

void gemm_v0(float* A, float* B, float* C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    gemm_kernel_v0<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

/************************************************************************************************************ */

void gemm_cublas(float* A, float* B, float* C, int M, int N, int K) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    float alpha = 1.0f;
    float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));
}
 
/************************************************************************************************************ */
// #define reg_tile_size 1
// #define reg_tile_size 2
#define reg_tile_size 4
// #define reg_tile_size 8

// __device__ void get_slice() {
//     for (int i = 0; i < reg_tile_size; ++i) {
//         array[i] = 
//     }
// }
// 
// __device__ void set_slice() {
// 
// }

/**
 * reg tile
 * see https://en.wikipedia.org/wiki/Block_matrix#Multiplication
 */
__global__ void gemm_kernel_v1(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float a[reg_tile_size];
    float b[reg_tile_size];
    float c[reg_tile_size][reg_tile_size] = {0};

    // supplementary comment: the important point is partioned matrix multiplication, not outer / inner product
    // because A is be partioned to col vector, and B is partiond to row vector,
    // so it seems like doing out product
    // use out prod to compute gemm
    for (int k = 0; k < K; ++k) {

        // get a, regard a as row vector
        for (int i = 0; i < reg_tile_size; ++i) {
            a[i] = A[(row * reg_tile_size + i) * K + k];
        }
        // get b, regard b as col vector
        for (int j = 0; j < reg_tile_size; ++j) {
            b[j] = B[k * N + (col * reg_tile_size + j)];
        }

        // out prod of matmul
        for (int y = 0; y < reg_tile_size; ++y) {
            for (int x = 0; x < reg_tile_size; ++x) {
                c[y][x] += a[y] * b[x];
            }
        }

    }

    // set C
    for (int y = 0; y < reg_tile_size; ++y) {
        for (int x = 0; x < reg_tile_size; ++x) {
            C[(row * reg_tile_size + y) * N + (col * reg_tile_size + x)] = c[y][x];
            // printf c[y][x]
            // printf("%f\n", c[y][x]);
        }
    }
}

/**
 * partition matrix by threads
 */
void gemm_v1(float* A, float* B, float* C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE / reg_tile_size, (M + TILE_SIZE - 1) / TILE_SIZE / reg_tile_size);
    
    gemm_kernel_v1<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}


/************************************************************************************************************ */

/**
 * shared memroy + reg tile
 */
__global__ void gemm_kernel_v2(float* A, float* B, float* C, int M, int N, int K) {
    // int yblock = blockIdx.y;
    // int xblock = blockIdx.x;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // local memory
    float a[reg_tile_size];
    float b[reg_tile_size];
    float c[reg_tile_size][reg_tile_size] = {0};

    // shared memory
    __shared__ float s_A[TILE_SIZE * reg_tile_size][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE * reg_tile_size];

    int S = blockDim.x;
    // int L = blockDim.y * reg_tile_size;

    for (int k = 0; k < K; k += S) {
        __syncthreads();
        // cooperate fetching memory of s_A and s_B to shared memory
        for (int i = 0; i < reg_tile_size; ++i) {
            // s_A[threadIdx.y + i][threadIdx.x] = A[(row * reg_tile_size + i) * K + (k + threadIdx.x)];
            s_A[threadIdx.y * reg_tile_size + i][threadIdx.x] = A[(row * reg_tile_size + i) * K + (k + threadIdx.x)];
            // assert s_A == 1
            // assert(s_A[threadIdx.y * reg_tile_size + i][threadIdx.x] == 1);
        }
        for (int i = 0; i < reg_tile_size; ++i) {
            // s_B[threadIdx.y][threadIdx.x + i] = B[(k + threadIdx.y) * N + (col * reg_tile_size + i)];
            s_B[threadIdx.y][threadIdx.x * reg_tile_size + i] = B[(k + threadIdx.y) * N + (col * reg_tile_size + i)];
            // assert(s_B[threadIdx.y][threadIdx.x * reg_tile_size + i] == 1);
        }
        __syncthreads();

        // get all vectors in shared memory and calculate out prod
        for (int i = 0; i < S; ++i) {
            // get a
            for (int j = 0; j < reg_tile_size; ++j) {
                // a[j] = s_A[threadIdx.y * reg_tile_size + j][threadIdx.x + i];
                a[j] = s_A[threadIdx.y * reg_tile_size + j][i];
                // assert(a[j] == 1);
            }
            // get b
            for (int j = 0; j < reg_tile_size; ++j) {
                // b[j] = s_B[threadIdx.y + i][threadIdx.x * reg_tile_size + j];
                b[j] = s_B[i][threadIdx.x * reg_tile_size + j];
                // assert(b[j] == 1);
            }

            // calculate out prod
            for (int y = 0; y < reg_tile_size; ++y) {
                for (int x = 0; x < reg_tile_size; ++x) {
                    c[y][x] += a[y] * b[x];
                }
            }
        }

    }

    // set C
    for (int y = 0; y < reg_tile_size; ++y) {
        for (int x = 0; x < reg_tile_size; ++x) {
            C[(row * reg_tile_size + y) * N + (col * reg_tile_size + x)] = c[y][x];
            // printf("%f\n", c[y][x]);
        }
    }
}

/**
 * shared memory tile
 * partition matrix by blocks, and then partition the submatrix by threads
 */
void gemm_v2(float* A, float* B, float* C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE / reg_tile_size, (M + TILE_SIZE - 1) / TILE_SIZE / reg_tile_size);
    // dim3 dimBlock(2, 2);
    // dim3 dimGrid(2, 2);
    
    gemm_kernel_v2<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}
