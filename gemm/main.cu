// reference: https://github.com/nicolaswilde/cuda-sgemm
#include "common.hpp"

#include <cassert>
#include <cstddef>
#include <ctime>
#include "omp.h"

template<typename dtype> void simt_v0(const dtype* lhs, const dtype* rhs, dtype* result, size_t M, size_t N, size_t K);
template<typename dtype> void simt_v1(const dtype* lhs, const dtype* rhs, dtype* result, size_t M, size_t N, size_t K);

/***********************************************************************************************************************************************************/
template <typename dtype>
void cpu_sgemm(const dtype* A, const dtype* B, dtype* C, size_t M, size_t N, size_t K) {
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
template <typename dtype>
void cublas_sgemm(const dtype* lhs, const dtype* rhs, dtype* result, size_t M, size_t N, size_t K) {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        float alpha = 1.0f;
        float beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, rhs, N, lhs, K, &beta, result, N));
        CUBLAS_CHECK(cublasDestroy(handle));
}

/***********************************************************************************************************************************************************/

/**
 * test the max error with cpu and cuda
 */
template <typename dtype, void (*func)(const dtype*, const dtype*, dtype*, size_t, size_t, size_t)>
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

    cpu_sgemm<float>(h_a, h_b, h_c, M, N, K);
    func(d_a, d_b, d_c, M, N, K);

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

template <typename dtype, void (*func)(const dtype*, const dtype*, dtype*, size_t, size_t, size_t)>
float testPerformance(int repeat, size_t M, size_t N, size_t K) {
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
        func(d_a, d_b, d_c, M, N, K);
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

typedef float (*TestMaxErrorFunc)(int, int, int);
TestMaxErrorFunc testMaxErrorFuncs[] = {
    // testMaxError<float, cublas_sgemm<float>>,
    // testMaxError<float, simt_v0<float>>,
    testMaxError<float, simt_v1<float>>
};

void testAllMaxError() {
    int M = 512, N = 512, K = 512;
    for (int j = 0; j < sizeof(testMaxErrorFuncs) / sizeof(TestMaxErrorFunc); j++) {
        float max_error = testMaxErrorFuncs[j](M, N, K);
        printf("M N K = %6d %6d %6d, Max Error = %10.8lf\n", M, N, K, max_error);
    }
}

// Define a type for the function pointers
typedef float (*TestFunc)(int, size_t, size_t, size_t);

// Array of function pointers
TestFunc testFuncs[] = {
    // testPerformance<float, cublas_sgemm<float>>,
    // testPerformance<float, simt_v0<float>>,
    testPerformance<float, simt_v1<float>>
};


void testAllPerformance() {
    const int TESTNUM = 15;
    const int M_list[TESTNUM] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[TESTNUM] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    // const int K_list[TESTNUM] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[TESTNUM] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};

    for (int j = 0; j < sizeof(testFuncs) / sizeof(TestFunc); j++) {
        printf("Test %d\n", j);
        for (int i = 0; i < TESTNUM; i++) {
            int M = M_list[i];
            int N = N_list[i];
            int K = K_list[i];
            float sec = testFuncs[j](10, M, N, K);
            double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / sec; 
            printf("M N K = %6d %6d %6d, AVG Performance = %10.4lf Gflops\n", M, N, K, avg_Gflops);
        }
    }
}

int main() {
    testAllMaxError();
    testAllPerformance();
    return 0;
}
