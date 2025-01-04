/**
 * @file main.cpp
 * @author Yangyang Zhu (1929772352@qq.com)
 * @version 0.1
 * @date 2024-07-31
 * 
 * @copyright Copyright (c) 2024
 * this file is used to test the performance of gemm methods
 */

#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

#define MEASURE_TIME(FUNC, DESC, SIZE) \
    start = std::chrono::high_resolution_clock::now(); \
    FUNC; \
    end = std::chrono::high_resolution_clock::now(); \
    duration = end - start; \
    gflops = (2.0 * SIZE * SIZE * SIZE) / (duration.count() * 1e9); \
    std::cout << DESC << " Time: " << duration.count() << " seconds"; \
    std::cout << ", GFLOPS: " << gflops << std::endl;

/**
 * generate matrices 
 */
void generate_matrices(float* A, float* B, int size) {
    for (int i = 0; i < size * size; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
        // A[i] = 1;
        // B[i] = 1;
    }
}

/**
 * compare if the results is correct
 */
bool compare_results(float* C1, float* C2, int size, float tol = 1e-2) {
    for (int i = 0; i < size * size; ++i) {
        if (fabs(C1[i] - C2[i]) > tol) {
            std::cout << C1[i] << " != " << C2[i] << std::endl;
            return false;
        }
    }
    return true;
}

/////////////////////////// gemm methods //////////////////////////////
// cpu
void gemm_cpu_naive(float* A, float* B, float* C, int M, int N, int K);
void gemm_cpu_out_prod(float* A, float* B, float* C, int M, int N, int K);
void gemm_cpu_cblas(float* A, float* B, float* C, int M, int N, int K);
// cuda
void gemm_cuda_naive(float* A, float* B, float* C, int M, int N, int K);
void gemm_cuda_reg_tile(float* A, float* B, float* C, int M, int N, int K);
void gemm_cuda_sm_tile(float* A, float* B, float* C, int M, int N, int K);
void gemm_cuda_cublas(float* A, float* B, float* C, int M, int N, int K);
///////////////////////////////////////////////////////////////////////

void benchmark_gemm(int size) {
    float *A = new float[size * size];
    float *B = new float[size * size];
    float *C_cpu_naive = new float[size * size];
    float *C_cpu_out_prod = new float[size * size];
    float *C_cpu_cblas = new float[size * size];
    float *C_cuda_naive = new float[size * size];
    float *C_cuda_reg_tile = new float[size * size];
    float *C_cuda_sm_tile = new float[size * size];
    float *C_cuda_cublas = new float[size * size];

    generate_matrices(A, B, size);

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration;
    float gflops;

    // MEASURE_TIME(gemm_cpu_naive(A, B, C_cpu_naive, size, size, size), "cpu naive", size);
    // MEASURE_TIME(gemm_cpu_out_prod(A, B, C_cpu_out_prod, size, size, size), "cpu out prod", size);
    // MEASURE_TIME(gemm_cpu_cblas(A, B, C_cpu_cblas, size, size, size), "cpu cblas", size);
    MEASURE_TIME(gemm_cuda_naive(A, B, C_cuda_naive, size, size, size), "cuda naive", size);
    MEASURE_TIME(gemm_cuda_reg_tile(A, B, C_cuda_reg_tile, size, size, size), "cuda reg tile", size);
    MEASURE_TIME(gemm_cuda_sm_tile(A, B, C_cuda_sm_tile, size, size, size), "cuda sm tile", size);
    MEASURE_TIME(gemm_cuda_cublas(A, B, C_cuda_cublas, size, size, size), "cuda cublas", size);

    // if (compare_results(C_cpu_naive, C_cpu_out_prod, size)) {
    // if (compare_results(C_cpu_naive, C_cuda_reg_tile, size)) {
    // if (compare_results(C_cuda_cublas, C_cuda_reg_tile, size)) {
    if (compare_results(C_cuda_cublas, C_cuda_sm_tile, size)) {
    // if (compare_results(C_cuda_naive, C_cuda_cublas, size)) {
        // if (compare_results(C_cpu_naive, C_cuda_cublas, size)) {
            std::cout << "Results are correct and match." << std::endl;
        // } else {
        //     std::cout << "Results are correct but do not match cuBLAS." << std::endl;
        // }
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] C_cpu_naive;
    delete[] C_cpu_out_prod;
    delete[] C_cpu_cblas;
    delete[] C_cuda_naive;
    delete[] C_cuda_reg_tile;
    delete[] C_cuda_sm_tile;
    delete[] C_cuda_cublas;
}

int main() {
    // int size = 4;
    // int size = 8;
    // int size = 32;  // Example size, you can vary this
    // int size = 64;
    // int size = 128;
    // int size = 256;
    // int size = 1024;
    // int size = 2048;
    // int size = 4096;
    // int size = 8192;
    // int size = 8192 * 2;
    for (int size = 128; size <= 8192 * 2; size *= 2) {
        std::cout << "Size: " << size << std::endl;
        benchmark_gemm(size);
    }
    return 0;
}
