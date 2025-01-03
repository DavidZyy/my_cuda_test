#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// Kernel declaration
__global__ void gemm_m8n8k16_mma(const int8_t* A, const int8_t* B, int32_t* C, 
                                 int M, int N, int K);

// CPU reference implementation of GEMM
void gemm_cpu(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Utility to print a matrix (for debugging)
template <typename T>
void print_matrix(const T* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << static_cast<int>(matrix[i * cols + j]) << " ";
        }
        std::cout << std::endl;
    }
}

// Main test bench
int main() {
    // Matrix dimensions (must be multiples of 8 and 16)
    int M = 16;  // Rows of A and C
    int N = 16;  // Columns of B and C
    int K = 16;  // Columns of A, Rows of B

    // Allocate host memory
    std::vector<int8_t> h_A(M * K, 0);
    std::vector<int8_t> h_B(K * N, 0);
    std::vector<int32_t> h_C(M * N, 0);
    std::vector<int32_t> h_C_ref(M * N, 0);

    // Initialize input matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = rand() % 4 - 2;  // Random values in range [-2, 2]
    for (int i = 0; i < K * N; ++i) h_B[i] = rand() % 4 - 2;

    // Allocate device memory
    int8_t *d_A, *d_B;
    int32_t *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(int32_t)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(int8_t), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 threadsPerBlock(32, 1);  // One warp per block
    dim3 blocksPerGrid((N + 7) / 8, (M + 7) / 8);
    gemm_m8n8k16_mma<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Compute reference result on CPU
    gemm_cpu(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    // Verify the results
    bool pass = true;
    for (int i = 0; i < M * N; ++i) {
        if (h_C[i] != h_C_ref[i]) {
            std::cerr << "Mismatch at index " << i << ": GPU result " << h_C[i]
                      << ", CPU result " << h_C_ref[i] << std::endl;
            pass = false;
            break;
        }
    }

    if (pass) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cerr << "Test failed!" << std::endl;
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}

