// a block have a warp threads
#include "common.hpp"

#define WARP_M 16
#define WARP_N 16
#define WARP_K 16

#define WARP_SIZE 32 // a warp have 32 threads

// total elements is 16 * 16 = 256, 256 / 32 = 8, each threads should fetch / calculate 8 elements
/**
 * write this function for compare with wmma m16n16k16 
 * @tparam dtype 
 */
template <typename dtype>
__global__ void simt_v2_kernel(const dtype* lhs, const dtype* rhs, dtype* result, size_t M, size_t N, size_t K) {
    int row_global = blockIdx.x;
    int col_global = blockIdx.y;

    // int row_warp   = threadIdx.x >> 2; // Row index 0 ~ 15
    // int col_warp   = threadIdx.x & 1; // fetch(matrix A, B) / calculate(matrix C) the first / last 8 elements in the row 
    int a_m = threadIdx.x >> 1; // bug: shift 1 bit, not 2 bits
    int a_k = threadIdx.x & 1;

    int b_k = threadIdx.x & 1;
    int b_n = threadIdx.x >> 1;

    int c_m = threadIdx.x >> 1;
    int c_n = threadIdx.x & 1;


    __shared__ dtype a[WARP_M][WARP_K], b[WARP_K][WARP_N];
    __shared__ dtype c[WARP_M][WARP_N];

    // set c to 0
    for (int i = 0; i < WARP_M; i++) {
        for (int j = 0; j < WARP_N; j++) {
            c[i][j] = 0;
        }
    }

    for (int k = 0; k < K; k += WARP_K) {
        for (int i=0; i<8; i++) {
            // fetch 8-len row vector for a
            a[a_m][a_k*8+i] = lhs[(row_global * WARP_M + a_m) * K + k + a_k* 8 + i];

            // fetch 8-len col vector for b
            b[b_k*8+i][b_n] = rhs[(k + b_k * 8 + i) * N + col_global * WARP_N + b_n];
        }

        // wait for all threads to finish fetching data
        __syncthreads();

        // threadIdx.x == 0, print a and b
//         if (threadIdx.x == 0) {
//             printf("a is: \n");
//             for (int i=0; i<WARP_M; i++) {
//                 for (int j=0; j<WARP_K; j++) {
//                     printf("%10d ", a[i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("\n");
// 
//             printf("b is: \n");
//             for (int i=0; i<WARP_K; i++) {
//                 for (int j=0; j<WARP_N; j++) {
//                     printf("%10d ", b[i][j]);
//                 }
//                 printf("\n");
//             }
//             printf("\n");
//         }


        for (int j=0; j<8; j++) {
            for (int i=0; i<WARP_K; i++) {
                c[c_m][c_n*8+j] += a[c_m][i] * b[i][c_n*8+j];
            }
        }

        // wait for all threads to finish calculating
        __syncthreads();

        // if threadIdx.x == 0, print c
        // if (threadIdx.x == 0) {
        //     printf("c is: \n");
        //     for (int i=0; i<WARP_M; i++) {
        //         for (int j=0; j<WARP_N; j++) {
        //             printf("%10d ", c[i][j]);
        //         }
        //         printf("\n");
        //     }
        //      printf("\n");
        // }
    }

    // write back to global memory
    for (int i=0; i<8; i++) {
        result[(row_global * WARP_M + c_m) * N + col_global * WARP_N + c_n*8+i] = c[c_m][c_n*8+i];
    }
}

template <typename dtype>
void simt_v2(const dtype* lhs, const dtype* rhs, dtype* result, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid((M + WARP_M - 1) / WARP_M, (N + WARP_N - 1) / WARP_N);

    simt_v2_kernel<dtype><<<grid, block>>>(lhs, rhs, result, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


template void simt_v2<float>(const float* lhs, const float* rhs, float* result, size_t M, size_t N, size_t K);
template void simt_v2<half>(const half* lhs, const half* rhs, half* result, size_t M, size_t N, size_t K);
template void simt_v2<int>(const int* lhs, const int* rhs, int* result, size_t M, size_t N, size_t K);
