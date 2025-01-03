__global__ void gemm_m8n8k16_mma(const int8_t* A, const int8_t* B, int32_t* C, 
                                 int M, int N, int K) {
    // Warp and lane identifiers
    int warp_id = threadIdx.x / 32;     // Warp index in the block
    int lane_id = threadIdx.x % 32;    // Lane index within the warp

    // Each warp processes one 8x8 block of C
    int block_row = (blockIdx.y * blockDim.y + warp_id) * 8;
    int block_col = blockIdx.x * 8;

    // Accumulators for the C tile
    int32_t accumulators[8] = {0};

    // Iterate over the K dimension in steps of 16
    for (int k = 0; k < K; k += 16) {
        // Load A and B fragments for this warp
        int4 a_frag, b_frag;

        if (block_row + lane_id / 8 < M && k + lane_id % 16 < K) {
            a_frag = *(reinterpret_cast<const int4*>(&A[(block_row + lane_id / 8) * K + k]));
        } else {
            a_frag = make_int4(0, 0, 0, 0);
        }

        if (block_col + lane_id / 8 < N && k + lane_id % 16 < K) {
            b_frag = *(reinterpret_cast<const int4*>(&B[k * N + block_col + lane_id / 8]));
        } else {
            b_frag = make_int4(0, 0, 0, 0);
        }

        // Perform the MMA operation
        asm volatile(
            "mma.m8n8k16.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, "
            "{%8, %9, %10, %11}, "
            "{%12, %13, %14, %15}, "
            "{%0, %1, %2, %3, %4, %5, %6, %7};\n"
            : "+r"(accumulators[0]), "+r"(accumulators[1]), "+r"(accumulators[2]), "+r"(accumulators[3]),
              "+r"(accumulators[4]), "+r"(accumulators[5]), "+r"(accumulators[6]), "+r"(accumulators[7])
            : "r"(a_frag.x), "r"(a_frag.y), "r"(a_frag.z), "r"(a_frag.w),
              "r"(b_frag.x), "r"(b_frag.y), "r"(b_frag.z), "r"(b_frag.w));
    }

    // Store the results back to global memory
    if (block_row + lane_id / 8 < M && block_col + lane_id % 8 < N) {
        for (int i = 0; i < 8; ++i) {
            C[(block_row + i) * N + block_col + lane_id % 8] = accumulators[i];
        }
    }
}

