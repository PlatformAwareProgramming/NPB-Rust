#include "../cgkernels.h"


#define WARP_SIZE 32

extern "C" {

__inline__ __device__ double warp_reduce_sum(double val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void matvecmul_A100(
    const double* __restrict__ a,
    const int* __restrict__ colidx,
    const int* __restrict__ rowstr,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_rows
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    if (warp_id < num_rows) {
        int row_start = rowstr[warp_id];
        int row_end = rowstr[warp_id + 1];

        double sum = 0.0;
        for (int jj = row_start + lane; jj < row_end; jj += WARP_SIZE) {
            sum += a[jj] * __ldg(&x[colidx[jj]]);
        }

        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            y[warp_id] = sum;
        }
    }
}


/* __global__ void matvecmul_A100(
    const double* __restrict__ a,
    const int* __restrict__ colidx,
    const int* __restrict__ rowstr,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        double sum = 0.0;
        int start = rowstr[row];
        int end = rowstr[row + 1];

        // Prefetch x[] values to avoid repeated global memory accesses
        #pragma unroll 4
        for (int i = start; i < end; ++i) {
            sum += a[i] * __ldg(&x[colidx[i]]);
        }

        y[row] = sum;
    }
}*/


/*__global__ void matvecmul_gpu(
    const double* a,
    const int* colidx,
    const int* rowstr,
    const double* x,
    double* y,
    int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        double sum = 0.0;
        int start = rowstr[row];
        int end = rowstr[row + 1];

        for (int i = start; i < end; ++i) {
            sum += a[i] * x[colidx[i]];
        }

        y[row] = sum;
    }

}
*/

void launch_matvecmul_A100(
    const double* d_aa,
    const int* d_colidx,
    const int* d_rowstr,
    const double* d_xx,
    double* d_yy,
    int nnz,
    int num_rows,
    int x_len
) {
    // Configuração do kernel
//    int blockSize /*= BLOCK_SIZE*/;
//    int gridSize /*= (num_rows + blockSize - 1) / blockSize */;

//    cudaOccupancyMaxPotentialBlockSize( &gridSize, &blockSize, matvecmul_A100, 0, 0); 

int blockSize = 256;  // Must be a multiple of 32
int warps_per_block = blockSize / 32;
int gridSize = (num_rows + warps_per_block - 1) / warps_per_block;

    matvecmul_A100<<<gridSize, blockSize>>>(
        d_aa, d_colidx, d_rowstr, d_xx, d_yy, num_rows
    );

    // Sincronizar GPU (garante conclusão)
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel
 }

}
