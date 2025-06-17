#include "../cgkernels.h"


#define BLOCK_SIZE 32
#define warpSize 32

extern "C" {

#include <cuda_runtime.h>

__inline__ __device__ double warp_reduce_sum(double val) {
    // Redução dentro do warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ __launch_bounds__(BLOCK_SIZE, 2)  // 2 blocos por SM como sugestão
void matvecmul_CC70(
    const double* __restrict__ a,
    const int* __restrict__ colidx,
    const int* __restrict__ rowstr,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_rows
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;
    int total_warps = (gridDim.x * blockDim.x) / warpSize;

    if (warp_id < num_rows) {
        for (int row = warp_id; row < num_rows; row += total_warps) {
            int row_start = __ldg(&rowstr[row]);
            int row_end = __ldg(&rowstr[row + 1]);

            double sum = 0.0;

            for (int jj = row_start + lane; jj < row_end; jj += warpSize) {
                double val_a = __ldg(&a[jj]);
                int col = __ldg(&colidx[jj]);
                double val_x = __ldg(&x[col]);
                sum += val_a * val_x;
            }

            sum = warp_reduce_sum(sum);

            if (lane == 0) {
                y[row] = sum;
            }
        }
    }
}


void launch_matvecmul_CC70(
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
 
    int blockSize = BLOCK_SIZE;
    int gridSize = (num_rows * warpSize + blockSize - 1) / blockSize;

    // cudaFuncSetAttribute(matvecmul_CC70, cudaFuncAttributePreferredSharedMemoryCarveout, 100); // 100% shared memory convertida em L1 cache

    matvecmul_CC70<<<gridSize, blockSize>>>(
        d_aa, d_colidx, d_rowstr, d_xx, d_yy, num_rows
    );

    // Sincronizar GPU (garante conclusão)
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel
 }

}
