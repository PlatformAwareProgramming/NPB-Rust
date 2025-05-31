#include "../cgkernels.h"

#define BLOCK_SIZE 4


extern "C" {


 __global__ void matvecmul_MX570A(
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
        #pragma unroll 8
        for (int i = start; i < end; ++i) {
            sum += a[i] * __ldg(&x[colidx[i]]);
        }

        y[row] = sum;
    }
}


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

void launch_matvecmul_MX570A(
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
    int blockSize /*= BLOCK_SIZE*/;
    int gridSize /*= (num_rows + blockSize - 1) / blockSize */;

    cudaOccupancyMaxPotentialBlockSize( &gridSize, &blockSize, matvecmul_MX570A, 0, 0); 

    matvecmul_MX570A<<<gridSize, blockSize>>>(
        d_aa, d_colidx, d_rowstr, d_xx, d_yy, num_rows
    );

    // Sincronizar GPU (garante conclusão)
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel
 }

}
