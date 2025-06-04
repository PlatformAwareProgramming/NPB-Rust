#include "../cgkernels.h"

extern "C" {

#define BLOCK_SIZE 32  // ou ajuste conforme capacidade

/*
 // VERY SLOW
__global__ void matvecmul_MX570A(
    double *values, 
    int *col_idx, 
    int *row_ptr, 
    double *x, 
    double *y, 
    int num_rows, 
    int x_size) 
{
    __shared__ double x_shared[BLOCK_SIZE];  // shared memory para parte de x

    int tid = threadIdx.x;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        double sum = 0.0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        // Processar múltiplos blocos de x
        for (int blk = 0; blk < x_size; blk += BLOCK_SIZE) {

            // Carregar o sub-bloco de x na shared memory
            int idx = blk + tid;
            if (idx < x_size && tid < BLOCK_SIZE) {
                x_shared[tid] = x[idx];
            }
            __syncthreads();

            // Para cada elemento não-nulo da linha
            for (int j = row_start; j < row_end; j++) {
                int col = col_idx[j];

                // Se col está dentro do bloco carregado → usar shared
                if (col >= blk && col < blk + BLOCK_SIZE) {
                    sum += values[j] * x_shared[col - blk];
                }
                // Caso contrário, acessar diretamente da memória global
                else if (blk == 0) { 
                    // Apenas uma vez, no primeiro bloco: acesso direto
                    sum += values[j] * x[col];
                }
            }

            __syncthreads();
        }

        y[row] = sum;
    }
}
*/

    /*

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
        #pragma unroll 4
        for (int i = start; i < end; ++i) {
            sum += a[i] * __ldg(&x[colidx[i]]);
        }

        y[row] = sum;
    }
}*/

__inline__ __device__ double warp_reduce_sum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void matvecmul_MX570A(
    const double* __restrict__ a,
    const int* __restrict__ colidx,
    const int* __restrict__ rowstr,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_rows
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;

    if (warp_id < num_rows) {
        int row_start = rowstr[warp_id];
        int row_end = rowstr[warp_id + 1];

        double sum = 0.0;
        for (int jj = row_start + lane; jj < row_end; jj += warpSize) {
            sum += a[jj] * __ldg(&x[colidx[jj]]);
        }

        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            y[warp_id] = sum;
        }
    }
}

/*
__global__ void matvecmul_MX570A(
    const double* a,
    const int* colidx,
    const int* rowstr,
    const double* x,
    double* y,
    int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0;
        int start = rowstr[row];
        int end = rowstr[row + 1];

        for (int i = start; i < end; ++i) {
            sum += (float)a[i] * (float)x[colidx[i]];
        }

        y[row] = sum;
    }
}
    */





void launch_matvecmul_MX570A(
     double* d_aa,
    int* d_colidx,
    int* d_rowstr,
    double* d_xx,
    double* d_yy,
    int nnz,
    int num_rows,
    int x_len
) {
    // Configuração do kernel
 //   int blockSize = BLOCK_SIZE;
 //   int gridSize = (num_rows + blockSize - 1) / blockSize ;

    int blockSize = BLOCK_SIZE;  // Must be a multiple of 32
    int warps_per_block = blockSize / 32;
    int gridSize = (num_rows + warps_per_block - 1) / warps_per_block;

  //  cudaOccupancyMaxPotentialBlockSize( &gridSize, &blockSize, matvecmul_MX570A, 0, 0); 

    matvecmul_MX570A<<<gridSize, blockSize>>>(
        d_aa, d_colidx, d_rowstr, d_xx, d_yy, num_rows
    );

    // Sincronizar GPU (garante conclusão)
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel
 }

}

