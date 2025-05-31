#include "../cgkernels.h"

#define BLOCK_SIZE 4

extern "C" {



__global__ void scalarvecmul1_gpu(double alpha, const double* x, double* y, int n) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < n) { 
        y[i] = x[i] + alpha*y[i];
    }
}

 
 void launch_scalarvecmul1_gpu(
    const double alpha, 
    const double* d_xx, 
    double* d_yy, 
    int n) {

        int blockSize /*= BLOCK_SIZE*/;
        int gridSize /*= (num_rows + blockSize - 1) / blockSize */;

        cudaOccupancyMaxPotentialBlockSize( &gridSize, &blockSize, scalarvecmul1_gpu, 0, 0); 
    
        if (blockSize & (blockSize - 1)) {
            fprintf(stderr, "Erro: o número de threads por bloco deve ser uma potência de 2.\n");
            exit(EXIT_FAILURE);
        }
    
        scalarvecmul1_gpu<<<gridSize, blockSize>>>(alpha, d_xx, d_yy, n);
      
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel
 }

 
}
