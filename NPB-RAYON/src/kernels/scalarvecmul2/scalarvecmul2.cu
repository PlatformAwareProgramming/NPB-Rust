#include "../cgkernels.h"

extern "C" {



__global__ void scalarvecmul2_gpu(double alpha, const double* x, double* y, int n) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < n) { 
        y[i] += alpha*x[i];
    }
}


 void launch_scalarvecmul2_gpu(
    const double alpha, 
    const double* d_xx, 
    double* d_yy, 
    int n) {

        int blockSize = BLOCK_SIZE;
        int gridSize = (n + blockSize - 1) / blockSize; // Ajusta o número de blocos dinamicamente
    
        if (blockSize & (blockSize - 1)) {
            fprintf(stderr, "Erro: o número de threads por bloco deve ser uma potência de 2.\n");
            exit(EXIT_FAILURE);
        }
    
        scalarvecmul2_gpu<<<gridSize, blockSize>>>(alpha, d_xx, d_yy, n);
      
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel
 }



}
