// dot_product.cu
#include "../cgkernels.h"

#define BLOCK_SIZE 4

extern "C" {

// Kernel CUDA para multiplicar os vetores
__global__ void vecvecmul_gpu(const double* x, const double* y, double* partial_sum, int n) {
    __shared__ double share_data[BLOCK_SIZE]; // Cache compartilhado para redução
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int local_id = threadIdx.x;

    share_data[local_id] = 0.0;

    if(thread_id >= n) { return; }

    share_data[threadIdx.x] = x[thread_id] * y[thread_id];

    __syncthreads();
    for(int i=blockDim.x/2; i>0; i>>=1){
        if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
        __syncthreads();
    }

    if(local_id == 0) { 
        partial_sum[blockIdx.x] = share_data[0]; 
    }
}

 
// Função wrapper para ser chamada do Rust
void launch_vecvecmul_gpu(const double* d_xx, 
                          const double* d_yy, 
                          double* result, 
                          int n) {

    int blockSize = BLOCK_SIZE;
    int gridSize = (n + blockSize - 1) / blockSize; // Ajusta o número de blocos dinamicamente
    double *d_partial_sum, *h_partial_sum;

    if (blockSize & (blockSize - 1)) {
        fprintf(stderr, "Erro (vecvecmul): o número de threads por bloco deve ser uma potência de 2.\n");
        exit(EXIT_FAILURE);
    }

    h_partial_sum = (double*) malloc(gridSize * sizeof(double));
    CUDA_CHECK(cudaMalloc((void**)&d_partial_sum, gridSize * sizeof(double)));
 
    vecvecmul_gpu<<<gridSize, blockSize>>>(d_xx, d_yy, d_partial_sum, n);
  
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel

    CUDA_CHECK(cudaMemcpy(h_partial_sum, d_partial_sum, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
 
    *result = 0.0;
    for (int i = 0; i < gridSize; i++) {
        *result += h_partial_sum[i];
    }

    CUDA_CHECK(cudaFree(d_partial_sum));
    free(h_partial_sum);
}



}
