// dot_product.cu
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

// Kernel CUDA para multiplicar os vetores
__global__ void dot_product_kernel(const double* x, const double* y, double* partial_sum, int n) {
    __shared__ double cache[256]; // Cache compartilhado para redução
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    double temp = 0.0;
    while (tid < n) {
        temp += x[tid] * y[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    // Redução paralela
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        partial_sum[blockIdx.x] = cache[0];
}

// Função wrapper para ser chamada do Rust
void dot_product_gpu(const double* x, const double* y, double* result, int n) {
    double *d_x, *d_y, *d_partial_sum;
    int blocks = 32;
    int threads = 256;

    cudaMalloc((void**)&d_x, n * sizeof(double));
    cudaMalloc((void**)&d_y, n * sizeof(double));
    cudaMalloc((void**)&d_partial_sum, blocks * sizeof(double));

    cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice);

    dot_product_kernel<<<blocks, threads>>>(d_x, d_y, d_partial_sum, n);

    double* h_partial_sum = (double*) malloc(blocks * sizeof(double));
    cudaMemcpy(h_partial_sum, d_partial_sum, blocks * sizeof(double), cudaMemcpyDeviceToHost);

    *result = 0.0;
    for (int i = 0; i < blocks; i++) {
        *result += h_partial_sum[i];
    }

    free(h_partial_sum);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partial_sum);
}

}
