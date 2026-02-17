// dot_product.cu
#include "../cgkernels.h"

#define BLOCK_SIZE 256

extern "C" {

 __inline__ __device__ double warp_reduce_sum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void vecvecmul_gpu(const double* __restrict__ x, const double* __restrict__ y, double* partial_sum, int n) {
    double sum = 0.0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Strided loop for coalesced global memory access and better load balancing
    for (int i = idx; i < n; i += stride) {
        sum += x[i] * y[i];
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    // Shared memory for inter-warp reduction
    __shared__ double shared[32];  // Max 32 warps per block

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) shared[warp_id] = sum;

    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < (blockDim.x / warpSize)) ? shared[lane] : 0.0;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            partial_sum[blockIdx.x] = sum;
        }
    }
}
   

/*    
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
*/
 
// Função wrapper para ser chamada do Rust
void launch_vecvecmul_gpu(const double* d_xx,
                          const double* d_yy,
                          double* result,
                          int n) {

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) num_blocks = 1024;  // Cap grid size for portability

    double *d_partial_sum;
    CUDA_CHECK(cudaMalloc((void**)&d_partial_sum, num_blocks * sizeof(double)));

    vecvecmul_gpu<<<num_blocks, BLOCK_SIZE>>>(d_xx, d_yy, d_partial_sum, n);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel

    double *h_partial_sum = (double*)malloc(num_blocks * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_partial_sum, d_partial_sum, num_blocks * sizeof(double), cudaMemcpyDeviceToHost));

    *result = 0.0;
    for (int i = 0; i < num_blocks; ++i) {
        *result += h_partial_sum[i];
    }

    CUDA_CHECK(cudaFree(d_partial_sum));
    free(h_partial_sum);
}



}
