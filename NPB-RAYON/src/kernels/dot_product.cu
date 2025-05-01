// dot_product.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

extern "C" {

// Kernel CUDA para multiplicar os vetores
__global__ void dot_product_kernel(const double* x, const double* y, double* partial_sum, int n, int NA) {
    __shared__ double share_data[256]; // Cache compartilhado para redução
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int local_id = threadIdx.x;

	share_data[local_id] = 0.0;

	if(thread_id >= NA){return;}

	share_data[threadIdx.x] = x[thread_id] * y[thread_id];

	__syncthreads();
	for(int i=blockDim.x/2; i>0; i>>=1){
		if(local_id<i){share_data[local_id]+=share_data[local_id+i];}
		__syncthreads();
	}

	if(local_id==0) { 
        partial_sum[blockIdx.x] = share_data[0]; 
    }

 /*   
   if (thread_id < NA) {

    int local_id = threadIdx.x;
    __syncthreads();

    double temp = 0.0;
    while (thread_id < n) {
        temp += x[thread_id] * y[thread_id];
        thread_id += blockDim.x * gridDim.x;
    }

    share_data[local_id] = temp;

    __syncthreads();

    // Redução paralela
    int i = blockDim.x/2;
    while (i != 0) {
        if (local_id < i) {
            share_data[local_id] += share_data[local_id + i];
        }
        __syncthreads();
        i /= 2;
    }   

    if (local_id == 0)
        partial_sum[blockIdx.x] = share_data[0];
    }
        */
}



double *d_x;
double *d_y;
double *d_partial_sum;
double* h_partial_sum;


void alloc_vectors_gpu(int n) {

    printf("ALLOC GPU %d\n", n);

    int threads = 256;
    int blocks = (n + threads - 1) / threads; // Ajusta o número de blocos dinamicamente
    int allthreads = threads*blocks;

    CUDA_CHECK(cudaMalloc((void**)&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_partial_sum, blocks * sizeof(double)));

    h_partial_sum = (double*) malloc(blocks * sizeof(double));

    printf("ALLOC GPU END\n");

}

void free_vectors_gpu() {

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_partial_sum));
    free(h_partial_sum);
}

int alloc = 0;

// Função wrapper para ser chamada do Rust
void dot_product_gpu(const double* x, const double* y, double* result, int n) {
    //double *d_x, *d_y, *d_partial_sum;
    int threads = 256;
    int blocks = (n + threads - 1) / threads; // Ajusta o número de blocos dinamicamente
    int allthreads = threads*blocks;

    if (threads & (threads - 1)) {
        fprintf(stderr, "Erro: o número de threads por bloco deve ser uma potência de 2.\n");
        exit(EXIT_FAILURE);
    }

 //      CUDA_CHECK(cudaMalloc((void**)&d_x, n * sizeof(double)));
 //      CUDA_CHECK(cudaMalloc((void**)&d_y, n * sizeof(double)));
 //      CUDA_CHECK(cudaMalloc((void**)&d_partial_sum, blocks * sizeof(double)));
 
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice));

    dot_product_kernel<<<blocks, threads>>>(d_x, d_y, d_partial_sum, n, allthreads);
  
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel
//    CUDA_CHECK(cudaDeviceSynchronize()); // Garante que o kernel terminou

//    double* h_partial_sum = (double*) malloc(blocks * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_partial_sum, d_partial_sum, blocks * sizeof(double), cudaMemcpyDeviceToHost));
 
    *result = 0.0;
    for (int i = 0; i < blocks; i++) {
        *result += h_partial_sum[i];
    }

//    CUDA_CHECK(cudaFree(d_x));
//    CUDA_CHECK(cudaFree(d_y));
//    CUDA_CHECK(cudaFree(d_partial_sum));
//    free(h_partial_sum);
}

}
