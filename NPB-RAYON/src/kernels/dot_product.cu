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

        if(thread_id >= NA) { return; }

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

    __global__ void csr_matvec_kernel(
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


double *d_a;
int* d_colidx,
int* d_rowstr;

double *d_x;
double *d_y;
double *d_partial_sum;
double* h_partial_sum;


void alloc_vectors_gpu(int n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads; // Ajusta o número de blocos dinamicamente
    int allthreads = threads*blocks;

    CUDA_CHECK(cudaMalloc((void**)&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_partial_sum, blocks * sizeof(double)));

    h_partial_sum = (double*) malloc(blocks * sizeof(double));

}

void free_vectors_gpu() {

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_partial_sum));
    free(h_partial_sum);
}

int alloc = 0;

// Função wrapper para ser chamada do Rust
void dot_product_gpu(const double* x, 
                     const double* y, 
                     double* result, 
                     int n) {

    int threads = 256;
    int blocks = (n + threads - 1) / threads; // Ajusta o número de blocos dinamicamente
    int allthreads = threads*blocks;

    if (threads & (threads - 1)) {
        fprintf(stderr, "Erro: o número de threads por bloco deve ser uma potência de 2.\n");
        exit(EXIT_FAILURE);
    }
 
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice));

    dot_product_kernel<<<blocks, threads>>>(d_x, d_y, d_partial_sum, n, allthreads);
  
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel

    CUDA_CHECK(cudaMemcpy(h_partial_sum, d_partial_sum, blocks * sizeof(double), cudaMemcpyDeviceToHost));
 
    *result = 0.0;
    for (int i = 0; i < blocks; i++) {
        *result += h_partial_sum[i];
    }
}

void launch_csr_matvec_mul(
    const int* h_colidx,
    const int* h_rowstr,
    const double* h_a,
    const double* h_x,
    double* h_y,
    int nnz,
    int num_rows,
    int x_len
) {
    // 1. Alocar memória no device
    double *d_a, *d_x, *d_y;
    int *d_colidx, *d_rowstr;

    cudaMalloc(&d_a, nnz * sizeof(double));
    cudaMalloc(&d_colidx, nnz * sizeof(int));
    cudaMalloc(&d_rowstr, (num_rows + 1) * sizeof(int));
    cudaMalloc(&d_x, x_len * sizeof(double));
    cudaMalloc(&d_y, num_rows * sizeof(double));

    // 2. Copiar dados do host para o device
    cudaMemcpy(d_a, h_a, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colidx, h_colidx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowstr, h_rowstr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, x_len * sizeof(double), cudaMemcpyHostToDevice);

    // 3. Configurar execução do kernel
    int blockSize = 256;
    int gridSize = (num_rows + blockSize - 1) / blockSize;

    // 4. Chamar o kernel
    csr_matvec_kernel<<<gridSize, blockSize>>>(
        d_a, d_colidx, d_rowstr, d_x, d_y, num_rows
    );

    // Sincronizar GPU (garante conclusão)
    cudaDeviceSynchronize();

    // 5. Copiar resultado de volta para o host
    cudaMemcpy(h_y, d_y, num_rows * sizeof(double), cudaMemcpyDeviceToHost);

    // 6. Liberar memória no device
    cudaFree(d_a);
    cudaFree(d_colidx);
    cudaFree(d_rowstr);
    cudaFree(d_x);
    cudaFree(d_y);
}

}
