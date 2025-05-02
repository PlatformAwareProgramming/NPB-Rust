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

    __global__ void init_x_gpu(double* x, int n) {

        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

        if(thread_id < n) {
            x[thread_id] = 1.0;
        }
    }

    __global__ void init_conj_grad_gpu(double* x, double* q, double* z, double* r, double* p, int n) {

        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

        if(thread_id < n) {
            q[thread_id] = 0;
            z[thread_id] = 0;
            r[thread_id] = x[thread_id];
            p[thread_id] = r[thread_id];
        }
    }
// Kernel CUDA para multiplicar os vetores
    __global__ void dot_product_kernel(const double* x, const double* y, double* partial_sum, int n) {
        __shared__ double share_data[256]; // Cache compartilhado para redução
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

    __global__ void scalarvecmul1_gpu(double alpha, const double* x, double* y, int n) {
        
        int i = threadIdx.x + blockIdx.x * blockDim.x;

        if(i < n) { 
            y[i] = x[i] + alpha*y[i];
        }
    }

    __global__ void scalarvecmul2_gpu(double alpha, const double* x, double* y, int n) {
        
        int i = threadIdx.x + blockIdx.x * blockDim.x;

        if(i < n) { 
            y[i] += alpha*x[i];
        }
    }

    __global__ void norm_gpu(const double* x, const double* y, double* partial_sum, int n) {

        __shared__ double share_data[256]; // Cache compartilhado para redução
        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        int local_id = threadIdx.x;

        share_data[local_id] = 0.0;

        if(thread_id >= n) { return; }

        { 
            double d;
            d = x[thread_id] - y[thread_id]; 
            share_data[threadIdx.x] = d * d;
        }

        __syncthreads();
        for(int i = blockDim.x/2; i>0; i>>=1) {
            if(local_id < i) { share_data[local_id] += share_data[local_id + i]; }
            __syncthreads();
        }

        if(local_id == 0) {
            partial_sum[blockIdx.x] = share_data[0]; 
        }
    }

    __global__ void update_x_gpu(double norm_temp2, const double* z, double* x, int n) {

        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

        if(thread_id < n) {
            x[thread_id] = norm_temp2 * z[thread_id];
        }
    }


int *d_colidx_;
int *d_rowstr_;
double *d_aa;
double *d_yy;
double *d_xx;
double *d_partial_sum;
double* h_partial_sum;

int *d_colidx;
void alloc_colidx_gpu(int** x, int m) {
    CUDA_CHECK(cudaMalloc((void**)x, m * sizeof(double)));
    d_colidx = *x;
}

int *d_rowstr;
void alloc_rowstr_gpu(int** x, int m) {
    CUDA_CHECK(cudaMalloc((void**)x, m * sizeof(double)));
    d_rowstr = *x;
}

double *d_a;
void alloc_a_gpu(double** x, int m) {
    CUDA_CHECK(cudaMalloc((void**)x, m * sizeof(double)));
    d_a = *x;
}

double *d_x;
void alloc_x_gpu(double** x, int m) {
    CUDA_CHECK(cudaMalloc((void**)x, m * sizeof(double)));
    d_x = *x;
}

double *d_z;
void alloc_z_gpu(double** x, int m) {
    CUDA_CHECK(cudaMalloc((void**)x, m * sizeof(double)));
    d_z = *x;
}

double *d_p;
void alloc_p_gpu(double** x, int m) {
    CUDA_CHECK(cudaMalloc((void**)x, m * sizeof(double)));
    d_p = *x;
}

double *d_q;
void alloc_q_gpu(double** x, int m) {
    CUDA_CHECK(cudaMalloc((void**)x, m * sizeof(double)));
    d_q = *x;
}

double *d_r;
void alloc_r_gpu(double** x, int m) {
    CUDA_CHECK(cudaMalloc((void**)x, m * sizeof(double)));
    d_r = *x;
}

void alloc_vectors_gpu(int m, int n) {

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize; // Ajusta o número de blocos dinamicamente

    CUDA_CHECK(cudaMalloc((void**)&d_xx, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_yy, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_partial_sum, gridSize * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_aa, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_colidx_, m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rowstr_, (n+1) * sizeof(int)));

    h_partial_sum = (double*) malloc(gridSize * sizeof(double));
}

void free_vectors_gpu() {

    CUDA_CHECK(cudaFree(d_colidx_));
    CUDA_CHECK(cudaFree(d_rowstr_));
    CUDA_CHECK(cudaFree(d_aa));
    CUDA_CHECK(cudaFree(d_xx));
    CUDA_CHECK(cudaFree(d_yy));
    CUDA_CHECK(cudaFree(d_partial_sum));
    
    free(h_partial_sum);
}

void launch_init_x_gpu(double* x, int n)
{
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize; // Ajusta o número de blocos dinamicamente

    if (blockSize & (blockSize - 1)) {
        fprintf(stderr, "Erro: o número de threads por bloco deve ser uma potência de 2.\n");
        exit(EXIT_FAILURE);
    }
 
    init_x_gpu<<<gridSize, blockSize>>>(x, n);
  
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel

}


void launch_init_conj_grad_gpu(double* x, double* q, double* z, double* r, double* p, int n)
{
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize; // Ajusta o número de blocos dinamicamente

    if (blockSize & (blockSize - 1)) {
        fprintf(stderr, "Erro: o número de threads por bloco deve ser uma potência de 2.\n");
        exit(EXIT_FAILURE);
    }
 
    init_conj_grad_gpu<<<gridSize, blockSize>>>(x, q, z, r, p, n);
  
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel

}

// Função wrapper para ser chamada do Rust
void dot_product_gpu(const double* x, 
                     const double* y, 
                     double* result, 
                     int n) {

    int blockSize = 256;
    int GridSize = (n + blockSize - 1) / blockSize; // Ajusta o número de blocos dinamicamente

    if (blockSize & (blockSize - 1)) {
        fprintf(stderr, "Erro: o número de threads por bloco deve ser uma potência de 2.\n");
        exit(EXIT_FAILURE);
    }
 
    CUDA_CHECK(cudaMemcpy(d_xx, x, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_yy, y, n * sizeof(double), cudaMemcpyHostToDevice));

    dot_product_kernel<<<GridSize, blockSize>>>(d_xx, d_yy, d_partial_sum, n);
  
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel

    CUDA_CHECK(cudaMemcpy(h_partial_sum, d_partial_sum, GridSize * sizeof(double), cudaMemcpyDeviceToHost));
 
    *result = 0.0;
    for (int i = 0; i < GridSize; i++) {
        *result += h_partial_sum[i];
    }
}

void move_a_to_device_gpu(const int* h_colidx, const int* h_rowstr, const double* h_a, int nnz, int num_rows) {
    CUDA_CHECK(cudaMemcpy(d_aa, h_a, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colidx_, h_colidx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowstr_, h_rowstr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
}

void launch_csr_matvec_mul(
    const double* h_a,
    const int* h_colidx,
    const int* h_rowstr,
    const double* h_x,
    double* h_y,
    int nnz,
    int num_rows,
    int x_len
) {
    // Alocar memória na GPU

    // Transferências de memória: host -> device (somente leitura)
    CUDA_CHECK(cudaMemcpy(d_aa, h_a, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colidx_, h_colidx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowstr_, h_rowstr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xx, h_x, x_len * sizeof(double), cudaMemcpyHostToDevice));

    // Configuração do kernel
    int blockSize = 256;
    int gridSize = (num_rows + blockSize - 1) / blockSize;

    csr_matvec_kernel<<<gridSize, blockSize>>>(
        d_aa, d_colidx_, d_rowstr_, d_xx, d_yy, num_rows
    );

    // Sincronizar GPU (garante conclusão)
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel

    // Transferência de resultado: device -> host
    CUDA_CHECK(cudaMemcpy(h_y, d_yy, num_rows * sizeof(double), cudaMemcpyDeviceToHost));

    // Liberar memória na GPU
 }

 void launch_scalarvecmul1_gpu(
    const double alpha, 
    const double* x, 
    double* y, 
    int n) {

        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize; // Ajusta o número de blocos dinamicamente
    
        if (blockSize & (blockSize - 1)) {
            fprintf(stderr, "Erro: o número de threads por bloco deve ser uma potência de 2.\n");
            exit(EXIT_FAILURE);
        }
     
        CUDA_CHECK(cudaMemcpy(d_xx, x, n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_yy, y, n * sizeof(double), cudaMemcpyHostToDevice));
    
        scalarvecmul1_gpu<<<gridSize, blockSize>>>(alpha, d_xx, d_yy, n);
      
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel
    
        CUDA_CHECK(cudaMemcpy(y, d_yy, n * sizeof(double), cudaMemcpyDeviceToHost));     
 }

 void launch_scalarvecmul2_gpu(
    const double alpha, 
    const double* x, 
    double* y, 
    int n) {

        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize; // Ajusta o número de blocos dinamicamente
    
        if (blockSize & (blockSize - 1)) {
            fprintf(stderr, "Erro: o número de threads por bloco deve ser uma potência de 2.\n");
            exit(EXIT_FAILURE);
        }
     
        CUDA_CHECK(cudaMemcpy(d_xx, x, n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_yy, y, n * sizeof(double), cudaMemcpyHostToDevice));
    
        scalarvecmul2_gpu<<<gridSize, blockSize>>>(alpha, d_xx, d_yy, n);
      
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel
    
        CUDA_CHECK(cudaMemcpy(y, d_yy, n * sizeof(double), cudaMemcpyDeviceToHost));        

 }

void launch_norm_gpu(const double* x, 
                     const double* y, 
                     double* result, 
                     int n) {

    int blockSize = 256;
    int GridSize = (n + blockSize - 1) / blockSize; // Ajusta o número de blocos dinamicamente

    if (blockSize & (blockSize - 1)) {
        fprintf(stderr, "Erro: o número de threads por bloco deve ser uma potência de 2.\n");
        exit(EXIT_FAILURE);
    }
 
    CUDA_CHECK(cudaMemcpy(d_xx, x, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_yy, y, n * sizeof(double), cudaMemcpyHostToDevice));

    norm_gpu<<<GridSize, blockSize>>>(d_xx, d_yy, d_partial_sum, n);
  
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel

    CUDA_CHECK(cudaMemcpy(h_partial_sum, d_partial_sum, GridSize * sizeof(double), cudaMemcpyDeviceToHost));
 
    *result = 0.0;
    for (int i = 0; i < GridSize; i++) {
        *result += h_partial_sum[i];
    }
}

void launch_update_x_gpu(double norm_temp2, const double* z, double* x, int n)
{
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize; // Ajusta o número de blocos dinamicamente

    if (blockSize & (blockSize - 1)) {
        fprintf(stderr, "Erro: o número de threads por bloco deve ser uma potência de 2.\n");
        exit(EXIT_FAILURE);
    }
 
    update_x_gpu<<<gridSize, blockSize>>>(norm_temp2, z, x, n);
  
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel
}

}
