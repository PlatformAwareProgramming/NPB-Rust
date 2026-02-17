#include "../cgkernels.h"

#include <cuda_runtime.h>
#include <cusparse.h>

#define BLOCK_SIZE 4

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
            q[thread_id] = 0.0;
            z[thread_id] = 0.0;
            r[thread_id] = x[thread_id];
            p[thread_id] = r[thread_id];
        }
    }

   __global__ void update_x_gpu(double norm_temp2, const double* z, double* x, int n) {

        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

        if(thread_id < n) {
            x[thread_id] = norm_temp2 * z[thread_id];
        }
    }


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

// cusparseSpMatDescr_t matA;



void launch_init_x_gpu(double* x, int n)
{
    int blockSize = BLOCK_SIZE;
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
    int blockSize = BLOCK_SIZE;
    int gridSize = (n + blockSize - 1) / blockSize; // Ajusta o número de blocos dinamicamente

    if (blockSize & (blockSize - 1)) {
        fprintf(stderr, "Erro: o número de threads por bloco deve ser uma potência de 2.\n");
        exit(EXIT_FAILURE);
    }
 
    init_conj_grad_gpu<<<gridSize, blockSize>>>(x, q, z, r, p, n);
  
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel

}

cusparseHandle_t handle;
cusparseSpMatDescr_t matA;
cusparseDnVecDescr_t vecY;
cusparseDnVecDescr_t vecX;
size_t bufferSize = 0;
void* dBuffer = nullptr;

void move_a_to_device_gpu(const int* h_colidx, const int* h_rowstr, const double* h_a, int nnz, int num_rows) {
    CUDA_CHECK(cudaMemcpy(d_a, h_a, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colidx, h_colidx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowstr, h_rowstr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Inicialização (uma vez)
    cusparseCreate(&handle);

    int x_len = num_rows;

    cusparseCreateCsr(
        &matA,
        num_rows,
        x_len,
        nnz,
        d_rowstr,
        d_colidx,
        d_a,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F
    );

    cusparseCreateDnVec(&vecX, x_len, d_p, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, num_rows, d_q, CUDA_R_64F);

    const double alpha = 1.0;
    const double beta = 0.0;

    cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        vecX,
        &beta,
        vecY,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize
    );
    
    cudaMalloc(&dBuffer, bufferSize);
}

void free_vectors_gpu() {
    cudaFree(dBuffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    CUDA_CHECK(cudaFree(d_colidx));
    CUDA_CHECK(cudaFree(d_rowstr));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_r));
 }

void launch_update_x_gpu(double norm_temp2, const double* z, double* x, int n)
{
    int blockSize = BLOCK_SIZE;
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
