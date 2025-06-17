#include "../cgkernels.h"

#include <cuda_runtime.h>
#include <cusparse.h>

extern cusparseHandle_t handle;
extern cusparseSpMatDescr_t matA;
extern cusparseDnVecDescr_t vecX;
extern cusparseDnVecDescr_t vecY;
extern void* dBuffer;
extern size_t bufferSize;

void matvecmul_CC60(
    double* d_aa,
    int* d_colidx,
    int* d_rowstr,
    double* d_xx,
    double* d_yy,
    int num_rows,
    int x_len,
    int nnz
) {

    cusparseDnVecSetValues(vecX, d_xx);
    cusparseDnVecSetValues(vecY, d_yy);

    const double alpha = 1.0;
    const double beta = 0.0;

    cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        vecX,
        &beta,
        vecY,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        dBuffer
    );
    
}

extern "C" {

void launch_matvecmul_CC60(
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

    matvecmul_CC60(d_aa, d_colidx, d_rowstr, d_xx, d_yy, num_rows, x_len, nnz);

    CUDA_CHECK(cudaDeviceSynchronize());
   // CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel
 }

}
