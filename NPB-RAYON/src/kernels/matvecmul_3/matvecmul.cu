#include "../cgkernels.h"

#define WARP_SIZE 32

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
    //cusparseHandle_t handle;
    //cusparseCreate(&handle);

    //cusparseSpMatDescr_t matA;
    //cusparseDnVecDescr_t vecX, vecY;

    //int nnz;
    //cudaMemcpy(&nnz, &d_rowstr[num_rows], sizeof(int), cudaMemcpyDeviceToHost);

    /*cusparseCreateCsr(
        &matA,
        num_rows,
        x_len,
        nnz,
        d_rowstr,
        d_colidx,
        d_aa,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F
    );*/

    cusparseDnVecSetValues(vecX, d_xx);
    cusparseDnVecSetValues(vecY, d_yy);

    const double alpha = 1.0;
    const double beta = 0.0;

 /*   size_t bufferSize = 0;
    void* dBuffer = nullptr;

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
*/
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

    /*cudaFree(dBuffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);*/
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
    CUDA_CHECK(cudaGetLastError()); // Verifica erros no lançamento do kernel
 }

}
