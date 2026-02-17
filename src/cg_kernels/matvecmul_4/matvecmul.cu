#include "../cgkernels.h"

#define WARP_SIZE 32

#include <cuda_runtime.h>
#include <cusparse.h>

void matvecmul_A100(
    double* d_aa,
    int* d_colidx,
    int* d_rowstr,
    double* d_xx,
    double* d_yy,
    int num_rows,
    int x_len,
    int nnz
) {
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    const double alpha = 1.0;
    const double beta = 0.0;

    // Execute y = alpha * A * x + beta * y
    cusparseDcsrmv(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        num_rows,
        x_len,
        nnz,
        &alpha,
        descrA,
        d_aa,
        d_rowstr,
        d_colidx,
        d_xx,
        &beta,
        d_yy
    );

    cusparseDestroyMatDescr(descrA);
    cusparseDestroy(handle);
}

extern "C" {

void launch_matvecmul_A100(
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

    matvecmul_A100(d_aa, d_colidx, d_rowstr, d_xx, d_yy, num_rows, x_len, nnz);

 }

}
