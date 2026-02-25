

use platform_aware_nvidia::CUDA;
use super::Kernels;

// Invocação da função CUDA
use std::ffi::c_double;
use std::os::raw::c_int;
unsafe extern "C" {
    
    // auxiliary kernels
    fn launch_init_x_gpu(x: *mut c_double, n: c_int);
    fn launch_update_x_gpu(norm_temp2:c_double, z: *const c_double, x: *mut c_double, n:c_int);
    fn launch_init_conj_grad_gpu(x: *mut c_double, q: *mut c_double, z: *mut c_double, r: *mut c_double, p: *mut c_double, n: c_int);

    // computation kernels
    fn launch_vecvecmul_gpu(x: *const c_double, y: *const c_double, result: *mut c_double, n: c_int);
    fn launch_matvecmul_cuda(h_a: *const f64, h_colidx: *const i32, h_rowstr: *const i32, h_x: *const f64, h_y: *mut f64, nnz: i32, num_rows: i32, x_len: i32);
    fn launch_matvecmul_CC35(h_a: *const f64, h_colidx: *const i32, h_rowstr: *const i32, h_x: *const f64, h_y: *mut f64, nnz: i32, num_rows: i32, x_len: i32);
    fn launch_matvecmul_CC70(h_a: *const f64, h_colidx: *const i32, h_rowstr: *const i32, h_x: *const f64, h_y: *mut f64, nnz: i32, num_rows: i32, x_len: i32);
    fn launch_matvecmul_CC60(h_a: *const f64, h_colidx: *const i32, h_rowstr: *const i32, h_x: *const f64, h_y: *mut f64, nnz: i32, num_rows: i32, x_len: i32);
    fn launch_scalarvecmul1_gpu(alpha:f64, x: *const f64, y: *mut f64, size: i32);
    fn launch_scalarvecmul2_gpu(alpha:f64, x: *const f64, y: *mut f64, size: i32);
    fn launch_norm_gpu(x: *const c_double, y: *const c_double, result: *mut c_double, n: c_int);
}

impl Kernels for KParams {

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA, 
                                                acc_cudatoolkit=(AtLeast{val:20}), 
                                                acc_cudacc=(AtLeast{val:13}), 
                                                acc_cudadriver=(AtLeast{val:17713}))]  // 177.13 -- 190.38
    fn matvecmul(self: &mut Self, colidx: &[i32], rowstr: &[i32], a: &[f64], x: &[f64], y: &mut[f64]) {
        let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
        let NA = self.NA;
        // println!("matvecmul - 2");
        let nnz = a.len() as i32;
        let num_rows = y.len() as i32;
        let x_len = x.len() as i32;
        unsafe {
            launch_matvecmul_cuda(
                a.as_ptr(),
                colidx.as_ptr(),
                rowstr.as_ptr(),
                x.as_ptr(),
                y.as_mut_ptr(),
                nnz,
                num_rows,
                x_len,
            );
        }
    }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA, 
                                                acc_cudatoolkit=(AtLeast{val:50}), 
                                                acc_cudacc=(AtLeast{val:35}), 
                                                acc_cudadriver=(AtLeast{val:31937}))]  // 319.37  -- 304.54
    fn matvecmul(self: &mut Self, colidx: &[i32], rowstr: &[i32], a: &[f64], x: &[f64], y: &mut[f64]) {
        let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
        let NA = self.NA;
        // println!("matvecmul - 3");
        let nnz = a.len() as i32;
        let num_rows = y.len() as i32;
        let x_len = x.len() as i32;
        unsafe {
            launch_matvecmul_CC35(
                a.as_ptr(),
                colidx.as_ptr(),
                rowstr.as_ptr(),
                x.as_ptr(),
                y.as_mut_ptr(),
                nnz,
                num_rows,
                x_len,
            );
        }
    }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA, 
                                                acc_cudatoolkit=(AtLeast{val:101}), 
                                                acc_cudacc=(AtLeast{val:60}), 
                                                acc_cudadriver=(AtLeast{val:41839}))]  // 418.39 -- ...
    fn matvecmul(self: &mut Self, colidx: &[i32], rowstr: &[i32], a: &[f64], x: &[f64], y: &mut[f64]) {
        let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
        let NA = self.NA;
        // println!("matvecmul - 4");
        let nnz = a.len() as i32;
        let num_rows = y.len() as i32;
        let x_len = x.len() as i32;

        unsafe {
            launch_matvecmul_CC60(
                a.as_ptr(),
                colidx.as_ptr(),
                rowstr.as_ptr(),
                x.as_ptr(),
                y.as_mut_ptr(),
                nnz,
                num_rows,
                x_len,
            );
        }
    }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA, 
                                                acc_cudatoolkit=(AtLeast{val:90}), 
                                                acc_cudacc=(AtLeast{val:70}), 
                                                acc_cudadriver=(AtLeast{val:38481}))]  // 384.81 -- ...
    fn matvecmul(self: &mut Self, colidx: &[i32], rowstr: &[i32], a: &[f64], x: &[f64], y: &mut[f64]) {
        let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
        let NA = self.NA;
        // println!("matvecmul - 5");
        let nnz = a.len() as i32;
        let num_rows = y.len() as i32;
        let x_len = x.len() as i32;

        unsafe {
            launch_matvecmul_CC70(
                a.as_ptr(),
                colidx.as_ptr(),
                rowstr.as_ptr(),
                x.as_ptr(),
                y.as_mut_ptr(),
                nnz,
                num_rows,
                x_len,
            );
        }
    }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    fn vecvecmul(self: &mut Self, x: &[f64], y: &[f64]) -> f64 {
        let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
        // println!("vecvecmul - 2");
        let mut result: f64 = 0.0;    
        unsafe { launch_vecvecmul_gpu(x.as_ptr(), y.as_ptr(), &mut result as *mut f64, COL_SIZE as i32); }
        result    
    }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    fn scalarvecmul2(self: &mut Self, alpha:f64, x: &[f64], y: &mut [f64]) {
        let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
        unsafe { launch_scalarvecmul2_gpu(alpha, x.as_ptr(), y.as_mut_ptr(), COL_SIZE as i32); }
    }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    fn scalarvecmul1(self: &mut Self, alpha:f64, x: &[f64], y: &mut [f64]) {
        let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
        unsafe { launch_scalarvecmul1_gpu(alpha, x.as_ptr(), y.as_mut_ptr(), COL_SIZE as i32); }
    }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    fn norm(self: &mut Self, x: &[f64], y: &[f64]) -> f64 {
        let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
        let mut sum: f64 = 0.0;    
        unsafe { launch_norm_gpu(x.as_ptr(), y.as_ptr(), &mut sum as *mut f64, COL_SIZE as i32); }    
        f64::sqrt(sum)
    }

}