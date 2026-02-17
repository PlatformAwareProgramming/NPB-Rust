
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