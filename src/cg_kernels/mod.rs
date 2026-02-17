use platform_aware::platformaware;

pub trait Kernels {
    fn matvecmul(self: &mut Self, colidx: &[i32], rowstr: &[i32], a: &[f64], x: &[f64], y: &mut[f64]);
    fn vecvecmul(self: &mut Self, x: &[f64], y: &[f64]) -> f64;
    fn scalarvecmul2(self: &mut Self, alpha:f64, x: &[f64], y: &mut [f64]);
    fn scalarvecmul1(self: &mut Self, alpha:f64, x: &[f64], y: &mut [f64]);
    fn norm(self: &mut Self, x: &[f64], y: &[f64]) -> f64;
}

#[platformaware]
pub mod compute {

    pub struct KParams {
        pub FIRSTCOL: i32,
        pub LASTCOL: i32,
        pub NA:i32
    }

    use rayon::prelude::*;
    use rayon::ThreadPoolBuilder;
    use std::env;
    use crate::class::*;
    use platform_aware::AtLeast;

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

        // y = a * x (sequential)
        #[kernelversion]
        fn matvecmul(self: &mut Self, colidx: &[i32], rowstr: &[i32], a: &[f64], x: &[f64], y: &mut[f64]) 
        {
            let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
            let NA = self.NA;
            // println!("matvecmul - 0");
            (&rowstr[0..NA as usize])
            .into_iter()
            .zip(&rowstr[1..NA as usize + 1])
            .zip(&mut y[0..COL_SIZE as usize])
            .for_each(|((j, j1), y)| {
                *y = (&a[*j as usize..*j1 as usize])
                    .into_iter()
                    .zip(&colidx[*j as usize..*j1 as usize])
                    .map(|(a, colidx)| a * x[*colidx as usize])
                    .sum();
            });    
        }

        // y = a * x (multithread)
        #[kernelversion(cpu_core_count=(AtLeast{val:2}))]
        fn matvecmul(self: &mut Self, colidx: &[i32], rowstr: &[i32], a: &[f64], x: &[f64], y: &mut[f64]) {
            let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
            let NA = self.NA;
            // println!("matvecmul - 1");
            (
                &rowstr[0..NA as usize],
                &rowstr[1..NA as usize + 1],
                &mut y[0..COL_SIZE as usize],
            )
                .into_par_iter()
                .for_each(|(j, j1, y)| {
                    *y = (&a[*j as usize..*j1 as usize])
                        .into_iter()
                        .zip(&colidx[*j as usize..*j1 as usize])
                        .map(|(a, colidx)| a * x[*colidx as usize])
                        .sum();
                });
            
        }

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


        // x * y (single thread)
        #[kernelversion]
        fn vecvecmul(self: &mut Self, x: &[f64], y: &[f64]) -> f64 {
             let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
           // println!("vecvecmul - 0");
            (&x[0..COL_SIZE as usize])
            .into_iter()
            .zip(&y[0..COL_SIZE as usize])
            .map(|(x, y)| *x * *y)
            .sum()
        }

        // x * y (multithread)
        #[kernelversion(cpu_core_count=(AtLeast{val:2}))]
        fn vecvecmul(self: &mut Self, x: &[f64], y: &[f64]) -> f64 {
             let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
           // println!("vecvecmul - 1");
            (
                &x[0..COL_SIZE as usize],
                &y[0..COL_SIZE as usize],
            )
                .into_par_iter()
                .map(|(x, y)| *x * *y)
                .sum()
        }

        #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
        fn vecvecmul(self: &mut Self, x: &[f64], y: &[f64]) -> f64 {
            let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
            // println!("vecvecmul - 2");
            let mut result: f64 = 0.0;    
            unsafe { launch_vecvecmul_gpu(x.as_ptr(), y.as_ptr(), &mut result as *mut f64, COL_SIZE as i32); }
            result    
        }
        
        // y = y + alpha * x  (single thread)
        #[kernelversion]
        fn scalarvecmul2(self: &mut Self, alpha:f64, x: &[f64], y: &mut [f64]) {
            let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
            for j in 0..COL_SIZE as usize {
                y[j] += alpha * x[j];
            }

        }

        // y = y + alpha * x   (multithread)
        #[kernelversion(cpu_core_count=(AtLeast{val:2}))]
        fn scalarvecmul2(self: &mut Self, alpha:f64, x: &[f64], y: &mut [f64]) {
            let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
                (
                    &mut y[0..COL_SIZE as usize],
                    &x[0..COL_SIZE as usize],
                )
                    .into_par_iter()
                    .for_each(|(y, x)| {
                        *y += alpha * *x;
                        });
        }

        #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
        fn scalarvecmul2(self: &mut Self, alpha:f64, x: &[f64], y: &mut [f64]) {
            let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
            unsafe { launch_scalarvecmul2_gpu(alpha, x.as_ptr(), y.as_mut_ptr(), COL_SIZE as i32); }
        }
        
        // y = x + alpha * y  (single thread)
        #[kernelversion]
        fn scalarvecmul1(self: &mut Self, alpha:f64, x: &[f64], y: &mut [f64]) {
            let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
            for j in 0..COL_SIZE as usize {
                y[j] = x[j] + alpha * y[j];
            }
        }

        // y = x + alpha * y (multithread)
        #[kernelversion(cpu_core_count=(AtLeast{val:2}))]
        fn scalarvecmul1(self: &mut Self, alpha:f64, x: &[f64], y: &mut [f64]) {
            let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
            (
                &mut y[0..COL_SIZE as usize],
                &x[0..COL_SIZE as usize],
            )
                .into_par_iter()
                .for_each(|(y, x)| {
                    *y = *x + alpha * *y;
                    });
        }

        #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
        fn scalarvecmul1(self: &mut Self, alpha:f64, x: &[f64], y: &mut [f64]) {
            let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
            unsafe { launch_scalarvecmul1_gpu(alpha, x.as_ptr(), y.as_mut_ptr(), COL_SIZE as i32); }
        }
        
        #[kernelversion]
        fn norm(self: &mut Self, x: &[f64], y: &[f64]) -> f64 {
                let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
                let sum = (&x[0..COL_SIZE as usize])
                .into_iter()
                .zip(&y[0..COL_SIZE as usize])
                .map(|(x, y)| {
                    let d = *x - *y;
                    d * d
            })
            .sum();
            f64::sqrt(sum)
        }

        #[kernelversion(cpu_core_count=(AtLeast{val:2}))]
        fn norm(self: &mut Self, x: &[f64], y: &[f64]) -> f64 {
           let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
           let sum = (&x[0..COL_SIZE as usize])
                .into_iter()
                .zip(&y[0..COL_SIZE as usize])
                .map(|(x, y)| {
                    let d = *x - *y;
                    d * d
            })
            .sum();
            f64::sqrt(sum)
        }

        #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
        fn norm(self: &mut Self, x: &[f64], y: &[f64]) -> f64 {
            let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
            let mut sum: f64 = 0.0;    
            unsafe { launch_norm_gpu(x.as_ptr(), y.as_ptr(), &mut sum as *mut f64, COL_SIZE as i32); }    
            f64::sqrt(sum)
        }
    }
}


#[platformaware]
pub mod aux {

    use rayon::prelude::*;
    use rayon::ThreadPoolBuilder;
    use std::env;
    use crate::class::*;
    use platform_aware_features::*;

    use platform_aware_nvidia::*;

    // Invocação da função CUDA
    use std::ffi::c_double;
    use std::os::raw::c_int;
    unsafe extern "C" {
        // auxiliary kernels
        fn launch_init_x_gpu(x: *mut c_double, n: c_int);
        fn launch_update_x_gpu(norm_temp2:c_double, z: *const c_double, x: *mut c_double, n:c_int);
        fn launch_init_conj_grad_gpu(x: *mut c_double, q: *mut c_double, z: *mut c_double, r: *mut c_double, p: *mut c_double, n: c_int);
    }

    #[kernelversion]
    pub fn init_x(x: &mut [f64], NA:i32) {
        x[0..NA as usize + 1].fill(1.0);
    }
    
    #[kernelversion(cpu_core_count=(AtLeast{val:2}))]
    pub fn init_x(x: &mut [f64], NA:i32) {
        x[0..NA as usize + 1].par_iter_mut().for_each(|x| *x = 1.0);
    }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    pub fn init_x(x: &mut [f64], NA:i32) {
        unsafe { launch_init_x_gpu(x.as_mut_ptr(), NA + 1) }
    }


    #[kernelversion]
    pub fn init_conj_grad(x: &mut [f64], q: &mut [f64], z: &mut [f64], r: &mut [f64], p: &mut [f64], COL_SIZE:i32) {
        q.fill(0.0);
        z.fill(0.0);
        (&mut r[..])
            .into_iter()
            .zip(&mut p[..])
            .zip(&x[..])
            .for_each(|((r, p), x)| {
                *r = *x;
                *p = *r;
            });
    }

    #[kernelversion(cpu_core_count=(AtLeast{val:2}))]
    pub fn init_conj_grad(x: &mut [f64], q: &mut [f64], z: &mut [f64], r: &mut [f64], p: &mut [f64], COL_SIZE:i32) {
        (&mut q[..], &mut z[..], &mut r[..], &mut p[..], &x[..])
            .into_par_iter()
            .for_each(|(q, z, r, p, x)| {
                *q = 0.0;
                *z = 0.0;
                *r = *x;
                *p = *r;
            });
    }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    pub fn init_conj_grad(x: &mut [f64], q: &mut [f64], z: &mut [f64], r: &mut [f64], p: &mut [f64], COL_SIZE:i32) {

        unsafe { 
            launch_init_conj_grad_gpu(x.as_mut_ptr(), q.as_mut_ptr(), z.as_mut_ptr(), r.as_mut_ptr(), p.as_mut_ptr(),  COL_SIZE) 
        }
    }



    #[kernelversion]
    pub fn announce_platform() { println!("=======> DEFAULT (serial)") }
    
    #[kernelversion(cpu_core_count=(AtLeast{val:2}))]
    // y = a * x (multithread)
    pub fn announce_platform() { 
        if let Ok(ray_num_threads_str) = env::var("RAY_NUM_THREADS") {
            if let Ok(ray_num_threads) = ray_num_threads_str.parse::<usize>() {
                ThreadPoolBuilder::new()
                    .num_threads(ray_num_threads)
                    .build_global()
                    .unwrap();
            } else {
                ThreadPoolBuilder::new().build_global().unwrap();
            }
        } else {
            ThreadPoolBuilder::new().build_global().unwrap();
        }
        println!("=======> MULTITHREADING (parallel with {} threads)", rayon::current_num_threads()) 
    }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA, 
                                                acc_cudatoolkit=(AtLeast{val:20}), 
                                                acc_cudacc=(AtLeast{val:13}), 
                                                acc_cudadriver=(AtLeast{val:17713}))]  // 177.13 -- 190.38
    pub fn announce_platform() { println!("=======> CUDA default") }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA, 
                                                acc_cudatoolkit=(AtLeast{val:50}), 
                                                acc_cudacc=(AtLeast{val:35}), 
                                                acc_cudadriver=(AtLeast{val:31937}))]  // 319.37  -- 304.54
    pub fn announce_platform() { println!("=======> CUDA 1") }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA, 
                                                acc_cudatoolkit=(AtLeast{val:101}), 
                                                acc_cudacc=(AtLeast{val:60}), 
                                                acc_cudadriver=(AtLeast{val:41839}))]  // 418.39 -- ...
    pub fn announce_platform() { println!("=======> CUDA 2") }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA, 
                                                acc_cudatoolkit=(AtLeast{val:90}), 
                                                acc_cudacc=(AtLeast{val:70}), 
                                                acc_cudadriver=(AtLeast{val:38481}), 
                                                problemclass = Class_C)]  // 384.81 -- ...
    pub fn announce_platform() { println!("=======> CUDA 3") }

            
    #[kernelversion]
    pub fn update_x(norm_temp2: f64, z: &[f64], x: &mut Vec<f64>, COL_SIZE:i32) {
        for j in 0..COL_SIZE as usize {
            x[j] = norm_temp2 * z[j];
        }
    }

    #[kernelversion(cpu_core_count=(AtLeast{val:2}))]
    pub fn update_x(norm_temp2: f64, z: &[f64], x: &mut Vec<f64>, COL_SIZE:i32) {
        z.par_iter()
         .map(|z| z * norm_temp2)
         .collect_into_vec(x);
    }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    pub fn update_x(norm_temp2: f64, z: &[f64], x: &mut Vec<f64>, COL_SIZE:i32) {
        { unsafe { launch_update_x_gpu(norm_temp2, z.as_ptr(), x.as_mut_ptr(), COL_SIZE as i32) } }
    }

} // mod cg