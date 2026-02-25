
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


#[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
pub fn init_x(x: &mut [f64], NA:i32) {
    unsafe { launch_init_x_gpu(x.as_mut_ptr(), NA + 1) }
}


#[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
pub fn init_conj_grad(x: &mut [f64], q: &mut [f64], z: &mut [f64], r: &mut [f64], p: &mut [f64], COL_SIZE:i32) {

    unsafe { 
        launch_init_conj_grad_gpu(x.as_mut_ptr(), q.as_mut_ptr(), z.as_mut_ptr(), r.as_mut_ptr(), p.as_mut_ptr(),  COL_SIZE) 
    }
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


#[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
pub fn update_x(norm_temp2: f64, z: &[f64], x: &mut Vec<f64>, COL_SIZE:i32) {
    { unsafe { launch_update_x_gpu(norm_temp2, z.as_ptr(), x.as_mut_ptr(), COL_SIZE as i32) } }
}
