
#[kernelversion(cpu_core_count=(AtLeast{val:2}))]
pub fn init_x(x: &mut [f64], NA:i32) {
    x[0..NA as usize + 1].par_iter_mut().for_each(|x| *x = 1.0);
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

#[kernelversion(cpu_core_count=(AtLeast{val:2}))]
pub fn update_x(norm_temp2: f64, z: &[f64], x: &mut Vec<f64>, COL_SIZE:i32) {
    z.par_iter()
        .map(|z| z * norm_temp2)
        .collect_into_vec(x);
}