mod common;
mod class;
mod cg_alloc;
mod cg_state;

use crate::cg_alloc::*;
use crate::cg_state::*;
use std::env;
use crate::class::ClassParams;
use crate::common::print_results::*;
use crate::common::randdp::*;
use crate::common::timers::*;

use crate::cg_alloc::Alloc;
use crate::cg_state::*;


fn main() {

    let mut timers = Timer::new();
    timers.clear(T_INIT);
    timers.clear(T_BENCH);
    timers.clear(T_CONJ_GRAD);
    timers.clear(T_VECVECMUL);
    timers.clear(T_MATVECMUL);

    timers.start(T_INIT);

    let mut tran: f64;

    /* initialize random number generator */
    tran = 314159265.0;
    randlc(&mut tran, AMULT);

    let mut cgst = CGstate::new(tran);

    cgst.cg(tran, timers);
    
}

use platform_aware::{platformaware};

#[cfg(safe = "true")]
pub const UNSAFE: bool = false;
#[cfg(not(safe = "true"))]
pub const UNSAFE: bool = true;

#[cfg(timers = "true")]
pub const TIMERS: bool = true;
#[cfg(not(timers = "true"))]
pub const TIMERS: bool = false;

/*
* ---------------------------------------------------------------------
* note: please observe that in the routine conj_grad three
* implementations of the sparse matrix-vector multiply have
* been supplied. the default matrix-vector multiply is not
* loop unrolled. the alternate implementations are unrolled
* to a depth of 2 and unrolled to a depth of 8. please
* experiment with these to find the fastest for your particular
* architecture. if reporting timing results, any of these three may
* be used without penalty.
* ---------------------------------------------------------------------
*/

pub const T_INIT: usize = 0;
pub const T_BENCH: usize = 1;
pub const T_CONJ_GRAD: usize = 2;
pub const T_LAST: usize = 3;
pub const T_VECVECMUL: usize = 4;
pub const T_MATVECMUL: usize = 5;

pub const EPSILON: f64 = 1.0e-10;

/* cg */

trait ConjGrad {
    fn cg(self: &mut Self, tran:f64, timers: Timer);
    fn conj_grad(self:&mut Self, rnorm: &mut f64,  timers: &mut Timer);
}

trait Kernels {
    fn init_x(self: &mut Self, x: &mut [f64]);
    fn update_x(self: &mut Self, norm_temp2: f64, z: &[f64], x: &mut Vec<f64>);
    fn init_conj_grad(self: &mut Self, x: &mut [f64], q: &mut [f64], z: &mut [f64], r: &mut [f64], p: &mut [f64]);
    fn announce_platform(self: &mut Self);
    fn matvecmul(self: &mut Self, colidx: &[i32], rowstr: &[i32], a: &[f64], x: &[f64], y: &mut[f64]);
    fn vecvecmul(self: &mut Self, x: &[f64], y: &[f64]) -> f64;
    fn scalarvecmul2(self: &mut Self, alpha:f64, x: &[f64], y: &mut [f64]);
    fn scalarvecmul1(self: &mut Self, alpha:f64, x: &[f64], y: &mut [f64]);
    fn norm(self: &mut Self, x: &[f64], y: &[f64]) -> f64;
    }

impl ConjGrad for CGstate {

    fn cg(self: &mut Self, mut tran:f64, mut timers: Timer) {

    let mut zeta: f64 = 0.0;
    let mut rnorm: f64 = 0.0;
    let (mut norm_temp1, mut norm_temp2): (f64, f64);
    let (mut t, mops, mut tmax): (f64, f64, f64);
    let verified: i8;       

    print!("\n\n NAS Parallel Benchmarks 4.1 Parallel Rust version with Rayon - CG Benchmark\n\n");
    
    let COL_SIZE = self.a.LASTCOL - self.a.FIRSTCOL + 1;
    let NA = self.params.NA;
    let NITER = self.params.NITER;
    let NONZER = self.params.NONZER;
    let SHIFT = self.params.SHIFT;
    let CLASS = self.params.CLASS;
    let ZETA_VERIFY = self.params.ZETA_VERIFY;

    print!(" Size: {:>11}\n", self.params.NA);
    print!(" Iterations: {:>5}\n", self.params.NITER);

    let mut kdata = KParams { FIRSTCOL: self.a.FIRSTCOL, LASTCOL: self.a.LASTCOL, NA: self.params.NA};

    kdata.announce_platform();

    /*
    * -------------------------------------------------------------------
    * ---->
    * do one iteration untimed to init all code and data page tables
    * ----> (then reinit, start timing, to niter its)
    * -------------------------------------------------------------------*/
    for _ in 0..1 {
        /* the call to the conjugate gradient routine */
        self.conj_grad(&mut rnorm, &mut timers);

        /*
        * --------------------------------------------------------------------
        * zeta = shift + 1/(x.z)
        * so, first: (x.z)
        * also, find norm of z
        * so, first: (z.z)
        * --------------------------------------------------------------------
        */

        let z = &mut self.z[..];
        let x = &mut self.x;

        norm_temp2 = kdata.vecvecmul(z, z);
        norm_temp2 = 1.0 / f64::sqrt(norm_temp2);

        /* normalize z to obtain x */
        kdata.update_x(norm_temp2, z, &mut self.x);
    } /* end of do one iteration untimed */

    /* set starting vector to (1, 1, .... 1) */
    kdata.init_x(&mut self.x[..]);

    timers.stop(T_INIT);

    print!(
        " Initialization time = {:>15.3} seconds\n",
        timers.read(T_INIT).as_secs_f64()
    );

    timers.start(T_BENCH);

    timers.clear(T_VECVECMUL);
    timers.clear(T_MATVECMUL);

    /*
    * --------------------------------------------------------------------
    * ---->
    * main iteration for inverse power method
    * ---->
    * --------------------------------------------------------------------
    */
    for it in 1..NITER + 1 {
        /* the call to the conjugate gradient routine */
        if TIMERS {
            timers.start(T_CONJ_GRAD);
        }

        self.conj_grad(&mut rnorm, &mut timers);

        if TIMERS {
            timers.stop(T_CONJ_GRAD);
        }

        /*
        * --------------------------------------------------------------------
        * zeta = shift + 1/(x.z)
        * so, first: (x.z)
        * also, find norm of z
        * so, first: (z.z)
        * --------------------------------------------------------------------
        */

        norm_temp1 = kdata.vecvecmul(&self.x[..], &self.z[..]);
        norm_temp2 = kdata.vecvecmul(&self.z[..], &self.z[..]);

        norm_temp2 = 1.0 / f64::sqrt(norm_temp2);

        zeta = SHIFT + 1.0 / norm_temp1;
        if it == 1 {
            println!("\n   iteration             ||r||                 zeta");
        }
        println!("   {:>5}       {:>20.14e}{:>20.13e}", it, rnorm, zeta);

        /* normalize z to obtain x */
        kdata.update_x(norm_temp2, &self.z[..], &mut self.x);

    } /* end of main iter inv pow meth */

    timers.stop(T_BENCH);

    /*
    * --------------------------------------------------------------------
    * end of timed section
    * --------------------------------------------------------------------
    */
    t = timers.read(T_BENCH).as_secs_f64();

    print!(" Benchmark completed\n");

    if CLASS != 'U' {
        let err = f64::abs(zeta - ZETA_VERIFY) / ZETA_VERIFY;
        if err <= EPSILON {
            verified = 1;
            print!(" VERIFICATION SUCCESSFUL {}\n", EPSILON);
            print!(" Zeta is    {:+20.13e}\n", zeta);
            print!(" Error is   {:+20.13e}\n", err);
        } else {
            verified = 0;
            print!(" VERIFICATION FAILED{}\n", EPSILON);
            print!(" Zeta                {:+20.13e}\n", zeta);
            print!(" The correct zeta is {:+20.13e}\n", ZETA_VERIFY);
        }
    } else {
        verified = 0;
        print!(" Problem size unknown\n");
        print!(" NO VERIFICATION PERFORMED\n");
    }
    if t != 0.0 {
        mops = ((NITER << 1) * NA) as f64
            * (3.0
                + (NONZER * (NONZER + 1)) as f64
                + 25.0 * (5.0 + (NONZER * (NONZER + 1)) as f64)
                + 3.0)
            / t
            / 1000000.0;
    } else {
        mops = 0.0;
    }

    let info = PrintInfo {
        name: String::from("CG"),
        class: CLASS.to_string(),
        size: (NA as usize, 0, 0),
        num_iter: NITER,
        time: t,
        mops,
        operation: String::from("Floating Point"),
        verified,
        num_threads: rayon::current_num_threads() as u32,
        //uns: UNSAFE
    };
    printer(info);


    println!("VECVECMUL timing: {}", timers.read(T_VECVECMUL).as_secs_f64());
    println!("MATVECMUL timing: {}", timers.read(T_MATVECMUL).as_secs_f64());

    /*
    * ---------------------------------------------------------------------
    * more timers
    * ---------------------------------------------------------------------
    */
    if TIMERS {
        let mut t_names: Vec<String> = vec![String::new(); 3];
        t_names[T_INIT] = String::from("init");
        t_names[T_BENCH] = String::from("benchmk");
        t_names[T_CONJ_GRAD] = String::from("conjgd");

        tmax = timers.read(T_BENCH).as_secs_f64();
        if tmax == 0.0 {
            tmax = 1.0;
        }
        print!("  SECTION   Time (secs)\n");
        for i in 0..T_LAST {
            t = timers.read(i).as_secs_f64();
            if i == T_INIT {
                print!("  {:>8}:{:>9.3}\n", t_names[i], t);
            } else {
                print!(
                    "  {:>8}:{:>9.3}  ({:>6.2}%)\n",
                    t_names[i],
                    t,
                    t * 100.0 / tmax
                );
                if i == T_CONJ_GRAD {
                    t = tmax - t;
                    print!(
                        "    --> {:>8}:{:>9.3}  ({:>6.2}%)\n",
                        "rest",
                        t,
                        t * 100.0 / tmax
                    );
                }
            }
        }
    }

    ClassParams::freevectors();


}

fn conj_grad(self:&mut Self, rnorm: &mut f64,  timers: &mut Timer) {

        let mut kdata = KParams { FIRSTCOL: self.a.FIRSTCOL, LASTCOL: self.a.LASTCOL, NA: self.params.NA};

        let colidx = &self.a.colidx[..];
        let rowstr = &self.a.rowstr[..];
        let x = &mut self.x[..];
        let z = &mut self.z[..];
        let a = &mut self.a.a[..];
        let p = &mut self.p[..];
        let q = &mut self.q[..];
        let r = &mut self.r[..];

        let cgitmax: i32 = 25;
        let (mut d, mut rho, mut rho0, mut alpha, mut beta): (f64, f64, f64, f64, f64);

        /* initialize the CG algorithm */
        kdata.init_conj_grad(&mut x[..], &mut q[..], &mut z[..], &mut r[..], &mut p[..]);

        /*
        * --------------------------------------------------------------------
        * rho = r.r
        * now, obtain the norm of r: First, sum squares of r elements locally...
        * --------------------------------------------------------------------
        */

        timers.start(T_VECVECMUL);
        rho = kdata.vecvecmul(r, r);
        timers.stop(T_VECVECMUL);
        
        /* the conj grad iteration loop */
        for _ in 1..cgitmax {
            /*
            * ---------------------------------------------------------------------
            * q = A.p
            * the partition submatrix-vector multiply: use workspace w
            * ---------------------------------------------------------------------
            *
            * note: this version of the multiply is actually (slightly: maybe %5)
            * faster on the sp2 on 16 nodes than is the unrolled-by-2 version
            * below. on the Cray t3d, the reverse is TRUE, i.e., the
            * unrolled-by-two version is some 10% faster.
            * the unrolled-by-8 version below is significantly faster
            * on the Cray t3d - overall speed of code is 1.5 times faster.
            */
            timers.start(T_MATVECMUL);
            kdata.matvecmul(colidx,rowstr, a, p, q);
            timers.stop(T_MATVECMUL);
    
            /*
            * --------------------------------------------------------------------
            * obtain p.q
            * --------------------------------------------------------------------
            */

            timers.start(T_VECVECMUL);
            d = kdata.vecvecmul(p, q);
            timers.stop(T_VECVECMUL);

            /*
            * --------------------------------------------------------------------
            * obtain alpha = rho / (p.q)
            * -------------------------------------------------------------------
            */
            alpha = rho / d;

            /*
            * --------------------------------------------------------------------
            * save a temporary of rho
            * --------------------------------------------------------------------
            */
            rho0 = rho;

            /*
            * ---------------------------------------------------------------------
            * obtain z = z + alpha*p
            * and    r = r - alpha*q
            * ---------------------------------------------------------------------
            */
        
            kdata.scalarvecmul2(alpha, p, z);
            kdata.scalarvecmul2(-alpha, q, r);

            timers.start(T_VECVECMUL);
            rho = kdata.vecvecmul(r, r);
            timers.stop(T_VECVECMUL);

            /*
            * ---------------------------------------------------------------------
            * obtain beta
            * ---------------------------------------------------------------------
            */
            beta = rho / rho0;

            /*
            * ---------------------------------------------------------------------
            * p = r + beta*p
            * ---------------------------------------------------------------------
            */
            kdata.scalarvecmul1(beta, r, p);
            
        } /* end of do cgit=1, cgitmax */

        /*
        * ---------------------------------------------------------------------
        * compute residual norm explicitly: ||r|| = ||x - A.z||
        * first, form A.z
        * the partition submatrix-vector multiply
        * ---------------------------------------------------------------------
        */
        timers.start(T_MATVECMUL);
        kdata.matvecmul(colidx, rowstr, a, z, r);
        timers.stop(T_MATVECMUL);

        /*
        * ---------------------------------------------------------------------
        * at this point, r contains A.z
        * ---------------------------------------------------------------------
        */

        *rnorm = kdata.norm(x, r) 

    }
        
}

struct KParams {
    FIRSTCOL: i32,
    LASTCOL: i32,
    NA:i32
}

impl Kernels for KParams {
    fn init_x(self: &mut Self, x: &mut [f64]) {
        let NA = self.NA;
        cg::init_x(x, NA);
    }
    
    fn update_x(self: &mut Self, norm_temp2: f64, z: &[f64], x: &mut Vec<f64>) { 
        let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
        cg::update_x(norm_temp2, z, x, COL_SIZE);
    }

    fn init_conj_grad(self: &mut Self, x: &mut [f64], q: &mut [f64], z: &mut [f64], r: &mut [f64], p: &mut [f64]) {
        let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
        cg::init_conj_grad(x, q, z, r, p, COL_SIZE);
    }
    
    fn announce_platform(self: &mut Self) { cg::announce_platform(); }
    
    fn matvecmul(self: &mut Self, colidx: &[i32], rowstr: &[i32], a: &[f64], x: &[f64], y: &mut[f64]) {
        let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
        let NA = self.NA;
        cg::matvecmul(colidx, rowstr, a, x, y, COL_SIZE, NA);
    }

    fn vecvecmul(self: &mut Self, x: &[f64], y: &[f64]) -> f64 {
        let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
        cg::vecvecmul(x, y, COL_SIZE)
    }
    
    fn scalarvecmul2(self: &mut Self, alpha:f64, x: &[f64], y: &mut [f64]) {
        let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
        cg::scalarvecmul2(alpha, x, y, COL_SIZE);
    }
    
    fn scalarvecmul1(self: &mut Self, alpha:f64, x: &[f64], y: &mut [f64]) {
        let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
        cg::scalarvecmul1(alpha, x, y, COL_SIZE);
    }
    
    fn norm(self: &mut Self, x: &[f64], y: &[f64]) -> f64 {
        let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
        cg::norm(x, y, COL_SIZE)
    }

    
}


#[platformaware(announce_platform,
                init_x, 
                update_x, 
                init_conj_grad,
                matvecmul, 
                vecvecmul, 
                scalarvecmul1, 
                scalarvecmul2, 
                norm
 )]
mod cg {

    use rayon::prelude::*;
    use rayon::ThreadPoolBuilder;
    use std::env;
    use crate::class::*;

    use platform_aware_nvidia::*;

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
                                                problemclass = ClassC)]  // 384.81 -- ...
    pub fn announce_platform() { println!("=======> CUDA 3") }

    // y = a * x (sequential)
    #[kernelversion]
    pub fn matvecmul(
        colidx: &[i32],
        rowstr: &[i32], 
        a: &[f64],
        x: &[f64],
        y: &mut[f64],
        COL_SIZE:i32,
        NA:i32
    ) 
    {
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
        });    }

    // y = a * x (multithread)
    #[kernelversion(cpu_core_count=(AtLeast{val:2}))]
    pub fn matvecmul(
        colidx: &[i32],
        rowstr: &[i32], 
        a: &[f64],
        x: &[f64],
        y: &mut[f64],
        COL_SIZE:i32,
        NA:i32
    ) {
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
    pub fn matvecmul(
        colidx: &[i32],
        rowstr: &[i32], 
        a: &[f64],
        x: &[f64],
        y: &mut[f64],
        COL_SIZE:i32,
        NA:i32
    ) {
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
    pub fn matvecmul(
        colidx: &[i32],
        rowstr: &[i32], 
        a: &[f64],
        x: &[f64],
        y: &mut[f64],
        COL_SIZE:i32,
        NA:i32
    ) {
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
    pub fn matvecmul(
        colidx: &[i32],
        rowstr: &[i32], 
        a: &[f64],
        x: &[f64],
        y: &mut[f64],
        COL_SIZE:i32,
        NA:i32
    ) {
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
    pub fn matvecmul(
        colidx: &[i32],
        rowstr: &[i32], 
        a: &[f64],
        x: &[f64],
        y: &mut[f64],
        COL_SIZE:i32,
        NA:i32
    ) {
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
    pub fn vecvecmul(x: &[f64], y: &[f64], COL_SIZE:i32) -> f64 {
        // println!("vecvecmul - 0");
        (&x[0..COL_SIZE as usize])
        .into_iter()
        .zip(&y[0..COL_SIZE as usize])
        .map(|(x, y)| *x * *y)
        .sum()
    }

    // x * y (multithread)
    #[kernelversion(cpu_core_count=(AtLeast{val:2}))]
    pub fn vecvecmul(x: &[f64], y: &[f64], COL_SIZE:i32) -> f64 {
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
    pub fn vecvecmul(x: &[f64], y: &[f64], COL_SIZE:i32) -> f64 {
        // println!("vecvecmul - 2");
        let mut result: f64 = 0.0;    
        unsafe { launch_vecvecmul_gpu(x.as_ptr(), y.as_ptr(), &mut result as *mut f64, COL_SIZE as i32); }
        result    
    }

    // y = y + alpha * x  (single thread)
    #[kernelversion]
    pub fn scalarvecmul2(alpha:f64, x: &[f64], y: &mut [f64], COL_SIZE:i32) {
        for j in 0..COL_SIZE as usize {
            y[j] += alpha * x[j];
        }

    }

    // y = y + alpha * x   (multithread)
    #[kernelversion(cpu_core_count=(AtLeast{val:2}))]
    pub fn scalarvecmul2(alpha:f64, x: &[f64], y: &mut [f64], COL_SIZE:i32) {
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
    pub fn scalarvecmul2(alpha:f64, x: &[f64], y: &mut [f64], COL_SIZE:i32) {
        unsafe { launch_scalarvecmul2_gpu(alpha, x.as_ptr(), y.as_mut_ptr(), COL_SIZE as i32); }
    }
            
            
    // y = x + alpha * y  (single thread)
    #[kernelversion]
    pub fn scalarvecmul1(alpha:f64, x: &[f64], y: &mut [f64], COL_SIZE:i32) {
        for j in 0..COL_SIZE as usize {
            y[j] = x[j] + alpha * y[j];
        }
    }

    // y = x + alpha * y (multithread)
    #[kernelversion(cpu_core_count=(AtLeast{val:2}))]
    pub fn scalarvecmul1(alpha:f64, x: &[f64], y: &mut [f64], COL_SIZE:i32) {
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
    pub fn scalarvecmul1(alpha:f64, x: &[f64], y: &mut [f64], COL_SIZE:i32) {
        unsafe { launch_scalarvecmul1_gpu(alpha, x.as_ptr(), y.as_mut_ptr(), COL_SIZE as i32); }
    }

    #[kernelversion]
    pub fn norm(x: &[f64], y: &[f64], COL_SIZE:i32) -> f64 {
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
    pub fn norm(x: &[f64], y: &[f64], COL_SIZE:i32) -> f64 {
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
    pub fn norm(x: &[f64], y: &[f64], COL_SIZE:i32) -> f64 {
        let mut sum: f64 = 0.0;    
        unsafe { launch_norm_gpu(x.as_ptr(), y.as_ptr(), &mut sum as *mut f64, COL_SIZE as i32); }    
        f64::sqrt(sum)
    }

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