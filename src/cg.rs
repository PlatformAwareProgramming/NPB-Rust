mod common;
mod class;
mod cg_alloc;
mod cg_state;
mod cg_kernels;

use crate::cg_alloc::*;
use crate::cg_state::*;
use crate::cg_kernels::*;
use crate::cg_kernels::compute::*;
use crate::cg_kernels::aux;

use std::env;
use crate::class::ClassParams;
use crate::common::print_results::*;
use crate::common::randdp::*;
use crate::common::timers::*;

use platform_aware::platformaware;


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

    cgst.cg(timers);
    
}


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
    fn announce_platform(self: &mut Self);
    fn cg(self: &mut Self, timers: Timer);
    fn conj_grad(self:&mut Self, rnorm: &mut f64,  timers: &mut Timer);
    fn init_x(self: &mut Self);
    fn update_x(self: &mut Self, norm_temp2: f64/*, z: &[f64], x: &mut Vec<f64>*/);
    fn init_conj_grad(self: &mut Self);
}



impl ConjGrad for CGstate {

    fn announce_platform(self: &mut Self) { aux::announce_platform(); }

    fn cg(self: &mut Self, mut timers: Timer) {

        let mut zeta: f64 = 0.0;
        let mut rnorm: f64 = 0.0;
        let (mut norm_temp1, mut norm_temp2): (f64, f64);
        let (mut t, mops, mut tmax): (f64, f64, f64);
        let verified: i8;       

        print!("\n\n NAS Parallel Benchmarks 4.1 Parallel Rust version with Rayon - CG Benchmark\n\n");
        
        let NA = self.params.NA;
        let NITER = self.params.NITER;
        let NONZER = self.params.NONZER;
        let SHIFT = self.params.SHIFT;
        let CLASS = self.params.CLASS;
        let ZETA_VERIFY = self.params.ZETA_VERIFY;

        print!(" Size: {:>11}\n", self.params.NA);
        print!(" Iterations: {:>5}\n", self.params.NITER);

        let mut kdata = KParams { FIRSTCOL: self.a.FIRSTCOL, LASTCOL: self.a.LASTCOL, NA: self.params.NA};

        self.announce_platform();

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

        
            norm_temp2 = kdata.vecvecmul(&self.z[..], &self.z[..]);
            norm_temp2 = 1.0 / f64::sqrt(norm_temp2);

            /* normalize z to obtain x */
            self.update_x(norm_temp2);
        } /* end of do one iteration untimed */

        /* set starting vector to (1, 1, .... 1) */
        self.init_x();

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
            self.update_x(norm_temp2);

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

        let cgitmax: i32 = 25;
        let (mut d, mut rho, mut rho0, mut alpha, mut beta): (f64, f64, f64, f64, f64);

        /* initialize the CG algorithm */
        self.init_conj_grad();

        let colidx = &self.a.colidx[..];
        let rowstr = &self.a.rowstr[..];
        let x = &mut self.x[..];
        let z = &mut self.z[..];
        let a = &mut self.a.a[..];
        let p = &mut self.p[..];
        let q = &mut self.q[..];
        let r = &mut self.r[..];

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
        
    fn init_x(self: &mut Self) {
        let NA = self.params.NA;
        aux::init_x(&mut self.x[..], NA);
    }
    
    fn update_x(self: &mut Self, norm_temp2: f64) { 
        let COL_SIZE = self.a.LASTCOL - self.a.FIRSTCOL + 1;
        aux::update_x(norm_temp2, &self.z[..], &mut self.x, COL_SIZE);
    }

    fn init_conj_grad(self: &mut Self) {
        let COL_SIZE = self.a.LASTCOL - self.a.FIRSTCOL + 1;
        aux::init_conj_grad(&mut self.x[..], &mut self.q[..], &mut self.z[..], &mut self.r[..], &mut self.p[..], COL_SIZE);
    }    

}




