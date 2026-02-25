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
        
        // y = y + alpha * x  (single thread)
        #[kernelversion]
        fn scalarvecmul2(self: &mut Self, alpha:f64, x: &[f64], y: &mut [f64]) {
            let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
            for j in 0..COL_SIZE as usize {
                y[j] += alpha * x[j];
            }

        }
        
        // y = x + alpha * y  (single thread)
        #[kernelversion]
        fn scalarvecmul1(self: &mut Self, alpha:f64, x: &[f64], y: &mut [f64]) {
            let COL_SIZE = self.LASTCOL - self.FIRSTCOL + 1;
            for j in 0..COL_SIZE as usize {
                y[j] = x[j] + alpha * y[j];
            }
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

    }
    platform!(multicore_compute_kernels);
    platform!(cuda_compute_kernels);

}


#[platformaware]
pub mod aux {

    use rayon::prelude::*;
    use rayon::ThreadPoolBuilder;
    use std::env;
    use crate::class::*;
    use platform_aware_features::*;

    #[kernelversion]
    pub fn init_x(x: &mut [f64], NA:i32) {
        x[0..NA as usize + 1].fill(1.0);
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

    #[kernelversion]
    pub fn announce_platform() { println!("=======> DEFAULT (serial)") }
          
    #[kernelversion]
    pub fn update_x(norm_temp2: f64, z: &[f64], x: &mut Vec<f64>, COL_SIZE:i32) {
        for j in 0..COL_SIZE as usize {
            x[j] = norm_temp2 * z[j];
        }
    }
 
    platform!(multicore_aux_kernels);
    platform!(cuda_aux_kernels);

} // mod cg