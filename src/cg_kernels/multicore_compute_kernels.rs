
impl Kernels for KParams {

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
    }
