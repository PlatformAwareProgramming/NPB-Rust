


    use crate::class::ClassParams;
    use crate::cg_state::*;

    use platform_aware::{platformaware};
    use crate::common::randdp::randlc;

    pub const AMULT: f64 = 1220703125.0;

    const fn icnvrt(x: f64, ipwr2: i32) -> i32 {
        return (x * ipwr2 as f64) as i32;
    }

    pub trait Alloc {
        fn create_sparse_matrix(self:Self, tran: f64) -> SparseMatrix;
        fn makea(self:Self, n: i32, nz: usize, a: &mut [f64], colidx: &mut [i32], rowstr: &mut [i32], arow: &mut [i32], acol: &mut [i32], aelt: &mut [f64], iv: &mut [i32], tran: &mut f64, FIRSTROW: i32, LASTROW: i32);
        fn sparse(self:Self, a: &mut [f64], colidx: &mut [i32], rowstr: &mut [i32], n: i32, nz: usize, arow: &mut [i32], acol: &mut [i32], aelt: &mut [f64], nzloc: &mut [i32], rcond: f64, FIRSTROW: i32, LASTROW: i32);
        fn sprnvc(self:Self, n: i32, nz: i32, nn1: i32, v: &mut [f64], iv: &mut [i32], tran: &mut f64);
        fn vecset(self:Self, v: &mut [f64], iv: &mut [i32], nzv: &mut i32, i: i32, val: f64);
        fn alloc_a_h(self:Self) -> Vec<f64>;
        fn alloc_a_d(self:Self) -> Vec<f64>;
        fn alloc_colidx_h(self:Self) -> Vec<i32>;
        fn alloc_colidx_d(self:Self) -> Vec<i32>;
        fn alloc_rowstr_h(self:Self) -> Vec<i32> ;
        fn alloc_rowstr_d(self:Self) -> Vec<i32>; 
        fn alloc_x(self:Self) -> Vec<f64>;
        fn alloc_z(self:Self) -> Vec<f64>;
        fn alloc_p(self:Self) -> Vec<f64>;
        fn alloc_q(self:Self) -> Vec<f64>;
        fn alloc_r(self:Self) -> Vec<f64>;
        fn alloc_iv  (self:Self) -> Vec<i32>;
        fn alloc_arow(self:Self) -> Vec<i32>;
        fn alloc_acol(self:Self) -> Vec<i32>;
        fn alloc_aelt(self:Self) -> Vec<f64>;
        fn freevectors();
    }

    impl Alloc for ClassParams {
        fn create_sparse_matrix(self:Self, mut tran: f64) -> SparseMatrix {
            
            let naa: i32 = self.NA;
            let nzz: usize = self.NZ();

            let mut colidx_d: Vec<i32> = self.alloc_colidx_d();
            let     colidx_h: Vec<i32> = self.alloc_colidx_h();
            let mut colidx: Vec<i32> = colidx_h;
            let mut rowstr_d: Vec<i32> = self.alloc_rowstr_d();
            let     rowstr_h: Vec<i32> = self.alloc_rowstr_h();
            let mut rowstr: Vec<i32> = rowstr_h;
            let     a_h: Vec<f64> = self.alloc_a_h();
            let mut a_d: Vec<f64> = self.alloc_a_d();
            let mut a: Vec<f64> = a_h;

            let mut arow: Vec<i32> = self.alloc_arow();
            let mut acol: Vec<i32> = self.alloc_acol();
            let mut aelt: Vec<f64> = self.alloc_aelt();
            let mut iv: Vec<i32> = self.alloc_iv();

            let FIRSTROW: i32 = 0;
            let LASTROW: i32 = self.NA - 1;
            let FIRSTCOL: i32 = 0;
            let LASTCOL: i32 = self.NA - 1;

            self.makea(
                    naa,
                    nzz,
                    &mut a[..],
                    &mut colidx[..],
                    &mut rowstr[..],
                    &mut arow[..],
                    &mut acol[..],
                    &mut aelt[..],
                    &mut iv[..],
                    &mut tran,
                    FIRSTROW,
                    LASTROW
            );


            (&rowstr[0..(LASTROW - FIRSTROW + 1) as usize])
            .into_iter()
            .zip(&rowstr[1..(LASTROW - FIRSTROW + 2) as usize])
            .for_each(|(j, j1)| {
                for k in *j..*j1 {
                    colidx[k as usize] -= FIRSTCOL;
                }
            });

            alloc::move_a_to_device(&colidx[..], &rowstr[..], &a[..], &mut colidx_d[..], &mut rowstr_d[..], &mut a_d[..]);

            SparseMatrix {
                a:a_d,
                colidx:
                colidx_d,
                rowstr:rowstr_d, 
                FIRSTROW:FIRSTROW, 
                LASTROW:LASTROW, 
                FIRSTCOL:FIRSTCOL, 
                LASTCOL:LASTCOL
            }

        }

        fn makea(self:Self,
                n: i32,
                nz: usize,
                a: &mut [f64],
                colidx: &mut [i32],
                rowstr: &mut [i32],
                arow: &mut [i32],
                acol: &mut [i32],
                aelt: &mut [f64],
                iv: &mut [i32],
                tran: &mut f64,
                FIRSTROW: i32,
                LASTROW: i32
        ) {

            let (mut nzv, mut nn1): (i32, i32);
            let mut ivc = vec![0; self.NONZER as usize + 1];
            let mut vc = vec![0.0; self.NONZER as usize + 1];

            /*
            * --------------------------------------------------------------------
            * nonzer is approximately  (int(sqrt(nnza /n)));
            * --------------------------------------------------------------------
            * nn1 is the smallest power of two not less than n
            * --------------------------------------------------------------------
            */
            nn1 = 1;
            while nn1 < n {
                nn1 *= 2;
            }

            /*
            * -------------------------------------------------------------------
            * generate nonzero positions and save for the use in sparse
            * -------------------------------------------------------------------
            */
            for iouter in 0..n {
                nzv = self.NONZER;
                self.sprnvc(n, nzv, nn1, &mut vc, &mut ivc, tran);
                self.vecset(&mut vc, &mut ivc, &mut nzv, iouter + 1, 0.5);
                arow[iouter as usize] = nzv;

                for ivelt in 0..nzv {
                    acol[(iouter * (self.NONZER + 1) + ivelt) as usize] = ivc[ivelt as usize] - 1;
                    aelt[(iouter * (self.NONZER + 1) + ivelt) as usize] = vc[ivelt as usize];
                }
            }

            /*
            * ---------------------------------------------------------------------
            * ... make the sparse matrix from list of elements with duplicates
            * (iv is used as  workspace)
            * ---------------------------------------------------------------------
            */
            self.sparse(a, colidx, rowstr, n, nz, &mut arow[..], &mut acol[..], &mut aelt[..], &mut iv[..], 0.1, FIRSTROW, LASTROW);
        }

        fn sparse(self:Self,
            a: &mut [f64],
            colidx: &mut [i32],
            rowstr: &mut [i32],
            n: i32,
            nz: usize,
            arow: &mut [i32],
            acol: &mut [i32],
            aelt: &mut [f64],
            nzloc: &mut [i32],
            rcond: f64,
            FIRSTROW: i32,
            LASTROW: i32
        ) {
        /*
        * ---------------------------------------------------
        * generate a sparse matrix from a list of
        * [col, row, element] tri
        * ---------------------------------------------------
        */
            let (mut j, mut j1, mut j2, mut nza, mut jcol, mut k_aux): (i32, i32, i32, i32, i32, i32);
            let (mut size, mut scale, ratio, mut va): (f64, f64, f64, f64);
            let mut goto_40: bool;
            k_aux = -1;

            /*
            * --------------------------------------------------------------------
            * how many rows of result
            * --------------------------------------------------------------------
            */
            let nrows: usize = (LASTROW - FIRSTROW + 1) as usize;

            /*
            * --------------------------------------------------------------------
            * ...count the number of triples in each row
            * --------------------------------------------------------------------
            */
            rowstr[0..(nrows + 1) as usize].fill(0);

            for i in 0..n {
                for nza in 0..arow[i as usize] {
                    j = acol[(i * (self.NONZER + 1) + nza) as usize] + 1;
                    rowstr[j as usize] += arow[i as usize];
                }
            }
            rowstr[0] = 0;
            for j in 1..nrows + 1 {
                rowstr[j] += rowstr[j - 1]
            }
            nza = rowstr[nrows] - 1;

            /*
            * ---------------------------------------------------------------------
            * ... rowstr(j) now is the location of the first nonzero
            * of row j of a
            * ---------------------------------------------------------------------
            */
            if nza as usize > nz {
                print!("Space for matrix elements exceeded in sparse\n");
                print!("nza, nzmax = {}, {}\n", nza, nz);
                std::process::exit(0);
            }

            /*
            * ---------------------------------------------------------------------
            * ... preload data pages
            * ---------------------------------------------------------------------
            */
            (&rowstr[0..nrows])
                .into_iter()
                .zip(&rowstr[1..nrows + 1])
                .zip(&mut nzloc[0..nrows])
                .for_each(|((j, j1), nzloc)| {
                    for k in *j..*j1 {
                        a[k as usize] = 0.0;
                        colidx[k as usize] = -1;
                    }
                    *nzloc = 0;
                });

            /*
            * ---------------------------------------------------------------------
            * ... generate actual values by summing duplicates
            * ---------------------------------------------------------------------
            */
            size = 1.0;
            ratio = f64::powf(rcond, 1.0 / n as f64);
            for i in 0..n {
                for nza in 0..arow[i as usize] {
                    j = acol[(i * (self.NONZER + 1) + nza) as usize];

                    scale = size * aelt[(i * (self.NONZER + 1) + nza) as usize];
                    for nzrow in 0..arow[i as usize] {
                        jcol = acol[(i * (self.NONZER + 1) + nzrow) as usize];
                        va = aelt[(i * (self.NONZER + 1) + nzrow) as usize] * scale;

                        /*
                        * --------------------------------------------------------------------
                        * ... add the identity * rcond to the generated matrix to bound
                        * the smallest eigenvalue from below by rcond
                        * --------------------------------------------------------------------
                        */
                        if jcol == j && j == i {
                            va = va + rcond - self.SHIFT;
                        }

                        goto_40 = false;
                        for k in rowstr[j as usize]..rowstr[(j + 1) as usize] {
                            if colidx[k as usize] > jcol {
                                /*
                                * ----------------------------------------------------------------
                                * ... insert colidx here orderly
                                * ----------------------------------------------------------------
                                */
                                for kk in (k..rowstr[j as usize + 1] - 1).rev() {
                                    if colidx[kk as usize] > -1 {
                                        a[(kk + 1) as usize] = a[kk as usize];
                                        colidx[(kk + 1) as usize] = colidx[kk as usize];
                                    }
                                }
                                colidx[k as usize] = jcol;
                                a[k as usize] = 0.0;
                                goto_40 = true;
                                k_aux = k;
                                break;
                            } else if colidx[k as usize] == -1 {
                                colidx[k as usize] = jcol;
                                goto_40 = true;
                                k_aux = k;
                                break;
                            } else if colidx[k as usize] == jcol {
                                /*
                                * --------------------------------------------------------------
                                * ... mark the duplicated entry
                                * -------------------------------------------------------------
                                */
                                nzloc[j as usize] = nzloc[j as usize] + 1;
                                goto_40 = true;
                                k_aux = k;
                                break;
                            }
                        }
                        if !goto_40 {
                            print!("internal error in sparse: i={}\n", i);
                            std::process::exit(0);
                        }
                        a[k_aux as usize] += va;
                    }
                }
                size *= ratio;
            }

            /*
            * ---------------------------------------------------------------------
            * ... remove empty entries and generate final results
            * ---------------------------------------------------------------------
            */
            for j in 1..nrows {
                nzloc[j as usize] += nzloc[(j - 1) as usize];
            }

            for j in 0..nrows {
                j1 = if j > 0 { rowstr[j] - nzloc[j - 1] } else { 0 };
                j2 = rowstr[j + 1] - nzloc[j];
                nza = rowstr[j];
                for k in j1..j2 {
                    a[k as usize] = a[nza as usize];
                    colidx[k as usize] = colidx[nza as usize];
                    nza += 1;
                }
            }
            for j in 1..nrows + 1 {
                rowstr[j] -= nzloc[j - 1];
            }
        }

        /*
        * ---------------------------------------------------------------------
        * generate a sparse n-vector (v, iv)
        * having nzv nonzeros
        *
        * mark(i) is set to 1 if position i is nonzero.
        * mark is all zero on entry and is reset to all zero before exit
        * this corrects a performance bug found by John G. Lewis, caused by
        * reinitialization of mark on every one of the n calls to sprnvc
        * ---------------------------------------------------------------------
        */
        fn sprnvc(self:Self, n: i32, nz: i32, nn1: i32, v: &mut [f64], iv: &mut [i32], tran: &mut f64) {
            let (mut nzv, mut i): (i32, i32);
            let (mut vecelt, mut vecloc): (f64, f64);

            nzv = 0;
            while nzv < nz {
                vecelt = randlc(tran, AMULT);

                /*
                * --------------------------------------------------------------------
                * generate an integer between 1 and n in a portable manner
                * --------------------------------------------------------------------
                */
                vecloc = randlc(tran, AMULT);
                i = icnvrt(vecloc, nn1) + 1;
                if i > n {
                    continue;
                }

                /*
                * --------------------------------------------------------------------
                * was this integer generated already?
                * --------------------------------------------------------------------
                */
                if iv[0..nzv as usize].iter().any(|v| *v == i) {
                    continue;
                }
                v[nzv as usize] = vecelt;
                iv[nzv as usize] = i;
                nzv += 1;
            }
        }

        /*
        * --------------------------------------------------------------------
        * set ith element of sparse vector (v, iv) with
        * nzv nonzeros to val
        * --------------------------------------------------------------------
        */
        fn vecset(self:Self, v: &mut [f64], iv: &mut [i32], nzv: &mut i32, i: i32, val: f64) {
            let mut set: bool = false;

            (&iv[0..*nzv as usize])
                .into_iter()
                .zip(&mut v[0..*nzv as usize])
                .for_each(|(iv, v)| {
                    if *iv == i {
                        *v = val;
                        set = true;
                    }
                });
            if !set {
                v[*nzv as usize] = val;
                iv[*nzv as usize] = i;
                *nzv += 1;
            }
        }

        fn alloc_a_h(self:Self) -> Vec<f64> { alloc::alloc_a_h(self.NZ()) }
        fn alloc_a_d(self:Self) -> Vec<f64> { alloc::alloc_a_d(self.NZ()) }
        fn alloc_colidx_h(self:Self) -> Vec<i32> { alloc::alloc_colidx_h(self.NZ()) }
        fn alloc_colidx_d(self:Self) -> Vec<i32> { alloc::alloc_colidx_d(self.NZ()) }
        fn alloc_rowstr_h(self:Self) -> Vec<i32>  { alloc::alloc_rowstr_h(self.NA) }
        fn alloc_rowstr_d(self:Self) -> Vec<i32>  { alloc::alloc_rowstr_d(self.NA) }
        fn alloc_x(self:Self) -> Vec<f64> { alloc::alloc_x(self.NA) }
        fn alloc_z(self:Self) -> Vec<f64> { alloc::alloc_z(self.NA) }
        fn alloc_p(self:Self) -> Vec<f64> { alloc::alloc_p(self.NA) }
        fn alloc_q(self:Self) -> Vec<f64> { alloc::alloc_q(self.NA) }
        fn alloc_r(self:Self) -> Vec<f64> { alloc::alloc_r(self.NA) }
        fn alloc_iv  (self:Self) -> Vec<i32> { alloc_iv(self.NA) }
        fn alloc_arow(self:Self) -> Vec<i32> {  alloc_arow(self.NA) }
        fn alloc_acol(self:Self) -> Vec<i32> {  alloc_acol(self.NAZ()) }
        fn alloc_aelt(self:Self) -> Vec<f64> {  alloc_aelt(self.NAZ()) }
        
        fn freevectors() {
            alloc::freevectors();
        }
    }

    fn alloc_iv(NA:i32) -> Vec<i32> { vec![0; NA as usize] }
    fn alloc_arow(NA:i32) -> Vec<i32> { vec![0; NA as usize] }
    fn alloc_acol(NAZ:i32) -> Vec<i32> { vec![0; NAZ as usize] }
    fn alloc_aelt(NAZ:i32) -> Vec<f64> { vec![0.0; NAZ as usize] }



#[platformaware]
pub mod alloc {

    use platform_aware_nvidia::*;
    use platform_aware_features::*;

    use std::ffi::c_double;
    use std::os::raw::c_int;

    unsafe extern "C" {
        
        // auxiliary kernels
        fn move_a_to_device_gpu (colidx: *const c_int, rowstr: *const c_int, a: *const c_double, nnz: c_int, num_rows: c_int);
        fn alloc_colidx_gpu(out_ptr: *mut *mut c_int, m:i32);
        fn alloc_rowstr_gpu(out_ptr: *mut *mut c_int, m:i32);
        fn alloc_a_gpu(out_ptr: *mut *mut c_double, m:i32);
        fn alloc_x_gpu(out_ptr: *mut *mut c_double, m:i32);
        fn alloc_p_gpu(out_ptr: *mut *mut c_double, m:i32);
        fn alloc_q_gpu(out_ptr: *mut *mut c_double, m:i32);
        fn alloc_r_gpu(out_ptr: *mut *mut c_double, m:i32);
        fn alloc_z_gpu(out_ptr: *mut *mut c_double, m:i32);
        fn free_vectors_gpu();
     }
     
    pub fn alloc_a_h(NZ:usize) -> Vec<f64> { vec![0.0; NZ] }

    #[kernelversion]
    pub fn alloc_a_d(NZ:usize) -> Vec<f64> { vec![0.0; NZ] }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    pub fn alloc_a_d(NZ:usize) -> Vec<f64> { 
        let mut ptr: *mut f64 = std::ptr::null_mut();
        unsafe { alloc_a_gpu(&mut ptr, NZ as i32) };
        unsafe { Vec::from_raw_parts(ptr, NZ, NZ) }
    }

    pub fn alloc_colidx_h(NZ:usize) -> Vec<i32> { vec![0; NZ] }
    
    #[kernelversion]
    pub fn alloc_colidx_d(NZ:usize) -> Vec<i32> { vec![0; NZ] }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    pub fn alloc_colidx_d(NZ:usize) -> Vec<i32> { 
        let mut ptr: *mut i32 = std::ptr::null_mut();
        unsafe { alloc_colidx_gpu(&mut ptr, NZ as i32) };
        unsafe { Vec::from_raw_parts(ptr, NZ, NZ) }
    }

    pub fn alloc_rowstr_h(NA:i32) -> Vec<i32>  { vec![0; (NA + 1) as usize] }
    
    #[kernelversion]
    pub fn alloc_rowstr_d(NA:i32) -> Vec<i32>  { vec![0; (NA + 1) as usize] }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    pub fn alloc_rowstr_d(NA:i32) -> Vec<i32>  { 
        let mut ptr: *mut i32 = std::ptr::null_mut();
        unsafe { alloc_rowstr_gpu(&mut ptr, NA + 1) };
        unsafe { Vec::from_raw_parts(ptr, (NA + 1) as usize, (NA + 1) as usize) }
    }

    #[kernelversion]
    pub fn alloc_x(NA:i32) -> Vec<f64> { vec![1.0; NA as usize + 2] }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    pub fn alloc_x(NA:i32) -> Vec<f64> { println!("alloc_x");
        let mut ptr: *mut f64 = std::ptr::null_mut();
        unsafe { alloc_x_gpu(&mut ptr, NA + 2) };
        unsafe { Vec::from_raw_parts(ptr, (NA + 2) as usize, (NA + 2) as usize) }
    }

    #[kernelversion]
    pub fn alloc_z(NA:i32) -> Vec<f64> { vec![0.0; NA as usize + 2] }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    pub fn alloc_z(NA:i32) -> Vec<f64> { println!("alloc_z");
        let mut ptr: *mut f64 = std::ptr::null_mut();
        unsafe { alloc_z_gpu(&mut ptr, NA + 2) };
        unsafe { Vec::from_raw_parts(ptr, (NA + 2) as usize, (NA + 2) as usize) }
    }

    #[kernelversion]
    pub fn alloc_p(NA:i32) -> Vec<f64> { vec![0.0; NA as usize + 2] }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    pub fn alloc_p(NA:i32) -> Vec<f64> { println!("alloc_p");
        let mut ptr: *mut f64 = std::ptr::null_mut();
        unsafe { alloc_p_gpu(&mut ptr, NA + 2) };
        unsafe { Vec::from_raw_parts(ptr, (NA + 2) as usize, (NA + 2) as usize) }
    }

    #[kernelversion]
    pub fn alloc_q(NA:i32) -> Vec<f64> { vec![0.0; NA as usize + 2] }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    pub fn alloc_q(NA:i32) -> Vec<f64> { println!("alloc_q");
        let mut ptr: *mut f64 = std::ptr::null_mut();
        unsafe { alloc_q_gpu(&mut ptr, NA + 2) };
        unsafe { Vec::from_raw_parts(ptr, (NA + 2) as usize, (NA + 2) as usize) }
    }

    #[kernelversion]
    pub fn alloc_r(NA:i32) -> Vec<f64> { vec![0.0; NA as usize + 2] }
 
    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    pub fn alloc_r(NA:i32) -> Vec<f64> { println!("alloc_r");
        let mut ptr: *mut f64 = std::ptr::null_mut();
        unsafe { alloc_r_gpu(&mut ptr, NA + 2) };
        unsafe { Vec::from_raw_parts(ptr, (NA + 2) as usize, (NA + 2) as usize) }
    }

    #[kernelversion]
    pub fn freevectors() {}

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    pub fn freevectors() { unsafe { free_vectors_gpu() } }

    #[kernelversion]
    pub fn move_a_to_device(colidx_h: &[i32],  rowstr_h: &[i32], a_h: &[f64], 
                        colidx_d: &mut [i32],  rowstr_d: &mut [i32], a_d: &mut [f64]) { 
                            colidx_d.copy_from_slice(colidx_h);
                            rowstr_d.copy_from_slice(rowstr_h);
                            a_d.copy_from_slice(a_h);
                        }

    #[kernelversion(acc_count=(AtLeast{val:1}), acc_backend=CUDA)]
    pub fn move_a_to_device(colidx_h: &[i32],  rowstr_h: &[i32], a_h: &[f64], 
                        colidx_d: &mut [i32],  rowstr_d: &mut [i32], a_d: &mut [f64]) {
        let nnz = a_h.len() as i32;
        let num_rows = rowstr_h.len() as i32;
        unsafe { move_a_to_device_gpu (colidx_h.as_ptr(), rowstr_h.as_ptr(), a_h.as_ptr(), nnz, num_rows) }
    }


}

pub mod teste {

}