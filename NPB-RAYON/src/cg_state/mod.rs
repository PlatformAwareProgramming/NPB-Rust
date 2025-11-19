    use crate::class::ClassParams;
    use crate::cg_alloc::Alloc;
    use crate::class::ClassValue;
    
    use platform_aware::CURRENT_FEATURES;
     
    pub struct SparseMatrix {
           pub colidx: Vec<i32>,
           pub rowstr: Vec<i32>,
           pub a: Vec<f64>,
           pub FIRSTROW: i32,
           pub LASTROW: i32,
           pub FIRSTCOL: i32,
           pub LASTCOL: i32
    }
    
    pub struct CGstate{
           pub params:ClassParams,
           pub a: SparseMatrix,
           pub x: Vec<f64>,
           pub z: Vec<f64>,
           pub p: Vec<f64>,
           pub q: Vec<f64>,
           pub r: Vec<f64>
    }

    impl CGstate {
        pub fn new(tran: f64) -> Self {
            let params = getparams();
            Self {
                params: params,
                a: params.create_sparse_matrix(tran),
                x: params.alloc_x(),
                z: params.alloc_z(),
                p: params.alloc_p(),
                q: params.alloc_q(),
                r: params.alloc_r()
            }
        }
    }

    fn getparams() -> ClassParams
    {
        let mut m = CURRENT_FEATURES.lock().unwrap();   
        let actual_class = m.get("problemclass");
        actual_class.expect("classU").string().params()
    }
