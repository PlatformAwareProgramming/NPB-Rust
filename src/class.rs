
use std::{sync::Arc, env, error::Error, fs, path::Path};
use ctor::ctor;
use platform_aware_features::{create_feature_hierarchy,insert_parameter, CURRENT_FEATURES};

create_feature_hierarchy! {  register_features ;"problemclass" : None :> ProblemClass :> Class_F :> 
                                                                                         Class_E :> 
                                                                                         Class_D :> 
                                                                                         Class_C :> 
                                                                                         Class_B :> 
                                                                                         Class_A :> 
                                                                                         Class_W :> 
                                                                                         Class_S; }


#[ctor]
fn insert_problem_class() {

        insert_parameter("problemclass".to_string(), Arc::new(ProblemClass));

        println!("+++++++++++ PROBLEM CLASS");
        let mut m0 = CURRENT_FEATURES.lock().unwrap();
        let m = readproblemclass();
   
        let actual_class = m.get("problemclass");
        let params = actual_class.expect("class not found").string().params();
        println!("NA = {}", params.NA);

        m0.extend(m.into_iter());
}




use platform_aware_features::{add_qualifier, PlatformFeatures, readplatform};
use serde::Deserialize;
use toml::to_string_pretty;


#[derive(Deserialize)]
struct Platform {
    pub kernel:Option<KernelParameters>,
}

#[derive(Deserialize)]
struct KernelParameters {
    pub problemclass:Option<String>,
}

pub fn readproblemclass() -> PlatformFeatures
{
    let t: Platform = match readplatform() {
        Ok(t) => t,
        Err(_) => todo!(),
    };

   let mut m: PlatformFeatures = ::std::collections::HashMap::new();

    match t.kernel {
        Some(u) => {
            match u.problemclass { Some(v) => { println!("=================> {}", v); add_qualifier(&mut m, "problemclass".to_string(), v); }, None => {} }
        },
        None => {},
    }

   return m;
}


pub trait ClassValue { fn params(self:Self) -> ClassParams; }

impl ClassValue for &str {
    fn params(self:Self) -> ClassParams {
        if self == "Class_S"      { ClassParams { CLASS: 'S', NA: 1400,    NONZER: 7,  NITER: 15,  SHIFT:   10.0, ZETA_VERIFY: 8.5971775078648 } }
        else if self == "Class_W" { ClassParams { CLASS: 'W', NA: 7000,    NONZER: 8,  NITER: 15,  SHIFT:   12.0, ZETA_VERIFY: 10.362595087124 } }
        else if self == "Class_A" { ClassParams { CLASS: 'A', NA: 14000,   NONZER: 11, NITER: 15,  SHIFT:   20.0, ZETA_VERIFY: 17.130235054029 } }
        else if self == "Class_B" { ClassParams { CLASS: 'B', NA: 75000,   NONZER: 13, NITER: 75,  SHIFT:   60.0, ZETA_VERIFY: 22.712745482631 } }
        else if self == "Class_C" { ClassParams { CLASS: 'C', NA: 150000,  NONZER: 15, NITER: 75,  SHIFT:  110.0, ZETA_VERIFY: 28.973605592845 } }
        else if self == "Class_D" { ClassParams { CLASS: 'D', NA: 1500000, NONZER: 21, NITER: 100, SHIFT:  500.0, ZETA_VERIFY: 52.514532105794 } }
        else if self == "Class_E" { ClassParams { CLASS: 'E', NA: 9000000, NONZER: 26, NITER: 100, SHIFT: 1500.0, ZETA_VERIFY: 77.522164599383 } }
        else                     { ClassParams { CLASS: 'U', NA: 1,       NONZER: 1,  NITER: 1,   SHIFT:    1.0, ZETA_VERIFY:  1.0            } }
    }
}


#[derive(Copy,Clone)]
pub struct ClassParams {
    pub CLASS: char,
    pub NA: i32,
    pub NONZER: i32,
    pub NITER: i32,
    pub SHIFT: f64,
    pub ZETA_VERIFY: f64
}

impl ClassParams {
    pub fn NZ(self:Self) -> usize{ self.NA as usize * (self.NONZER + 1) as usize * (self.NONZER + 1) as usize }
    pub fn NAZ(self:Self) -> i32 { self.NA * (self.NONZER + 1) }
}