

use std::{sync::Arc, env, error::Error, fs, path::Path};
use ctor::ctor;
use platform_aware_features::{insert_parameter, insert_feature, PlatformParameter, Feature, CURRENT_FEATURES};

pub struct ProblemClass;

impl Feature for ProblemClass { 
    fn is_top(self:&Self) -> bool { true }
    fn string(self:&Self) -> &'static str { "ProblemClass" } 
    fn feature_class(self:&Self) -> std::option::Option<PlatformParameter> { Some("problemclass".to_string()) } 
}

pub struct ClassS;

impl Feature for ClassS {
    fn string(&self) -> &'static str { "ClassS" }
    fn supertype(&self) -> Option<Box<dyn Feature>> { Some(Box::new(ClassW)) }
    fn feature_class(&self) -> Option<PlatformParameter> { Some("problemclass".to_string()) }
}

pub struct ClassW;

impl Feature for ClassW {
    fn string(&self) -> &'static str { "ClassW" }
    fn supertype(&self) -> Option<Box<dyn Feature>> { Some(Box::new(ClassA)) }
    fn feature_class(&self) -> Option<PlatformParameter> { Some("problemclass".to_string()) }
}

pub struct ClassA;

impl Feature for ClassA {
    fn string(&self) -> &'static str { "ClassA" }
    fn supertype(&self) -> Option<Box<dyn Feature>> { Some(Box::new(ClassB)) }
    fn feature_class(&self) -> Option<PlatformParameter> { Some("problemclass".to_string()) }
}

pub struct ClassB;

impl Feature for ClassB {
    fn string(&self) -> &'static str { "ClassB" }
    fn supertype(&self) -> Option<Box<dyn Feature>> { Some(Box::new(ClassC)) }
    fn feature_class(&self) -> Option<PlatformParameter> { Some("problemclass".to_string()) }
}

pub struct ClassC;

impl Feature for ClassC {
    fn string(&self) -> &'static str { "ClassC" }
    fn supertype(&self) -> Option<Box<dyn Feature>> { Some(Box::new(ClassD)) }
    fn feature_class(&self) -> Option<PlatformParameter> { Some("problemclass".to_string()) }
}

pub struct ClassD;

impl Feature for ClassD {
    fn string(&self) -> &'static str { "ClassD" }
    fn supertype(&self) -> Option<Box<dyn Feature>> { Some(Box::new(ClassE)) }
    fn feature_class(&self) -> Option<PlatformParameter> { Some("problemclass".to_string()) }
}

pub struct ClassE;

impl Feature for ClassE {
    fn string(&self) -> &'static str { "ClassE" }
    fn supertype(&self) -> Option<Box<dyn Feature>> { Some(Box::new(ClassF)) }
    fn feature_class(&self) -> Option<PlatformParameter> { Some("problemclass".to_string()) }
}

pub struct ClassF;

impl Feature for ClassF {
    fn string(&self) -> &'static str { "ClassF" }
    fn supertype(&self) -> Option<Box<dyn Feature>> { Some(Box::new(ProblemClass)) }
    fn feature_class(&self) -> Option<PlatformParameter> { Some("problemclass".to_string()) }
}



#[ctor]
fn module_initializer() {

        insert_feature(Arc::new(ProblemClass));
        insert_feature(Arc::new(ClassS));
        insert_feature(Arc::new(ClassW));
        insert_feature(Arc::new(ClassA));
        insert_feature(Arc::new(ClassB));
        insert_feature(Arc::new(ClassC));
        insert_feature(Arc::new(ClassD));
        insert_feature(Arc::new(ClassE));
        insert_feature(Arc::new(ClassF));

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
        if self == "ClassS"      { ClassParams { CLASS: 'S', NA: 1400,    NONZER: 7,  NITER: 15,  SHIFT:   10.0, ZETA_VERIFY: 8.5971775078648 } }
        else if self == "ClassW" { ClassParams { CLASS: 'W', NA: 7000,    NONZER: 8,  NITER: 15,  SHIFT:   12.0, ZETA_VERIFY: 10.362595087124 } }
        else if self == "ClassA" { ClassParams { CLASS: 'A', NA: 14000,   NONZER: 11, NITER: 15,  SHIFT:   20.0, ZETA_VERIFY: 17.130235054029 } }
        else if self == "ClassB" { ClassParams { CLASS: 'B', NA: 75000,   NONZER: 13, NITER: 75,  SHIFT:   60.0, ZETA_VERIFY: 22.712745482631 } }
        else if self == "ClassC" { ClassParams { CLASS: 'C', NA: 150000,  NONZER: 15, NITER: 75,  SHIFT:  110.0, ZETA_VERIFY: 28.973605592845 } }
        else if self == "ClassD" { ClassParams { CLASS: 'D', NA: 1500000, NONZER: 21, NITER: 100, SHIFT:  500.0, ZETA_VERIFY: 52.514532105794 } }
        else if self == "ClassE" { ClassParams { CLASS: 'E', NA: 9000000, NONZER: 26, NITER: 100, SHIFT: 1500.0, ZETA_VERIFY: 77.522164599383 } }
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