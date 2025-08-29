

use std::{sync::Arc, env, error::Error, fs, path::Path};
use ctor::ctor;
use platform_aware_features::{insert_parameter, insert_feature, PlatformParameter, Feature, CURRENT_FEATURES};

pub struct ProblemClass;

impl Feature for ProblemClass { 
    fn is_top(self:&Self) -> bool { true }
    fn string(self:&Self) -> &'static str { "ProblemClass" } 
    fn feature_class(self:&Self) -> std::option::Option<PlatformParameter> { Some("npb_problemclass".to_string()) } 
}

pub struct ClassS;

impl Feature for ClassS {
    fn string(&self) -> &'static str { "ClassS" }
    fn supertype(&self) -> Option<Box<dyn Feature>> { Some(Box::new(ClassW)) }
    fn feature_class(&self) -> Option<PlatformParameter> { Some("npb_problemclass".to_string()) }
}

pub struct ClassW;

impl Feature for ClassW {
    fn string(&self) -> &'static str { "ClassW" }
    fn supertype(&self) -> Option<Box<dyn Feature>> { Some(Box::new(ClassA)) }
    fn feature_class(&self) -> Option<PlatformParameter> { Some("npb_problemclass".to_string()) }
}

pub struct ClassA;

impl Feature for ClassA {
    fn string(&self) -> &'static str { "ClassA" }
    fn supertype(&self) -> Option<Box<dyn Feature>> { Some(Box::new(ClassB)) }
    fn feature_class(&self) -> Option<PlatformParameter> { Some("npb_problemclass".to_string()) }
}

pub struct ClassB;

impl Feature for ClassB {
    fn string(&self) -> &'static str { "ClassB" }
    fn supertype(&self) -> Option<Box<dyn Feature>> { Some(Box::new(ClassC)) }
    fn feature_class(&self) -> Option<PlatformParameter> { Some("npb_problemclass".to_string()) }
}

pub struct ClassC;

impl Feature for ClassC {
    fn string(&self) -> &'static str { "ClassC" }
    fn supertype(&self) -> Option<Box<dyn Feature>> { Some(Box::new(ClassD)) }
    fn feature_class(&self) -> Option<PlatformParameter> { Some("npb_problemclass".to_string()) }
}

pub struct ClassD;

impl Feature for ClassD {
    fn string(&self) -> &'static str { "ClassD" }
    fn supertype(&self) -> Option<Box<dyn Feature>> { Some(Box::new(ClassE)) }
    fn feature_class(&self) -> Option<PlatformParameter> { Some("npb_problemclass".to_string()) }
}

pub struct ClassE;

impl Feature for ClassE {
    fn string(&self) -> &'static str { "ClassE" }
    fn supertype(&self) -> Option<Box<dyn Feature>> { Some(Box::new(ClassF)) }
    fn feature_class(&self) -> Option<PlatformParameter> { Some("npb_problemclass".to_string()) }
}

pub struct ClassF;

impl Feature for ClassF {
    fn string(&self) -> &'static str { "ClassF" }
    fn supertype(&self) -> Option<Box<dyn Feature>> { Some(Box::new(ProblemClass)) }
    fn feature_class(&self) -> Option<PlatformParameter> { Some("npb_problemclass".to_string()) }
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

        insert_parameter("npb_problemclass".to_string(), Arc::new(ProblemClass));

        let mut m0 = CURRENT_FEATURES.lock().unwrap();
        let m = readproblemclass();
   
        m0.extend(m.into_iter());
}




use platform_aware_features::{add_qualifier, PlatformFeatures};
use serde::Deserialize;


#[derive(Deserialize)]
struct Platform {
    pub kernel:Option<KernelParameters>,
}

#[derive(Deserialize)]
struct KernelParameters {
    pub npb_problemclass:Option<String>,
}

fn readplatform() -> Result<Platform,Box<dyn Error>> {

    let platform_path = match env::var("PLATFORM_DESCRIPTION") {
        Ok(var) => var,
        Err(_) => env::var("PWD")?
    };

    let contents: String = fs::read_to_string(Path::new(&platform_path).join("Platform.toml"))?;

    Ok(toml::from_str(&contents).unwrap())
}

pub fn readproblemclass() -> PlatformFeatures
{
    let t = match readplatform() {
        Ok(t) => t,
        Err(_) => todo!(),
    };

   let mut m: PlatformFeatures = ::std::collections::HashMap::new();

    match t.kernel {
        Some(u) => {
            match u.npb_problemclass { Some(v) => { println!("=================> {}", v); add_qualifier(&mut m, "npb_problemclass".to_string(), v); }, None => {} }
        },
        None => {},
    }

   return m;
}