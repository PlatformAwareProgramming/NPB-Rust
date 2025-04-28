fn main() {
    println!("cargo:rustc-link-search=.");
    println!("cargo:rustc-link-lib=dylib=dot_product");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart"); 
}
