use std::ffi::c_double;
use std::os::raw::c_int;

extern "C" {
    fn dot_product_gpu(x: *const c_double, y: *const c_double, result: *mut c_double, n: c_int);
}

// Função Rust
fn vecvecmul(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mut result: f64 = 0.0;

    unsafe {
        dot_product_gpu(x.as_ptr(), y.as_ptr(), &mut result as *mut f64, x.len() as i32);
    }

    result
}

fn main() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let dot = vecvecmul(&a, &b);
    println!("Produto interno: {}", dot); // Deve imprimir 32.0
}
