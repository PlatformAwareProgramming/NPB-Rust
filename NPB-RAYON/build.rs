use std::process::Command;
use std::env;

fn main() {
    // Define o diretório onde o Makefile está localizado
    let kernel_dir = "src/kernels";

    // Executa o Makefile para compilar a biblioteca C
    let make_status = Command::new("make")
        .arg("-C") // Altera o diretório antes de executar o make
        .arg(kernel_dir)
        .status(); // Executa o comando e espera por ele

    match make_status {
        Ok(status) => {
            if !status.success() {
                panic!("Makefile failed with status: {}", status);
            }
        }
        Err(e) => {
            panic!("Failed to execute make: {}", e);
        }
    }

    // Informa ao Cargo para linkar a biblioteca C
    // O nome da biblioteca deve ser o nome do arquivo sem o prefixo 'lib' e a extensão (.so)
    // Por exemplo, para libdot_product.so, o nome é dot_product
    println!("cargo:rustc-link-lib=matvecmul");
    println!("cargo:rustc-link-lib=vecvecmul");
    println!("cargo:rustc-link-lib=scalarvecmul1");
    println!("cargo:rustc-link-lib=scalarvecmul2");
    println!("cargo:rustc-link-lib=vectors");
    println!("cargo:rustc-link-lib=norm");

    // Informa ao Cargo onde encontrar a biblioteca para linkar
    // Este é o diretório onde o Makefile gera a libdot_product.so
    let library_dir = env::current_dir().unwrap().join(kernel_dir);
    println!("cargo:rustc-link-search=native={}", library_dir.display());

    // --- Adição para libcudart.so ---
    // Informa ao Cargo para linkar a biblioteca 'cudart' (para libcudart.so)
    println!("cargo:rustc-link-lib=cudart");

    // Informa ao Cargo onde encontrar a libcudart.so
    // Use 'native=' para indicar que é um caminho do sistema
    println!("cargo:rustc-link-search=native=/usr/local/cuda-12.9/targets/x86_64-linux/lib");
    // --------------------------------

    // Opcional: Informar ao Cargo para recompilar se os arquivos C ou o Makefile mudarem
    println!("cargo:rerun-if-changed={}/cgkernels.h", kernel_dir);
    println!("cargo:rerun-if-changed={}/matvecmul/matvecmul.cu", kernel_dir);
    println!("cargo:rerun-if-changed={}/vecvecmul/vecvecmul.cu", kernel_dir);
    println!("cargo:rerun-if-changed={}/norm/norm.cu", kernel_dir);
    println!("cargo:rerun-if-changed={}/scalarvecmul1/scalarvecmul1.cu", kernel_dir);
    println!("cargo:rerun-if-changed={}/scalarvecmul2/scalarvecmul2.cu", kernel_dir);
    println!("cargo:rerun-if-changed={}/vectors/vectors.cu", kernel_dir);
    println!("cargo:rerun-if-changed={}/Makefile", kernel_dir);
}
