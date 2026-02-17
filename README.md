# NPB-Rust/CG (platform-aware programming version)

This is a fork of the NAS Parallel Benchmarks in Rust repository ([NPB-Rust](https://github.com/GMAP/NPB-Rust)), originally developed by the Parallel Applications Modelling Group (GMAP) at PUCRS, in Brazil.

It includes a platform-aware programming refactoring of the CG kernel, using the [platform-aware](https://github.com/PlatformAwareProgramming/platform-aware) crate, which is still experimental. 

Both NPB-Rust/CG and the platform-aware crate have been developed by researchers from the HPC research group of the graduate program in Computer Science of the Federal University of Ceará.

NPB-Rust/CG is originally a monolithic program, like the original Fortran version from which it was derived. To build a platform-aware version, the code has been refactored to identify functions that implement basic linear algebra operations, called kernel functions. 

The set of identified kernel functions includes: 
* _matvecmul_;
* _vecvecmul_;
* _scalarvecmul1_;
* _scalarvecmul2_;
* _norm_.

The _matvecmul_ kernel corresponds to most of the execution time of the CG kernel (e.g., > 95% for class C). 

Using platform-aware programming techniques via the platform-aware crate, each kernel function has at least three versions, with self-explanatory names: _serial_, _multicore_ (using Rayon), and _CUDA/GPU_ (using FFI for direct calls to low-level CUDA kernels). 

For _matvecmul_, there are three additional CUDA versions that exploit the characteristics of NVIDIA GPUs based on their compute capability, CUDA driver version (runtime), and CUDA toolkit version (compilation). 

The more appropriate version of each kernel function is selected dynamically at startup based on the contents of a _Platform.toml_ file, which declares the current features of the underlying execution platform. 

A resolution algorithm matches kernel assumptions with the actual platform features.


## Software requirements

* Rust Toolchain version 1.85.0 or higher;
* CUDA Toolkit 12.9.

## How to compile

Enter the directory from the version desired and execute:

CUDA_LIBDIR=<*cuda runtime directory*> ```cargo build --release --bin cg```

e.g., <*cuda runtime directory*> = /usr/local/cuda-12.9/targets/x86_64-linux/lib

WORKLOADs are:

```
S: small for quick test purposes
W: workstation size (a 90's workstation; now likely too small)	
A, B, C: standard test problems; ~4X size increase going from one class to the next	
D, E, F: large test problems; ~16X size increase from each of the previous Classes  
```


## How to execute

Binaries are generated in the `target/release` or `target/debug` folder, depending on the build configuration.

Execution command example:

```
LD_LIBRARY_PATH=./src/cg_kernels target/release/cg
```


## Parallel execution details

To configure the number of threads when using NPB-Rayon, set the `RAY_NUM_THREADS` environment variable to the desired number of threads. If not set, the machine's maximum number of threads will be used.

Command example:

```
export RAY_NUM_THREADS=32
```

## How to cite this work

```
@inproceedings{sblp,
 author = {Francisco Heron de Carvalho-Junior and José Mykael Nogueira and João Marcelon Uchoa de Alencar},
 title = { Structured platform-aware programming for Rust},
 booktitle = {Anais do XXIX Simpósio Brasileiro de Linguagens de Programação},
 location = {Recife/PE},
 year = {2025},
 issn = {0000-0000},
 pages = {51--58},
 publisher = {SBC},
 address = {Porto Alegre, RS, Brasil},
 doi = {10.5753/sblp.2025.11166},
 url = {https://sol.sbc.org.br/index.php/sblp/article/view/36948}
}

```
