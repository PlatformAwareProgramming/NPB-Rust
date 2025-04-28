# NPB-Rust

NAS Parallel Benchmarks in Rust 

*This is a repository aimed at providing parallel codes with different Rust parallel programming APIs for the NAS Parallel Benchmarks (NPB). You can also contribute to this project by writing issues and pull requests.*

	===================================================================
	This project was conducted in the Parallel Applications
	Modelling Group (GMAP) at PUCRS - Brazil.
	
	Project leader:
		Dalvan Griebler (PUCRS)
	
	Code contributors: 
		Eduardo M. Martins (PUCRS)
	
	In case of questions or problems, please send an e-mail to us:	
		dalvan.griebler@pucrs.br
  		e.martins01@edu.pucrs.br
		
	We would like to thank the following researchers for the fruitful discussions:
		Renato B. Hoffmann (PUCRS)
	 	Leonardo G. Faé (PUCRS)
	  	Lucas Bianchessi (PUCRS)
      	
	===================================================================


## Folders inside the project

NPB-RUST - This directory contains the sequential version.

NPB-RAYON - This directory contains the parallel version implemented with Rayon.


## Software requirements

Rust Toolchain version 1.85.0 or higher


## How to compile

Enter the directory from the version desired and execute:

```
RUSTFLAGS='--cfg class="_WORKLOAD"' cargo build --release
```

_WORKLOADs are:

```
S: small for quick test purposes
W: workstation size (a 90's workstation; now likely too small)	
A, B, C: standard test problems; ~4X size increase going from one class to the next	
D, E, F: large test problems; ~16X size increase from each of the previous Classes  
```

The command above will compile all applications of the chosen version. To compile a specific application, it is possible to add the `--bin` flag:

```
RUSTFLAGS='--cfg class="_WORKLOAD"' cargo build --release --bin _BENCHMARK
```

_BENCHMARKs are:

```
ep, cg, mg, is, ft, bt, sp and lu 
```

Command example:

```
RUSTFLAGS='--cfg class="A"' cargo build --release --bin ep
```


## How to execute

Binaries are generated inside the folders `target/release` or `target/debug`, depending on the compilation.

Execution command example:

```
./target/release/ep
```


## Parallel execution details

To configure the number of threads when using NPB-Rayon, set the `RAY_NUM_THREADS` environment variable to the desired number of threads. If not set, the maximum number of threads available on the machine will be used.

Command example:

```
export RAY_NUM_THREADS=32
```


## The five kernels and three pseudo-applications

### Kernels

```
EP - Embarrassingly Parallel, floating-point operation capacity
MG - Multi-Grid, non-local memory accesses, short- and long-distance communication
CG - Conjugate Gradient, irregular memory accesses and communication
FT - discrete 3D fast Fourier Transform, intensive long-distance communication
IS - Integer Sort, integer computation and communication
```


### Pseudo-applications

```
BT - Block Tri-diagonal solver
SP - Scalar Penta-diagonal solver
LU - Lower-Upper Gauss-Seidel solver
```


## How to cite our work

Our [paper](https://arxiv.org/abs/2502.15536) contains abundant information on how the porting was conducted and discusses the outcome performance we obtained with NPB-Rust. We compare our implementation with consolidated sequential and parallel versions of NPB in Fortran and C++.

```
@misc{MARTINS:NPB-RUST:ARXIV:25,
	author={Eduardo M. Martins and Leonardo G. Faé and Renato B. Hoffmann and Lucas S. Bianchessi and Dalvan Griebler},
	title={{NPB-Rust: NAS Parallel Benchmarks in Rust}}, 
	url={https://arxiv.org/abs/2502.15536},
	month={February},
	year={2025},
	eprint={2502.15536},
	archivePrefix={arXiv},
	primaryClass={cs.DC},  
}
```
