# Caminhos para os arquivos
CUDA_SRC = src/dot_product.cu
CUDA_OBJ = dot_product.o
CUDA_LIB = libdot_product.so
RUST_TARGET = target/debug/meu-projeto-rust  # Substitua com o nome do seu pacote Rust

# Compilador CUDA
NVCC = nvcc
CC = gcc
CARGO = cargo

# Flags de compilação
CUDA_FLAGS = --compiler-options '-fPIC' -c
CC_FLAGS = -shared -lcudart

# Compilação CUDA: cria o arquivo de objeto
$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(CUDA_FLAGS) $(CUDA_SRC) -o $(CUDA_OBJ)

# Criação da biblioteca compartilhada a partir do objeto compilado
$(CUDA_LIB): $(CUDA_OBJ)
	$(CC) $(CC_FLAGS) $(CUDA_OBJ) -o $(CUDA_LIB)

# Compilar o código Rust e vincular a biblioteca compartilhada
rust: $(CUDA_LIB)
	cargo build

# Limpeza de arquivos gerados
clean:
	rm -f $(CUDA_OBJ) $(CUDA_LIB)
	cargo clean

# Compilar tudo
all: rust

.PHONY: clean all
