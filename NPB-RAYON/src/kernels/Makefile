# Caminhos para os arquivos
CUDA_SRC = dot_product.cu
CUDA_OBJ = dot_product.o
CUDA_LIB = libdot_product.so

# Compilador CUDA
NVCC = nvcc
CC = gcc

# Flags de compilação
CUDA_FLAGS = --compiler-options '-fPIC' -c
CC_FLAGS = -shared -lcudart

# Compilação CUDA: cria o arquivo de objeto
$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(CUDA_FLAGS) $(CUDA_SRC) -o $(CUDA_OBJ)

# Criação da biblioteca compartilhada a partir do objeto compilado
$(CUDA_LIB): $(CUDA_OBJ)
	$(CC) $(CC_FLAGS) $(CUDA_OBJ) -o $(CUDA_LIB)

clean:
	rm -f $(CUDA_OBJ) $(CUDA_LIB)

# Compilar tudo
all: $(CUDA_LIB)
