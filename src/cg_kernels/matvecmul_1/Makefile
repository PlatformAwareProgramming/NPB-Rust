# Caminhos para os arquivos
CUDA_SRC = matvecmul.cu 
CUDA_OBJ = matvecmul1.o
CUDA_LIB = libmatvecmul1.so

# Compilador CUDA
NVCC = nvcc
CC = gcc

# Flags de compilação
CUDA_FLAGS = --compiler-options '-fPIC' -c -Wno-deprecated-gpu-targets \
  -gencode arch=compute_35,code=sm_35 \
  -gencode arch=compute_50,code=sm_50 \
  -gencode arch=compute_52,code=sm_52
CC_FLAGS = -shared -lcudart -L$(CUDA_LIBDIR)

# Criação da biblioteca compartilhada a partir do objeto compilado
$(CUDA_LIB): $(CUDA_OBJ)
	$(CC)  ../$(CUDA_OBJ) -o ../$(CUDA_LIB) $(CC_FLAGS)

# Compilação CUDA: cria o arquivo de objeto
$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(CUDA_FLAGS) $(CUDA_SRC) -o ../$(CUDA_OBJ)

clean:
	rm -f ../$(CUDA_OBJ) ../$(CUDA_LIB)

# Compilar tudo
all: $(CUDA_LIB)
