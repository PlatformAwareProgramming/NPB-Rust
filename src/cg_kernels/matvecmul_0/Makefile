# Caminhos para os arquivos
CUDA_SRC = matvecmul.cu 
CUDA_OBJ = matvecmul0.o
CUDA_LIB = libmatvecmul0.so

# Compilador CUDA
NVCC = nvcc
CC = gcc

# -gencode arch=compute_13,code=sm_13 \
# -gencode arch=compute_20,code=sm_20 \
# -gencode arch=compute_21,code=sm_21 \
#  -gencode arch=compute_30,code=sm_30 \
#
# Flags de compilação
CUDA_FLAGS = --compiler-options '-fPIC' -c -Wno-deprecated-gpu-targets \
  -gencode arch=compute_35,code=sm_35
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
