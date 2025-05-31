# Caminhos para os arquivos
CUDA_SRC = scalarvecmul2.cu
CUDA_OBJ = scalarvecmul2.o
CUDA_LIB = libscalarvecmul2.so

# Compilador CUDA
NVCC = nvcc
CC = gcc

# Flags de compilação
CUDA_FLAGS = -arch=sm_86 --compiler-options '-fPIC' -c -Wno-deprecated-gpu-targets
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
