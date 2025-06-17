# Caminhos para os arquivos
CUDA_SRC = matvecmul.cu 
CUDA_OBJ = matvecmul3.o
CUDA_LIB = libmatvecmul3.so

# Compilador CUDA
NVCC = nvcc
CC = gcc

# Flags de compilação
CUDA_FLAGS = -O3 -use_fast_math -Xptxas="-dlcm=ca" -lineinfo --compiler-options '-fPIC' -c -Wno-deprecated-gpu-targets \
  -gencode arch=compute_60,code=sm_60 \
  -gencode arch=compute_61,code=sm_61
CC_FLAGS = -shared -lcudart -lcusparse -L$(CUDA_LIBDIR)

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
