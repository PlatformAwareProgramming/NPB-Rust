# Caminhos para os arquivos
CUDA_SRC = matvecmul.cu 
CUDA_OBJ = matvecmul2.o
CUDA_LIB = libmatvecmul2.so

# Compilador CUDA
NVCC = nvcc
CC = gcc

# Flags de compilação
CUDA_FLAGS = -O3 -use_fast_math -Xptxas="-dlcm=ca" -lineinfo --compiler-options '-fPIC' -c -Wno-deprecated-gpu-targets \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 # \
# -gencode arch=compute_89,code=sm_89 \
#  -gencode arch=compute_90,code=sm_90
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
