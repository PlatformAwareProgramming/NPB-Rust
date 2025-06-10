# Caminhos para os arquivos
CUDA_SRC = scalarvecmul1.cu
CUDA_OBJ = scalarvecmul1.o
CUDA_LIB = libscalarvecmul1.so

# Compilador CUDA
ifdef DOCKER_CUDA
NVCC = docker run -v "$$(dirname "$$PWD")":"$$(dirname "$$PWD")" -w "$$PWD" --rm --gpus all -it cuda-10-2-dev nvcc
CC = docker run -v "$$(dirname "$$PWD")":"$$(dirname "$$PWD")" -w "$$PWD" --rm --gpus all -it cuda-10-2-dev gcc
CUDA_FLAGS = -arch=sm_35 --compiler-options '-fPIC' -c -Wno-deprecated-gpu-targets
CC_FLAGS = -shared -lcudart -L/usr/local/cuda/lib64
else
NVCC = nvcc
CC = gcc
CUDA_FLAGS = -arch=sm_35 --compiler-options '-fPIC' -c -Wno-deprecated-gpu-targets
CC_FLAGS = -shared -lcudart -L$(CUDA_LIBDIR)
endif

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
