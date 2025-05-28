# Lista de subdiretórios, cada um com seu próprio Makefile
DIRS = matvecmul vecvecmul scalarvecmul1 scalarvecmul2 norm vectors

# Regra padrão: compilar todos os programas
all:
	@for dir in $(DIRS); do \
		echo "Compilando em $$dir..."; \
		$(MAKE) -C $$dir; \
	done

# Limpeza: limpa em todos os subdiretórios
clean:
	@for dir in $(DIRS); do \
		echo "Limpando em $$dir..."; \
		$(MAKE) -C $$dir clean; \
	done

