# Strassen Library Compilation
# Author: grisun0
#
# Compiles all Strassen implementations and creates symlinks for expected names

CC = gcc
CFLAGS = -O3 -ffast-math -march=native -fPIC -shared
CFLAGS_OMP = $(CFLAGS) -fopenmp
CFLAGS_AVX = $(CFLAGS) -mavx2
CFLAGS_AVX512 = $(CFLAGS) -mavx512f -mavx512vl
CFLAGS_BLAS = $(CFLAGS) -lopenblas

# Output libraries
LIBS = libstrassen.so libstrassen_turbo.so libstrassen_optimal.so
ALIASES = strassen_hybrid.so strassen_production_final.so

.PHONY: all clean aliases

all: $(LIBS) aliases

# Basic Strassen implementation
libstrassen.so: strassen_c.c
	$(CC) $(CFLAGS) -o $@ $<

# Turbo implementation with OpenMP and AVX2
libstrassen_turbo.so: strassen_turbo.c
	$(CC) $(CFLAGS_OMP) $(CFLAGS_AVX) -o $@ $<

# Optimal implementation with BLAS fallback
libstrassen_optimal.so: strassen_optimal.c
	$(CC) $(CFLAGS_BLAS) -o $@ $< || echo "Warning: BLAS not available, skipping optimal build"

# AVX512 versions (optional, may not compile on all systems)
libstrassen_avx512.so: strassen_c.c
	$(CC) $(CFLAGS_AVX512) -o $@ $< 2>/dev/null || echo "AVX-512 not supported on this system"

libstrassen_hybrid_avx512.so: strassen_turbo.c
	$(CC) $(CFLAGS_OMP) $(CFLAGS_AVX512) -o $@ $< 2>/dev/null || echo "AVX-512 not supported on this system"

# Create symlinks for expected library names
aliases:
	ln -sf libstrassen_turbo.so strassen_hybrid.so
	ln -sf libstrassen_turbo.so strassen_production_final.so

# Install to benchmark and training directories
install: all aliases
	cp strassen_hybrid.so ../benchmarks/
	cp strassen_production_final.so ../training/

clean:
	rm -f *.so

# Rebuild everything
rebuild: clean all
