# Copyright (c) 2011,2014, Gerhard Zumbusch
# All rights reserved.

CC = nvcc


OPT = -O3 -use_fast_math
#OPT += -Xptxas -v

# native compile:
OPT += $(shell ./cuda_sm)
#OPT += -arch=sm_10 -DPROC=24
#OPT += -arch=sm_20 -DPROC=16
#OPT += -arch=sm_30 -DPROC=8
#OPT += -arch=sm_35 -DPROC=13
#OPT += -arch=sm_50 -DPROC=5

DEF = -DFLOAT

LIB = -lrt # provides clock_gettime()

default:
	./run.pl

run: bench_fd1
	$(RUN) ./bench_fd1
	rm -f bench_fd1 bench_fd1.o

cuda_sm: cuda_sm.cu
	$(CC) cuda_sm.cu -o cuda_sm

bench_fd1: bench_fd1.o
	$(CC) bench_fd1.o $(LIB) -o bench_fd1

%.o: %.cc Makefile
	$(CC) $(OPT) $(DEF) $*.cc -c

%.o: %.cu Makefile
	$(CC) $(OPT) $(DEF) $*.cu -c

clean:
	rm -rf *.o bench_fd1
	#rm -rf *.o bench_fd1 cuda_sm

tar:
	tar czvf bench_fd1.tgz *.cu *.hpp *.pl Makefile


bench_fd1.o: util.hpp gpu_cuda.hpp cuda_sm

