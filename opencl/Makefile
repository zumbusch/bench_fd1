# Copyright (c) 2011,2014, Gerhard Zumbusch
# All rights reserved.

CC = g++


OPT = -O3

DEF = -DFLOAT

# INC = -I/opt/AMDAPP/include
# LIB = -L/opt/AMDAPP/lib/x86_64 -lOpenCL -fPIC

INC = -I/usr/local/cuda/include
LIB = -lOpenCL

LIB += -lrt # provides clock_gettime()


default:
	./run.pl

run: bench_fd1 bench_fd1.cl
	$(RUN) ./bench_fd1
	rm -f bench_fd1 bench_fd1.o

bench_fd1: bench_fd1.o
	$(CC) bench_fd1.o $(LIB) -o bench_fd1

%.o: %.cc Makefile
	$(CC) $(OPT) $(INC) $(DEF) $*.cc -c

clean:
	rm -rf *.o bench_fd1

tar:
	tar czvf bench_fd1.tgz *.cc *.hpp *.cl *.pl Makefile


bench_fd1.o: util.hpp gpu_ocl.hpp

