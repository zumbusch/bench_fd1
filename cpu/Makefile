# Copyright (c) 2011,2014, Gerhard Zumbusch
# All rights reserved.

#CC = g++
#CC = clang++
#CC = /opt/intel/bin/icpc
#CC = arm-linux-gnueabihf-g++-4.8
#CC = powerpc-linux-gnu-g++-4.8 

#OPT = -O3 
#OPT += -fopenmp -DOPENMP
#OPT += -msse4 -DSSE
#OPT += -mavx -DAVX
#OPT += -mavx512f -DAVX512
#OPT += -mfma -DFMA
#OPT += -mfma4 -DFMA4
#OPT += -mmic -DPHI
#OPT += -marm -mfpu=neon -DNEON
#OPT += -maltivec -DALTIVEC

#DEF = -DFLOAT

LIB = -lrt # provides clock_gettime()

default:
	./run.pl

run: bench_fd1
	sleep 2
	$(RUN) ./bench_fd1
	rm -f bench_fd1 bench_fd1.o

bench_fd1: bench_fd1.o util.o
	$(CC) $(OPT) bench_fd1.o util.o $(LIB) -o bench_fd1

simd_check: simd_check.o
	$(CC) $(OPT) simd_check.o $(LIB) -o simd_check

%.o: %.cc Makefile
	$(CC) $(OPT) $(DEF) $*.cc -c

clean:
	rm -rf *.o bench_fd1 simd_check

tar:
	tar czvf bench_fd1.tgz *.cc *.hpp *.pl *.sh *.txt LICENSE Makefile README


bench_fd1.o: util.hpp simd.hpp
util.o: util.hpp simd.hpp
simd_check.o: simd.hpp


