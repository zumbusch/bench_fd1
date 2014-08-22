# Copyright (c) 2011,2014, Gerhard Zumbusch
# All rights reserved.


default:
	./run.pl

run: bench_fd1
	$(RUN) ./bench_fd1
	rm -f bench_fd1

bench_fd1: bench_fd1.cc
	$(CC) $(OPT) $(DEF) bench_fd1.cc $(LIB) -o bench_fd1

clean:
	rm -rf *.o bench_fd1

