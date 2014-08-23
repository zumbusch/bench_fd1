// *-*-C++-*-*
// ----------------------------------------------------------------------
//
// Copyright (c) 2011, 2014, Gerhard Zumbusch.
// All rights reserved.
//
// ----------------------------------------------------------------------

// find CUDA compute capabilities
// number of multi processors
// compile with nvcc

#include <stdio.h>

int main(int argc, char **argv) {
  int n=0, n0=0;
  int pc = 0;
  if (cudaGetDeviceCount(&n)) return 1;
  int c[n];
  for (int i=0; i<n; ++i) {
    if (cudaSetDevice(i)) return 1;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, i)) return 1;
    int d = p.major * 10 + p.minor;
    pc = p.multiProcessorCount;
    for (int j=0; j<n0; ++j)
      if (d == c[j]) {
	d = 0;
	break;
      }
    if (d)
      c[n0++] = d;
  }
  for (int i=0; i<n0; ++i) {
    if (i>0)
      printf(" ");
    printf("-gencode arch=compute_%d,code=sm_%d", c[i], c[i]);
  }
  if (pc > 0)
    printf(" -DPROC=%d", pc);
  cudaDeviceReset();
  return 0;
}
