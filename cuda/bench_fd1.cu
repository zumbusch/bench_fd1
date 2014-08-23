/* -*- mode: c++ -*-  */

// Copyright (c) 2011, 2012, 2014, Gerhard Zumbusch
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.

// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.

// * The names of its contributors may not be used to endorse or promote
//   products derived from this software without specific prior written
//   permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


// 1D FD, 3pt stencil, periodic b.c.
// space-time slicing

// change parameters and look for output flop=...
// optimize the kernel "diagvuu"

// size of local O(memory/(2*sizeof(real))
// tuning parameter, larger is better, fill device memory


// number of GPUs
#define DEV_MAX 1

// check for small iteration numbers
// #define CHECK
// #define PRINT

// number of (multi-) processors on a GPU 
#ifndef PROC
#define PROC 5
#endif

#include "gpu_cuda.hpp"
#include <stdlib.h>
#include "stdio.h"

using namespace std;

#ifdef FLOAT
typedef float real;
#else
typedef double real;
#endif

// algorithm: even number of time steps
#ifndef TIMESTEP
#define TIMESTEP 50
#endif

// algorithm: number of vectors width
#ifndef WIDTH
#define WIDTH 50
#endif

// number of threads on a GPU = algorithm vector length
#ifndef LOCAL
#define LOCAL 256
#endif

// large LOCAL and large WIDTH exceed GPU register limit

// algorithm: approx grid size 
#ifndef GRIDSIZE
#define GRIDSIZE 314572800/8
#endif


// round to even multiple of LOCAL * WIDTH * PROC
#define GRID_LOCAL (((GRIDSIZE) + (LOCAL) * (WIDTH) * (PROC) - 1) / (2 * (LOCAL) * (WIDTH) * (PROC)) * 2)


#if (GRID_LOCAL < 2 * TIMESTEP)
#error "overlap too large"
#endif

#if ((WIDTH) <= 1)
#error "WIDTH must be greater 1"
#endif

#if ((LOCAL)%32 != 0)
#warning "LOCAL should be a multiple of 32"
#endif

#if ((TIMESTEP)%2 != 0)
#error "TIMESTEP must be even"
#endif

#if ((GRID_LOCAL)%2 != 0)
#warning "even GRID_LOCAL"
#endif


// ----------------------------------------------------------------------
// initial data
// ----------------------------------------------------------------------

extern "C" __global__ void kernel0 (real *gx, int m, int dev_no) {
  int i = threadIdx.x;
  int k0 = blockIdx.x *LOCAL*WIDTH*GRID_LOCAL;
  int o = 2*TIMESTEP*LOCAL;
  for (int n=0; n<GRID_LOCAL; n++)
    for (int ii=0; ii<WIDTH; ii++) {
      real y = ((m * dev_no / LOCAL + blockIdx.x *WIDTH*GRID_LOCAL+ n*WIDTH + ii) + i*((m*DEV_MAX)/LOCAL)) / (real) (m * DEV_MAX);
      y = y*y;
      if (DEV_MAX!=1) {
  	int j = k0 + n*WIDTH*LOCAL + ii*LOCAL + i;
  	gx[j] = y;
      } else {
  	for (int j1=0; j1<2+o/m; j1++) {
  	  int j = j1*m + k0 + n*WIDTH*LOCAL + ii*LOCAL + (i+j1)%LOCAL;
  	  if (j<o+m) {
  	    gx[j] = y;
	  }
  	}
      }
    }
}


// ----------------------------------------------------------------------
// rotate boundary condition vectors
// ----------------------------------------------------------------------

extern "C" __global__ void kernel1 (real *gx, int m) {
  int i = threadIdx.x;
  int k0 = blockIdx.x *LOCAL + m;
  gx[k0 + i] = gx[k0 + (i+LOCAL-1)%LOCAL];
}

// ----------------------------------------------------------------------

__device__  __host__ inline real kern (real a, real b, real c) {
  // 3pt stencil
  real d = .5f * b + .25f * (a + c);
  return d;
}

__device__ void diagvu2 (real *ap, real *bp, int m) {
  // 2*unrolled space-time slice
  int id = threadIdx.x;
  real d0 = bp [0+id];
  real d1 = bp [LOCAL+id];
  for (int i=0; i<2*m*LOCAL; i+=2*LOCAL) {
    real a0 = ap [i+id];
    real a1 = ap [i+LOCAL+id];
    real e0 = kern (a0, a1, d0);
    real e1 = kern (a1, d0, d1);
    bp [i+2*LOCAL+id] = e0;
    bp [i+3*LOCAL+id] = e1;
    d0 = e0;
    d1 = e1;
  }
}

__device__ void diagvuu (real *ap, real *bp, real *ip, real *jp) {
  // u*unrolled in space, 2*in time, space-time slice
  int id = threadIdx.x;
  ap[id] = ip [id];
  ap[LOCAL+id] = ip [LOCAL+id];
  real d[WIDTH+2];
  for (int k=0; k<WIDTH; k++)
    d[k+2] = ip [LOCAL*(k+2)+id];
  for (int i=0; i<2*TIMESTEP*LOCAL; i+=4*LOCAL) {
    d[0] = ap [i+id];
    d[1] = ap [i+LOCAL+id];
    real e[WIDTH+2];
    for (int k=0; k<WIDTH; k++)
      e[k+2] = kern (d[k], d[k+1], d[k+2]);
    bp [i+2*LOCAL+id] = e[WIDTH];
    bp [i+3*LOCAL+id] = e[WIDTH+1];
    e[0] = ap [i+2*LOCAL+id];
    e[1] = ap [i+3*LOCAL+id];
    for (int k=0; k<WIDTH; k++)
      d[k+2] = kern (e[k], e[k+1], e[k+2]);
    bp [i+4*LOCAL+id] = d[WIDTH];
    bp [i+5*LOCAL+id] = d[WIDTH+1];
  }
  for (int k=0; k<WIDTH; k++)
    jp [LOCAL*k+id] = d[k+2];
}


extern "C" __global__ void kernel2 (real *s0, real *s1,
				    real *ga, real *gb) {
  int id = threadIdx.x;
  int k1 = blockIdx.x *LOCAL*2*(TIMESTEP+1);

  real *a = &ga[k1];
  real *b = &gb[k1];
  int i0 = blockIdx.x *LOCAL*GRID_LOCAL*WIDTH;
  int i1 = (blockIdx.x+1) *LOCAL*GRID_LOCAL*WIDTH;
  a [0+id] = s0 [i0+id];
  a [LOCAL+id] = s0 [i0+LOCAL+id];
  for (int i=1; i<TIMESTEP; i++) { // 2* unroll initial
    b [0+id] = s0 [2*LOCAL*i+i0+id];
    b [LOCAL+id] = s0 [2*LOCAL*i+LOCAL+i0+id];
    diagvu2 (a, b, i);
    real *c = a;
    a = b;
    b = c;
  }

  // if (GRID_LOCAL%2 == 1) {
  //   for (int i=i0; i<i1; i+=LOCAL*(WIDTH)) { // u* unroll block
  //     real *ip = &s0 [i+2*LOCAL*(TIMESTEP-1)];
  //     real *jp = &s1 [i];
  //     diagvuu (a, b, ip, jp);
  //     real *c = a;
  //     a = b;
  //     b = c;
  //   }
  // } else {
    for (int i=i0; i<i1; i+=2*LOCAL*(WIDTH)) { // u* unroll block
      real *ip = &s0 [i+2*LOCAL*(TIMESTEP-1)];
      real *jp = &s1 [i];
      diagvuu (a, b, ip, jp);
      ip = &s0 [i+2*LOCAL*(TIMESTEP-1)+LOCAL*(WIDTH)];
      jp = &s1 [i+LOCAL*(WIDTH)];
      diagvuu (b, a, ip, jp);
    }
  // }
}


int comp (const void*x, const void*y) {
  double xx = *(double*)x;
  double yy = *(double*)y;
  return xx<yy;
}


real* init_cpu (int n, int o, int local) {
  real *x = (real*)malloc ((n+o) * sizeof (real));
  if (!x) pferror ("malloc");
  for (int j=0; j<n/local; j++)
    for (int i=0; i<local; i++) {
      real y = (j+i*(n/local)) / (real)n;
      y = y*y;
      x[j*local+i] = y;
    }
  for (int j=0; j<o/local; j++)
    for (int i=0; i<local; i++) {
      x[n+j*local+(i+1)%local] = x[j*local+i];
    }
  return x;
}

void iterate_cpu (real *x, int n, int local, int iter) {
  int m = n / local;
  for (int it=0; it<iter; it++) {
    for (int j=0; j<m-2-it; j++)
      for (int i=0; i<local; i++) {
	int l = j*local+i;
	x[l] = kern (x[l], x[l+local], x[l+2*local]);
      }
  }
}

real* read (real * xd, int n, int th) {
  real *x = (real*)malloc (n * sizeof (real));
  if (!x) pferror ("malloc");
  pfgpu[th].read (xd, x, 0, n);
  return x;
}

void print (real *x, int n) {
  for (int i=0; i<n; i++)
    cout<<x[i]<<" ";
  cout<<"\n";
}

void print (real *x, real *y, int n) {
  for (int i=0; i<n; i++)
    cout<<x[i]-y[i]<<" ";
  cout<<"\n";
}

void diff (real *x0, real *x1, int n) {
  real s1 = 0.f, s2 = 0.f, si = 0.f;
  for (int i=0; i<n; i++) {
    real y = fabs(x0[i]-x1[i]);
    s1 += y;
    s2 += y*y;
    si = fmaxf(si, y);
  }
  s1 = s1 / n;
  s2 = sqrtf (s2 / n);
  cout<<"error l1="<<s1<<"  error l2="<<s2<<"  error max="<<si<<"\n";
}

void init_gpu (int argc, char *argv[]) {
  int p = DEV_MAX;
  for (int i=0; i<p; i++)
    pfgpu[i].init (argc, argv, i);
}

int main (int argc, char *argv[]) {
  const int local = LOCAL, iter=TIMESTEP, width=WIDTH, grid=GRID_LOCAL, proc=PROC, maxthread=DEV_MAX;
  init_gpu (argc, argv);

  const int p = maxthread;
  int n = proc*width*local*grid;
  int o = local*2*iter;
  if (grid<2*iter) pferror ("overlap too large");
  real *x[maxthread], *y[maxthread], *a[maxthread], *b[maxthread];
  real* x_buf[maxthread];
  for (int i=0; i<p; i++) {
    x[i] = pfgpu[i].alloc<real> (n+o);
    y[i] = pfgpu[i].alloc<real> (n);
    a[i] = pfgpu[i].alloc<real> (2*(iter+2)*local*proc);
    b[i] = pfgpu[i].alloc<real> (2*(iter+2)*local*proc);
    x_buf[i] = (real*)malloc(sizeof(real)*o);
    if (!x_buf[i]) pferror("malloc");
  }

#ifdef CHECK
#define IT 1
#else
#define IT 1
#endif

#ifdef CHECK
  cout<<"n="<<n<<"\n"<<"o="<<o<<"\n"<<"p="<<p<<"\n"<<"iter="<<iter<<"\n";
  real *xh = init_cpu (n*p, o, local);
#endif // CHECK

  double fl[IT];
  for (int it=0; it<IT; it++) {

    for (int i=0; i<p; i++) {
      // cout << "line " << __LINE__ << "\n";
      pfgpu[i].start ();
      kernel0 <<<proc, local>>> (x[i], n, i);
      // cout << "line " << __LINE__ << "\n";
      pfgpu[i].sync ();
    }

    realtime r;
    r.start ();


    if (p>1) {
      for (int i=0; i<p; i+=2)
       	pfgpu[i].copy (x[(i+1)%p], pfgpu[(i+1)%p], x[i], 0, n, o);

      for (int i=1; i<p; i+=2)
       	pfgpu[i].copy (x[(i+1)%p], pfgpu[(i+1)%p], x[i], 0, n, o);

      for (int i=0; i<p; i++) {
	// cout << "line " << __LINE__ << "\n";
	pfgpu[i].start ();
	kernel1 <<<2*iter, local>>> (x[i], n);
      }

      for (int i=0; i<p; i++) {
	// cout << "line " << __LINE__ << "\n";
	pfgpu[i].sync ();
      }

    }

#ifdef PRINT
    for (int i=0; i<p; i++) {
      real *xd = read (x[i], n+o, i);
      cout << "init"<<i<<"\n";
      print (xh+n*i, n+o);
      print (xd, n+o);
      free (xd);
    }
#endif // PRINT

    for (int i=0; i<p; i++) {
      // cout << "line " << __LINE__ << "\n";
      pfgpu[i].start ();
      kernel2 <<<proc, local>>> (x[i], y[i], a[i], b[i]);
    }

    for (int i=0; i<p; i++) {
      // cout << "line " << __LINE__ << "\n";
      pfgpu[i].sync ();
    }

    r.stop ();

    // if (p==1) 
    //   fl[it] = pfgpu[0].time();
    // else
    fl[it] = r.elapsed (); // pfgpu[0].time();
    std::cout <<"t_host="<<r.elapsed ()
	      <<"  t_gpu="<<pfgpu[0].time()
	      <<std::endl;

#ifdef CHECK
    iterate_cpu (xh, n*p+o, local, iter);
    for (int i=0; i<p; i++) {
      real *yd = read (y[i], n, i);
#ifdef PRINT
      cout << "res"<<i<<"\n";
      print (xh+n*i, n);
      print (yd, n);
#endif // PRINT
      diff (xh+n*i, yd, n);
      free (yd);
    }
#endif // CHECK
  }
  qsort (&fl[0], IT, sizeof (fl[0]), comp);
  std::cout<<"flop="<<(p*iter*4.*width*proc*local*grid) / fl[IT/2]<<std::endl;


  for (int i=0; i<p; i++) {
    free(x_buf[i]);
    pfgpu[i].free (b[i]);
    pfgpu[i].free (a[i]);
    pfgpu[i].free (y[i]);
    pfgpu[i].free (x[i]);
    pfgpu[i].close ();
  }
  cudaDeviceReset ();
  return 0;
}
