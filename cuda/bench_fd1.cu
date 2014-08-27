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
#ifndef DEV_MAX
#define DEV_MAX 1
#endif

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

//#define REAL2
//#define WRP 32

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

extern "C" __global__ void kernel0 (real *gx, uint m, uint dev_no) {
  int i = threadIdx.x;
  int k0 = blockIdx.x * LOCAL*WIDTH*GRID_LOCAL;
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

extern "C" __global__ void kernel1 (real *gx, uint m) {
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


//----------------------------------------------------------------------
// alternative versions of kernel2
//----------------------------------------------------------------------

#ifndef WRP
#ifdef REAL2
//----------------------------------------------------------------------

#ifdef FLOAT
typedef float2 real2;
#else
typedef double2 real2;
#endif

__device__ void diagvu2 (real2 *ap, real2 *bp, uint m) {
  // 2*unrolled space-time slice
  int id = threadIdx.x;
  real2 dd = bp [0+id];
  real d0 = dd.x;
  real d1 = dd.y;
  for (int i=0; i<m*LOCAL; i+=LOCAL) {
    real2 ad = ap [i+id];
    real a0 = ad.x;
    real a1 = ad.y;
    real e0 = kern (a0, a1, d0);
    real e1 = kern (a1, d0, d1);
    bp [i+LOCAL+id] = (real2){e0, e1};
    d0 = e0;
    d1 = e1;
  }
}

__device__ void diagvuu (real2 *ap, real2 *bp, real *ip, real *jp) {
  // u*unrolled in space, 2*in time, space-time slice
  int id = threadIdx.x;
  ap[id] = (real2){ip [id], ip [LOCAL+id]};
  real d[WIDTH+2];
  for (int k=0; k<WIDTH; k++)
    d[k+2] = ip [LOCAL*(k+2)+id];
  for (int i=0; i<TIMESTEP*LOCAL; i+=2*LOCAL) {
    real2 dd = ap [i+id];
    d[0] = dd.x;
    d[1] = dd.y;
    real e[WIDTH+2];
    for (int k=0; k<WIDTH; k++)
      e[k+2] = kern (d[k], d[k+1], d[k+2]);
    bp [i+LOCAL+id] = (real2){e[WIDTH], e[WIDTH+1]};
    real2 ed = ap [i+LOCAL+id];
    e[0] = ed.x;
    e[1] = ed.y;
    for (int k=0; k<WIDTH; k++)
      d[k+2] = kern (e[k], e[k+1], e[k+2]);
    bp [i+2*LOCAL+id] = (real2){d[WIDTH], d[WIDTH+1]};
  }
  for (int k=0; k<WIDTH; k++)
    jp [LOCAL*k+id] = d[k+2];
}


extern "C" __global__ void kernel2 (real *s0, real *s1,
				    real *ga, real *gb) {
  int id = threadIdx.x;
  int k1 = blockIdx.x *LOCAL*2*(TIMESTEP+1);

  real2 *a = (real2*)&ga[k1];
  real2 *b = (real2*)&gb[k1];
  int i0 = blockIdx.x *LOCAL*GRID_LOCAL*WIDTH;
  int i1 = (blockIdx.x+1) *LOCAL*GRID_LOCAL*WIDTH;
  a [0+id] = (real2){s0 [i0+id], s0 [i0+LOCAL+id]};
  for (int i=1; i<TIMESTEP; i++) { // 2* unroll initial
    b [0+id] = (real2){s0 [2*LOCAL*i+i0+id], s0 [2*LOCAL*i+LOCAL+i0+id]};
    diagvu2 (a, b, i);
    real2 *c = a;
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

//----------------------------------------------------------------------
#else //REAL2
//----------------------------------------------------------------------


__device__ void diagvu2 (real *ap, real *bp, uint m) {
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

//----------------------------------------------------------------------
#endif //REAL2
#else // WRP
#ifdef REAL2
//----------------------------------------------------------------------

#ifdef FLOAT
typedef float2 real2;
#else
typedef double2 real2;
#endif

__device__ void diagvu2 (real2 *ap, real2 *bp, uint m) {
  // 2*unrolled space-time slice
  int ida = threadIdx.x + __mul24((int)WRP*(TIMESTEP+1), threadIdx.y);

  real2 dd = bp [0+ida];
  real d0 = dd.x;
  real d1 = dd.y;
  for (int i=0; i<m*WRP; i+=WRP) {
    real2 ad = ap [i+ida];
    real a0 = ad.x;
    real a1 = ad.y;
    real e0 = kern (a0, a1, d0);
    real e1 = kern (a1, d0, d1);
    bp [i+WRP+ida] = (real2){e0, e1};
    d0 = e0;
    d1 = e1;
  }
}

__device__ void diagvuu (real2 *ap, real2 *bp, real *ip, real *jp) {
  // u*unrolled in space, 2*in time, space-time slice
  int id = threadIdx.x + __mul24((int)WRP, threadIdx.y);
  int ida = threadIdx.x + __mul24((int)WRP*(TIMESTEP+1), threadIdx.y);

  ap[ida] = (real2){ip [id], ip [LOCAL+id]};
  real d[WIDTH+2];
  for (int k=0; k<WIDTH; k++)
    d[k+2] = ip [LOCAL*(k+2)+id];
  for (int i=0; i<TIMESTEP*WRP; i+=2*WRP) {
    real2 dd = ap [i+ida];
    d[0] = dd.x;
    d[1] = dd.y;
    real e[WIDTH+2];
    for (int k=0; k<WIDTH; k++)
      e[k+2] = kern (d[k], d[k+1], d[k+2]);
    bp [i+WRP+ida] = (real2){e[WIDTH], e[WIDTH+1]};
    real2 ed = ap [i+WRP+ida];
    e[0] = ed.x;
    e[1] = ed.y;
    for (int k=0; k<WIDTH; k++)
      d[k+2] = kern (e[k], e[k+1], e[k+2]);
    bp [i+2*WRP+ida] = (real2){d[WIDTH], d[WIDTH+1]};
  }
  for (int k=0; k<WIDTH; k++)
    jp [LOCAL*k+id] = d[k+2];
}


extern "C" __global__ void kernel2 (real *s0, real *s1,
				    real *ga, real *gb) {
  int id = threadIdx.x + __mul24((int)WRP, threadIdx.y);
  int ida = threadIdx.x + __mul24((int)WRP*(TIMESTEP+1), threadIdx.y);
  int k1 = blockIdx.x *LOCAL*2*(TIMESTEP+1);

  real2 *a = (real2*)&ga[k1];
  real2 *b = (real2*)&gb[k1];
  int i0 = blockIdx.x *LOCAL*GRID_LOCAL*WIDTH;
  int i1 = (blockIdx.x+1) *LOCAL*GRID_LOCAL*WIDTH;
  a [0+ida] = (real2){s0 [i0+id], s0 [i0+LOCAL+id]};
  for (int i=1; i<TIMESTEP; i++) { // 2* unroll initial
    b [0+ida] = (real2){s0 [2*LOCAL*i+i0+id], s0 [2*LOCAL*i+LOCAL+i0+id]};
    diagvu2 (a, b, i);
    real2 *c = a;
    a = b;
    b = c;
  }

  // if (GRID_LOCAL%2 == 1) {
  //   for (int i=i0; i<i1; i+=LOCAL*(WIDTH)) { // u* unroll block
  //     real *ip = &s0 [i+2*LOCAL*(TIMESTEP-1)];
  //     real *jp = &s1 [i];
  //     diagvuu (a, b, ip, jp);
  //     real2 *c = a;
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

//----------------------------------------------------------------------
#else //REAL2
//----------------------------------------------------------------------

__device__ void diagvu2 (real *ap, real *bp, uint m) {
  // 2*unrolled space-time slice
  int ida = threadIdx.x + __mul24((int)WRP*2*(TIMESTEP+1), threadIdx.y);

  real d0 = bp [0+ida];
  real d1 = bp [WRP+ida];
  for (int i=0; i<2*m*WRP; i+=2*WRP) {
    real a0 = ap [i+ida];
    real a1 = ap [i+WRP+ida];
    real e0 = kern (a0, a1, d0);
    real e1 = kern (a1, d0, d1);
    bp [i+2*WRP+ida] = e0;
    bp [i+3*WRP+ida] = e1;
    d0 = e0;
    d1 = e1;
  }
}

__device__ void diagvuu (real *ap, real *bp, real *ip, real *jp) {
  // u*unrolled in space, 2*in time, space-time slice
  int id = threadIdx.x + __mul24((int)WRP, threadIdx.y);
  int ida = threadIdx.x + __mul24((int)WRP*2*(TIMESTEP+1), threadIdx.y);

  ap[ida] = ip [id];
  ap[WRP+ida] = ip [LOCAL+id];
  real d[WIDTH+2];
  for (int k=0; k<WIDTH; k++)
    d[k+2] = ip [LOCAL*(k+2)+id];
  for (int i=0; i<2*TIMESTEP*WRP; i+=4*WRP) {
    d[0] = ap [i+ida];
    d[1] = ap [i+WRP+ida];
    real e[WIDTH+2];
    for (int k=0; k<WIDTH; k++)
      e[k+2] = kern (d[k], d[k+1], d[k+2]);
    bp [i+2*WRP+ida] = e[WIDTH];
    bp [i+3*WRP+ida] = e[WIDTH+1];
    e[0] = ap [i+2*WRP+ida];
    e[1] = ap [i+3*WRP+ida];
    for (int k=0; k<WIDTH; k++)
      d[k+2] = kern (e[k], e[k+1], e[k+2]);
    bp [i+4*WRP+ida] = d[WIDTH];
    bp [i+5*WRP+ida] = d[WIDTH+1];
  }
  for (int k=0; k<WIDTH; k++)
    jp [LOCAL*k+id] = d[k+2];
}


extern "C" __global__ void kernel2 (real *s0, real *s1,
				    real *ga, real *gb) {
  int id = threadIdx.x + __mul24((int)WRP, threadIdx.y);
  int ida = threadIdx.x + __mul24((int)WRP*2*(TIMESTEP+1), threadIdx.y);
  int k1 = blockIdx.x *LOCAL*2*(TIMESTEP+1);

  real *a = &ga[k1];
  real *b = &gb[k1];
  int i0 = blockIdx.x *LOCAL*GRID_LOCAL*WIDTH;
  int i1 = (blockIdx.x+1) *LOCAL*GRID_LOCAL*WIDTH;
  a [0+ida] = s0 [i0+id];
  a [WRP+ida] = s0 [i0+LOCAL+id];
  for (int i=1; i<TIMESTEP; i++) { // 2* unroll initial
    b [0+ida] = s0 [2*LOCAL*i+i0+id];
    b [WRP+ida] = s0 [2*LOCAL*i+LOCAL+i0+id];
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

//----------------------------------------------------------------------
#endif //REAL2
#endif // WRP


int comp (const void*x, const void*y) {
  double xx = *(double*)x;
  double yy = *(double*)y;
  return xx<yy;
}


real* init_cpu (uint n, uint o, uint local) {
  real *x = (real*)malloc ((n+o) * sizeof (real));
  if (!x) pferror ("malloc");
  for (uint j=0; j<n/local; j++)
    for (uint i=0; i<local; i++) {
      real y = (j+i*(n/local)) / (real)n;
      y = y*y;
      x[j*local+i] = y;
    }
  for (uint j=0; j<o/local; j++)
    for (uint i=0; i<local; i++) {
      x[n+j*local+(i+1)%local] = x[j*local+i];
    }
  return x;
}

void iterate_cpu (real *x, uint n, uint local, uint iter) {
  uint m = n / local;
  for (uint it=0; it<iter; it++) {
    for (uint j=0; j<m-2-it; j++)
      for (uint i=0; i<local; i++) {
	uint l = j*local+i;
	x[l] = kern (x[l], x[l+local], x[l+2*local]);
      }
  }
}

real* read (real * xd, uint n, uint th) {
  real *x = (real*)malloc (n * sizeof (real));
  if (!x) pferror ("malloc");
  pfgpu[th].read (xd, x, 0, n);
  return x;
}

void print (real *x, uint n) {
  for (uint i=0; i<n; i++)
    cout<<x[i]<<" ";
  cout<<"\n";
}

void print (real *x, real *y, uint n) {
  for (uint i=0; i<n; i++)
    cout<<x[i]-y[i]<<" ";
  cout<<"\n";
}

void diff (real *x0, real *x1, uint n) {
  real s1 = 0.f, s2 = 0.f, si = 0.f;
  for (uint i=0; i<n; i++) {
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
  uint p = DEV_MAX;
  for (uint i=0; i<p; i++)
    pfgpu[i].init (argc, argv, i);
}

int main (int argc, char *argv[]) {
  const uint local = LOCAL, iter=TIMESTEP, width=WIDTH, grid=GRID_LOCAL, proc=PROC, maxthread=DEV_MAX;
  init_gpu (argc, argv);

  const uint p = maxthread;
  uint n = proc*width*local*grid;
  uint o = local*2*iter;
  if (grid<2*iter) pferror ("overlap too large");
  real *x[maxthread], *y[maxthread], *a[maxthread], *b[maxthread];
  real* x_buf[maxthread];
  for (uint i=0; i<p; i++) {
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
  for (uint it=0; it<IT; it++) {

    for (uint i=0; i<p; i++) {
      // cout << "line " << __LINE__ << "\n";
      pfgpu[i].start ();
      kernel0 <<<proc, local>>> (x[i], n, i);
      // cout << "line " << __LINE__ << "\n";
      pfgpu[i].sync ();
    }

    realtime r;
    r.start ();


    if (p>1) {
      for (uint i=0; i<p; i+=2)
       	pfgpu[i].copy (x[(i+1)%p], pfgpu[(i+1)%p], x[i], 0, n, o);

      for (uint i=1; i<p; i+=2)
       	pfgpu[i].copy (x[(i+1)%p], pfgpu[(i+1)%p], x[i], 0, n, o);

      for (uint i=0; i<p; i++) {
	// cout << "line " << __LINE__ << "\n";
	pfgpu[i].start ();
	kernel1 <<<2*iter, local>>> (x[i], n);
      }

      for (uint i=0; i<p; i++) {
	// cout << "line " << __LINE__ << "\n";
	pfgpu[i].sync ();
      }

    }

#ifdef PRINT
    for (uint i=0; i<p; i++) {
      real *xd = read (x[i], n+o, i);
      cout << "init"<<i<<"\n";
      print (xh+n*i, n+o);
      print (xd, n+o);
      free (xd);
    }
#endif // PRINT

    for (uint i=0; i<p; i++) {
      // cout << "line " << __LINE__ << "\n";
      pfgpu[i].start ();
#ifndef WRP
      kernel2 <<<proc, local>>> (x[i], y[i], a[i], b[i]);
#else // WRP
      dim3 p2(proc, 1);
      dim3 l2(WRP, local/WRP);
      kernel2 <<<p2, l2>>> (x[i], y[i], a[i], b[i]);
#endif // WRP
    }

    for (uint i=0; i<p; i++) {
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
    for (uint i=0; i<p; i++) {
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


  for (uint i=0; i<p; i++) {
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
