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

// size of local O(memory/(2*sizeof(real))
// tuning parameter, larger is better, fill device memory


// number of devices
#ifndef DEV_MAX
#define DEV_MAX 1
#endif

// check for small iteration numbers
//#define CHECK
//#define PRINT

// number of (multi-) processors
#ifndef PROC
#define PROC 8
#endif

#include "gpu_ocl.hpp"
#include <stdlib.h>
#include "stdio.h"
#include "math.h"

using namespace std;

//#define REAL2
//#define WRP 32
//#define FLOAT

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

// number of threads = algorithm vector length
#ifndef LOCAL
#define LOCAL 256
#endif

// large LOCAL and large WIDTH exceed register limit

// algorithm: approx grid size 
#ifndef GRIDSIZE
#define GRIDSIZE 314572800
#endif


// round to even multiple of LOCAL * WIDTH * PROC
#define GRID_LOCAL (((GRIDSIZE) + (LOCAL) * (WIDTH) * (PROC) - 1) / (2 * (LOCAL) * (WIDTH) * (PROC)) * 2)


#if (GRID_LOCAL < 2 * TIMESTEP)
#error "overlap too large"
#endif

#if ((WIDTH) <= 1)
#error "WIDTH must be greater 1"
#endif

#if ((LOCAL) <= 0)
#error "LOCAL must be positive"
#endif

#if ((TIMESTEP)%2 != 0 || (TIMESTEP)<=0)
#error "TIMESTEP must be even"
#endif

#if ((GRID_LOCAL)%2 != 0)
#warning "even GRID_LOCAL"
#endif



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

inline real kern (real a, real b, real c) { // CPU version
  // 3pt stencil
  real d = .5f * b + .25f * (a + c);
  return d;
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

real* read (cl_mem xd, int n, int th) {
  real *x = (real*)malloc (n * sizeof (real));
  if (!x) pferror ("malloc");
  pfgpu[th].read<real> (xd, x, 0, n);
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
  char* fla = (char*)malloc(200);
  int j = sprintf (fla, "-DLOCAL=%d -DTIMESTEP=%d -DWIDTH=%d -DGRID_LOCAL=%d -DDEV_MAX=%d",
    LOCAL, TIMESTEP, WIDTH, GRID_LOCAL, DEV_MAX);
#ifdef FLOAT
  j += sprintf (fla+j, " -DFLOAT");
#endif
#ifdef REAL2
  j += sprintf (fla+j, " -DREAL2");
#endif
#ifdef WRP
  j += sprintf (fla+j, " -DWRP=%d", WRP);
#endif
  for (int i=0; i<p; i++) {
    pfgpu[i].init (argc, argv, i);
    // name of the executable file + suffix "cl" 
    std::cout << "compile " << argv[0] << ".cl " << fla << "\n";
    pfgpu[i].compile (argv[0], fla);
  }
}

int main (int argc, char *argv[]) {
  const int local = LOCAL, iter=TIMESTEP, width=WIDTH, grid=GRID_LOCAL, proc=PROC, maxthread=DEV_MAX;
  init_gpu (argc, argv);

  const unsigned int p = maxthread;
  unsigned int n = proc*width*local*grid;
  unsigned int o = local*2*iter;
  if (grid<2*iter) pferror ("overlap too large");
  cl_mem x[maxthread], y[maxthread], a[maxthread], b[maxthread];
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

    for (unsigned int i=0; i<p; i++) {
      pfgpu[i].addArg (0, x[i]);
      pfgpu[i].addArg (0, n);
      pfgpu[i].addArg (0, i);
      pfgpu[i].launch (0, proc*local, local);
      pfgpu[i].sync ();
    }

    realtime r;
    r.start ();


    if (p>1) {
      for (int i=0; i<p; i++)
       	pfgpu[i].copy<real> (x[(i+1)%p], pfgpu[(i+1)%p], x[i], 0, n, o);

      for (int i=0; i<p; i++) {
	pfgpu[i].addArg (1, x[i]);
	pfgpu[i].addArg (1, n);
	pfgpu[i].launch (1, 2*iter*local, local);
      }

      for (int i=0; i<p; i++) {
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
      pfgpu[i].addArg (2, x[i]);
      pfgpu[i].addArg (2, y[i]);
      pfgpu[i].addArg (2, a[i]);
      pfgpu[i].addArg (2, b[i]);
#ifndef WRP
      pfgpu[i].launch (2, proc*local, local);
#else // WRP
      pfgpu[i].launch (2, proc*WRP, local/WRP, WRP, local/WRP);
#endif // WRP
    }

    for (int i=0; i<p; i++) {
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
  return 0;
}
