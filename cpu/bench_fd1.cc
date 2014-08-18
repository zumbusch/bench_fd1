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
// C++ wrapped SIMD vectors, space-time slicing

// change parameters and look for output flop=...
// optimize the kernel "diagvuu"

// number of float values per MPI process
// size of local O(memory/(2*sizeof(real))
// tuning parameter, larger is better, fill main memory
// multi threading: really large
#ifndef GRIDSIZE
#define GRIDSIZE 1024*1024*1024
#endif

// tuning parameter, fill (L1) cache, roughly 768 float, 320 double
// vector overhead O(TIMESTEP^2)
// even number
#ifndef TIMESTEP
#define TIMESTEP 768 
#endif

// number of vectors in flight
// is related to the number of registers
// 7 seems optimal for many x86_64
// 15 for MIC
// 1 < WIDTH <= 24
#ifndef WIDTH
#define WIDTH 7 
#endif

// unroll kernel code
#define UNROLLED 

// repeat algorithm several times
#define TIMEBLK 1 

// turn on for verification
//#define CHECK 

// further defines, parallel
//#define OPENMP
//#define PF_MPI

// further defines, data type, simd instructions
//#define FLOAT
//#define SSE
//#define AVX
//#define PHI
//#define AVX512
//#define NEON
//#define ALTIVEC
//#define FMA
//#define FMA4

// --------------------------


#include "util.hpp"
#include "simd.hpp"


// --------------------
// 3pt FD stencil 
// --------------------

template <class A>
A kern (A a, A b, A c) { // 3pt stencil
  //  A d = .5f * b + .25f * (a + c);
  A d = fma (.25, a + c, .5 * b);
  return d;
}


// --------------------------


using namespace std;

template <class A>
void print (A &a) {
  int o =A::length;
  typename A::ptr x = (typename A::ptr)&a;
  for (int i=0; i<o; i++) {
    cout<<x[i];
    if (i<o-1)
      cout<<" ";
  }
}

template <class A>
void print (A a, int n) {
  for (int i=0; i<n; i++)
    cout<<a [i]<<"\t";
  cout<<"\n";
}

template <class A>
void printv (A* av, int n) {
  typename A::ptr a = (typename A::ptr)av;
  int o = A::length;
  for (int i=0; i<n; i+=o) {
    for (int j=0; j<o; j++) {
      cout<<a [i+j];
      if (j<o-1)
	cout<<"\t";
    }
    cout<<"\n";
  }
  cout<<"\n";
}

// --------------------

template <class A>
void swap2 (A &a, A &b) {
  A t = a; 
  a = b;
  b = t;
}

// --------------------

template <class A>
void copy (A a, int n, int o) { // overlap o
  for (int i=0; i<o; i++)
    a[n+i] = a[i];
}

template <class A>
void init (A* a, int n, int o) { // initial data, t=0, overlap o
  A y = 2.f / (n * pfcomm.pr);
  y = y*y;
# ifdef OPENMP
# pragma omp parallel for
# endif
  for (int i=0; i<n; i++) {
    A x = ( (A) (/* n/2- */ i + n * pfcomm.id));
    a [i] = x*x*y;
  }
  copy (a, n, o);
}

template <class A>
void copyv (A* av, int n, int nn) { // overlap o
  typename A::ptr a = (typename A::ptr)av;
  int o = A::length;
  for (int i=0; i<nn; i+=o)
    *(A*)&a[n+i] = lrotate (*(A*)&a[i]);
}

template <class A>
void initv (A* av, int n, int nn) { // initial data, t=0, overlap o
  typename A::ptr a = (typename A::ptr)av;
  int o = A::length;
  int ng = n * pfcomm.pr;
  int ig = n * pfcomm.id;
  typename A::base y = 2.f / ng;
  y = y*y;
  A xx;
  xx.set_inc ();
  xx = xx * ((typename A::base)ng / A::length);

# ifdef OPENMP
# pragma omp parallel for
# endif
  for (int i=0; i<n; i+=o) {
    A x(xx + (typename A::base) (i+ig) / A::length);
    *(A*)&a [i] = x*x*y;
  }
  copyv (av, n, nn);
}

// --------------------

template <class A>
void sol (A* &av, A* &bv, int n, int m) { // time slice
  // scalar in memory
  // in: a initial
  // out: a final
  // m steps
  // grid points 0,1,..,n,n+1
  typename A::ptr a = (typename A::ptr)av;
  typename A::ptr b = (typename A::ptr)bv;
  for (int j=0; j<m; j++) {
    pfcomm.send (&a[0], &a[n+2], 1);
#   ifdef OPENMP
#   pragma omp parallel for
#   endif
    for (int i=1; i<=n; i++)
      b [i] = kern (a [i-1], a [i], a [i+1]);
    b[0] = b[n];
    b[n+1] = b[1];
    swap2 (a, b);
  }
}

// --------------------

template <class A>
void comp2v (A* av, A* bv, int n, int m) { // comp scalar a, vec b
  typename A::ptr a = (typename A::ptr)av;
  typename A::ptr b = (typename A::ptr)bv;
  int o = A::length;
  n *= o;
  // cout<<"scalar :";
  // print (a, n);
  // cout<<"vector :";
  // print (b, n);
  typename A::base s = 0.f, ma = 0.f;
  for (int i=0; i<o; i++) {
    for (int j=0; j<n/o; j++) {
      // cout<<i* (n/o)+j<<"\t"<<i+j*o<<"\n";
      typename A::base d = abs (a [(i* (n/o)+j+m) % n] - b [i+j*o]);
      // cout<<a [(i* (n/o)+j+m)%n]<<" "<<b [i+j*o]<<"\n";
      s += d;
      ma = max (ma, d);
    }
  }
  cout<<"error max="<<ma<<" error sum="<<s<<"\n";
}

// --------------------

void part (int &i0, int &i1, int n) {
# ifdef OPENMP
  int p = omp_get_num_threads ();
  int j = omp_get_thread_num ();
  i0 = (j* (long)n)/p;
  i1 = ( (j+1)* (long)n)/p;
# else
  i0 = 0;
  i1 = n;
# endif
}

template <class A>
void diagvu2 (A* av, A* bv, int m) { // 2*unrolled space-time slice
  typename A::ptr ap = (typename A::ptr)av;
  typename A::ptr bp = (typename A::ptr)bv;
  int o = A::length;
  A d0 = *(A*)&bp [0];
  A d1 = *(A*)&bp [o];
  for (int i=0; i<2*m*o; i+=2*o) {
    A a0 = *(A*)&ap [i];
    A a1 = *(A*)&ap [i+o];
    A e0 = kern (a0, a1, d0);
    A e1 = kern (a1, d0, d1);
    *(A*)&bp [i+2*o] = e0;
    *(A*)&bp [i+3*o] = e1;
    d0 = e0;
    d1 = e1;
  }
}

// ---------------
// optimize the following kernel
// ---------------

#ifndef UNROLLED
template <class A>
void diagvuu (A* av, A* bv, A* iv, A* jv, int m) {
  typename A::ptr ap = (typename A::ptr)av;
  typename A::ptr bp = (typename A::ptr)bv;
  typename A::ptr ip = (typename A::ptr)iv;
  typename A::ptr jp = (typename A::ptr)jv;
  int o =A::length;
  assert (m%2 == 0);
  *(A*)&ap[0] = *(A*)&ip [-2*o];
  *(A*)&ap[o] = *(A*)&ip [-o];
  A d[WIDTH+2];
#pragma unroll
  for (int k=0; k<WIDTH; k++)
    d[k+2] = *(A*)&ip [o*k];
  for (int i=0; i<2*m*o; i+=4*o) {
    d[0] = *(A*)&ap [i];
    d[1] = *(A*)&ap [i+o];
    A e[WIDTH+2];
    e[0] = *(A*)&ap [i+2*o];
    e[1] = *(A*)&ap [i+3*o];
#pragma unroll
    for (int k=0; k<WIDTH; k++)
      e[k+2] = kern (d[k], d[k+1], d[k+2]);
#pragma unroll
    for (int k=0; k<WIDTH; k++)
      d[k+2] = kern (e[k], e[k+1], e[k+2]);
    *(A*)&bp [i+2*o] = e[WIDTH];
    *(A*)&bp [i+3*o] = e[WIDTH+1];
    *(A*)&bp [i+4*o] = d[WIDTH];
    *(A*)&bp [i+5*o] = d[WIDTH+1];
  }
#pragma unroll
  for (int k=0; k<WIDTH; k++)
    *(A*)&jp [o*k] = d[k+2];
}

#else //  UNROLLED

#define UGT(a,b) if ((a)<(WIDTH)) {b;}
#define UEQ(a,b) if ((a)==(WIDTH)) {b;}

template <class A>
void diagvuu (A* av, A* bv, A* iv, A* jv, int m) { // u*unrolled in space, 2*in time, space-time slice
  typename A::ptr ap = (typename A::ptr)av;
  typename A::ptr bp = (typename A::ptr)bv;
  typename A::ptr ip = (typename A::ptr)iv;
  typename A::ptr jp = (typename A::ptr)jv;
  int o =A::length;
  assert (m%2 == 0);
  *(A*)&ap[0] = *(A*)&ip [-2*o];
  *(A*)&ap[o] = *(A*)&ip [-o];
  A d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11;
  A d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23;
  UGT(0, d0 = *(A*)&ip [o*0]);
  UGT(1, d1 = *(A*)&ip [o*1]);
  UGT(2, d2 = *(A*)&ip [o*2]);
  UGT(3, d3 = *(A*)&ip [o*3]);
  UGT(4, d4 = *(A*)&ip [o*4]);
  UGT(5, d5 = *(A*)&ip [o*5]);
  UGT(6, d6 = *(A*)&ip [o*6]);
  UGT(7, d7 = *(A*)&ip [o*7]);
  UGT(8, d8 = *(A*)&ip [o*8]);
  UGT(9, d9 = *(A*)&ip [o*9]);
  UGT(10, d10 = *(A*)&ip [o*10]);
  UGT(11, d11 = *(A*)&ip [o*11]);
  UGT(12, d12 = *(A*)&ip [o*12]);
  UGT(13, d13 = *(A*)&ip [o*13]);
  UGT(14, d14 = *(A*)&ip [o*14]);
  UGT(15, d15 = *(A*)&ip [o*15]);
  UGT(16, d16 = *(A*)&ip [o*16]);
  UGT(17, d17 = *(A*)&ip [o*17]);
  UGT(18, d18 = *(A*)&ip [o*18]);
  UGT(19, d19 = *(A*)&ip [o*19]);
  UGT(20, d20 = *(A*)&ip [o*20]);
  UGT(21, d21 = *(A*)&ip [o*21]);
  UGT(22, d22 = *(A*)&ip [o*22]);
  UGT(23, d23 = *(A*)&ip [o*23]);
  for (int i=0; i<2*m*o; i+=4*o) {
    A a0 = *(A*)&ap [i];
    A a1 = *(A*)&ap [i+o];
    A a2 = *(A*)&ap [i+2*o];
    A a3 = *(A*)&ap [i+3*o];
    A e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11;
    A e12, e13, e14, e15, e16, e17, e18, e19, e20, e21, e22, e23;
    UGT(0, e0 = kern (a0, a1, d0));
    UGT(1, e1 = kern (a1, d0, d1));
    UGT(2, e2 = kern (d0, d1, d2));
    UGT(3, e3 = kern (d1, d2, d3));
    UGT(4, e4 = kern (d2, d3, d4));
    UGT(5, e5 = kern (d3, d4, d5));
    UGT(6, e6 = kern (d4, d5, d6));
    UGT(7, e7 = kern (d5, d6, d7));
    UGT(8, e8 = kern (d6, d7, d8));
    UGT(9, e9 = kern (d7, d8, d9));
    UGT(10, e10 = kern (d8, d9, d10));
    UGT(11, e11 = kern (d9, d10, d11));
    UGT(12, e12 = kern (d10, d11, d12));
    UGT(13, e13 = kern (d11, d12, d13));
    UGT(14, e14 = kern (d12, d13, d14));
    UGT(15, e15 = kern (d13, d14, d15));
    UGT(16, e16 = kern (d14, d15, d16));
    UGT(17, e17 = kern (d15, d16, d17));
    UGT(18, e18 = kern (d16, d17, d18));
    UGT(19, e19 = kern (d17, d18, d19));
    UGT(20, e20 = kern (d18, d19, d20));
    UGT(21, e21 = kern (d19, d20, d21));
    UGT(22, e22 = kern (d20, d21, d22));
    UGT(23, e23 = kern (d21, d22, d23));

    UGT(0, d0 = kern (a2, a3, e0));
    UGT(1, d1 = kern (a3, e0, e1));
    UGT(2, d2 = kern (e0, e1, e2));
    UGT(3, d3 = kern (e1, e2, e3));
    UGT(4, d4 = kern (e2, e3, e4));
    UGT(5, d5 = kern (e3, e4, e5));
    UGT(6, d6 = kern (e4, e5, e6));
    UGT(7, d7 = kern (e5, e6, e7));
    UGT(8, d8 = kern (e6, e7, e8));
    UGT(9, d9 = kern (e7, e8, e9));
    UGT(10, d10 = kern (e8, e9, e10));
    UGT(11, d11 = kern (e9, e10, e11));
    UGT(12, d12 = kern (e10, e11, e12));
    UGT(13, d13 = kern (e11, e12, e13));
    UGT(14, d14 = kern (e12, e13, e14));
    UGT(15, d15 = kern (e13, e14, e15));
    UGT(16, d16 = kern (e14, e15, e16));
    UGT(17, d17 = kern (e15, e16, e17));
    UGT(18, d18 = kern (e16, e17, e18));
    UGT(19, d19 = kern (e17, e18, e19));
    UGT(20, d20 = kern (e18, e19, e20));
    UGT(21, d21 = kern (e19, e20, e21));
    UGT(22, d22 = kern (e20, e21, e22));
    UGT(23, d23 = kern (e21, e22, e23));

    UEQ(2, *(A*)&bp [i+2*o] = e0; *(A*)&bp [i+3*o] = e1);
    UEQ(3, *(A*)&bp [i+2*o] = e1; *(A*)&bp [i+3*o] = e2);
    UEQ(4, *(A*)&bp [i+2*o] = e2; *(A*)&bp [i+3*o] = e3);
    UEQ(5, *(A*)&bp [i+2*o] = e3; *(A*)&bp [i+3*o] = e4);
    UEQ(6, *(A*)&bp [i+2*o] = e4; *(A*)&bp [i+3*o] = e5);
    UEQ(7, *(A*)&bp [i+2*o] = e5; *(A*)&bp [i+3*o] = e6);
    UEQ(8, *(A*)&bp [i+2*o] = e6; *(A*)&bp [i+3*o] = e7);
    UEQ(9, *(A*)&bp [i+2*o] = e7; *(A*)&bp [i+3*o] = e8);
    UEQ(10, *(A*)&bp [i+2*o] = e8; *(A*)&bp [i+3*o] = e9);
    UEQ(11, *(A*)&bp [i+2*o] = e9; *(A*)&bp [i+3*o] = e10);
    UEQ(12, *(A*)&bp [i+2*o] = e10; *(A*)&bp [i+3*o] = e11);
    UEQ(13, *(A*)&bp [i+2*o] = e11; *(A*)&bp [i+3*o] = e12);
    UEQ(14, *(A*)&bp [i+2*o] = e12; *(A*)&bp [i+3*o] = e13);
    UEQ(15, *(A*)&bp [i+2*o] = e13; *(A*)&bp [i+3*o] = e14);
    UEQ(16, *(A*)&bp [i+2*o] = e14; *(A*)&bp [i+3*o] = e15);
    UEQ(17, *(A*)&bp [i+2*o] = e15; *(A*)&bp [i+3*o] = e16);
    UEQ(18, *(A*)&bp [i+2*o] = e16; *(A*)&bp [i+3*o] = e17);
    UEQ(19, *(A*)&bp [i+2*o] = e17; *(A*)&bp [i+3*o] = e18);
    UEQ(20, *(A*)&bp [i+2*o] = e18; *(A*)&bp [i+3*o] = e19);
    UEQ(21, *(A*)&bp [i+2*o] = e19; *(A*)&bp [i+3*o] = e20);
    UEQ(22, *(A*)&bp [i+2*o] = e20; *(A*)&bp [i+3*o] = e21);
    UEQ(23, *(A*)&bp [i+2*o] = e21; *(A*)&bp [i+3*o] = e22);
    UEQ(24, *(A*)&bp [i+2*o] = e22; *(A*)&bp [i+3*o] = e23);

    UEQ(2, *(A*)&bp [i+4*o] = d0; *(A*)&bp [i+5*o] = d1);
    UEQ(3, *(A*)&bp [i+4*o] = d1; *(A*)&bp [i+5*o] = d2);
    UEQ(4, *(A*)&bp [i+4*o] = d2; *(A*)&bp [i+5*o] = d3);
    UEQ(5, *(A*)&bp [i+4*o] = d3; *(A*)&bp [i+5*o] = d4);
    UEQ(6, *(A*)&bp [i+4*o] = d4; *(A*)&bp [i+5*o] = d5);
    UEQ(7, *(A*)&bp [i+4*o] = d5; *(A*)&bp [i+5*o] = d6);
    UEQ(8, *(A*)&bp [i+4*o] = d6; *(A*)&bp [i+5*o] = d7);
    UEQ(9, *(A*)&bp [i+4*o] = d7; *(A*)&bp [i+5*o] = d8);
    UEQ(10, *(A*)&bp [i+4*o] = d8; *(A*)&bp [i+5*o] = d9);
    UEQ(11, *(A*)&bp [i+4*o] = d9; *(A*)&bp [i+5*o] = d10);
    UEQ(12, *(A*)&bp [i+4*o] = d10; *(A*)&bp [i+5*o] = d11);
    UEQ(13, *(A*)&bp [i+4*o] = d11; *(A*)&bp [i+5*o] = d12);
    UEQ(14, *(A*)&bp [i+4*o] = d12; *(A*)&bp [i+5*o] = d13);
    UEQ(15, *(A*)&bp [i+4*o] = d13; *(A*)&bp [i+5*o] = d14);
    UEQ(16, *(A*)&bp [i+4*o] = d14; *(A*)&bp [i+5*o] = d15);
    UEQ(17, *(A*)&bp [i+4*o] = d15; *(A*)&bp [i+5*o] = d16);
    UEQ(18, *(A*)&bp [i+4*o] = d16; *(A*)&bp [i+5*o] = d17);
    UEQ(19, *(A*)&bp [i+4*o] = d17; *(A*)&bp [i+5*o] = d18);
    UEQ(20, *(A*)&bp [i+4*o] = d18; *(A*)&bp [i+5*o] = d19);
    UEQ(21, *(A*)&bp [i+4*o] = d19; *(A*)&bp [i+5*o] = d20);
    UEQ(22, *(A*)&bp [i+4*o] = d20; *(A*)&bp [i+5*o] = d21);
    UEQ(23, *(A*)&bp [i+4*o] = d21; *(A*)&bp [i+5*o] = d22);
    UEQ(24, *(A*)&bp [i+4*o] = d22; *(A*)&bp [i+5*o] = d23);
  }
  UGT(0, *(A*)&jp [o*0] = d0);
  UGT(1, *(A*)&jp [o*1] = d1);
  UGT(2, *(A*)&jp [o*2] = d2);
  UGT(3, *(A*)&jp [o*3] = d3);
  UGT(4, *(A*)&jp [o*4] = d4);
  UGT(5, *(A*)&jp [o*5] = d5);
  UGT(6, *(A*)&jp [o*6] = d6);
  UGT(7, *(A*)&jp [o*7] = d7);
  UGT(8, *(A*)&jp [o*8] = d8);
  UGT(9, *(A*)&jp [o*9] = d9);
  UGT(10, *(A*)&jp [o*10] = d10);
  UGT(11, *(A*)&jp [o*11] = d11);
  UGT(12, *(A*)&jp [o*12] = d12);
  UGT(13, *(A*)&jp [o*13] = d13);
  UGT(14, *(A*)&jp [o*14] = d14);
  UGT(15, *(A*)&jp [o*15] = d15);
  UGT(16, *(A*)&jp [o*16] = d16);
  UGT(17, *(A*)&jp [o*17] = d17);
  UGT(18, *(A*)&jp [o*18] = d18);
  UGT(19, *(A*)&jp [o*19] = d19);
  UGT(20, *(A*)&jp [o*20] = d20);
  UGT(21, *(A*)&jp [o*21] = d21);
  UGT(22, *(A*)&jp [o*22] = d22);
  UGT(23, *(A*)&jp [o*23] = d23);
}
#endif // UNROLLED

// ---------------

template <class A>
void calcvu (A* s0v, A* s1v, int n, int m) {
  // in:  s0 initial, points 0..n-1+2*m
  // out: s1 final, points 0..n-1
  // m steps
  typename A::ptr s0 = (typename A::ptr)s0v;
  typename A::ptr s1 = (typename A::ptr)s1v;
# ifdef OPENMP
  int p1 = omp_get_max_threads ();
# pragma omp parallel for
  for (int p0=0; p0<p1; p0++) 
# endif
    {
      int o =A::length;
      typename A::ptr a, b;
      a = allocate<A> ((m+1)*2);
      b = allocate<A> ((m+1)*2);
      int i0, i1;
      part (i0, i1, n);
      assert (m*o<i1-i0);
      *(A*)&a [0] = *(A*)&s0 [i0];
      *(A*)&a [o] = *(A*)&s0 [i0+o];
      for (int i=1; i<m; i++) { // 2* unroll initial
	*(A*)&b [0] = *(A*)&s0 [2*o*i+i0];
	*(A*)&b [o] = *(A*)&s0 [2*o*i+o+i0];
	diagvu2 ((A*)a, (A*)b, i);
	swap2 (a, b);
      }
      for (int i=i0; i<i1; i+=o* (WIDTH)) { // u* unroll block
	typename A::ptr ip = &s0 [i+2*o*m];
	typename A::ptr jp = &s1 [i];
	diagvuu ((A*)a, (A*)b, (A*)ip, (A*)jp, m);
	swap2 (a, b);
      }
    }
}
// --------------------


template <class A>
A* run1 (int n, int m) {
  typename A::ptr s0, s1;
  s0 = allocate<A> (n+2);
  s1 = allocate<A> (n+2);
  n *= A::length;
  init (s0, n, 2);
  realtime r;
  r.start ();
  for (int t=0; t<TIMEBLK; t++)
    sol (s0, s1, n, m);
  r.stop ();
  if (pfcomm.id == 0)
    cout << "t="<<r.elapsed ()
	 << " flop=" << n * 4. * m / r.elapsed ()
	 << "\n";
  return (A*)s0;
}

template <class A>
A* run2vu (int n, int m) {
  typename A::ptr s2, s3;
  int o = A::length;
  s2 = allocate<A> (n+2*m);
  s3 = allocate<A> (n+2*m);
  n *= o;
  assert(n% (o* (WIDTH)) == 0);
  initv ( (A*)s2, n, 2*o*m);
  realtime r;
  r.start ();
  for (int t=0; t<TIMEBLK; t++) {
    calcvu ( (A*)s2, (A*)s3, n, m);
    swap2 (s2, s3);
    if (t+1<TIMEBLK)
      copyv ( (A*)s2, n, 2*o*m);
  }
  r.stop ();
  if (pfcomm.id == 0)
    cout << "t="<<r.elapsed ()
	 << " flop=" << n * 4. * m / r.elapsed ()
      //<< " [fltot=" << (n+m*o) * 4. * m / r.elapsed ()  << "]"
	 << "\n";
  return (A*)s2;
}

// --------------------

int main (int argc, char *argv[]) {
  pfcomm.init(argc, argv);

  // single or double precision
  // choose one of the vector instructions

#ifdef SCALAR
#ifdef FLOAT
  typedef real32 vec;
#else // FLOAT
  typedef real64 vec;
#endif // FLOAT
#endif

#if defined(SSE) || defined(ALTIVEC) || defined(NEON)
#ifdef FLOAT
  typedef SVec<real32, 4> vec;
#else // FLOAT
  typedef SVec<real64, 2> vec;
#endif // FLOAT
#endif

#ifdef AVX
#ifdef FLOAT
  typedef SVec<real32, 8> vec;
#else // FLOAT
  typedef SVec<real64, 4> vec;
#endif // FLOAT
#endif

#if defined(PHI) || defined(AVX512)
#ifdef FLOAT
  typedef SVec<real32, 16> vec;
#else // FLOAT
  typedef SVec<real64, 8> vec;
#endif // FLOAT
#endif

  int o = vec::length;
  assert ((WIDTH) > 1 && (WIDTH) <= 24);
  int n = GRIDSIZE;
  int p = WIDTH;
# ifdef OPENMP
  p *= omp_get_max_threads ();
# endif
  n = (n + p * o - 1) / (p * o);
  n *= p * pfcomm.pr;
  int m = TIMESTEP;
  m = (m+1) & ~ 1; // even
  // cout << "n=" << n << "\n";

#ifdef CHECK
  // scalar time slice vs unrolled vector space-time slice
  // check small gridsizes and small number of timesteps
  comp2v (run1<vec> (n,m), run2vu<vec> (n,m), n, m);
#else // CHECK
  // run bechmark
  // large gridsizes
  run2vu<vec> (n,m);
#endif // CHECK
  pfcomm.finalize();
  return 0;
}
