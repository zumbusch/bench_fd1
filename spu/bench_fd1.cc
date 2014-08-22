
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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <spu_intrinsics.h>
#include <simdmath.h>

#define ITER 10000000

class realtime {
private:
  time_t tp0, tp1;
  //clock_t tp0, tp1;
public:
  realtime () {
  }
  ~realtime () {}
  float r;
  float res() {
    return 1e-9;
  }
  void start () {
   tp0 = time(0);
   //tp0 = clock();
  }
  void stop () {
   tp1 = time(0);
   //tp1 = clock();
  }
  float elapsed () {
    return (tp1 - tp0) / (float)(ITER);
  }
};

float* allocate (size_t s) {
  float* a;
  a = (float*)malloc (s * 4*sizeof (float));
  return a;
}


// number of float values per process
// size of local O(memory/(2*sizeof(real))
// tuning parameter, larger is better, fill SPU memory

#ifndef GRIDSIZE
#define GRIDSIZE 1024*20
#endif

// tuning parameter, fill (L1) cache
// vector overhead O(TIMESTEP^2)
// even number
#ifndef TIMESTEP
#define TIMESTEP 2
#endif

// number of vectors in flight
// 1 < WIDTH <= 24
// is related to the
//     number of registers/
//     pipeline length/
//     in or out of order execution
// SPU:    15, 20
#ifndef WIDTH
#define WIDTH 20
#endif

// unroll kernel code
#define UNROLLED 

// repeat algorithm several times
#define TIMEBLK 1 


// --------------------------


typedef vec_float4 vec;
#define A vec

// --------------------
// 3pt FD stencil 
// --------------------
A kern (A a, A b, A c) {
  return spu_madd (spu_madd (spu_splats (.5f), b, spu_splats (0.f)),
		   spu_add (a, c),
		   spu_splats (.25f));
}

// --------------------------

template <class B>
void swap2 (B &a, B &b) {
  B t = a; 
  a = b;
  b = t;
}

// --------------------


void copyv (A* av, int n, int nn) { // overlap o
  float* a = (float*)av;
  int o = 4;
  for (int i=0; i<nn; i+=o) {
    __vector unsigned char c = {4,5,6,7, 8,9,10,11, 12,13,14,15, 0,1,2,3};
    A b = *(A*)&a[i];
    *(A*)&a[n+i] = spu_shuffle (b, b, c);
  }
}

void initv (A* av, int n, int nn) { // initial data, t=0, overlap o
  float* a = (float*)av;
  int o = 4;
  int ng = n * 1;
  int ig = n * 0;
  float y = 2.f / ng;
  y = y*y;
  A xx = {0.f, 1.f, 2.f, 3.f};
  xx = spu_madd(xx, spu_splats((float)ng / 4), spu_splats(0.f));

  for (int i=0; i<n; i+=o) {
    A x = spu_add(xx, spu_splats((float) (i+ig) / 4));
    *(A*)&a [i] = spu_madd(x, spu_madd(x, spu_splats(y), spu_splats(0.f)), spu_splats(0.f));
  }
  copyv (av, n, nn);
}

// --------------------


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

inline void diagvu2 (A* av, A* bv, int m) { // 2*unrolled space-time slice
  float* ap = (float*)av;
  float* bp = (float*)bv;
  int o = 4;
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


#define UGT(a,b) if ((a)<(WIDTH)) {b;}
#define UEQ(a,b) if ((a)==(WIDTH)) {b;}

inline void diagvuu (A* av, A* bv, A* iv, A* jv) { // u*unrolled in space, 2*in time, space-time slice
  int m = TIMESTEP;
  float* ap = (float*)av;
  float* bp = (float*)bv;
  float* ip = (float*)iv;
  float* jp = (float*)jv;
  int o = 4;

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
#pragma unroll
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

// ---------------

void calcvu (A* s0v, A* s1v, int n) {
  // in:  s0 initial, points 0..n-1+2*m
  // out: s1 final, points 0..n-1
  // m steps
  int m = TIMESTEP;
  float* s0 = (float*)s0v;
  float* s1 = (float*)s1v;
  int p0=0;
      int o =4;
      float* a, *b;
      a = allocate ((m+1)*2);
      b = allocate ((m+1)*2);
      int i0, i1;
      part (i0, i1, n);
      assert (m*o<i1-i0);
      for (int iter = 0; iter<ITER; iter++) {
      *(A*)&a [0] = *(A*)&s0 [i0];
      *(A*)&a [o] = *(A*)&s0 [i0+o];
      for (int i=1; i<m; i++) { // 2* unroll initial
	*(A*)&b [0] = *(A*)&s0 [2*o*i+i0];
	*(A*)&b [o] = *(A*)&s0 [2*o*i+o+i0];
	diagvu2 ((A*)a, (A*)b, i);
	swap2 (a, b);
      /*
      for (int i=i0; i<i1; i+=o* (WIDTH)) { // u* unroll block
	float* ip = &s0 [i+2*o*m];
	float* jp = &s1 [i];
	diagvuu ((A*)a, (A*)b, (A*)ip, (A*)jp);
	swap2 (a, b);
      }
      */
      for (int i=i0; i<i1; i+=2*o* (WIDTH)) { // u* unroll block
	float* ip = &s0 [i+2*o*m];
	float* jp = &s1 [i];
	diagvuu ((A*)a, (A*)b, (A*)ip, (A*)jp);
	diagvuu ((A*)b, (A*)a, (A*)(ip+o*(WIDTH)), (A*)(jp+o*(WIDTH)));
      }

    }
  }
}
// --------------------


A* run2vu (int n) {
  int m = TIMESTEP;
  float* s2, *s3;
  int o = 4;
  s2 = allocate (n+2*m);
  s3 = allocate (n+2*m);
  n *= o;
  assert(n% (o* (WIDTH)) == 0);
  initv ( (A*)s2, n, 2*o*m);
  realtime r;
  r.start ();
  for (int t=0; t<TIMEBLK; t++) {
    calcvu ( (A*)s2, (A*)s3, n);
    swap2 (s2, s3);
//    if (t+1<TIMEBLK)
//      copyv ( (A*)s2, n, 2*o*m);
  }
  r.stop ();
  printf ("t=%g flop=%g\n", r.elapsed (), n * 4. * m / r.elapsed ());
      //<< " [fltot=" << (n+m*o) * 4. * m / r.elapsed ()  << "]"
  return (A*)s2;
}

// --------------------

int main (int argc, char *argv[]) {
  int o = 4;
  assert ((WIDTH) > 1 && (WIDTH) <= 24);
  int n = GRIDSIZE;
  int p = WIDTH;
  n = (n + p * o * 2 - 1) / (p * o * 2);
  n *= p ;
  int m = TIMESTEP;
  assert(TIMESTEP % 2 == 0);

  // run bechmark
  run2vu (n);
  return 0;
}
