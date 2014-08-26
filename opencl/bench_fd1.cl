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

// optimize the kernel "diagvuu"

#ifdef FLOAT
 #define real float
 #define real2 float2
#else
 #pragma OPENCL EXTENSION cl_khr_fp64 : enable
 #define real double
 #define real2 double2
#endif


// ----------------------------------------------------------------------
// initial data
// ----------------------------------------------------------------------

__kernel void kernel0 (__global real *gx, int m, int dev_no) {
  int i = get_local_id (0);
  int k0 = get_group_id (0) *LOCAL*WIDTH*GRID_LOCAL;
  int o = 2*TIMESTEP*LOCAL;
  for (int n=0; n<GRID_LOCAL; n++)
    for (int ii=0; ii<WIDTH; ii++) {
      real y = ((m * dev_no / LOCAL + get_group_id (0) *WIDTH*GRID_LOCAL+ n*WIDTH + ii) + i*((m*DEV_MAX)/LOCAL)) / (real) (m * DEV_MAX);
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

__kernel void kernel1 (__global real *gx, int m) {
  int i = get_local_id (0);
  int k0 = get_group_id (0) *LOCAL + m;
  gx[k0 + i] = gx[k0 + (i+LOCAL-1)%LOCAL];
}

// ----------------------------------------------------------------------

inline real kern (real a, real b, real c) {
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

void diagvu2 (__global real2 *ap, __global real2 *bp, int m) {
  // 2*unrolled space-time slice
  int id = get_local_id (0);
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

void diagvuu (__global real2 *ap, __global real2 *bp, __global real *ip, __global real *jp) {
  // u*unrolled in space, 2*in time, space-time slice
  int id = get_local_id (0);
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


__kernel void kernel2 (__global real *s0, __global real *s1,
		       __global real *ga, __global real *gb) {
  int id = get_local_id (0);
  int k1 = get_group_id (0) *LOCAL*2*(TIMESTEP+1);

  __global real2 *a = (__global real2*)&ga[k1];
  __global real2 *b = (__global real2*)&gb[k1];
  int i0 = get_group_id (0) *LOCAL*GRID_LOCAL*WIDTH;
  int i1 = (get_group_id (0)+1) *LOCAL*GRID_LOCAL*WIDTH;
  a [0+id] = (real2){s0 [i0+id], s0 [i0+LOCAL+id]};
  for (int i=1; i<TIMESTEP; i++) { // 2* unroll initial
    b [0+id] = (real2){s0 [2*LOCAL*i+i0+id], s0 [2*LOCAL*i+LOCAL+i0+id]};
    diagvu2 (a, b, i);
    __global real2 *c = a;
    a = b;
    b = c;
  }

  // if (GRID_LOCAL%2 == 1) {
  //   for (int i=i0; i<i1; i+=LOCAL*(WIDTH)) { // u* unroll block
  //     __global real *ip = &s0 [i+2*LOCAL*(TIMESTEP-1)];
  //     __global real *jp = &s1 [i];
  //     diagvuu (a, b, ip, jp);
  //     __global real *c = a;
  //     a = b;
  //     b = c;
  //   }
  // } else {
  for (int i=i0; i<i1; i+=2*LOCAL*(WIDTH)) { // u* unroll block
    __global real *ip = &s0 [i+2*LOCAL*(TIMESTEP-1)];
    __global real *jp = &s1 [i];
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


void diagvu2 (__global real *ap, __global real *bp, int m) {
  // 2*unrolled space-time slice
  int id = get_local_id (0);
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

void diagvuu (__global real *ap, __global real *bp, __global real *ip, __global real *jp) {
  // u*unrolled in space, 2*in time, space-time slice
  int id = get_local_id (0);
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


__kernel void kernel2 (__global real *s0, __global real *s1,
		       __global real *ga, __global real *gb) {
  int id = get_local_id (0);
  int k1 = get_group_id (0) *LOCAL*2*(TIMESTEP+1);

  __global real *a = &ga[k1];
  __global real *b = &gb[k1];
  int i0 = get_group_id (0) *LOCAL*GRID_LOCAL*WIDTH;
  int i1 = (get_group_id (0)+1) *LOCAL*GRID_LOCAL*WIDTH;
  a [0+id] = s0 [i0+id];
  a [LOCAL+id] = s0 [i0+LOCAL+id];
  for (int i=1; i<TIMESTEP; i++) { // 2* unroll initial
    b [0+id] = s0 [2*LOCAL*i+i0+id];
    b [LOCAL+id] = s0 [2*LOCAL*i+LOCAL+i0+id];
    diagvu2 (a, b, i);
    __global real *c = a;
    a = b;
    b = c;
  }

  // if (GRID_LOCAL%2 == 1) {
  //   for (int i=i0; i<i1; i+=LOCAL*(WIDTH)) { // u* unroll block
  //     __global real *ip = &s0 [i+2*LOCAL*(TIMESTEP-1)];
  //     __global real *jp = &s1 [i];
  //     diagvuu (a, b, ip, jp);
  //     __global real *c = a;
  //     a = b;
  //     b = c;
  //   }
  // } else {
  for (int i=i0; i<i1; i+=2*LOCAL*(WIDTH)) { // u* unroll block
    __global real *ip = &s0 [i+2*LOCAL*(TIMESTEP-1)];
    __global real *jp = &s1 [i];
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

void diagvu2 (__global real2 *ap, __global real2 *bp, int m) {
  // 2*unrolled space-time slice
  int ida = get_local_id (0) + mul24(WRP*(TIMESTEP+1), get_local_id (1));

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

void diagvuu (__global real2 *ap, __global real2 *bp, __global real *ip, __global real *jp) {
  // u*unrolled in space, 2*in time, space-time slice
  int id = get_local_id (0) + mul24(WRP, get_local_id (1));
  int ida = get_local_id (0) + mul24(WRP*(TIMESTEP+1), get_local_id (1));

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


__kernel void kernel2 (__global real *s0, __global real *s1,
		       __global real *ga, __global real *gb) {
  int id = get_local_id (0) + mul24(WRP, get_local_id (1));
  int ida = get_local_id (0) + mul24(WRP*(TIMESTEP+1), get_local_id (1));
  int k1 = get_group_id (0) *LOCAL*2*(TIMESTEP+1);

  __global real2 *a = (__global real2*)&ga[k1];
  __global real2 *b = (__global real2*)&gb[k1];
  int i0 = get_group_id (0) *LOCAL*GRID_LOCAL*WIDTH;
  int i1 = (get_group_id (0)+1) *LOCAL*GRID_LOCAL*WIDTH;
  a [0+ida] = (real2){s0 [i0+id], s0 [i0+LOCAL+id]};
  for (int i=1; i<TIMESTEP; i++) { // 2* unroll initial
    b [0+ida] = (real2){s0 [2*LOCAL*i+i0+id], s0 [2*LOCAL*i+LOCAL+i0+id]};
    diagvu2 (a, b, i);
    __global real2 *c = a;
    a = b;
    b = c;
  }

  // if (GRID_LOCAL%2 == 1) {
  //   for (int i=i0; i<i1; i+=LOCAL*(WIDTH)) { // u* unroll block
  //     __global real *ip = &s0 [i+2*LOCAL*(TIMESTEP-1)];
  //     __global real *jp = &s1 [i];
  //     diagvuu (a, b, ip, jp);
  //     __global real2 *c = a;
  //     a = b;
  //     b = c;
  //   }
  // } else {
  for (int i=i0; i<i1; i+=2*LOCAL*(WIDTH)) { // u* unroll block
    __global real *ip = &s0 [i+2*LOCAL*(TIMESTEP-1)];
    __global real *jp = &s1 [i];
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

void diagvu2 (__global real *ap, __global real *bp, int m) {
  // 2*unrolled space-time slice
  int ida = get_local_id (0) + mul24(WRP*2*(TIMESTEP+1), get_local_id (1));

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

void diagvuu (__global real *ap, __global real *bp, __global real *ip, __global real *jp) {
  // u*unrolled in space, 2*in time, space-time slice
  int id = get_local_id (0) + mul24(WRP, get_local_id (1));
  int ida = get_local_id (0) + mul24(WRP*2*(TIMESTEP+1), get_local_id (1));

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


__kernel void kernel2 (__global real *s0, __global real *s1,
		       __global real *ga, __global real *gb) {
  int id = get_local_id (0) + mul24(WRP, get_local_id (1));
  int ida = get_local_id (0) + mul24(WRP*2*(TIMESTEP+1), get_local_id (1));
  int k1 = get_group_id (0) *LOCAL*2*(TIMESTEP+1);

  __global real *a = &ga[k1];
  __global real *b = &gb[k1];
  int i0 = get_group_id (0) *LOCAL*GRID_LOCAL*WIDTH;
  int i1 = (get_group_id (0)+1) *LOCAL*GRID_LOCAL*WIDTH;
  a [0+ida] = s0 [i0+id];
  a [WRP+ida] = s0 [i0+LOCAL+id];
  for (int i=1; i<TIMESTEP; i++) { // 2* unroll initial
    b [0+ida] = s0 [2*LOCAL*i+i0+id];
    b [WRP+ida] = s0 [2*LOCAL*i+LOCAL+i0+id];
    diagvu2 (a, b, i);
    __global real *c = a;
    a = b;
    b = c;
  }

  // if (GRID_LOCAL%2 == 1) {
  //   for (int i=i0; i<i1; i+=LOCAL*(WIDTH)) { // u* unroll block
  //     __global real *ip = &s0 [i+2*LOCAL*(TIMESTEP-1)];
  //     __global real *jp = &s1 [i];
  //     diagvuu (a, b, ip, jp);
  //     __global real *c = a;
  //     a = b;
  //     b = c;
  //   }
  // } else {
  for (int i=i0; i<i1; i+=2*LOCAL*(WIDTH)) { // u* unroll block
    __global real *ip = &s0 [i+2*LOCAL*(TIMESTEP-1)];
    __global real *jp = &s1 [i];
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

