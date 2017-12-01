// Copyright (c) 2017, Gerhard Zumbusch
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



// --------------------

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "simd.hpp"

int main (int argc, char *argv[]) {
  // single or double precision
  // choose one of the vector instructions

#ifdef SCALAR
#ifdef FLOAT
  typedef real32 vec;
#else // FLOAT
  typedef real64 vec;
#endif // FLOAT
#endif

#if defined(SSE) || defined(ALTIVEC) || defined(SPU) || defined(NEON)
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
  typedef vec::base base;
  typedef base::data_t f;
  vec x, l, r;
  x.set_inc();
  x += base(1);
  l = lrotate(x);
  int err = 0;
  f* p = (f*)&l;
  for (int i=0; i<o; i++)
    if (p[i] != 1+((1+i)%o))
      err++;
  if (err)
    std::cout<< "x = " << x << "\nlrotate " << l << "\n";

  r = rrotate(x);
  err = 0;
  p = (f*)&r;
  for (int i=0; i<o; i++)
    if (p[i] != 1+((o+i-1)%o))
      err++;
  if (err)
    std::cout<< "x = " << x << "\nrrotate " << r << "\n";

  return 0;
}

// make simd_check CC="g++ -DSSE -msse4 -DFLOAT";./simd_check
// make simd_check CC="g++ -DSSE -msse4";./simd_check
// make simd_check CC="g++ -DAVX -mavx2 -DFLOAT";./simd_check
// make simd_check CC="g++ -DAVX -mavx2";./simd_check
// make simd_check CC="g++ -DAVX512 -mavx512f -DFLOAT";./simd_check
// make simd_check CC="g++ -DAVX512 -mavx512f";./simd_check
