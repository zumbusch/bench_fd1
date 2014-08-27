#ifndef SIMD_HPP
#define SIMD_HPP

// Copyright (c) 2011-2014, Gerhard Zumbusch
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//     * Redistributions of source code must retain the above
//       copyright notice, this list of conditions and the following
//       disclaimer.
//     * The names of its contributors may not be used to endorse or
//       promote products derived from this software without specific
//       prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.


#include <math.h>

// a C++ SIMD vector wrapper
//  based on
// x86 SSE 128 bit
// x86 AVX 256 bit
// MIC/Phi 512 bit
// ARM NEON 128 bit
// partial support PowerPC Altivec 128 bit, Cell SPU, AVX512

//----------
//! SIMD vector class
//----------


//! dummy vector
template <class BASE, /*!< base type */
	  int N /*!< length */
	  >
class SVec {
public:
  SVec () {}
};


//! scalar simd class
/*!
  single 32 bit float
*/
class real32 {
public:
  typedef float data_t;
  typedef real32 base;
  typedef real32 *ptr;
  data_t data;
  static const int length = 1;
  static const int size = 4;
  real32 () {}
  ~real32 () {}
  real32 (float m) {
    data = m;
  }
  static const char* name () {
    return "real32";
  }
  friend real32 operator + (const real32 &a, const real32 &b) {
    return a.data + b.data; 
  }
  friend real32 operator - (const real32 &a, const real32 &b) {
    return a.data - b.data; 
  }
  friend real32 operator * (const real32 &a, const real32 &b) {
    return a.data * b.data; 
  }
  friend real32 operator / (const real32 &a, const real32 &b) {
    return a.data / b.data; 
  }
  friend real32 fma (const real32 &a, const real32 &b, const real32 &c) {
    return c + a * b;
  }
  friend real32 fms (const real32 &a, const real32 &b, const real32 &c) {
    return c - a * b;
  }
  real32& operator += (const real32 &a) {
    return *this = data + a.data; 
  }
  real32& operator -= (const real32 &a) {
    return *this = data - a.data; 
  }
  real32& operator *= (const real32 &a) {
    return *this = data * a.data; 
  } 
  real32& operator /= (const real32 &a) {
    return *this = data / a.data; 
  }
  real32 operator - () const {
    return -data; 
  }
  void set_zero () {
    data = 0.f; 
  }
  void set_inc () {
    data = 0.f; 
  }
  void set (const real32& a) {
    data = a.data; 
  }
  friend real32 lshift (const real32 &a, const real32 &b) {
    return b;
  }
  friend real32 rshift (const real32 &a, const real32 &b) {
    return b;
  }
  friend real32 lrotate (const real32 &a) {
    return a;
  }
  friend real32 rrotate (const real32 &a) {
    return a;
  }
  friend real32 min (const real32 &a, const real32 &b) {
    return fminf (a.data, b.data); 
  }
  friend real32 max (const real32 &a, const real32 &b) {
    return fmaxf (a.data, b.data); 
  }
  friend real32 abs (const real32 &a) {
    return fabs (a.data); 
  }	 
  friend real32 sqrt (const real32 &a) {
    return sqrtf (a.data); 
  }	 
  friend real32 rcp (const real32 &a) {
    return 1.f / a.data; 
  }	 
  friend real32 ceil (const real32 &a)	{
    return ceilf (a.data); 
  }	 
  friend real32 floor (const real32 &a)	{
    return floorf (a.data); 
  }	 
  friend real32 round (const real32 &a)	{
    return roundf (a.data); 
  }	 
#ifndef SPU
  void print (std::ostream& j) const {
    j <<data;
  }
  friend std::ostream& operator << (std::ostream& i, real32& a) {
    a.print (i); return i; 
  }
#endif
};

#ifndef SPU
//! scalar simd class
/*!
  single 64 bit float
*/
class real64 {
public:
  typedef double data_t;
  typedef real64 base;
  typedef real64 *ptr;
  data_t data;
  static const int length = 1;
  static const int size = 8;
  real64 () {}
  ~real64 () {}
  real64 (double m) {
    data = m;
  }
  static const char* name () {
    return "real64";
  }
  friend real64 operator + (const real64 &a, const real64 &b) {
    return a.data + b.data; 
  }
  friend real64 operator - (const real64 &a, const real64 &b) {
    return a.data - b.data; 
  }
  friend real64 operator * (const real64 &a, const real64 &b) {
    return a.data * b.data; 
  }
  friend real64 operator / (const real64 &a, const real64 &b) {
    return a.data / b.data; 
  }
  friend real64 fma (const real64 &a, const real64 &b, const real64 &c) {
    return c + a * b;
  }
  friend real64 fms (const real64 &a, const real64 &b, const real64 &c) {
    return c - a * b;
  }
  real64& operator += (const real64 &a) {
    return *this = data + a.data; 
  }
  real64& operator -= (const real64 &a) {
    return *this = data - a.data; 
  }
  real64& operator *= (const real64 &a) {
    return *this = data * a.data; 
  } 
  real64& operator /= (const real64 &a) {
    return *this = data / a.data; 
  }
  real64 operator - () const {
    return -data; 
  }
  void set_zero () {
    data = 0.; 
  }
  void set_inc () {
    data = 0.; 
  }
  void set (const real64& a) {
    data = a.data; 
  }
  friend real64 lshift (const real64 &a, real64 b) {
    return b;
  }
  friend real64 rshift (const real64 &a, real64 b) {
    return b;
  }
  friend real64 lrotate (const real64 &a) {
    return a;
  }
  friend real64 rrotate (const real64 &a) {
    return a;
  }
  friend real64 ceil (const real64 &a)	{
    return ceil (a.data); 
  }	 
  friend real64 floor (const real64 &a)	{
    return floor (a.data); 
  }	 
  friend real64 round (const real64 &a)	{
    return round (a.data); 
  }	 
  friend real64 min (const real64 &a, const real64 &b) {
    return fmin (a.data, b.data); 
  }
  friend real64 max (const real64 &a, const real64 &b) {
    return fmax (a.data, b.data); 
  }
  friend real64 abs (const real64 &a) {
    return abs (a.data); 
  }	 
  friend real64 sqrt (const real64 &a) {
    return sqrtf (a.data); 
  }	 
  friend real64 rcp (const real64 &a) {
    return 1. / a.data; 
  }	 
  void print (std::ostream& j) const {
    j <<data;
  }
  friend std::ostream& operator << (std::ostream& i, real64& a) {
    a.print (i); return i; 
  }
};
#endif // ndef SPU

#if defined (SSE) || defined (AVX)
#include <x86intrin.h>

//! simd vector simd class
/*!
  x86 SSE simd 128 bit register, vector of 4 single 32 bit floats
*/
template <>
class SVec<real32, 4> {
public:
  typedef __m128 data_t;
  typedef real32 base;
  typedef real32 *ptr;
  __m128 data;
public:
  static const int length = 4;
  static const int size = 4;
  SVec () {}
  ~SVec () {}
  SVec (real32 f)	{
    data = _mm_set1_ps (f.data); 
  }
  SVec (float f)	{
    data = _mm_set1_ps (f); 
  }
  SVec (data_t m) {
    data = m; 
  }
  SVec& operator= (real32 f) {
    data = _mm_set1_ps (f.data);
    return *this;
  }
  SVec& operator= (float f) {
    data = _mm_set1_ps (f);
    return *this;
  }
  static const char* name () {
    return "SVec";
  }
  friend SVec operator + (const SVec &a, const SVec &b) {
    return _mm_add_ps (a.data, b.data); 
  }
  friend SVec operator - (const SVec &a, const SVec &b) {
    return _mm_sub_ps (a.data, b.data); 
  } 
  friend SVec operator * (const SVec &a, const SVec &b) {
    return _mm_mul_ps (a.data, b.data); 
  } 
  friend SVec operator / (const SVec &a, const SVec &b) {
    return _mm_div_ps (a.data, b.data); 
  }
  friend SVec fma (const SVec &c, const SVec &b, const SVec &a) {
#ifdef FMA4
    return _mm_macc_ps (c.data, b.data, a.data);
#else
#ifdef FMA
    return _mm_fmadd_ps (c.data, b.data, a.data);
#else
    return _mm_add_ps (a.data, _mm_mul_ps (b.data, c.data));
#endif //FMA
#endif //FMA4
  }
  friend SVec fms (const SVec &c, const SVec &b, const SVec &a) {
#ifdef FMA4
    return _mm_nmacc_ps (c.data, b.data, a.data);
#else
#ifdef FMA
    return _mm_fnmadd_ps (c.data, b.data, a.data);
#else
    return _mm_sub_ps (a.data, _mm_mul_ps (b.data, c.data));
#endif //FMA
#endif //FMA4
  }
  SVec& operator += (const SVec &a) {
    return *this = _mm_add_ps (data, a.data); 
  }
  SVec& operator -= (const SVec &a) {
    return *this = _mm_sub_ps (data, a.data); 
  }
  SVec& operator *= (const SVec &a) {
    return *this = _mm_mul_ps (data, a.data); 
  } 
  SVec& operator /= (const SVec &a) {
    return *this = _mm_div_ps (data, a.data); 
  }
  SVec& operator += (const real32 &f) {
    return *this = _mm_add_ps (data, _mm_set1_ps (f.data)); 
  }
  SVec& operator -= (const real32 &f) {
    return *this = _mm_sub_ps (data, _mm_set1_ps (f.data)); 
  }
  SVec& operator *= (const real32 &f) {
    return *this = _mm_mul_ps (data, _mm_set1_ps (f.data)); 
  }
  SVec& operator /= (const real32 &f) {
    return *this = _mm_div_ps (data, _mm_set1_ps (f.data)); 
  }
  friend SVec operator + (const SVec &a, const real32 &f) {
    return _mm_add_ps (a.data, _mm_set1_ps (f.data)); 
  }
  friend SVec operator - (const SVec &a, const real32 &f) {
    return _mm_sub_ps (a.data, _mm_set1_ps (f.data)); 
  }
  friend SVec operator * (const SVec &a, const real32 &f) {
    return _mm_mul_ps (a.data, _mm_set1_ps (f.data)); 
  }
  friend SVec operator / (const SVec &a, const real32 &f) {
    return _mm_div_ps (a.data, _mm_set1_ps (f.data)); 
  }
  SVec operator - () const {
    return  _mm_xor_ps (_mm_set1_ps (-0.0), data); 
  }
  void set_zero () {
    data = _mm_setzero_ps (); 
  }
  void set_inc () {
    data = (data_t) {0.f, 1.f, 2.f, 3.f};
  }
  void set (const SVec& a) {
    data = a.data; 
  }
#ifdef __SSE4_1__
  // shift left in memory == shift right in little endian register
  friend SVec lshift (const SVec &a, real32 b) {
    SVec ftemp =
#ifdef __AVX__
      _mm_permute_ps (a.data, 0xf9)
#else
      _mm_shuffle_ps (a.data, a.data, 0xf9)
#endif
      ;
    return _mm_insert_ps (ftemp.data, _mm_set1_ps (b.data), 0x30);
  }
  // shift right in memory == shift left in little endian register
  friend SVec rshift (const SVec &a, real32 b) {
    SVec ftemp =
#ifdef __AVX__
      _mm_permute_ps (a.data, 0x90)
#else
      _mm_shuffle_ps (a.data, a.data, 0x90)
#endif
      ;
    return _mm_insert_ps (ftemp.data, _mm_set1_ps (b.data), 0x0 );
    // _mm_shuffle_pd (_mm_set1_pd (b.data), a.data, _MM_SHUFFLE2 (0,1));
  }
  friend SVec lrotate (const SVec &a) {
    SVec ftemp =
#ifdef __AVX__
      _mm_permute_ps (a.data, 0x39)
#else
      _mm_shuffle_ps (a.data, a.data, 0x39)
#endif
      ;
    return ftemp;
  }
  friend SVec rrotate (const SVec &a) {
    SVec ftemp =
#ifdef __AVX__
      _mm_permute_ps (a.data, 0x93)
#else
      _mm_shuffle_ps (a.data, a.data, 0x93)
#endif
      ;
    return ftemp;
  }
#endif // __SSE4_1__
  friend SVec min (const SVec &a, const SVec &b) {
    return _mm_min_ps (a.data, b.data); 
  }
  friend SVec max (const SVec &a, const SVec &b) {
    return _mm_max_ps (a.data, b.data); 
  }
	
  friend SVec abs (const SVec &a) {
    int i = 0x7fffffff;
    return _mm_and_ps (a.data, _mm_set1_ps ( * (float*)&i) ); 
  }	 
  friend SVec sqrt (const SVec &a) {
    return _mm_sqrt_ps (a.data); 
  }	 
  // friend SVec ceil (const SVec &a)	{
  //   return _mm_svml_ceil_ps (a.data); 
  // }	 
  // friend SVec floor (const SVec &a)	{
  //   return _mm_svml_floor_ps (a.data); 
  // }	 
  // friend SVec round (const SVec &a)	{
  //   return _mm_svml_round_ps (a.data); 
  // }	 
  void print (std::ostream& j) const {
    float *x = (float*)&data;
    for (int i=0; i <length; i++)
      j <<x[i] <<" ";
  }
  friend std::ostream& operator << (std::ostream& i, SVec& a) {
    a.print (i); return i; 
  }
};

//! simd vector simd class
/*!
  x86 SSE simd 128 bit register, vector of 2 double 64 bit floats
*/
template <>
class SVec<real64, 2> {
public:
  typedef __m128d data_t;
  typedef real64 base;
  typedef real64 *ptr;
  __m128d data;
public:
  static const int length = 2;
  static const int size = 8;
  SVec () {}
  ~SVec () {}
  SVec (real64 f)	{
    data = _mm_set1_pd (f.data); 
  }
  SVec (double f)	{
    data = _mm_set1_pd (f); 
  }
  SVec (data_t m) {
    data = m; 
  }
  SVec& operator= (real64 f) {
    data = _mm_set1_pd (f.data);
    return *this;
  }
  SVec& operator= (double f) {
    data = _mm_set1_pd (f);
    return *this;
  }
  static const char* name () {
    return "SVec";
  }
  friend SVec operator + (const SVec &a, const SVec &b) {
    return _mm_add_pd (a.data, b.data); 
  }
  friend SVec operator - (const SVec &a, const SVec &b) {
    return _mm_sub_pd (a.data, b.data); 
  } 
  friend SVec operator * (const SVec &a, const SVec &b) {
    return _mm_mul_pd (a.data, b.data); 
  } 
  friend SVec operator / (const SVec &a, const SVec &b) {
    return _mm_div_pd (a.data, b.data); 
  }
  friend SVec fma (const SVec &c, const SVec &b, const SVec &a) {
#ifdef FMA4
    return _mm_macc_pd (c.data, b.data, a.data);
#else
#ifdef FMA
    return _mm_fmadd_pd (c.data, b.data, a.data);
#else
    return _mm_add_pd (a.data, _mm_mul_pd (b.data, c.data));
#endif //FMA
#endif //FMA4
  }
  friend SVec fms (const SVec &c, const SVec &b, const SVec &a) {
#ifdef FMA4
    return _mm_nmacc_pd (c.data, b.data, a.data);
#else
#ifdef FMA
    return _mm_fnmadd_pd (c.data, b.data, a.data);
#else
    return _mm_sub_pd (a.data, _mm_mul_pd (b.data, c.data));
#endif //FMA
#endif //FMA4
  }
  SVec& operator += (const SVec &a) {
    return *this = _mm_add_pd (data, a.data); 
  }
  SVec& operator -= (const SVec &a) {
    return *this = _mm_sub_pd (data, a.data); 
  }
  SVec& operator *= (const SVec &a) {
    return *this = _mm_mul_pd (data, a.data); 
  } 
  SVec& operator /= (const SVec &a) {
    return *this = _mm_div_pd (data, a.data); 
  }
  SVec& operator += (const real64 &f) {
    return *this = _mm_add_pd (data, _mm_set1_pd (f.data)); 
  }
  SVec& operator -= (const real64 &f) {
    return *this = _mm_sub_pd (data, _mm_set1_pd (f.data)); 
  }
  SVec& operator *= (const real64 &f) {
    return *this = _mm_mul_pd (data, _mm_set1_pd (f.data)); 
  }
  SVec& operator /= (const real64 &f) {
    return *this = _mm_div_pd (data, _mm_set1_pd (f.data)); 
  }
  friend SVec operator + (const SVec &a, const real64 &f) {
    return _mm_add_pd (a.data, _mm_set1_pd (f.data)); 
  }
  friend SVec operator - (const SVec &a, const real64 &f) {
    return _mm_sub_pd (a.data, _mm_set1_pd (f.data)); 
  }
  friend SVec operator * (const SVec &a, const real64 &f) {
    return _mm_mul_pd (a.data, _mm_set1_pd (f.data)); 
  }
  friend SVec operator / (const SVec &a, const real64 &f) {
    return _mm_div_pd (a.data, _mm_set1_pd (f.data)); 
  }
  SVec operator - () const {
    return  _mm_xor_pd (_mm_set1_pd (-0.0), data); 
  }
  void set_zero () {
    data = _mm_setzero_pd (); 
  }
  void set_inc () {
    data = (data_t) {0., 1.};
  }
  void set (const SVec& a) {
    data = a.data; 
  }
  // shift left in memory == shift right in little endian register
  friend SVec lshift (const SVec &a, real64 b) {
    return _mm_shuffle_pd (a.data, _mm_set1_pd (b.data), 0x1); 
  }
  // shift right in memory == shift left in little endian register
  friend SVec rshift (const SVec &a, real64 b) {
    return _mm_shuffle_pd (_mm_set1_pd (b.data), a.data, 0x1);
  }
  friend SVec lrotate (const SVec &a) {
    std::cout << "lrotate";
    return _mm_shuffle_pd (a.data, a.data, 0x1); 
  }
  friend SVec rrotate (const SVec &a) {
    return _mm_shuffle_pd (a.data, a.data, 0x1); 
  }
  friend SVec min (const SVec &a, const SVec &b) {
    return _mm_min_pd (a.data, b.data); 
  }
  friend SVec max (const SVec &a, const SVec &b) {
    return _mm_max_pd (a.data, b.data); 
  }
  friend SVec abs (const SVec &a) {
    int i[4] =  {0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff};
    return _mm_and_pd (a.data, * (__m128d*)&i ); 
  }	 
  friend SVec sqrt (const SVec &a) {
    return _mm_sqrt_pd (a.data); 
  }	 
  // friend SVec ceil (const SVec &a)	{
  //   return _mm_svml_ceil_pd (a.data); 
  // }	 
  // friend SVec floor (const SVec &a)	{
  //   return _mm_svml_floor_pd (a.data); 
  // }	 
  // friend SVec round (const SVec &a)	{
  //   return _mm_svml_round_pd (a.data); 
  // }	 
  void print (std::ostream& j) const {
    double *x = (double*)&data;
    for (int i=0; i <length; i++)
      j <<x[i] <<" ";
  }
  friend std::ostream& operator << (std::ostream& i, SVec& a) {
    a.print (i); return i; 
  }
};
#endif // SSE

#ifdef AVX
#include <x86intrin.h>

//! simd vector simd class
/*!
  x86 AVX simd 256 bit register, vector of 8 single 32 bit floats
*/
template <>
class SVec<real32, 8> {
public:
  typedef __m256 data_t;
  typedef real32 base;
  typedef real32 *ptr;
  __m256 data;
public:
  static const int length = 8;
  static const int size = 4;
  SVec () {}
  ~SVec () {}
  SVec (real32 f)	{
    data = _mm256_set1_ps (f.data); 
  }
  SVec (float f)	{
    data = _mm256_set1_ps (f); 
  }
  SVec (data_t m) {
    data = m; 
  }
  SVec& operator= (real32 f) {
    data = _mm256_set1_ps (f.data);
    return *this;
  }
  SVec& operator= (float f) {
    data = _mm256_set1_ps (f);
    return *this;
  }
  static const char* name () {
    return "SVec";
  }
  friend SVec operator + (const SVec &a, const SVec &b) {
    return _mm256_add_ps (a.data, b.data); 
  }
  friend SVec operator - (const SVec &a, const SVec &b) {
    return _mm256_sub_ps (a.data, b.data); 
  } 
  friend SVec operator * (const SVec &a, const SVec &b) {
    return _mm256_mul_ps (a.data, b.data); 
  } 
  friend SVec operator / (const SVec &a, const SVec &b) {
    return _mm256_div_ps (a.data, b.data); 
  }
  friend SVec fma (const SVec &c, const SVec &b, const SVec &a) {
#ifdef FMA4
    return _mm256_macc_ps (c.data, b.data, a.data);
#else
#ifdef FMA
    return _mm256_fmadd_ps (c.data, b.data, a.data);
#else
    return _mm256_add_ps (a.data, _mm256_mul_ps (b.data, c.data));
#endif //FMA
#endif //FMA4
  }
  friend SVec fms (const SVec &c, const SVec &b, const SVec &a) {
#ifdef FMA4
    return _mm256_nmacc_ps (c.data, b.data, a.data);
#else
#ifdef FMA
    return _mm256_fnmadd_ps (c.data, b.data, a.data);
#else
    return _mm256_sub_ps (a.data, _mm256_mul_ps (b.data, c.data));
#endif //FMA
#endif //FMA4
  }
  SVec& operator += (const SVec &a) {
    return *this = _mm256_add_ps (data, a.data); 
  }
  SVec& operator -= (const SVec &a) {
    return *this = _mm256_sub_ps (data, a.data); 
  }
  SVec& operator *= (const SVec &a) {
    return *this = _mm256_mul_ps (data, a.data); 
  } 
  SVec& operator /= (const SVec &a) {
    return *this = _mm256_div_ps (data, a.data); 
  }
  SVec& operator += (const real32 &f) {
    return *this = _mm256_add_ps (data, _mm256_set1_ps (f.data)); 
  }
  SVec& operator -= (const real32 &f) {
    return *this = _mm256_sub_ps (data, _mm256_set1_ps (f.data)); 
  }
  SVec& operator *= (const real32 &f) {
    return *this = _mm256_mul_ps (data, _mm256_set1_ps (f.data)); 
  }
  SVec& operator /= (const real32 &f) {
    return *this = _mm256_div_ps (data, _mm256_set1_ps (f.data)); 
  }
  friend SVec operator + (const SVec &a, const real32 &f) {
    return _mm256_add_ps (a.data, _mm256_set1_ps (f.data)); 
  }
  friend SVec operator - (const SVec &a, const real32 &f) {
    return _mm256_sub_ps (a.data, _mm256_set1_ps (f.data)); 
  }
  friend SVec operator * (const SVec &a, const real32 &f) {
    return _mm256_mul_ps (a.data, _mm256_set1_ps (f.data)); 
  }
  friend SVec operator / (const SVec &a, const real32 &f) {
    return _mm256_div_ps (a.data, _mm256_set1_ps (f.data)); 
  }
  SVec operator - () const {
    return  _mm256_xor_ps (_mm256_set1_ps (-0.0), data); 
  }
  void set_zero () {
    data = _mm256_setzero_ps (); 
  }
  void set_inc () {
    data = (data_t) {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  }
  void set (const SVec& a) {
    data = a.data; 
  }
  friend SVec lshift (const SVec &a, real32 b) {
    SVec ftemp = _mm256_permute_ps (a.data, 0x39);
    SVec<real32,4> h = _mm_insert_ps (_mm256_extractf128_ps (ftemp.data, 1),
				      _mm_set1_ps (b.data), 0x30);
    SVec<real32,4> l = _mm_insert_ps (_mm256_castps256_ps128 (ftemp.data),
				      _mm_set1_ps (_mm_cvtss_f32 (_mm256_extractf128_ps (a.data, 1))),
				      0x30);
    return _mm256_insertf128_ps (_mm256_castps128_ps256 (l.data), h.data, 0x1);
  }
  friend SVec rshift (const SVec &a, real32 b) {
    SVec ftemp = _mm256_permute_ps (a.data, 0x93);
    SVec<real32,4> l = _mm256_castps256_ps128 (ftemp.data);
    SVec<real32,4> h = _mm_insert_ps (_mm256_extractf128_ps (ftemp.data, 1),
				      _mm_set1_ps (_mm_cvtss_f32 (l.data)), 0x0 );
    SVec<real32,4> l1 = _mm_insert_ps (l.data, _mm_set1_ps (b.data), 0x0 );
    return _mm256_insertf128_ps (_mm256_castps128_ps256 (l1.data), h.data, 0x1);
  }
  friend SVec lrotate (const SVec &a) {
    SVec ftemp = _mm256_permute_ps (a.data, 0x39);
    SVec<real32,4> h = _mm_insert_ps (_mm256_extractf128_ps (ftemp.data, 1),
				      _mm256_castps256_ps128 (a.data), 0x30);
    SVec<real32,4> l = _mm_insert_ps (_mm256_castps256_ps128 (ftemp.data),
				      _mm_set1_ps (_mm_cvtss_f32 (_mm256_extractf128_ps (a.data, 1))),
				      0x30);
    return _mm256_insertf128_ps (_mm256_castps128_ps256 (l.data), h.data, 0x1);
  }
  friend SVec rrotate (const SVec &a) {
    SVec ftemp = _mm256_permute_ps (a.data, 0x93);
    SVec<real32,4> l = _mm256_castps256_ps128 (ftemp.data);
    SVec<real32,4> h = _mm_insert_ps (_mm256_extractf128_ps (ftemp.data, 1),
				      _mm_set1_ps (_mm_cvtss_f32 (l.data)), 0x0 );
    SVec<real32,4> l1 = _mm_insert_ps (l.data, _mm256_extractf128_ps (ftemp.data, 1), 0x0 );
    return _mm256_insertf128_ps (_mm256_castps128_ps256 (l1.data), h.data, 0x1);
  }
  friend SVec min (const SVec &a, const SVec &b) {
    return _mm256_min_ps (a.data, b.data); 
  }
  friend SVec max (const SVec &a, const SVec &b) {
    return _mm256_max_ps (a.data, b.data); 
  }
  // friend SVec ceil (const SVec &a)	{
  //   return _mm256_svml_ceil_ps (a.data); 
  // }	 
  // friend SVec floor (const SVec &a)	{
  //   return _mm256_svml_floor_ps (a.data); 
  // }	 
  // friend SVec round (const SVec &a)	{
  //   return _mm256_svml_round_ps (a.data); 
  // }	 
  friend SVec sqrt (const SVec &a) {
    return _mm256_sqrt_ps (a.data); 
  }	 
  friend SVec abs (const SVec &a) {
    int i = 0x7fffffff;
    return _mm256_and_ps (a.data, _mm256_set1_ps ( * (float*)&i) ); 
  }	 
  void print (std::ostream& j) const {
    float *x = (float*)&data;
    for (int i=0; i <length; i++)
      j <<x[i] <<" ";
  }
  friend std::ostream& operator << (std::ostream& i, SVec& a) {
    a.print (i); return i; 
  }
};

//! simd vector simd class
/*!
  x86 AVX simd 256 bit register, vector of 4 double 64 bit floats
*/
template <>
class SVec<real64, 4> {
public:
  typedef __m256d data_t;
  typedef real64 base;
  typedef real64 *ptr;
  __m256d data;
public:
  static const int length = 4;
  static const int size = 8;
  SVec () {}
  ~SVec () {}
  SVec (real64 f)	{
    data = _mm256_set1_pd (f.data); 
  }
  SVec (double f)	{
    data = _mm256_set1_pd (f); 
  }
  SVec (data_t m) {
    data = m; 
  }
  SVec& operator= (real64 f) {
    data = _mm256_set1_pd (f.data);
    return *this;
  }
  SVec& operator= (double f) {
    data = _mm256_set1_pd (f);
    return *this;
  }
  static const char* name () {
    return "SVec";
  }
  friend SVec operator + (const SVec &a, const SVec &b) {
    return _mm256_add_pd (a.data, b.data); 
  }
  friend SVec operator - (const SVec &a, const SVec &b) {
    return _mm256_sub_pd (a.data, b.data); 
  } 
  friend SVec operator * (const SVec &a, const SVec &b) {
    return _mm256_mul_pd (a.data, b.data); 
  } 
  friend SVec operator / (const SVec &a, const SVec &b) {
    return _mm256_div_pd (a.data, b.data); 
  }
  friend SVec fma (const SVec &c, const SVec &b, const SVec &a) {
#ifdef FMA4
    return _mm256_macc_pd (c.data, b.data, a.data);
#else
#ifdef FMA
    return _mm256_fmadd_pd (c.data, b.data, a.data);
#else
    return _mm256_add_pd (a.data, _mm256_mul_pd (b.data, c.data));
#endif //FMA
#endif //FMA4
  }
  friend SVec fms (const SVec &c, const SVec &b, const SVec &a) {
#ifdef FMA4
    return _mm256_nmacc_pd (c.data, b.data, a.data);
#else
#ifdef FMA
    return _mm256_fnmadd_pd (c.data, b.data, a.data);
#else
    return _mm256_sub_pd (a.data, _mm256_mul_pd (b.data, c.data));
#endif //FMA
#endif //FMA4
  }
  SVec& operator += (const SVec &a) {
    return *this = _mm256_add_pd (data, a.data); 
  }
  SVec& operator -= (const SVec &a) {
    return *this = _mm256_sub_pd (data, a.data); 
  }
  SVec& operator *= (const SVec &a) {
    return *this = _mm256_mul_pd (data, a.data); 
  } 
  SVec& operator /= (const SVec &a) {
    return *this = _mm256_div_pd (data, a.data); 
  }
  SVec& operator += (const real64 &f) {
    return *this = _mm256_add_pd (data, _mm256_set1_pd (f.data)); 
  }
  SVec& operator -= (const real64 &f) {
    return *this = _mm256_sub_pd (data, _mm256_set1_pd (f.data)); 
  }
  SVec& operator *= (const real64 &f) {
    return *this = _mm256_mul_pd (data, _mm256_set1_pd (f.data)); 
  }
  SVec& operator /= (const real64 &f) {
    return *this = _mm256_div_pd (data, _mm256_set1_pd (f.data)); 
  }
  friend SVec operator + (const SVec &a, const real64 &f) {
    return _mm256_add_pd (a.data, _mm256_set1_pd (f.data)); 
  }
  friend SVec operator - (const SVec &a, const real64 &f) {
    return _mm256_sub_pd (a.data, _mm256_set1_pd (f.data)); 
  }
  friend SVec operator * (const SVec &a, const real64 &f) {
    return _mm256_mul_pd (a.data, _mm256_set1_pd (f.data)); 
  }
  friend SVec operator / (const SVec &a, const real64 &f) {
    return _mm256_div_pd (a.data, _mm256_set1_pd (f.data)); 
  }
  SVec operator - () const {
    return  _mm256_xor_pd (_mm256_set1_pd (-0.0), data); 
  }
  void set_zero () {
    data = _mm256_setzero_pd (); 
  }
  void set_inc () {
    data = (data_t) {0., 1., 2., 3.};
  }
  void set (const SVec& a) {
    data = a.data; 
  }
  friend SVec lshift (const SVec &a, real64 b) {
    SVec ftemp = _mm256_permute_pd (a.data, 0x5);
    SVec<real64,2> h = _mm_shuffle_pd (_mm256_extractf128_pd (ftemp.data, 1),
				       _mm_set1_pd (b.data), 0x2);
    SVec<real64,2> l = _mm_shuffle_pd (_mm256_castpd256_pd128 (ftemp.data),
				       _mm_set1_pd (_mm_cvtsd_f64 (_mm256_extractf128_pd (a.data, 1))),
				       0x2);
    return _mm256_insertf128_pd (_mm256_castpd128_pd256 (l.data), h.data, 0x1);
  }
  friend SVec rshift (const SVec &a, real64 b) {
    SVec ftemp = _mm256_permute_pd (a.data, 0x5);
    SVec<real64,2> l = _mm256_castpd256_pd128 (ftemp.data);
    SVec<real64,2> h = _mm_shuffle_pd (_mm_set1_pd (_mm_cvtsd_f64 (l.data)),
				       _mm256_extractf128_pd (ftemp.data, 1), 0x2);
    SVec<real64,2> l1 = _mm_shuffle_pd (_mm_set1_pd (b.data), l.data, 0x2);
    return _mm256_insertf128_pd (_mm256_castpd128_pd256 (l1.data), h.data, 0x1);
  }
  friend SVec lrotate (const SVec &a) {
    SVec ftemp = _mm256_permute_pd (a.data, 0x5);
    SVec<real64,2> h = _mm_shuffle_pd (_mm256_extractf128_pd (ftemp.data, 1),
				       _mm256_castpd256_pd128 (a.data), 0x30);
    SVec<real64,2> l = _mm_shuffle_pd (_mm256_castpd256_pd128 (ftemp.data),
				       _mm_set1_pd (_mm_cvtsd_f64 (_mm256_extractf128_pd (a.data, 1))),
				       0x30);
    return _mm256_insertf128_pd (_mm256_castpd128_pd256 (l.data), h.data, 0x1);
  }
  friend SVec rrotate (const SVec &a) {
    SVec ftemp = _mm256_permute_pd (a.data, 0x5);
    SVec<real64,2> l = _mm256_castpd256_pd128 (ftemp.data);
    SVec<real64,2> h = _mm256_extractf128_pd (ftemp.data, 1);
    SVec<real64,2> l1 = _mm_shuffle_pd (_mm_set1_pd (_mm_cvtsd_f64 (h.data)), l.data, 0x2);
    SVec<real64,2> h1 = _mm_shuffle_pd (_mm_set1_pd (_mm_cvtsd_f64 (l.data)), h.data, 0x2);
    return _mm256_insertf128_pd (_mm256_castpd128_pd256 (l1.data), h1.data, 0x1);
  }
  friend SVec min (const SVec &a, const SVec &b) {
    return _mm256_min_pd (a.data, b.data); 
  }
  friend SVec max (const SVec &a, const SVec &b) {
    return _mm256_max_pd (a.data, b.data); 
  }
	
  friend SVec sqrt (const SVec &a) {
    return _mm256_sqrt_pd (a.data); 
  }	 
  // friend SVec ceil (const SVec &a)	{
  //   return _mm256_svml_ceil_pd (a.data); 
  // }	 
  // friend SVec floor (const SVec &a)	{
  //   return _mm256_svml_floor_pd (a.data); 
  // }	 
  // friend SVec round (const SVec &a)	{
  //   return _mm256_svml_round_pd (a.data); 
  // }	 
  friend SVec abs (const SVec &a) {
    int i[8] = { 0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff,
		 0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff};
    return _mm256_and_pd (a.data,  * (__m256d*)&i); 
  }	 
  void print (std::ostream& j) const {
    double *x = (double*)&data;
    for (int i=0; i <length; i++)
      j <<x[i] <<" ";
  }
  friend std::ostream& operator << (std::ostream& i, SVec& a) {
    a.print (i); return i; 
  }
};
#endif // AVX


#if defined (PHI) || defined (AVX512)
#include <x86intrin.h>

//! simd vector simd class
/*!
  x86 MIC/Phi simd 512 bit register, vector of 16 single 32 bit floats
*/

// inline __m512 _mm512_set1_ps (float A) {
//   return (__m512){ A, A, A, A, A, A, A, A,
//       A, A, A, A, A, A, A, A };
// }

// inline __m512d _mm512_set1_pd (double A) {
//   return (__m512d){ A, A, A, A, A, A, A, A };
// }


template <>
class SVec<real32, 16> {
public:
  typedef __m512 data_t;
  typedef real32 base;
  typedef real32 *ptr;
  __m512 data;
public:
  static const int length = 16;
  static const int size = 4;
  SVec () {}
  ~SVec () {}
  SVec (real32 f)	{
    data = _mm512_set1_ps (f.data); 
  }
  SVec (float f) {
    data = _mm512_set1_ps (f);
  }
  SVec (data_t m) {
    data = m; 
  }
  SVec& operator= (real32 f) {
    data = _mm512_set1_ps (f.data);
    return *this;
  }
  SVec& operator= (float f) {
    data = _mm512_set1_ps (f);
    return *this;
  }
  static const char* name () {
    return "SVec";
  }
  friend SVec operator + (const SVec &a, const SVec &b) {
    return _mm512_add_ps (a.data, b.data); 
  }
  friend SVec operator - (const SVec &a, const SVec &b) {
    return _mm512_sub_ps (a.data, b.data); 
  } 
  friend SVec operator * (const SVec &a, const SVec &b) {
    return _mm512_mul_ps (a.data, b.data); 
  } 
  friend SVec operator / (const SVec &a, const SVec &b) {
    return _mm512_div_ps (a.data, b.data); 
  }
  friend SVec fma (const SVec &a, const SVec &b, const SVec &c) {
    return _mm512_fmadd_ps (a.data, b.data, c.data);
  }
  friend SVec fms (const SVec &a, const SVec &b, const SVec &c) {
    return _mm512_fnmadd_ps (a.data, b.data, c.data);
  }
  SVec& operator += (const SVec &a) {
    return *this = _mm512_add_ps (data, a.data); 
  }
  SVec& operator -= (const SVec &a) {
    return *this = _mm512_sub_ps (data, a.data); 
  }
  SVec& operator *= (const SVec &a) {
    return *this = _mm512_mul_ps (data, a.data); 
  } 
  SVec& operator /= (const SVec &a) {
    return *this = _mm512_div_ps (data, a.data); 
  }
  SVec& operator += (const real32 &f) {
    return *this = _mm512_add_ps (data, _mm512_set1_ps (f.data)); 
  }
  SVec& operator -= (const real32 &f) {
    return *this = _mm512_sub_ps (data, _mm512_set1_ps (f.data)); 
  }
  SVec& operator *= (const real32 &f) {
    return *this = _mm512_mul_ps (data, _mm512_set1_ps (f.data)); 
  }
  SVec& operator /= (const real32 &f) {
    return *this = _mm512_div_ps (data, _mm512_set1_ps (f.data)); 
  }
  friend SVec operator + (const SVec &a, const real32 &f) {
    return _mm512_add_ps (a.data, _mm512_set1_ps (f.data)); 
  }
  friend SVec operator - (const SVec &a, const real32 &f) {
    return _mm512_sub_ps (a.data, _mm512_set1_ps (f.data)); 
  }
  friend SVec operator * (const SVec &a, const real32 &f) {
    return _mm512_mul_ps (a.data, _mm512_set1_ps (f.data)); 
  }
  friend SVec operator / (const SVec &a, const real32 &f) {
    return _mm512_div_ps (a.data, _mm512_set1_ps (f.data)); 
  }
  SVec operator - () const {
    static const int negfloat16mask[16] __attribute__ ( (__aligned__ (64))) =
      {0x80000000, 0x80000000, 0x80000000, 0x80000000,
       0x80000000, 0x80000000, 0x80000000, 0x80000000,
       0x80000000, 0x80000000, 0x80000000, 0x80000000,
       0x80000000, 0x80000000, 0x80000000, 0x80000000};
    return (__m512) _mm512_xor_epi64 ( (__m512i)data, * (__m512i*)negfloat16mask);
  }
  void set_zero () {
    data = _mm512_setzero_ps (); 
  }
  void set_inc () {
    data = (data_t) {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
		     8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f};
  }
  void set (const SVec& a) {
    data = a.data; 
  }
#ifdef PHI
  friend SVec lrotate (const SVec &a) {
    SVec b = _mm512_shuffle_epi32 ( (__m512i)a.data, _MM_PERM_ADCB);
    SVec c = _mm512_mask_permute4f128_ps (b.data, 0x8888, b.data, _MM_PERM_ADCB);
    return c;
  }
  friend SVec rrotate (const SVec &a) {
    SVec b = _mm512_shuffle_epi32 ( (__m512i)a.data, _MM_PERM_CBAD);
    SVec c = _mm512_mask_permute4f128_ps (b.data, 0x1111, b.data, _MM_PERM_CBAD);
    return c;
  }
#else
  friend SVec lrotate (const SVec &a) {
    SVec b = _mm512_shuffle_ps (a.data, a.data, _MM_PERM_ADCB);
    SVec c = _mm512_permute_ps (b.data, _MM_PERM_ADCB);
    return c;
  }
  friend SVec rrotate (const SVec &a) {
    SVec b = _mm512_shuffle_ps (a.data, a.data, _MM_PERM_CBAD);
    SVec c = _mm512_permute_ps (b.data, _MM_PERM_CBAD);
    return c;
  }
#endif
  friend SVec min (const SVec &a, const SVec &b) {
    return _mm512_min_ps (a.data, b.data); 
  }
  friend SVec max (const SVec &a, const SVec &b) {
    return _mm512_max_ps (a.data, b.data); 
  }
  friend SVec sqrt (const SVec &a) {
    return _mm512_sqrt_ps (a.data); 
  }
#ifdef PHI
  friend SVec abs (const SVec &a) {
    return _mm512_abs_ps (a.data);
  }
#endif
  friend SVec ceil (const SVec &a)	{
    return _mm512_ceil_ps (a.data); 
  }	 
  friend SVec floor (const SVec &a)	{
    return _mm512_floor_ps (a.data); 
  }	 
  friend SVec round (const SVec &a)	{
    return _mm512_floor_ps (_mm512_add_ps (a.data, _mm512_set1_ps (.5)));
  }	 
  void print (std::ostream& j) const {
    float *x = (float*)&data;
    for (int i=0; i <length; i++)
      j <<x[i] <<" ";
  }
  friend std::ostream& operator << (std::ostream& i, SVec& a) {
    a.print (i); return i; 
  }
};

//! simd vector simd class
/*!
  x86 MIC/Phi simd 512 bit register, vector of 8 double 64 bit floats
*/
template <>
class SVec<real64, 8> {
public:
  typedef __m512d data_t;
  typedef real64 base;
  typedef real64 *ptr;
  __m512d data;
public:
  static const int length = 8;
  static const int size = 8;
  SVec () {}
  ~SVec () {}
  SVec (real64 f)	{
    data = _mm512_set1_pd (f.data); 
  }
  SVec (double f)	{
    data = _mm512_set1_pd (f); 
  }
  SVec (data_t m) {
    data = m; 
  }
  SVec& operator= (real64 f) {
    data = _mm512_set1_pd (f.data);
    return *this;
  }
  SVec& operator= (double f) {
    data = _mm512_set1_pd (f);
    return *this;
  }
  static const char* name () {
    return "SVec";
  }
  friend SVec operator + (const SVec &a, const SVec &b) {
    return _mm512_add_pd (a.data, b.data); 
  }
  friend SVec operator - (const SVec &a, const SVec &b) {
    return _mm512_sub_pd (a.data, b.data); 
  } 
  friend SVec operator * (const SVec &a, const SVec &b) {
    return _mm512_mul_pd (a.data, b.data); 
  } 
  friend SVec operator / (const SVec &a, const SVec &b) {
    return _mm512_div_pd (a.data, b.data); 
  }
  friend SVec fma (const SVec &a, const SVec &b, const SVec &c) {
    return _mm512_fmadd_pd (a.data, b.data, c.data);
  }
  friend SVec fms (const SVec &a, const SVec &b, const SVec &c) {
    return _mm512_fnmadd_pd (a.data, b.data, c.data);
  }
  SVec& operator += (const SVec &a) {
    return *this = _mm512_add_pd (data, a.data); 
  }
  SVec& operator -= (const SVec &a) {
    return *this = _mm512_sub_pd (data, a.data); 
  }
  SVec& operator *= (const SVec &a) {
    return *this = _mm512_mul_pd (data, a.data); 
  } 
  SVec& operator /= (const SVec &a) {
    return *this = _mm512_div_pd (data, a.data); 
  }
  SVec& operator += (const real64 &f) {
    return *this = _mm512_add_pd (data, _mm512_set1_pd (f.data)); 
  }
  SVec& operator -= (const real64 &f) {
    return *this = _mm512_sub_pd (data, _mm512_set1_pd (f.data)); 
  }
  SVec& operator *= (const real64 &f) {
    return *this = _mm512_mul_pd (data, _mm512_set1_pd (f.data)); 
  }
  SVec& operator /= (const real64 &f) {
    return *this = _mm512_div_pd (data, _mm512_set1_pd (f.data)); 
  }
  friend SVec operator + (const SVec &a, const real64 &f) {
    return _mm512_add_pd (a.data, _mm512_set1_pd (f.data)); 
  }
  friend SVec operator - (const SVec &a, const real64 &f) {
    return _mm512_sub_pd (a.data, _mm512_set1_pd (f.data)); 
  }
  friend SVec operator * (const SVec &a, const real64 &f) {
    return _mm512_mul_pd (a.data, _mm512_set1_pd (f.data)); 
  }
  friend SVec operator / (const SVec &a, const real64 &f) {
    return _mm512_div_pd (a.data, _mm512_set1_pd (f.data)); 
  }
  SVec operator - () const {
    static const int negdouble8mask[16] __attribute__ ( (__aligned__ (64))) =
      {0x0, 0x80000000, 0x0, 0x80000000,
       0x0, 0x80000000, 0x0, 0x80000000,
       0x0, 0x80000000, 0x0, 0x80000000,
       0x0, 0x80000000, 0x0, 0x80000000};
    return (__m512d) _mm512_xor_epi64 ( (__m512i)data, * (__m512i*)negdouble8mask);
  }
  void set_zero () {
    data = _mm512_setzero_pd (); 
  }
  void set_inc () {
    data = (data_t) {0., 1., 2., 3., 4., 5., 6., 7.};
  }
  void set (const SVec& a) {
    data = a.data; 
  }
#ifdef PHI
  friend SVec lrotate (const SVec &a) {
    __m512 b = (__m512)_mm512_shuffle_epi32 ( (__m512i)a.data, _MM_PERM_BADC);
    SVec c = _mm512_mask_permute4f128_ps (b, 0xcccc, b, _MM_PERM_ADCB);
    return c;
  }
  friend SVec rrotate (const SVec &a) {
    __m512 b = (__m512)_mm512_shuffle_epi32 ( (__m512i)a.data, _MM_PERM_BADC);
    SVec c = _mm512_mask_permute4f128_ps (b, 0x3333, b, _MM_PERM_CBAD);
    return c;
  }
#else
  friend SVec lrotate (const SVec &a) {
    SVec b = _mm512_shuffle_pd (a.data, a.data, _MM_PERM_ADCB);
    SVec c = _mm512_permute_pd (b.data, _MM_PERM_ADCB);
    return c;
  }
  friend SVec rrotate (const SVec &a) {
    SVec b = _mm512_shuffle_pd (a.data, a.data, _MM_PERM_CBAD);
    SVec c = _mm512_permute_pd (b.data, _MM_PERM_CBAD);
    return c;
  }
#endif
  friend SVec min (const SVec &a, const SVec &b) {
    return _mm512_min_pd (a.data, b.data); 
  }
  friend SVec max (const SVec &a, const SVec &b) {
    return _mm512_max_pd (a.data, b.data); 
  }
	
  friend SVec sqrt (const SVec &a) {
    return _mm512_sqrt_pd (a.data); 
  }	 
#ifdef PHI
  friend SVec abs (const SVec &a) {
    return _mm512_abs_pd (a.data);
  }
#endif
  friend SVec ceil (const SVec &a)	{
    return _mm512_ceil_pd (a.data); 
  }	 
  friend SVec floor (const SVec &a)	{
    return _mm512_floor_pd (a.data); 
  }	 
  friend SVec round (const SVec &a)	{
    return _mm512_floor_pd (_mm512_add_pd (a.data, _mm512_set1_pd (.5)));
  }	 
  void print (std::ostream& j) const {
    double *x = (double*)&data;
    for (int i=0; i <length; i++)
      j <<x[i] <<" ";
  }
  friend std::ostream& operator << (std::ostream& i, SVec& a) {
    a.print (i); return i; 
  }
};
#endif // PHI


#ifdef NEON
#include <arm_neon.h>

//! simd vector simd class
/*!
  ARM NEON simd 128 bit register, vector of 4 single 32 bit floats
*/
template <>
class SVec<real32, 4> {
public:
  typedef float32x4_t data_t;
  typedef real32 base;
  typedef real32 *ptr;
  float32x4_t data;
public:
  static const int length = 4;
  static const int size = 4;
  SVec () {}
  ~SVec () {}
  SVec (real32 f)	{
    data = vdupq_n_f32 (f.data); 
  }
  SVec (float f)	{
    data = vdupq_n_f32 (f); 
  }
  SVec (data_t m) {
    data = m; 
  }
  SVec& operator= (real32 f) {
    data = vdupq_n_f32 (f.data);
    return *this;
  }
  SVec& operator= (float f) {
    data = vdupq_n_f32 (f);
    return *this;
  }
  static const char* name () {
    return "SVec";
  }
  friend SVec operator + (const SVec &a, const SVec &b) {
    return vaddq_f32 (a.data, b.data); 
  }
  friend SVec operator - (const SVec &a, const SVec &b) {
    return vsubq_f32 (a.data, b.data); 
  } 
  friend SVec operator * (const SVec &a, const SVec &b) {
    return vmulq_f32 (a.data, b.data); 
  } 
  friend SVec operator / (const SVec &a, const SVec &b) {
    SVec x0 = vrecpeq_f32 (b.data);
    // 1 iteration
    SVec x1 = vrecpsq_f32 (x0.data, b.data);
    return vmulq_f32 (vmulq_f32 (x0.data, a.data), x1.data);
  }
  friend SVec fma (const SVec &a, const SVec &b, const SVec &c) {
#ifdef FMA
    return vmlaq_f32 (c.data, b.data, a.data);
#else
    return a * b + c;
#endif
  }
  friend SVec fms (const SVec &a, const SVec &b, const SVec &c) {
#ifdef FMA
    return vmlsq_f32 (c.data, b.data, a.data);
#else
    return c - a * b;
#endif
  }
  SVec& operator += (const SVec &a) {
    return *this = vaddq_f32 (data, a.data); 
  }
  SVec& operator -= (const SVec &a) {
    return *this = vsubq_f32 (data, a.data); 
  }
  SVec& operator *= (const SVec &a) {
    return *this = vmulq_f32 (data, a.data); 
  } 
  SVec& operator /= (const SVec &a) {
    return *this = (*this) / a;
  }
  SVec& operator += (const real32 &f) {
    return *this = vaddq_f32 (data, vdupq_n_f32 (f.data)); 
  }
  SVec& operator -= (const real32 &f) {
    return *this = vsubq_f32 (data, vdupq_n_f32 (f.data)); 
  }
  SVec& operator *= (const real32 &f) {
    return *this = vmulq_f32 (data, vdupq_n_f32 (f.data)); 
  }
  SVec& operator /= (const real32 &f) {
    // return *this = (*this) / vdupq_n_f32 (f.data);
    return *this = vmulq_f32 (data, vdupq_n_f32 (1.f / f.data)); 
  }
  friend SVec operator + (const SVec &a, const real32 &f) {
    return vaddq_f32 (a.data, vdupq_n_f32 (f.data)); 
  }
  friend SVec operator - (const SVec &a, const real32 &f) {
    return vsubq_f32 (a.data, vdupq_n_f32 (f.data)); 
  }
  friend SVec operator * (const SVec &a, const real32 &f) {
    return vmulq_f32 (a.data, vdupq_n_f32 (f.data)); 
  }
  friend SVec operator / (const SVec &a, const real32 &f) {
    // return a / vdupq_n_f32 (f.data);
    return vmulq_f32 (a.data, vdupq_n_f32 (1.f / f.data)); 
  }
  SVec operator - () const {
    return vnegq_f32 (data);
  }
  void set_zero () {
    data = vdupq_n_f32 (0.f);
  }
  void set_inc () {
    data = (data_t) {0.f, 1.f, 2.f, 3.f};
  }
  void set (const SVec& a) {
    data = a.data; 
  }
  friend SVec lrotate (const SVec &a) {
    return vextq_f32 (a.data, a.data, 1);
  }
  friend SVec rrotate (const SVec &a) {
    return vextq_f32 (a.data, a.data, 3);
  }
  friend SVec min (const SVec &a, const SVec &b) {
    return vminq_f32 (a.data, b.data); 
  }
  friend SVec max (const SVec &a, const SVec &b) {
    return vmaxq_f32 (a.data, b.data); 
  }
  friend SVec abs (const SVec &a) {
    return vabsq_f32 (a.data);
  }	 
  friend SVec rcp (const SVec &b) {
    SVec x0 = vrecpeq_f32 (b.data);
    // 1 iteration
    SVec x1 = vrecpsq_f32 (x0.data, b.data);
    return vmulq_f32 (x0.data, x1.data);
  }	 
  friend SVec rsqrt (const SVec &b) {
    SVec x0 = vrsqrteq_f32 (b.data);
    // 1 iteration
    return vmulq_f32 (x0.data, vrsqrtsq_f32 (vmulq_f32 (x0.data, x0.data), b.data));
  }
  friend SVec sqrt (const SVec &a) {
    return rcp (rsqrt (a)); 
  }	 
  friend SVec ceil (const SVec &a) {
    SVec inf = vdupq_n_f32 (8388608.f); // 2^23
    SVec zero = vdupq_n_f32 (0.f);
    SVec ainf = vsubq_f32 (inf.data, 
			   vcvtq_f32_s32 (vcvtq_s32_f32 (vsubq_f32 (inf.data, a.data))));
    SVec a0 = vcvtq_f32_s32 (vcvtq_s32_f32 (a.data));
    return vbslq_f32 (vcgeq_f32 (a.data, zero.data), ainf.data, a0.data);
  }
  friend SVec floor (const SVec &a) {
    SVec inf = vdupq_n_f32 (8388608.f);
    SVec zero = vdupq_n_f32 (0.f);
    SVec aminf = vsubq_f32 (vcvtq_f32_s32 (vcvtq_s32_f32 (vaddq_f32 (a.data, inf.data))),
			    inf.data);
    SVec a0 = vcvtq_f32_s32 (vcvtq_s32_f32 (a.data));
    return vbslq_f32 (vcgeq_f32 (a.data, zero.data), a0.data, aminf.data);
  }
  friend SVec round (const SVec &a) {
    SVec inf = vdupq_n_f32 (8388608.f);
    SVec zero = vdupq_n_f32 (0.f);
    SVec half = vdupq_n_f32 (.5f);
    SVec a5 = vaddq_f32 (a.data, half.data);
    SVec aminf = vsubq_f32 (vcvtq_f32_s32 (vcvtq_s32_f32 (vaddq_f32 (a5.data, inf.data))),
			    inf.data);
    SVec a0 = vcvtq_f32_s32 (vcvtq_s32_f32 (a5.data));
    return vbslq_f32 (vcgeq_f32 (a5.data, zero.data), a0.data, aminf.data);
  }
  void print (std::ostream& j) const {
    float *x = (float*)&data;
    for (int i=0; i <length; i++)
      j <<x[i] <<" ";
  }
  friend std::ostream& operator << (std::ostream& i, SVec& a) {
    a.print (i); return i; 
  }
};
#endif // NEON


#ifdef ALTIVEC
#include <altivec.h>

//! simd vector simd class
/*!
  ALTIVEC simd 128 bit register, vector of 4 single 32 bit floats
*/
template <>
class SVec<real32, 4> {
public:
  typedef __vector float data_t;
  typedef real32 base;
  typedef real32 *ptr;
  __vector float data;
public:
  static const int length = 4;
  static const int size = 4;
  SVec () {}
  ~SVec () {}
  SVec (real32 f)	{
    data = vec_splats (f.data); 
  }
  SVec (float f)	{
    data = vec_splats (f); 
  }
  SVec (data_t m) {
    data = m; 
  }
  SVec& operator= (real32 f) {
    data = vec_splats (f.data);
    return *this;
  }
  SVec& operator= (float f) {
    data = vec_splats (f);
    return *this;
  }
  static const char* name () {
    return "SVec";
  }
  friend SVec operator + (const SVec &a, const SVec &b) {
    return vec_add (a.data, b.data); 
  }
  friend SVec operator - (const SVec &a, const SVec &b) {
    return vec_sub (a.data, b.data); 
  } 
  friend SVec operator * (const SVec &a, const SVec &b) {
    return fma (a, b, vec_splats (0.f)); 
  } 
  friend SVec operator / (const SVec &a, const SVec &b) {
    return a * rcp (b); 
  }
  friend SVec fma (const SVec &c, const SVec &b, const SVec &a) {
    return vec_madd (c.data, b.data, a.data);
  }
  friend SVec fms (const SVec &c, const SVec &b, const SVec &a) {
    return vec_nmsub (c.data, b.data, a.data);
  }
  SVec& operator += (const SVec &a) {
    return *this = vec_add (data, a.data); 
  }
  SVec& operator -= (const SVec &a) {
    return *this = vec_sub (data, a.data); 
  }
  SVec& operator *= (const SVec &a) {
    return *this = fma (*this, a, vec_splats (0.f)); 
  } 
  SVec& operator /= (const SVec &a) {
    return *this = (*this) * rcp (a); 
  }
  SVec& operator += (const real32 &f) {
    return *this = vec_add (data, vec_splats (f.data)); 
  }
  SVec& operator -= (const real32 &f) {
    return *this = vec_sub (data, vec_splats (f.data)); 
  }
  SVec& operator *= (const real32 &f) {
    return *this = fma (*this, vec_splats (f.data), vec_splats (0.f)); 
  }
  SVec& operator /= (const real32 &f) {
    return *this = fma (*this, vec_splats (1.f / f.data), vec_splats (0.f)); 
  }
  friend SVec operator + (const SVec &a, const real32 &f) {
    return vec_add (a.data, vec_splats (f.data)); 
  }
  friend SVec operator - (const SVec &a, const real32 &f) {
    return vec_sub (a.data, vec_splats (f.data)); 
  }
  friend SVec operator * (const SVec &a, const real32 &f) {
    return fma (a, vec_splats (f.data), vec_splats (0.f)); 
  }
  friend SVec operator / (const SVec &a, const real32 &f) {
    return fma (a, vec_splats (1.f / f.data), vec_splats (0.f)); 
  }
  SVec operator - () const {
    return vec_xor (vec_splats (-0.f), data); 
  }
  void set_zero () {
    data = vec_splats (0.f); 
  }
  void set_inc () {
    data = (data_t) {0.f, 1.f, 2.f, 3.f};
  }
  void set (const SVec& a) {
    data = a.data; 
  }
  friend SVec lrotate (const SVec &a) {
    return vec_sld (a.data, a.data, 4);
  }
  friend SVec rrotate (const SVec &a) {
    return vec_sld (a.data, a.data, 12);
  }
  friend SVec min (const SVec &a, const SVec &b) {
    return vec_min (a.data, b.data); 
  }
  friend SVec max (const SVec &a, const SVec &b) {
    return vec_max (a.data, b.data); 
  }
	
  friend SVec abs (const SVec &a) {
    int i = 0x7fffffff;
    return vec_and (a.data, vec_splats ( * (float*)&i) ); 
  }
  friend SVec rcp (const SVec &b) {
    SVec x0 = vec_re (b.data);
    return vec_madd (x0.data, vec_nmsub (x0.data, b.data, vec_splats (1.f)), x0.data);
  }	 
  friend SVec sqrt (const SVec &a) {
    return rcp (rsqrt (a)); 
  }	 
  friend SVec rsqrt (const SVec &b) {
    SVec x0 = vec_rsqrte (b.data);
    SVec x2 = x0 * x0;
    SVec x1 = vec_nmsub (x2.data, b.data, vec_splats (3.f));
    return x0 * x1 * (real32).5f;
  }	 
  friend SVec ceil (const SVec &a) {
    return vec_ceil (a.data); 
  }	 
  friend SVec floor (const SVec &a) {
    return vec_floor (a.data); 
  }	 
  friend SVec round (const SVec &a) {
    return vec_round (a.data); 
  }	 
  void print (std::ostream& j) const {
    float *x = (float*)&data;
    for (int i=0; i <length; i++)
      j <<x[i] <<" ";
  }
  friend std::ostream& operator << (std::ostream& i, SVec& a) {
    a.print (i); return i; 
  }
};

#endif // ALTIVEC

#ifdef SPU
#include <spu_intrinsics.h>
#include <simdmath.h>

//! simd vector simd class
/*!
  Cell BE SPU simd 128 bit register, vector of 4 single 32 bit floats
*/
template <>
class SVec<real32, 4> {
public:
  typedef __vector float data_t;
  typedef real32 base;
  typedef real32 *ptr;
  __vector float data;
public:
  static const int length = 4;
  static const int size = 4;
  SVec () {}
  ~SVec () {}
  SVec (real32 f)	{
    data = spu_splats (f.data); 
  }
  SVec (float f)	{
    data = spu_splats (f); 
  }
  SVec (data_t m) {
    data = m; 
  }
  SVec& operator= (real32 f) {
    data = spu_splats (f.data);
    return *this;
  }
  SVec& operator= (float f) {
    data = spu_splats (f);
    return *this;
  }
  static const char* name () {
    return "SVec";
  }
  friend SVec operator + (const SVec &a, const SVec &b) {
    return spu_add (a.data, b.data); 
  }
  friend SVec operator - (const SVec &a, const SVec &b) {
    return spu_sub (a.data, b.data); 
  } 
  friend SVec operator * (const SVec &a, const SVec &b) {
    return fma (a, b, spu_splats (0.f)); 
  } 
  friend SVec operator / (const SVec &a, const SVec &b) {
    return a * rcp (b); 
  }
  friend SVec fma (const SVec &c, const SVec &b, const SVec &a) {
    return spu_madd (c.data, b.data, a.data);
  }
  friend SVec fms (const SVec &c, const SVec &b, const SVec &a) {
    return spu_nmsub (c.data, b.data, a.data);
  }
  SVec& operator += (const SVec &a) {
    return *this = spu_add (data, a.data); 
  }
  SVec& operator -= (const SVec &a) {
    return *this = spu_sub (data, a.data); 
  }
  SVec& operator *= (const SVec &a) {
    return *this = fma (*this, a, spu_splats (0.f)); 
  } 
  SVec& operator /= (const SVec &a) {
    return *this = (*this) * rcp (a); 
  }
  SVec& operator += (const real32 &f) {
    return *this = spu_add (data, spu_splats (f.data)); 
  }
  SVec& operator -= (const real32 &f) {
    return *this = spu_sub (data, spu_splats (f.data)); 
  }
  SVec& operator *= (const real32 &f) {
    return *this = fma (*this, spu_splats (f.data), spu_splats (0.f)); 
  }
  SVec& operator /= (const real32 &f) {
    return *this = fma (*this, spu_splats (1.f / f.data), spu_splats (0.f)); 
  }
  friend SVec operator + (const SVec &a, const real32 &f) {
    return spu_add (a.data, spu_splats (f.data)); 
  }
  friend SVec operator - (const SVec &a, const real32 &f) {
    return spu_sub (a.data, spu_splats (f.data)); 
  }
  friend SVec operator * (const SVec &a, const real32 &f) {
    return fma (a, spu_splats (f.data), spu_splats (0.f)); 
  }
  friend SVec operator / (const SVec &a, const real32 &f) {
    return fma (a, spu_splats (1.f / f.data), spu_splats (0.f)); 
  }
  SVec operator - () const {
    return spu_xor (spu_splats (-0.f), data); 
  }
  void set_zero () {
    data = spu_splats (0.f); 
  }
  void set_inc () {
    data = (data_t) {0.f, 1.f, 2.f, 3.f};
  }
  void set (const SVec& a) {
    data = a.data; 
  }
  friend SVec lrotate (const SVec &a) {
    __vector unsigned char c = {4,5,6,7, 8,9,10,11, 12,13,14,15, 0,1,2,3};
    return spu_shuffle (a.data, a.data, c);
  }
  friend SVec rrotate (const SVec &a) {
    __vector unsigned char c = {12,13,14,15, 0,1,2,3, 4,5,6,7, 8,9,10,11};
    return spu_shuffle (a.data, a.data, c);
  }
  friend SVec min (const SVec &a, const SVec &b) {
    return (spu_sel (a.data, b.data, spu_cmpgt (a.data, b.data)));
  }
  friend SVec max (const SVec &a, const SVec &b) {
    return (spu_sel (b.data, a.data, spu_cmpgt (a.data, b.data)));
  }
	
  friend SVec abs (const SVec &a) {
    return (data_t) (spu_rlmask (spu_sl ( (__vector unsigned int) (a.data), 1), -1));
  }
  friend SVec rcp (const SVec &b) {
    SVec x0 = spu_re (b.data);
    return spu_madd (x0.data, spu_nmsub (x0.data, b.data, spu_splats (1.f)), x0.data);
  }	 
  friend SVec sqrt (const SVec &a) {
    return rcp (rsqrt (a)); 
  }	 
  friend SVec rsqrt (const SVec &b) {
    SVec x0 = spu_rsqrte (b.data);
    SVec x2 = x0 * x0;
    SVec x1 = spu_nmsub (x2.data, b.data, spu_splats (3.f));
    return x0 * x1 * (real32).5f;
  }	 
  friend SVec ceil (const SVec &aa) { // (c) sony
    data_t a = spu_add (aa.data, (data_t) (spu_and (spu_xor (spu_rlmaska ( (vec_int4)a, -31), -1), spu_splats ( (signed int)0x3F7FFFFF))));
    vec_int4 exp = spu_sub (127, (vec_int4) (spu_and (spu_rlmask ( (vec_uint4)(a), -23), 0xFF)));
    vec_uint4 mask = spu_rlmask (spu_splats ( (unsigned int)0x7FFFFF), exp);
    mask = spu_sel (spu_splats ( (unsigned int)0), mask, spu_cmpgt (exp, -31));
    mask = spu_or (mask, spu_xor ( (vec_uint4)(spu_rlmaska (spu_add (exp, -1), -31)), -1));
    return (data_t)spu_andc ( (vec_uint4)(a), mask);
  }	 
  friend SVec floor (const SVec &aa) { // (c) sony
    data_t a = spu_sub (aa.data, (data_t)(spu_and (spu_rlmaska ((vec_int4)a, -31), spu_splats ((signed int)0x3F7FFFFF))));
    vec_int4 exp = spu_sub (127, (vec_int4) (spu_and (spu_rlmask ( (vec_uint4)(a), -23), 0xFF)));
    vec_uint4 mask = spu_rlmask (spu_splats ((unsigned int)0x7FFFFF), exp);
    mask = spu_sel (spu_splats ((unsigned int)0), mask, spu_cmpgt (exp, -31));
    mask = spu_or (mask, spu_xor ((vec_uint4)(spu_rlmaska (spu_add (exp, -1), -31)), -1));
    return (data_t)spu_andc ((vec_uint4)(a), mask);
  }	 
  friend SVec round (const SVec &a) {
    SVec b = spu_add (spu_splats (.5f), a.data);
    return floor (b);
  }	 
  /*
    void print (std::ostream& j) const {
    float *x = (float*)&data;
    for (int i=0; i <length; i++)
    j <<x[i] <<" ";
    }
    friend std::ostream& operator << (std::ostream& i, SVec& a) {
    a.print (i); return i; 
    }
  */
};

#endif // SPU


#endif // SIMD_HPP
