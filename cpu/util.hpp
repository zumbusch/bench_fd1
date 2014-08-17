#ifndef UTIL_HPP
#define UTIL_HPP

// Copyright (c) 2011-2012, Gerhard Zumbusch
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


#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>


#ifdef _OPENMP // OpenMP enabled
#include <omp.h>
#endif

#ifdef PF_MPI
#include <mpi.h>
#endif // PF_MPI

// MPI wrapper
class pfcomm_ { 
public:
  int id; // my position
  int l, h; // with upper/lower neigbors
  int pr; // nr of proc
  pfcomm_ ();
  ~pfcomm_ ();
  void init (int &argc, char **&argv);
  void finalize ();
  void barrier ();
  double reduce_max (double r);
  void send (class real32 *xl, class real32 *xr, int j);
  void send (class real64 *xl, class real64 *xr, int j);
};

static pfcomm_ pfcomm;


void pferror(const char*n, int s=0);

//----------
// POSIX TIMERS clock, resolution 10^-9 s
//----------

class realtime {
private:
  struct timespec tp0, tp1;
  double r;
public:
  realtime () {
    tp0.tv_sec = tp1.tv_sec = 0;
    tp0.tv_nsec = tp1.tv_nsec = 0;
    r = 0;
  }
  ~realtime () {}
  double res() {
    return 1e-9;
  }
  void start () {
    pfcomm.barrier();
    clock_gettime (CLOCK_REALTIME, &tp0); //  CLOCK_PROCESS_CPUTIME_ID
  }
  void stop () {
    clock_gettime (CLOCK_REALTIME, &tp1); //  CLOCK_PROCESS_CPUTIME_ID
    double r0 = (tp1.tv_sec - tp0.tv_sec) + 1e-9 * (tp1.tv_nsec - tp0.tv_nsec);
    r = pfcomm.reduce_max (r0);
  }
  double elapsed () {
    return r;
  }
};

#endif // UTIL_HPP
