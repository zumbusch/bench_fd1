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

#include <stdlib.h>


#ifdef PF_MPI
#include <mpi.h>
#endif // PF_MPI

void pferror(const char*n, int s=0);

#include <time.h>
//----------
//! clock, resolution 10^-9 s
//----------

class realtime {
private:
  struct timespec tp0, tp1;
public:
  realtime () {
    tp0.tv_sec = tp1.tv_sec = 0;
    tp0.tv_nsec = tp1.tv_nsec = 0;
  }
  ~realtime () {}
  double res() {
    return 1e-9;
  }
  void start () {
#ifdef PF_MPI
    MPI_Barrier (MPI_COMM_WORLD);
#endif // PF_MPI
    clock_gettime (CLOCK_PROCESS_CPUTIME_ID, // CLOCK_REALTIME
		   &tp0);
  }
  void stop () {
#ifdef PF_MPI
    MPI_Barrier (MPI_COMM_WORLD);
#endif // PF_MPI
    clock_gettime (CLOCK_PROCESS_CPUTIME_ID,
		   &tp1);
  }
  double elapsed () {
    return (tp1.tv_sec - tp0.tv_sec) + 1e-9 * (tp1.tv_nsec - tp0.tv_nsec);
  }
};

#endif // UTIL_HPP
