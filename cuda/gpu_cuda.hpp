#ifndef GPU_HPP
#define GPU_HPP

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


#include <iostream>
#include <cuda.h>

#include "util.hpp"

class pfgpu_ {
public:
  cudaEvent_t evt_start, evt_stop;
  int dev, numdev;
  pfgpu_ () : dev (0), numdev (0) {
  }
  ~pfgpu_ () {}
  void init (int argc, char *argv[], int th) {
    cudaError_t status;
    status = cudaGetDeviceCount (&numdev); // cutilSafeCall
    if (status != cudaSuccess)
      pferror ("cudaGetDeviceCount failed", status);
#ifdef PF_MPI
    int id;
    MPI_Comm_rank (MPI_COMM_WORLD, &id);
    dev = id % numdev; // assume numdev processes per compute node
    std::cout<<"proc "<<id<<" device"<<id%numdev<<"/"<<numdev;
#else // PF_MPI
    dev = th % numdev;
    std::cout<<"device "<<th % numdev<<"/"<<numdev;
#endif // PF_MPI
    setDevice ();
    cudaDeviceProp deviceProp;
    status = cudaGetDeviceProperties (&deviceProp, 0);
    if (status != cudaSuccess)
      pferror ("cudaGetDeviceProperties failed", status);
    std::cout<<" "<<deviceProp.name<<", "<<proc ()<<" proc, " // <<vec_len ()<<" vec"<<", "
	     <<group_len ()<<" thrd, "<<shmem_size ()<<" lmem, "
	     << (double)mem_size ()<<" dmem\n";
#ifndef CHECK
    if (proc () != PROC)
      pferror ("wrong number of processors", PROC);
#endif
    status = cudaEventCreate (&evt_start);
    if (status != cudaSuccess)
      pferror ("cudaEventCreate failed", status);
    status = cudaEventCreate (&evt_stop);
    if (status != cudaSuccess)
      pferror ("cudaEventCreate failed", status);
    status = cudaDeviceSetCacheConfig (cudaFuncCachePreferL1);
    if (status != cudaSuccess)
      pferror ("cudaDeviceSetCacheConfig failed", status);
  }
  void setDevice () {
    cudaError_t status;
    status = cudaSetDevice (dev);
    if (status != cudaSuccess)
      pferror ("cudaSetDevice failed", status);
  }
  int proc () {
    cudaDeviceProp deviceProp;
    cudaError_t status;
    status = cudaGetDeviceProperties (&deviceProp, 0);
    if (status != cudaSuccess)
      pferror ("cudaGetDeviceProperties failed", status);
    return deviceProp.multiProcessorCount;
  }
  int vec_len () {
    cudaDeviceProp deviceProp;
    cudaError_t status;
    status = cudaGetDeviceProperties (&deviceProp, 0);
    if (status != cudaSuccess)
      pferror ("cudaGetDeviceProperties failed", status);
    return deviceProp.maxThreadsDim[0];
  }
  size_t group_len () {
    cudaDeviceProp deviceProp;
    cudaError_t status;
    status = cudaGetDeviceProperties (&deviceProp, 0);
    if (status != cudaSuccess)
      pferror ("cudaGetDeviceProperties failed", status);
    return deviceProp.maxThreadsPerBlock;
  }
  long shmem_size () {
    cudaDeviceProp deviceProp;
    cudaError_t status;
    status = cudaGetDeviceProperties (&deviceProp, 0);
    if (status != cudaSuccess)
      pferror ("cudaGetDeviceProperties failed", status);
    size_t m = deviceProp.sharedMemPerBlock;
    return m;
  }
  long mem_size () {
    cudaDeviceProp deviceProp;
    cudaError_t status;
    status = cudaGetDeviceProperties (&deviceProp, 0);
    if (status != cudaSuccess)
      pferror ("cudaGetDeviceProperties failed", status);
    size_t m = deviceProp.totalGlobalMem;
    long s = 1;
    while (s<m) s *= 2;
    return s;
  }
  template <typename A>
  A* alloc (int n) {
    setDevice ();
    A *p;
    cudaError_t status;
    status = cudaMalloc ((void**)&p, sizeof (A) * n);
    if (status != cudaSuccess)
      pferror ("cudaMalloc failed", status);
    return p;
  }
  template <typename A>
  void free (A* x) {
    if (x) {
      setDevice ();
      cudaError_t status = cudaFree (x);
      if (status != cudaSuccess)
	pferror ("cudaFree failed", status);
      x = 0;
    }
  }
  template <typename A>
  void copy (const A* xsrc, pfgpu_& src, A* xdst, int offsrc, int offdst, int n) {
    // setDevice ();
    cudaError_t status;
    status = cudaMemcpyPeer (offdst+xdst, dev, offsrc+xsrc, src.dev, sizeof (A) * n);
    if (status != cudaSuccess)
      pferror ("cudaMemcpyP2P failed", status);
  }
  template <typename A>
  void read (const A* xdev, A* xhost, int off, int n) {
    setDevice ();
    cudaError_t status;
    status = cudaMemcpy (off+xhost, off+(A*)xdev, sizeof (A) * n, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
      pferror ("cudaMemcpyD2H failed", status);
  }
  template <typename A>
  void write (const A* xhost, A* xdev, int off, int n) {
    setDevice ();
    cudaError_t status;
    status = cudaMemcpy (off+(A*)xdev, off+xhost, sizeof (A) * n, cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
      pferror ("cudaMemcpyH2D failed", status);
  }
  void start () {
    setDevice ();
    cudaError_t status;
    status = cudaEventRecord (evt_start, 0);
    if (status != cudaSuccess)
      pferror ("cudaEventRecord failed", status);
  }
  void sync () {
    setDevice ();
    lastError ();
    cudaError_t status;
    status = cudaEventRecord (evt_stop, 0);
    if (status != cudaSuccess)
      pferror ("cudaEventRecord failed", status);
    status = cudaEventSynchronize (evt_stop);
    if (status != cudaSuccess)
      pferror ("cudaEventSynchronize failed", status);
  }
  double time () {
    setDevice ();
    cudaError_t status;
    float time;
    status = cudaEventElapsedTime (&time, evt_start, evt_stop);
    if (status != cudaSuccess)
      pferror ("cudaEventElapsedTime failed", status);
    return time * 1.0e-3;
  }
  void close () {
    setDevice ();
    cudaError_t status;
    status = cudaEventDestroy (evt_start);
    if (status != cudaSuccess)
      pferror ("cudaEventDestroy failed", status);
    status = cudaEventDestroy (evt_stop);
    if (status != cudaSuccess)
      pferror ("cudaEventDestroy failed", status);
  }
  void lastError () {
    cudaError_t status = cudaGetLastError ();
    if (status != cudaSuccess)
      pferror ("cudaGetLastError failed", status);
  }
};

pfgpu_ pfgpu[DEV_MAX];

void pferror (const char*n, int s) {
#ifdef PF_MPI
  int id;
  MPI_Comm_rank (MPI_COMM_WORLD, &id);
  std::cerr <<" proc " << id << ": ";
#endif // PF_MPI
  std::cerr << n;
  if (s) {
    switch (s) {
    case CUDA_ERROR_INVALID_VALUE:
      std::cerr<<" invalid value";
      break;
    case CUDA_ERROR_OUT_OF_MEMORY:
      std::cerr<<" out of memory";
      break;
    case CUDA_ERROR_NOT_INITIALIZED:
      std::cerr<<" not initialized";
      break;
    case CUDA_ERROR_DEINITIALIZED:
      std::cerr<<" deinitialized";
      break;
    case CUDA_ERROR_PROFILER_ALREADY_STARTED:
      std::cerr<<" profiler already started";
      break;
    case CUDA_ERROR_NO_DEVICE:
      std::cerr<<" no device";
      break;
    case CUDA_ERROR_INVALID_DEVICE:
      std::cerr<<" invalid device";
      break;
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
      std::cerr<<" launch out of resources";
      break;
    default:
      std::cerr << " code " << s;
    }
  }
  std::cerr << std::endl;
  std::cout << "CUDA_ERROR" << std::endl;
#ifdef PF_MPI
  MPI_Finalize ();
#endif // PF_MPI
  cudaDeviceReset ();
  exit (-1);
}

#endif // GPU_HPP
