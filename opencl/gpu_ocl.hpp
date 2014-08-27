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

#ifndef GPU_OCL_HPP
#define GPU_OCL_HPP

#include "util.hpp"

#include <iostream>
#include <fstream>
#include <string.h>
#include <string.h>
#include <assert.h>

#include <CL/opencl.h>

class pfgpu_ {
public:
#define maxdevice 8  // max devices per compute node
#define maxkernels 10  // max kernels
#define maxplat 15  // max plat
  cl_context context;
  cl_platform_id platform [maxplat];
  cl_device_id device, devicev [maxdevice];
  cl_command_queue commandQueue;
  cl_program program;
  cl_event event;
  cl_kernel kernels [maxkernels];
  cl_uint num_kernels;
  cl_uint work_dim;
  size_t global_work_size [3], local_work_size [3];
  cl_uint arg_count;
  char* source;

  pfgpu_ () {
    source = 0;
    num_kernels = 0;
    for (int i=0; i<maxkernels; i++)
      kernels [i] = 0;
    program = 0;
    commandQueue = 0;
    context = 0;
  }

  ~pfgpu_ () {}

  void init (int argc, char *argv [], int th) {
#ifdef PF_MPI
    int id;
    MPI_Comm_rank (MPI_COMM_WORLD, &id);
#endif // PF_MPI
    cl_int status;
    platform [0] = NULL;
    cl_uint m;
    status = clGetPlatformIDs (maxplat, &platform [0], &m);
    std::cout<<m<<(m>1 ? " platforms: " :  " platform: ");
    if (status != CL_SUCCESS)
      pferror ("clGetPlatformIDs failed", status);

    int p = -1;
    for (int i=0; i<m; i++) {
      cl_uint n;
      status = clGetDeviceIDs (platform [i],
#ifdef PFCPU
			       CL_DEVICE_TYPE_CPU,
#else
			       CL_DEVICE_TYPE_GPU,
#endif
			       maxdevice, &devicev [0], &n);
      if (status == CL_SUCCESS) {
 	p = i;
 	break;
      }
    }
    if (p == -1)
#ifdef PFCPU
      pferror ("no OpenCL CPU platform");
#else
    pferror ("no OpenCL GPU platform");
#endif
    cl_uint numdev;
    status = clGetDeviceIDs (platform [p], /* CL_DEVICE_TYPE_ALL */
#ifdef PFCPU
			     CL_DEVICE_TYPE_CPU,
#else
			     CL_DEVICE_TYPE_GPU,
#endif
			     maxdevice, &devicev [0], &numdev);
    if (status != CL_SUCCESS)
      pferror ("clGetDeviceIDs failed", status);
#ifdef PF_MPI
    device = devicev [id % numdev]; // assume numdev processes per compute node
    std::cout<<"proc "<<id<<" device"<<id%numdev<<"/"<<numdev;
#else // PF_MPI
    device = devicev [th % numdev];
    std::cout<<"device "<<th % numdev<<"/"<<numdev;
#endif // PF_MPI
#undef maxdevice
    char fname [200];
    status = clGetDeviceInfo (device, CL_DEVICE_NAME,
			      sizeof (fname), fname, NULL);
    if (status != CL_SUCCESS)
      pferror ("clGetDeviceInfo failed", status);
    std::cout<<" "<<fname<<", "<<proc ()<<" proc, " // <<vec_len ()<<" vec"<<", "
	     <<group_len ()<<" thrd, "<<shmem_size ()<<" lmem, "
	     << (double)mem_size ()<<" dmem\n";
    context = clCreateContext (0, 1, &device, NULL, NULL, &status);
    if (status != CL_SUCCESS)
      pferror ("clCreateContext failed", status);
    commandQueue = clCreateCommandQueue (context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    if (status != CL_SUCCESS)
      pferror ("clCreateCommandQueue failed", status);
  }

  void compile (const char* arg, const char* opt) {
    cl_int status;
    char fname [200];
    strcpy (fname, arg);
    char *f = strchr (fname, 0);
    strcpy (f, ".cl");
    FILE* pFileStream = fopen (fname, "rb");
    if (pFileStream == 0) {
      std::cerr<<"file open "<<fname;
      perror (" failed");
    }
    fseek (pFileStream, 0, SEEK_END); 
    size_t kernelLength = ftell (pFileStream);
    fseek (pFileStream, 0, SEEK_SET); 
    source = (char *)malloc (kernelLength + 1); 
    if (fread (source, kernelLength, 1, pFileStream) != 1)
      perror ("fread failed");
    fclose (pFileStream);
    source [kernelLength] = '\0';
    program = clCreateProgramWithSource (context, 1, (const char **)&source,
					 &kernelLength, &status);
    if (status != CL_SUCCESS)
      pferror ("clCreateProgramWithSource failed", status);

    char flags [1000];
    strcpy (flags, "-cl-single-precision-constant -cl-fast-relaxed-math -cl-mad-enable ");
    f = strchr (flags, 0);
    strcpy (f, opt);
    status = clBuildProgram (program, 1, &device, flags, NULL, NULL);
    if (status != CL_SUCCESS) {
      char cBuildLog [10240];
      clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, 
			     sizeof (cBuildLog), cBuildLog, NULL );
      std::cerr<<cBuildLog<<"\n";
      pferror ("clBuildProgram failed", status);
    }

    cl_kernel ks [maxkernels];
    status = clCreateKernelsInProgram (program, maxkernels, &ks [0], &num_kernels);
    if (status != CL_SUCCESS)
      pferror ("clCreateKernelsInProgram failed", status);
    char *name = new char [15];
    for (int i=0; i<maxkernels; i++)
      kernels [i] = 0;
    for (int i=0; i<num_kernels; i++) { // sort kernels
      size_t s;
      status = clGetKernelInfo (ks [i], CL_KERNEL_FUNCTION_NAME, 15, (void*)name, &s);
      if (status != CL_SUCCESS)
	pferror ("clGetKernelInfo failed", status);
      int j;
      sscanf (name, "kernel%d", &j); // kernel names
      kernels [j] = ks [i];
    }
    delete [] name;

#undef maxkernels
    arg_count = 0;
  }

  void syncQueue () {
    cl_int status;
    status = clFinish (commandQueue);
    if (status != CL_SUCCESS)
      pferror ("clFinish failed", status);
  }

  void sync () {
    cl_int status;
    // status = clFinish (commandQueue);
    // if (status != CL_SUCCESS)
    //   pferror ("clFinish failed", status);
    status = clWaitForEvents (1, &event);
    if (status != CL_SUCCESS)
      pferror ("clWaitForEvents failed", status);
  }

  cl_uint proc () {
    cl_int status;
    cl_uint r;
    status = clGetDeviceInfo (device, CL_DEVICE_MAX_COMPUTE_UNITS,
			      sizeof (r), &r, NULL);
    if (status != CL_SUCCESS)
      pferror ("clGetDeviceInfo failed", status);
    return r;
  }

  cl_uint vec_len () {
    cl_int status;
    cl_uint r;
    status = clGetDeviceInfo (device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
			      sizeof (r), &r, NULL);
    if (status != CL_SUCCESS)
      pferror ("clGetDeviceInfo failed", status);
    return r;
  }

  size_t group_len () {
    cl_int status;
    size_t r;
    status = clGetDeviceInfo (device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
			      sizeof (r), &r, NULL);
    if (status != CL_SUCCESS)
      pferror ("clGetDeviceInfo failed", status);
    return r;
  }

  cl_ulong shmem_size () {
    cl_int status;
    cl_ulong r;
    status = clGetDeviceInfo (device, CL_DEVICE_LOCAL_MEM_SIZE,
			      sizeof (r), &r, NULL);
    if (status != CL_SUCCESS)
      pferror ("clGetDeviceInfo failed", status);
    return r;
  }

  cl_ulong mem_size () {
    cl_int status;
    cl_ulong r;
    status = clGetDeviceInfo (device, CL_DEVICE_GLOBAL_MEM_SIZE,
			      sizeof (r), &r, NULL);
    if (status != CL_SUCCESS)
      pferror ("clGetDeviceInfo failed", status);
    return r;
  }

  template <typename A>
  cl_mem alloc (int n);

  void free (cl_mem x) {
    cl_int status;
    status = clReleaseMemObject (x);
    if (status != CL_SUCCESS)
      pferror ("clReleaseMemObject failed", status);
  }

  template <typename A>
  void read (const cl_mem xd, A* xh, int off, int n);

  template <typename A>
  void write (const A* xh, cl_mem xd, int off, int n);

  template <typename A>
  void copy (const cl_mem xsrc, pfgpu_& src, cl_mem xdst, int offsrc, int offdst, int n) {
    // copy device -> host -> device,
    // there is no "peer to peer copy" + memory allocation on a specific device
    // at the same time in OpenCL 
    A *h = new A [n];
    src.read (xsrc, h, offsrc, n);
    src.syncQueue ();
    write (h, xdst, offdst, n);
    syncQueue ();
    delete [] h;
  }


  void addArg (int i, int v) {
    cl_int status;
    status = clSetKernelArg (kernels [i], arg_count, sizeof (cl_int), (void*)&v);
    if (status != CL_SUCCESS)
      pferror ("clSetKernelArg int failed", status);
    arg_count++;
  }

  void addArg (int i, unsigned int v) {
    cl_int status;
    status = clSetKernelArg (kernels [i], arg_count, sizeof (cl_uint), (void*)&v);
    if (status != CL_SUCCESS)
      pferror ("clSetKernelArg int failed", status);
    arg_count++;
  }

  void addArg (int i, cl_mem& v) {
    cl_int status;
    status = clSetKernelArg (kernels [i], arg_count, sizeof (cl_mem), (void*)&v);
    if (status != CL_SUCCESS)
      pferror ("clSetKernelArg mem failed", status);
    arg_count++;
  }

  double time () {
    cl_ulong start = 0, end = 0;
    clGetEventProfilingInfo (event, CL_PROFILING_COMMAND_START,
			     sizeof (cl_ulong), &start, NULL);
    clGetEventProfilingInfo (event, CL_PROFILING_COMMAND_END,
			     sizeof (cl_ulong), &end, NULL);
    long long t0 = start, t1 = end;
    return (double) (t1 - t0) * (double)1.0e-9; // Nvidia GPU
  }

  void launch (int i) {
    if (i<0 || i>=num_kernels || ! kernels [i])
      pferror ("no kernel");
    assert (i < num_kernels);
    cl_int status;
    status = clEnqueueNDRangeKernel (commandQueue, kernels [i], work_dim, NULL,
				     &global_work_size [0], &local_work_size [0],
				     0, NULL, &event);
    // std::cout<<"kernel "<<i<<" dim "<< work_dim<<": global "<<global_work_size [0]<<" "<<global_work_size [1]<<", local "<<local_work_size [0]<<" "<<local_work_size [1]<<"\n";

    if (status != CL_SUCCESS)
      pferror ("clEnqueueNDRangeKernel failed", status);
    arg_count = 0;
  }

  void launch (int i, int s1
#ifdef PF_MPI
	       , int i1
#endif // PF_MPI
	       , int l1) {
#ifdef PF_MPI
    addArg (i, i1);
#endif // PF_MPI
    work_dim = 1;
    global_work_size [0] = s1;
    local_work_size [0] = l1;
    launch (i);
  }

  void launch (int i, int s1
#ifdef PF_MPI
	       , int i1
#endif // PF_MPI
	       , int s2
#ifdef PF_MPI
	       , int i2
#endif // PF_MPI
	       , int l1, int l2) {
#ifdef PF_MPI
    addArg (i, i1);
    addArg (i, i2);
#endif // PF_MPI
    work_dim = 2;
    global_work_size [0] = s1;
    global_work_size [1] = s2;
    local_work_size [0] = l1;
    local_work_size [1] = l2;
    launch (i);
  }

  void launch (int i, int s1
#ifdef PF_MPI
	       , int i1
#endif // PF_MPI
	       , int s2
#ifdef PF_MPI
	       , int i2
#endif // PF_MPI
	       , int s3
#ifdef PF_MPI
	       , int i3
#endif // PF_MPI
	       , int l1, int l2, int l3) {
#ifdef PF_MPI
    addArg (i, i1);
    addArg (i, i2);
    addArg (i, i3);
#endif // PF_MPI
    work_dim = 3;
    global_work_size [0] = s1;
    global_work_size [1] = s2;
    global_work_size [2] = s3;
    local_work_size [0] = l1;
    local_work_size [1] = l2;
    local_work_size [2] = l3;
    launch (i);
  }

  void close () {
    if (source) ::free (source);
    for (int i=0; i<num_kernels; i++)
      if (kernels [i]) clReleaseKernel (kernels [i]);  
    if (program) clReleaseProgram (program);
    if (commandQueue) clReleaseCommandQueue (commandQueue);
    if (context) clReleaseContext (context);
  }
};


template <>
cl_mem pfgpu_::alloc<float> (int n) {
  cl_int status;
  cl_mem x;
  x = clCreateBuffer (context, CL_MEM_READ_WRITE,
		      sizeof (cl_float) * n, NULL, &status);
  if (status != CL_SUCCESS)
    pferror ("clCreateBuffer failed", status);
  return x;
}

template <>
cl_mem pfgpu_::alloc<double> (int n) {
  cl_int status;
  cl_mem x;
  x = clCreateBuffer (context, CL_MEM_READ_WRITE,
		      sizeof (cl_double) * n, NULL, &status);
  if (status != CL_SUCCESS)
    pferror ("clCreateBuffer failed", status);
  return x;
}

template <>
void pfgpu_::read (const cl_mem xd, float* xh, int off, int n) {
  cl_int status;
  status = clEnqueueReadBuffer (commandQueue, xd, CL_TRUE,
				sizeof (cl_float) * off, sizeof (cl_float) * n, &xh [0], 0, NULL, NULL);
  if (status != CL_SUCCESS)
    pferror ("clEnqueueReadBuffer failed", status);
}

template <>
void pfgpu_::read (const cl_mem xd, double* xh, int off, int n) {
  cl_int status;
  status = clEnqueueReadBuffer (commandQueue, xd, CL_TRUE,
				sizeof (cl_double) * off, sizeof (cl_double) * n, &xh [0], 0, NULL, NULL);
  if (status != CL_SUCCESS)
    pferror ("clEnqueueReadBuffer failed", status);
}

template <>
void pfgpu_::write (const float* xh, cl_mem xd, int off, int n) {
  cl_int status;
  status = clEnqueueWriteBuffer (commandQueue, xd, CL_TRUE,
				 sizeof (cl_float) * off, sizeof (cl_float) * n, &xh [0], 0, NULL, NULL);
  if (status != CL_SUCCESS)
    pferror ("clEnqueueWriteBuffer failed", status);
}

template <>
void pfgpu_::write (const double* xh, cl_mem xd, int off, int n) {
  cl_int status;
  status = clEnqueueWriteBuffer (commandQueue, xd, CL_TRUE,
				 sizeof (cl_double) * off, sizeof (cl_double) * n, &xh [0], 0, NULL, NULL);
  if (status != CL_SUCCESS)
    pferror ("clEnqueueWriteBuffer failed", status);
}


pfgpu_ pfgpu [DEV_MAX];


void pferror (const char*n, int s) {
#ifdef PF_MPI
  int id;
  MPI_Comm_rank (MPI_COMM_WORLD, &id);
  std::cerr <<" proc " << id << ": ";
#endif // PF_MPI
  std::cerr << n;
  if (s) {
    switch (s) {
    case CL_OUT_OF_RESOURCES:
      std::cerr << ": CL out of resources ";
      break;
    case CL_INVALID_WORK_ITEM_SIZE:
      std::cerr << ": CL invalid work item size ";
      break;
    case CL_INVALID_BUFFER_SIZE:
      std::cerr << ": CL invalid buffer size ";
      break;
    case CL_BUILD_PROGRAM_FAILURE:
      std::cerr << ": CL build program failure ";
      break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      std::cerr << ": CL mem object allocation failure ";
      break;
    case CL_INVALID_CONTEXT:
      std::cerr << ": CL invalid context ";
      break;
    case CL_INVALID_KERNEL_DEFINITION:
      std::cerr << ": CL invalid kernel definition ";
      break;
    case CL_INVALID_WORK_GROUP_SIZE:
      std::cerr << ": CL invalid work group size ";
      break;
    case CL_INVALID_COMMAND_QUEUE:
      std::cerr << ": CL invalid command queue ";
      break;
    default:
      std::cerr << " code " << s;
    }
  }
  std::cerr << std::endl;
#ifdef PF_MPI
  MPI_Finalize ();
#endif // PF_MPI
  exit (-1);
}


#endif // GPU_OCL_HPP
