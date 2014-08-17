#include "util.hpp"
#include "simd.hpp"

void pferror(const char*n, int s) {
#ifdef PF_MPI
  std::cerr <<" proc " << pfcomm.id << ": ";
#endif // PF_MPI
  std::cerr << n;
  if (s)
    std::cerr << " code " << s;
  std::cerr << std::endl;
  pfcomm.finalize();
  exit(-1);
}

pfcomm_::pfcomm_ () {}

pfcomm_::~pfcomm_ () {}

void pfcomm_::init (int &argc, char **&argv) {
#ifdef PF_MPI
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &id);
  MPI_Comm_size (MPI_COMM_WORLD, &pr);
#else // PF_MPI
  id = 0;
  pr = 1;
#endif // PF_MPI
  l = (id+pr-1)%pr;
  h = (id+1)%pr;
}

void pfcomm_::finalize () {
#ifdef PF_MPI
  MPI_Finalize ();
#endif // PF_MPI
}

void pfcomm_::barrier () {
#ifdef PF_MPI
  MPI_Barrier (MPI_COMM_WORLD);
#endif // PF_MPI
}

double pfcomm_::reduce_max (double r) {
  double r0;
#ifdef PF_MPI
  MPI_Allreduce (&r, &r0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#else
  r0 = r;
#endif // PF_MPI
  return r0;
}

void pfcomm_::send (real32 *xl, real32 *xr, int j) { //vec [xl,..,xr[
#ifdef PF_MPI
  MPI_Status status;
  MPI_Sendrecv (xr-2*j, j, MPI_FLOAT, pfcomm.r, 7,
		xl,     j, MPI_FLOAT, pfcomm.l, 7, 
		MPI_COMM_WORLD, &status);
  MPI_Sendrecv (xl+j, j, MPI_FLOAT, pfcomm.l, 8,
		xr-j, j, MPI_FLOAT, pfcomm.r, 8, 
		MPI_COMM_WORLD, &status);
#else
  for (int i=0; i<j; i++)
    xr[-j+i] = xl[j+i];
  for (int i=0; i<j; i++)
    xl[i] = xr[-2*j+i];
#endif // PF_MPI
}

void pfcomm_::send (real64 *xl, real64 *xr, int j) { //vec [xl,..,xr[
#ifdef PF_MPI
  MPI_Status status;
  MPI_Sendrecv (xr-2*j, j, MPI_DOUBLE, pfcomm.r, 7,
		xl,     j, MPI_DOUBLE, pfcomm.l, 7, 
		MPI_COMM_WORLD, &status);
  MPI_Sendrecv (xl+j, j, MPI_DOUBLE, pfcomm.l, 8,
		xr-j, j, MPI_DOUBLE, pfcomm.r, 8, 
		MPI_COMM_WORLD, &status);
#else
  for (int i=0; i<j; i++)
    xr[-j+i] = xl[j+i];
  for (int i=0; i<j; i++)
    xl[i] = xr[-2*j+i];
#endif // PF_MPI
}
