/* dev_R1grid.cu
 * R1 under discretization (discretize functor) to a grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161026
 * */
#include "dev_R1grid.h"

__constant__ int dev_Ld[1];

//__constant__ float dev_Deltat[1];


__host__ dev_Grid1d::dev_Grid1d( dim3 Ld_in) : Ld(Ld_in)
{
	HANDLE_ERROR(
		cudaMalloc((void**)&this->dev_f, this->NFLAT()*sizeof(float) ));

	HANDLE_ERROR(
		cudaMalloc((void**)&this->dev_foverline, this->NFLAT()*sizeof(float) ));

	HANDLE_ERROR(
		cudaMalloc((void**)&this->dev_f_out, this->NFLAT()*sizeof(float) ));

}

__host__ __device__ int dev_Grid1d :: NFLAT() {
	return Ld.x ;
}

