/* dev_R1grid.h
 * R1 under discretization (discretize) functor to a thread block
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161026
 * */

#ifndef __DEV_R1GRID_H__
#define __DEV_R1GRID_H__

#include "../commonlib/errors.h"

extern __constant__ int dev_Ld[1]; // L^{d=1} = L_x \in \mathbb{N}

extern __constant__ float dev_Deltat[1]; // \Delta t, on the device


class dev_Grid1d
{
	public:
		dim3 Ld;
		
		// this has to be a pointer to a float or "old-school" float array 
		// in order to be cuda copied back to the host
		float *dev_f;

		float *dev_foverline;
		float *dev_f_out;
		
		// constructor
		__host__ dev_Grid1d( dim3 );
		
		__host__ __device__ int NFLAT();
};


#endif // __DEV_R1GRID_H__
