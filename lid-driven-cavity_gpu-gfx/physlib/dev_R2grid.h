/* dev_R2grid.h
 * R2 under discretization (discretize functor) to a thread block
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160728
 */
#ifndef __DEV_R2GRID_H__
#define __DEV_R2GRID_H__

#include "../commonlib/checkerror.h" // checkCudaErrors(

class Dev_Grid2d
{
	public:
		dim3 Ld;

		float* F;
		float* G;
		float* u;
		float* v;
	
		float* pres_red;
		float* pres_black;
		

		// constructor
		__host__ Dev_Grid2d( dim3 );

		// destructor
//		__host__ ~dev_Grid3d();

		__host__ int NFLAT();
		
		__host__ int staggered_NFLAT();
};


#endif // __DEV_R2GRID_H__
