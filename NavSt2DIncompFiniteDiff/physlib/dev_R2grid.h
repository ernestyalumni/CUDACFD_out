/* dev_R2grid.h
 * R2 under discretization (discretize functor) to a (staggered) grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161114
 * 
 * compilation tip: (compile separately)
 * nvcc -std=c++11 -c ./physlib/dev_R2grid.cu -o dev_R2grid.o
 * 
 */
#ifndef __DEV_R2GRID_H__
#define __DEV_R2GRID_H__

#include <thrust/device_vector.h>

#include "../commonlib/checkerror.h"  // checkCudaErrors

class Dev_Grid2d
{
	public:
		dim3 Ld;  // Ld.x,Ld.y = L_x, L_y or i.e. imax,jmax 
		dim3 staggered_Ld; // Ld[0]+2,Ld[1]+2 = L_x+2,L_y+2, or i.e. imax+2,jmax+2 

		// thrust::device_vector for pressure, and other scalar fields
		thrust::device_vector<float> p;
		thrust::device_vector<float> p_temp;
		thrust::device_vector<float> F;
		thrust::device_vector<float> G;
		thrust::device_vector<float> RHS;

		// thrust:device_vector for velocity
		thrust::device_vector<float> u;
		thrust::device_vector<float> v;

		float* p_arr ; 
		float* p_temp_arr ; 
		float* F_arr ; 
		float* G_arr ; 
		float* RHS_arr ; 
		float* u_arr ; 
		float* v_arr ; 
	

		// Constructor
		/* --------------------------------------------------------- */
		/* Sets the initial values for velocity u, p                 */
		/* --------------------------------------------------------- */
		__host__ Dev_Grid2d( dim3 );

		// destructor
		__host__ ~Dev_Grid2d();

		__host__ int NFLAT();
		
		// __host__ int staggered_SIZE() - returns the staggered grid size
		/* this would correspond to Griebel's notation of 
		 * (imax+1)*(jmax+1)
		 */
		__host__ int staggered_SIZE();
		
		__host__ int flatten(const int i_x, const int i_y ) ;

		__host__ int staggered_flatten(const int i_x, const int i_y ) ;

};




#endif // __DEV_R2GRID_H__
