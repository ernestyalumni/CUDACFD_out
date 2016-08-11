/* dev_R3grid.cu
 * R3 under discretization (discretize functor) to a grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160630
 */
#include "dev_R3grid.h"

__device__ dev_Grid3d :: dev_Grid3d(unsigned int N_x, unsigned int N_y, unsigned int N_z ) : N_is {N_x,N_y,N_z} 
{}


__device__ unsigned int dev_Grid3d :: flatten(
							unsigned int i_x, unsigned int i_y, unsigned int i_z) {
	return i_x+i_y*N_is[0]+i_z*N_is[0]*N_is[1];
}	
	
