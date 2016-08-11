/* dev_R3grid.h
 * R3 under discretization (discretize functor) to a thread block
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160701
 */
#ifndef __DEV_R3GRID_H__
#define __DEV_R3GRID_H__

class dev_Grid3d
{
	public:
		unsigned int  N_is[3];

		__device__ dev_Grid3d(unsigned int N_x, unsigned int N_y, unsigned int N_z);

		__device__ unsigned int flatten(unsigned int i_x, unsigned int i_y, unsigned int i_z);
};

#endif // __DEV_R3GRID_H__

 
