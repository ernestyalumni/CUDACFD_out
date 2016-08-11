/* sharedmem.h
 * shared memory to a 1-dim. grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160710
 */
#ifndef __SHAREDMEM_H__
#define __SHAREDMEM_H__

#include "finitediff.h"

#include "../physlib/dev_R3grid.h"

namespace sharedmem {

extern __constant__ int RAD[1] ;  // radius of the stencil; helps to deal with "boundary conditions" at (thread) block's ends



__device__ int idxClip( int idx, int idxMax) ;
		
__device__ int flatten(int k_x, int k_y, int k_z, int L_x, int L_y, int L_z) ;

__device__ void sh_dev_div1( float* dev_rho, float3* dev_u, float &result  ) ;

__device__ float sh_dev_div2( float* dev_rho, float3* dev_u  ) ;

__device__ float sh_dev_div3( float* dev_rho, float3* dev_u  ) ;

__device__ float sh_dev_div4( float* dev_rho, float3* dev_u  ) ;


}

#endif // __SHAREDMEM_H__

		

		
		
