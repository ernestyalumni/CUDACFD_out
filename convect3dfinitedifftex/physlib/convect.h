/* convect.h
 * 3-dim. convection by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160720
 */
#ifndef __CONVECT_H__
#define __CONVECT_H__

#include "../commonlib/finitediff.h"
#include "../commonlib/sharedmem.h"

#include "../physlib/dev_R3grid.h"

extern __constant__ float dev_Deltat[1] ; // Deltat

// convect_fd_naive_sh - convection with finite difference and naive shared memory scheme
__global__ void convect_fd_naive_sh( float* dev_rho, float3* dev_u ) ;

__global__ void convect_sh( float* dev_rho, float3* dev_u);

#endif // __CONVECT_H__
