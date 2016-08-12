/* convect.h
 * 2-dim. Euler eq. (heat eq.) by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160804
 */
#ifndef __CONVECT_H__
#define __CONVECT_H__

#include "../commonlib/finitediff.h"
#include "../commonlib/sharedmem.h" // __device__ int flatten
#include "dev_phys_param.h" // dev_Deltat, dev_phys_params
#include "dev_R2grid.h" // dev_Ld
 
// convect finite difference, for stencil of "size NU"
//__device__ float pressure(float, float, float2 );

__global__ void convect_fd( float* dev_rho, float2* dev_u ) ;

__global__ void convect_fd2( float* dev_rho, float2* dev_u ) ;

__global__ void convect_fd_sh( float* dev_rho, float2* dev_u ) ;


__global__ void Euler2dp1( float* dev_rho, float2* dev_p, float2* dev_u, float* dev_E ) ; 

__global__ void Euler2dp2( float* dev_rho, float2* dev_p, float2* dev_u, float* dev_E ) ; 

 
#endif // __CONVECT_H__
