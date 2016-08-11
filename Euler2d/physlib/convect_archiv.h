/* convect.h
 * 2-dim. Euler eq. (heat eq.) by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160804
 */
#ifndef __CONVECT_H__
#define __CONVECT_H__

#include "../commonlib/finitediff.h"
#include "../commonlib/sharedmem.h" // __device__ int flatten
#include "dev_phys_param.h" // dev_Deltat
#include "dev_R2grid.h" // dev_Ld
 
__global__ void convectKernel(float *dev_rho, float *dev_rho_out, float2 *dev_u) ; 

__global__ void convectKernelb(float *dev_rho, float2 *dev_u) ; 

__global__ void convectKernelc(float *dev_rho, float2 *dev_u) ; 


__global__ void convect_fd_naive_sh( float* dev_rho, float* dev_rho_out, float2* dev_u ) ;

// convect finite difference, "naive" shared method, for stencil of "size 2"
__global__ void convect_fd_naive_sh2( float* dev_rho, float2* dev_u ) ;

__global__ void convect_fd_naive_sh2b( float* dev_rho, float* dev_rho_out, float2* dev_u ) ;

__global__ void convect_fd2( float* dev_rho, float2* dev_u ) ;

 
#endif // __CONVECT_H__
