/* Euler2d.h
 * 2-dim. Euler eq. (heat eq.) by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160804
 */
#ifndef __EULER2D_H__
#define __EULER2D_H__

#include "../commonlib/sharedmem.h" // __global__ void float_to_char
#include "convect.h"  // __global__void convectKernel



void kernelLauncher(uchar4 *d_out, 
	float *dev_rho, 
	float2 *dev_u,
	dim3 Ld, dim3 M_in) ;

void kernelLauncher2(uchar4 *d_out, 
	float *dev_rho, 
	float2 *dev_u,
	dim3 Ld, dim3 M_in) ;	


void EulerLauncher(uchar4 *d_out, 
	float *dev_rho, float2 *dev_p,
	float2 *dev_u, float *dev_E,
	dim3 Ld, dim3 M_in) ;	


#endif // __EULER2D_H__
