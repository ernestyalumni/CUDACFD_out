/* Euler2d.h
 * 2-dim. Euler eq. (heat eq.) by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160804
 */
#ifndef __EULER2D_H__
#define __EULER2D_H__

#include "../commonlib/finitediff.h"
#include "dev_R2grid.h"  // dev_Ld

extern __constant__ float dev_Deltat[1]; // Deltat

extern __constant__ float dev_gas_params[2] ; // dev_gas_params[0] = heat capacity ratio

void kernelLauncher(uchar4 *d_out, 
	float *dev_rho, 
	float2 *dev_u,
	float2 *dev_p,
	float *dev_E, 
	dim3 Ld, dim3 M_in) ;
	
void kernelLauncher2(uchar4 *d_out, 
	float *dev_rho, 
	float2 *dev_u,
	float2 *dev_p,
	float *dev_E, 
	dim3 Ld, dim3 M_in) ;

void kernelLauncher3(uchar4 *d_out, 
	float *dev_rho, 
	float2 *dev_u,
	float2 *dev_p,
	float *dev_E, 
	dim3 Ld, dim3 M_in) ;

void kernelLauncher4(uchar4 *d_out, 
	float *dev_rho, 
	float2 *dev_u,
	float2 *dev_p,
	float *dev_E, 
	dim3 Ld, dim3 M_in) ;

#endif // __EULER2D_H__
