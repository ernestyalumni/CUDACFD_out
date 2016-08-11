/* convect.cu
 * 3-dim. convection by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160720
 */
#include "convect.h"

__constant__ float dev_Deltat[1] ;

// convect_fd_naive_sh - convection with finite difference and naive shared memory scheme
__global__ void convect_fd_naive_sh( float* dev_rho, float2* dev_u ) {

	const int NU = 2;
	
	// map from threadIdx/blockIdx to x grid position
	int k_x = threadIdx.x + blockIdx.x * blockDim.x;
	int k_y = threadIdx.y + blockIdx.y * blockDim.y;
	
	int k = k_x + k_y*blockDim.x*gridDim.x  ;

	__shared__ int2 stencilindicesplus[NU] ;
	__shared__ int2 stencilindicesminus[NU] ;

	for (int nu = 0; nu < NU; ++nu ) {
		stencilindicesplus[  nu ].x = k + (nu + 1) ; 
		stencilindicesminus[ nu ].x = k - (nu + 1) ; 
		stencilindicesplus[  nu ].y = k + (nu + 1)*dev_Ld[0] ; 
		stencilindicesminus[ nu ].y = k - (nu + 1)*dev_Ld[0] ; 
	}

	int XI = 0;

	// check boundary conditions
	for (int nu = 0; nu < NU; ++nu) {
		if (k_x == nu ) {
			XI = NU - nu ;
			for (int xi = 0; xi < XI; ++xi ) {
				stencilindicesminus[ NU - 1 - xi ].x += XI- xi ;  
			}
		}
	
		if (k_y == nu ) {
			XI = NU - nu ;
			for (int xi = 0; xi < XI; ++xi) { 
				stencilindicesminus[ NU - 1 - xi ].y += (XI - xi)*dev_Ld[0] ;
			}
		}
		
	
		if (k_x == (dev_Ld[0] - (nu + 1) ) ) {
			XI = NU - nu ;
			for (int xi = 0; xi < XI; ++xi ) {
				stencilindicesplus[ NU - 1 - xi].x -= XI-xi ;
			}
		}
		
		if (k_y == (dev_Ld[1] - (nu + 1) ) ) {
			XI = NU - nu ;
			for (int xi = 0; xi < XI; ++xi ) {
				stencilindicesplus[ NU - 1 - xi].y -= (XI-xi)*dev_Ld[0] ;
			}
		}
		
		
	}

	__syncthreads();
	
	__shared__ float2 stencil[NU][2] ; 
	
	for (int nu = 0 ; nu < NU; ++nu ) {
		stencil[nu][0].x = dev_rho[stencilindicesminus[nu].x]*dev_u[stencilindicesminus[nu].x].x  ;
		stencil[nu][1].x = dev_rho[stencilindicesplus[nu].x]*dev_u[stencilindicesplus[nu].x].x  ;
		stencil[nu][0].y = dev_rho[stencilindicesminus[nu].y]*dev_u[stencilindicesminus[nu].y].y  ;
		stencil[nu][1].y = dev_rho[stencilindicesplus[nu].y]*dev_u[stencilindicesplus[nu].y].y  ;
	}	
	
	float div_value { dev_div2( stencil ) } ;
	
	__syncthreads();
	
	dev_rho[k] +=  dev_Deltat[0] * (-1.f) * div_value ;		
			
//	__syncthreads();		
			
}






