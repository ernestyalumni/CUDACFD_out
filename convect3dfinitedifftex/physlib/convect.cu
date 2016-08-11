/* convect.cu
 * 3-dim. convection by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160720
 */
#include "convect.h"

__constant__ float dev_Deltat[1] ;

// convect_fd_naive_sh - convection with finite difference and naive shared memory scheme
__global__ void convect_fd_naive_sh( float* dev_rho, float3* dev_u ) {

	const int NU = 2;
	
	// map from threadIdx/blockIdx to x grid position
	int k_x = threadIdx.x + blockIdx.x * blockDim.x;
	int k_y = threadIdx.y + blockIdx.y * blockDim.y;
	int k_z = threadIdx.z + blockIdx.z * blockDim.z;
	
	int k = k_x + k_y*blockDim.x*gridDim.x + k_z*blockDim.x*gridDim.x*blockDim.y*gridDim.y ;

	__shared__ int3 stencilindicesplus[NU] ;
	__shared__ int3 stencilindicesminus[NU] ;

	for (int nu = 0; nu < NU; ++nu ) {
		stencilindicesplus[  nu ].x = k + (nu + 1) ; 
		stencilindicesminus[ nu ].x = k - (nu + 1) ; 
		stencilindicesplus[  nu ].y = k + (nu + 1)*dev_Ld[0] ; 
		stencilindicesminus[ nu ].y = k - (nu + 1)*dev_Ld[0] ; 
		stencilindicesplus[  nu ].z = k + (nu + 1)*dev_Ld[0]*dev_Ld[1] ; 
		stencilindicesminus[ nu ].z = k - (nu + 1)*dev_Ld[0]*dev_Ld[1] ; 
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
		
		if (k_z == nu) {
			XI = NU - nu ;
			for (int xi = 0 ; xi < XI; ++xi ) {
				stencilindicesminus[ NU - 1 - xi ].z += (XI - xi)*dev_Ld[0]*dev_Ld[1] ;  
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
		
		if (k_z == (dev_Ld[2] - (nu + 1) ) ) {
			XI = NU - nu ;
			for (int xi = 0; xi < XI; ++xi ) {
				stencilindicesplus[ NU - 1 - xi].z -= (XI-xi)*dev_Ld[0]*dev_Ld[1] ;
			}
		}
		
	}

	__syncthreads();
	
	__shared__ float3 stencil[NU][2] ; 
	
	for (int nu = 0 ; nu < NU; ++nu ) {
		stencil[nu][0].x = dev_rho[stencilindicesminus[nu].x]*dev_u[stencilindicesminus[nu].x].x  ;
		stencil[nu][1].x = dev_rho[stencilindicesplus[nu].x]*dev_u[stencilindicesplus[nu].x].x  ;
		stencil[nu][0].y = dev_rho[stencilindicesminus[nu].y]*dev_u[stencilindicesminus[nu].y].y  ;
		stencil[nu][1].y = dev_rho[stencilindicesplus[nu].y]*dev_u[stencilindicesplus[nu].y].y  ;
		stencil[nu][0].z = dev_rho[stencilindicesminus[nu].z]*dev_u[stencilindicesminus[nu].z].z  ;
		stencil[nu][1].z = dev_rho[stencilindicesplus[nu].z]*dev_u[stencilindicesplus[nu].z].z  ;
	}	
	
	float div_value { dev_div2( stencil ) } ;
	
	__syncthreads();
	
	dev_rho[k] +=  dev_Deltat[0] * (-1.f) * div_value ;		
			
//	__syncthreads();		
			
}

__global__ void convect_sh( float* dev_rho, float3* dev_u ) {
	const int k_x = threadIdx.x + blockIdx.x * blockDim.x; 
	const int k_y = threadIdx.y + blockIdx.y * blockDim.y; 
	const int k_z = threadIdx.z + blockIdx.z * blockDim.z; 
	
	const int k = k_x + k_y*dev_Ld[0]+k_z*dev_Ld[0]*dev_Ld[1]   ;
	
	float div_value = sharedmem::sh_dev_div2( dev_rho, dev_u) ;

	__syncthreads();
	
	dev_rho[k] += dev_Deltat[0] * (1.f) * div_value ;

//	__syncthreads();
}





