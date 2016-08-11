/* convect.cu
 * 2-dim. Euler eq. (heat eq.) by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160804
 */
#include "convect.h"


__global__ void convect_fd( float* dev_rho, float2* dev_u ) {
	const int NU = 1;
	
	// map from threadIdx/blockIdx to x grid position
	int k_x = threadIdx.x + blockIdx.x * blockDim.x;
	int k_y = threadIdx.y + blockIdx.y * blockDim.y;
	
	int k = k_x + k_y*blockDim.x*gridDim.x ;
	
	int2 stencilindicesplus[NU] ;
	int2 stencilindicesminus[NU] ;

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
	
	float div_value { dev_div1( stencil ) } ;

	__syncthreads();
	
	dev_rho[k] +=  dev_Deltat[0] * (-1.f) * div_value ;		
			
//	__syncthreads();		
			
}


__global__ void convect_fd2( float* dev_rho, float2* dev_u ) {
	const int NU = 2;
	
	// map from threadIdx/blockIdx to x grid position
	int k_x = threadIdx.x + blockIdx.x * blockDim.x;
	int k_y = threadIdx.y + blockIdx.y * blockDim.y;
	
	int k = k_x + k_y*blockDim.x*gridDim.x ;
	
	int2 stencilindicesplus[NU] ;
	int2 stencilindicesminus[NU] ;

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

__global__ void convect_fd_sh( float* dev_rho, float2* dev_u ) {
	const int NU = 1;
	
	// map from threadIdx/blockIdx to x grid position
	int k_x = threadIdx.x + blockIdx.x * blockDim.x;
	int k_y = threadIdx.y + blockIdx.y * blockDim.y;
	
	int k = k_x + k_y*blockDim.x*gridDim.x ;
	
	int2 stencilindicesplus[NU] ;
	int2 stencilindicesminus[NU] ;

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

	__shared__ float2 sh_rho[NU][2] ; 
	__shared__ float2 sh_u[NU][2]  ;

	for (int nu = 0 ; nu < NU; ++nu ) {
		sh_rho[nu][0].x = dev_rho[stencilindicesminus[nu].x] ; 
		sh_u[nu][0].x = dev_u[stencilindicesminus[nu].x].x ;
		sh_rho[nu][1].x = dev_rho[stencilindicesplus[nu].x] ; 
		sh_u[nu][1].x = dev_u[stencilindicesplus[nu].x].x ;
		sh_rho[nu][0].y = dev_rho[stencilindicesminus[nu].y] ; 
		sh_u[nu][0].y = dev_u[stencilindicesminus[nu].y].y ; 
		sh_rho[nu][1].y = dev_rho[stencilindicesplus[nu].y] ; 
		sh_u[nu][1].y = dev_u[stencilindicesplus[nu].y].y ;
	}
	
	__shared__ float2 rhoustencil[NU][2] ; 
	
	for (int nu = 0 ; nu < NU; ++nu ) {
		rhoustencil[nu][0].x = sh_rho[nu][0].x * sh_u[nu][0].x ; 
//		dev_rho[stencilindicesminus[nu].x]*dev_u[stencilindicesminus[nu].x].x  ;
		rhoustencil[nu][1].x = sh_rho[nu][1].x * sh_u[nu][1].x ; 
//		dev_rho[stencilindicesplus[nu].x]*dev_u[stencilindicesplus[nu].x].x  ;
		rhoustencil[nu][0].y = sh_rho[nu][0].y * sh_u[nu][0].y ; 
		//dev_rho[stencilindicesminus[nu].y]*dev_u[stencilindicesminus[nu].y].y  ;
		rhoustencil[nu][1].y = sh_rho[nu][1].y * sh_u[nu][1].y ; 
		//dev_rho[stencilindicesplus[nu].y]*dev_u[stencilindicesplus[nu].y].y  ;
	}	
	
	float div_value { dev_div1( rhoustencil ) } ;

	__syncthreads();
	
	dev_rho[k] +=  dev_Deltat[0] * (-1.f) * div_value ;		
			
//	__syncthreads();		
			
}


__global__ void Euler2dp1( float* dev_rho, float2* dev_u, float* dev_E ) {
	const int NU = 1;
	
	// map from threadIdx/blockIdx to x grid position
	int k_x = threadIdx.x + blockIdx.x * blockDim.x;
	int k_y = threadIdx.y + blockIdx.y * blockDim.y;
	
	int k = k_x + k_y*blockDim.x*gridDim.x ;
	
	int2 stencilindicesplus[NU] ;
	int2 stencilindicesminus[NU] ;

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
	
	float div_value { dev_div1( stencil ) } ;

	// energy stencils
	
	__shared__ float2 Estencil[NU][2] ; 
	
	for (int nu = 0 ; nu < NU; ++nu ) {
		Estencil[nu][0].x = dev_gas_params[0] * 
			dev_E[stencilindicesminus[nu].x]*dev_u[stencilindicesminus[nu].x].x  ;
		Estencil[nu][1].x = dev_gas_params[0] * 
			dev_E[stencilindicesplus[nu].x]*dev_u[stencilindicesplus[nu].x].x  ;
		Estencil[nu][0].y = dev_gas_params[0] * 
			dev_E[stencilindicesminus[nu].y]*dev_u[stencilindicesminus[nu].y].y  ;
		Estencil[nu][1].y = dev_gas_params[0] *
			dev_E[stencilindicesplus[nu].y]*dev_u[stencilindicesplus[nu].y].y  ;
	}	
	
	__shared__ float2 KEstencil[NU][2] ; 
	
	for (int nu = 0 ; nu < NU; ++nu ) {
		KEstencil[nu][0].x = (dev_gas_params[0]-1.f) * 0.5f *
			dev_rho[stencilindicesminus[nu].x]*dev_u[stencilindicesminus[nu].x].x  ;
		KEstencil[nu][1].x = (dev_gas_params[0] -1.f) * 0.5f *
			dev_rho[stencilindicesplus[nu].x]*dev_u[stencilindicesplus[nu].x].x  ;
		KEstencil[nu][0].y = (dev_gas_params[0] -1.f) * 0.5f *
			dev_rho[stencilindicesminus[nu].y]*dev_u[stencilindicesminus[nu].y].y  ;
		KEstencil[nu][1].y = (dev_gas_params[0] - 1.f) * 0.5f *
			dev_rho[stencilindicesplus[nu].y]*dev_u[stencilindicesplus[nu].y].y  ;
	}	
	
	__syncthreads();
	
	dev_rho[k] +=  dev_Deltat[0] * (-1.f) * div_value ;		
			
//	__syncthreads();		
			
}






