/* convect.cu
 * 2-dim. Euler eq. (heat eq.) by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160804
 */
#include "convect.h"

/*
__device__ float pressure( float energy, float rho, float2 u ) {
	float pressureval { (dev_gas_params[0] - 1.f ) * ( energy - 0.5f * rho * (u.x*u.x+u.y*u.y) ) } ;
	return pressureval ; 
}*/

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


__global__ void Euler2dp1( float* dev_rho, float2* dev_p, float2* dev_u, float* dev_E ) {
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
	__shared__ float2 sh_usq[NU][2] ; // u^2
	__shared__ float2 sh_E[NU][2] ; 
	__shared__ float2 sh_px[NU][2] ; 
	__shared__ float2 sh_py[NU][2] ; 
	

	for (int nu = 0 ; nu < NU; ++nu ) {
		sh_rho[nu][0].x = dev_rho[stencilindicesminus[nu].x] ; 
		sh_u[nu][0].x = dev_u[stencilindicesminus[nu].x].x ;
		sh_usq[nu][0].x = dev_u[stencilindicesminus[nu].x].x * dev_u[stencilindicesminus[nu].x].x + 
					dev_u[stencilindicesminus[nu].x].y * dev_u[stencilindicesminus[nu].x].y ;
		sh_E[nu][0].x = dev_E[stencilindicesminus[nu].x] ;
		sh_px[nu][0].x = dev_p[stencilindicesminus[nu].x].x ;
		sh_py[nu][0].x = dev_p[stencilindicesminus[nu].x].y ;


		sh_rho[nu][1].x = dev_rho[stencilindicesplus[nu].x] ; 
		sh_u[nu][1].x = dev_u[stencilindicesplus[nu].x].x ;
		sh_usq[nu][1].x = dev_u[stencilindicesplus[nu].x].x * dev_u[stencilindicesplus[nu].x].x + 
					dev_u[stencilindicesplus[nu].x].y * dev_u[stencilindicesplus[nu].x].y ;
		sh_E[nu][1].x = dev_E[stencilindicesplus[nu].x] ;
		sh_px[nu][1].x = dev_p[stencilindicesplus[nu].x].x ;
		sh_py[nu][1].x = dev_p[stencilindicesplus[nu].x].y ;


		sh_rho[nu][0].y = dev_rho[stencilindicesminus[nu].y] ; 
		sh_u[nu][0].y = dev_u[stencilindicesminus[nu].y].y ; 
		sh_usq[nu][0].y = dev_u[stencilindicesminus[nu].y].x * dev_u[stencilindicesminus[nu].y].x + 
				dev_u[stencilindicesminus[nu].y].y * dev_u[stencilindicesminus[nu].y].y ; 
		sh_E[nu][0].y = dev_E[stencilindicesminus[nu].y] ;
		sh_px[nu][0].y = dev_p[stencilindicesminus[nu].y].x ;
		sh_py[nu][0].y = dev_p[stencilindicesminus[nu].y].y ;


		sh_rho[nu][1].y = dev_rho[stencilindicesplus[nu].y] ; 
		sh_u[nu][1].y = dev_u[stencilindicesplus[nu].y].y ;
		sh_usq[nu][1].y = dev_u[stencilindicesplus[nu].y].x * dev_u[stencilindicesplus[nu].y].x + 
				dev_u[stencilindicesplus[nu].y].y * dev_u[stencilindicesplus[nu].y].y ;
		sh_E[nu][1].y = dev_E[stencilindicesplus[nu].y] ;
		sh_px[nu][1].y = dev_p[stencilindicesplus[nu].y].x ;
		sh_py[nu][1].y = dev_p[stencilindicesplus[nu].y].y ;

	}

	
	__shared__ float2 rhoustencil[NU][2] ; 
	
	for (int nu = 0 ; nu < NU; ++nu ) {
		rhoustencil[nu][0].x = sh_rho[nu][0].x * sh_u[nu][0].x ; 
		rhoustencil[nu][1].x = sh_rho[nu][1].x * sh_u[nu][1].x ; 
		rhoustencil[nu][0].y = sh_rho[nu][0].y * sh_u[nu][0].y ; 
		rhoustencil[nu][1].y = sh_rho[nu][1].y * sh_u[nu][1].y ; 
	}	
	
	float div_value { dev_div1( rhoustencil ) } ;

	__shared__ float2 Estencil[NU][2] ; 
	
	for (int nu = 0 ; nu < NU; ++nu ) {
		Estencil[nu][0].x = dev_gas_params[0] * sh_E[nu][0].x * sh_u[nu][0].x ; 
		Estencil[nu][1].x = dev_gas_params[0] * sh_E[nu][1].x * sh_u[nu][1].x ; 
		Estencil[nu][0].y = dev_gas_params[0] * sh_E[nu][0].y * sh_u[nu][0].y ; 
		Estencil[nu][1].y = dev_gas_params[0] * sh_E[nu][1].y * sh_u[nu][1].y ; 
	}	
	
	float div_Eval { dev_div1( Estencil ) } ; 

	__shared__ float2 KEstencil[NU][2] ; 
	
	for (int nu = 0 ; nu < NU; ++nu ) {
		KEstencil[nu][0].x = 0.5f*(dev_gas_params[0]-1.f) * 
						sh_rho[nu][0].x * sh_usq[nu][0].x * sh_u[nu][0].x ; 
		KEstencil[nu][1].x = 0.5f*(dev_gas_params[0]-1.f) * 
						sh_rho[nu][1].x * sh_usq[nu][1].x * sh_u[nu][1].x ; 
		KEstencil[nu][0].y = 0.5f*(dev_gas_params[0]-1.f) * 
						sh_rho[nu][0].y * sh_usq[nu][0].y * sh_u[nu][0].y ; 
		KEstencil[nu][1].y = 0.5f*(dev_gas_params[0]-1.f) * 
						sh_rho[nu][1].y * sh_usq[nu][1].y * sh_u[nu][1].y ; 
	}	
	
	float div_KEval { dev_div1( KEstencil ) } ; 

	__shared__ float2 pxstencil[NU][2] ; 

	for (int nu = 0 ; nu < NU; ++nu ) {
		pxstencil[nu][0].x = sh_px[nu][0].x * sh_u[nu][0].x ; 
		pxstencil[nu][1].x = sh_px[nu][1].x * sh_u[nu][1].x ; 
		pxstencil[nu][0].y = sh_px[nu][0].y * sh_u[nu][0].y ; 
		pxstencil[nu][1].y = sh_px[nu][1].y * sh_u[nu][1].y ; 
	}
	
	float div_pxval { dev_div1( pxstencil ) } ; 

	__shared__ float2 pystencil[NU][2] ; 

	for (int nu = 0 ; nu < NU; ++nu ) {
		pystencil[nu][0].x = sh_py[nu][0].x * sh_u[nu][0].x ; 
		pystencil[nu][1].x = sh_py[nu][1].x * sh_u[nu][1].x ; 
		pystencil[nu][0].y = sh_py[nu][0].y * sh_u[nu][0].y ; 
		pystencil[nu][1].y = sh_py[nu][1].y * sh_u[nu][1].y ; 
	}

	float div_pyval { dev_div1( pystencil ) };

	__shared__ float2 pressurestencil[NU][2] ; 

	for (int nu = 0 ; nu < NU; ++nu ) {
		pressurestencil[nu][0].x = pressure( sh_E[nu][0].x, sh_rho[nu][0].x, sh_usq[nu][0].x ) ; 
		pressurestencil[nu][1].x = pressure( sh_E[nu][1].x, sh_rho[nu][1].x, sh_usq[nu][1].x )  ; 
		pressurestencil[nu][0].y = pressure( sh_E[nu][0].y, sh_rho[nu][0].y, sh_usq[nu][0].y )  ; 
		pressurestencil[nu][1].y = pressure( sh_E[nu][1].y, sh_rho[nu][1].y, sh_usq[nu][1].y )  ; 
	}

	float2 grad_pressureval = dev_grad1( pressurestencil ) ;
	
	__syncthreads() ;
	
	dev_rho[k] +=  dev_Deltat[0] * (-1.f) * div_value ;		
	dev_E[k] += dev_Deltat[0] * ( (-1.f) * div_Eval + div_KEval ) ; 		
	dev_p[k].x += dev_Deltat[0] * (-1.f) * ( div_pxval + grad_pressureval.x ) ; 
	dev_p[k].y += dev_Deltat[0] * (-1.f) * ( div_pyval + grad_pressureval.y ) ; 

	if (dev_rho[k] != 0.f ) {
		float2 temp_u { dev_p[k].x / dev_rho[k] , dev_p[k].y / dev_rho[k] } ;
	}

			
//	__syncthreads();		
			
}






