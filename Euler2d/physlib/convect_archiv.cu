/* convect.cu
 * 2-dim. Euler eq. (heat eq.) by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160804
 */
#include "convect.h"

__global__ void convectKernel(float *dev_rho, float *dev_rho_out, float2 *dev_u) {
	constexpr int NUS = 1;
	constexpr int radius = NUS;
	
	extern __shared__ float sh_rho[] ; // shared rho, shared mass density
	extern __shared__ float2 sh_u[]  ; // shared u, shared velocity vector field
	
	// global indices
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x ;
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y ; 
	if ((k_x >= dev_Ld[0]) || (k_y >= dev_Ld[1]) ) return;
	const int k = flatten(k_x,k_y,dev_Ld[0],dev_Ld[1]) ;
	// local width and height
	const int2 S = { static_cast<int>(blockDim.x + 2 * radius), 
						static_cast<int>(blockDim.y + 2 * radius) };
	
	// local indices
	const int s_x = threadIdx.x + radius;
	const int s_y = threadIdx.y + radius;
	const int s_k = flatten(s_x,s_y,S.x,S.y);
	// assign default color values for d_out (black)
	
	// Load regular cells
	sh_rho[s_k] = dev_rho[k] ; 
	sh_u[s_k]   = dev_u[k]; 
	// Load halo cels
	if (threadIdx.x < radius) {
		sh_rho[flatten(s_x-radius,s_y,S.x,S.y)] = 
			dev_rho[flatten(k_x-radius,k_y,dev_Ld[0], dev_Ld[1])];
		sh_u[flatten(s_x-radius,s_y,S.x,S.y)] = 
			dev_u[flatten(k_x-radius,k_y,dev_Ld[0], dev_Ld[1])];
		sh_rho[flatten(s_x+blockDim.x,s_y,S.x,S.y)] = 
			dev_rho[flatten(k_x+blockDim.x,k_y,dev_Ld[0],dev_Ld[1])];
		sh_u[flatten(s_x+blockDim.x,s_y,S.x,S.y)] = 
			dev_u[flatten(k_x+blockDim.x,k_y,dev_Ld[0],dev_Ld[1])];
	}

	if (threadIdx.y < radius) {
		sh_rho[flatten(s_x,s_y-radius,S.x,S.y)] = 
			dev_rho[flatten(k_x,k_y-radius,dev_Ld[0], dev_Ld[1])];
		sh_u[flatten(s_x-radius,s_y,S.x,S.y)] = 
			dev_u[flatten(k_x-radius,k_y,dev_Ld[0], dev_Ld[1])];
		sh_rho[flatten(s_x,s_y+blockDim.y,S.x,S.y)] = 
			dev_rho[flatten(k_x,k_y+blockDim.y,dev_Ld[0],dev_Ld[1])];
		sh_u[flatten(s_x,s_y+blockDim.y,S.x,S.y)] = 
			dev_u[flatten(k_x,k_y+blockDim.y,dev_Ld[0],dev_Ld[1])];
	}
	
	float2 stencil[NUS][2];
	
	for (int nu = 0 ; nu < NUS; ++nu) {
		stencil[nu][0].x = sh_rho[flatten(s_x-(nu+1),s_y,S.x,S.y)] * 
								sh_u[flatten(s_x-(nu+1),s_y,S.x,S.y)].x ;
		stencil[nu][1].x = sh_rho[flatten(s_x+(nu+1),s_y,S.x,S.y)] * 
								sh_u[flatten(s_x+(nu+1),s_y,S.x,S.y)].x ;
		stencil[nu][0].y = sh_rho[flatten(s_x,s_y-(nu+1),S.x,S.y)] * 
								sh_u[flatten(s_x,s_y-(nu+1),S.x,S.y)].x ;
		stencil[nu][1].y = sh_rho[flatten(s_x,s_y+(nu+1),S.x,S.y)] * 
								sh_u[flatten(s_x,s_y+(nu+1),S.x,S.y)].x ;
	}
	
	float divrhouval { dev_div1( stencil ) }; // value of the divergence of rho * u
	
	dev_rho_out[k] -= dev_Deltat[0]*divrhouval;
	
}

__global__ void convectKernelb(float *dev_rho, float2 *dev_u) {
	constexpr int NUS = 2;
	constexpr int radius = NUS;
	
	extern __shared__ float sh_rho[] ; // shared rho, shared mass density
	extern __shared__ float2 sh_u[]  ; // shared u, shared velocity vector field
	
	// global indices
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x ;
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y ; 
	if ((k_x >= dev_Ld[0]) || (k_y >= dev_Ld[1]) ) return;
	const int k = flatten(k_x,k_y,dev_Ld[0],dev_Ld[1]) ;
	// local width and height
	const int2 S = { static_cast<int>(blockDim.x + 2 * radius), 
						static_cast<int>(blockDim.y + 2 * radius) };
	
	// local indices
	const int s_x = threadIdx.x + radius;
	const int s_y = threadIdx.y + radius;
	const int s_k = flatten(s_x,s_y,S.x,S.y);
	// assign default color values for d_out (black)
	
	// Load regular cells
	sh_rho[s_k] = dev_rho[k] ; 
	sh_u[s_k]   = dev_u[k]; 
	// Load halo cels
	if (threadIdx.x < radius) {
		sh_rho[flatten(s_x-radius,s_y,S.x,S.y)] = 
			dev_rho[flatten(k_x-radius,k_y,dev_Ld[0], dev_Ld[1])];
		sh_u[flatten(s_x-radius,s_y,S.x,S.y)] = 
			dev_u[flatten(k_x-radius,k_y,dev_Ld[0], dev_Ld[1])];
		sh_rho[flatten(s_x+blockDim.x,s_y,S.x,S.y)] = 
			dev_rho[flatten(k_x+blockDim.x,k_y,dev_Ld[0],dev_Ld[1])];
		sh_u[flatten(s_x+blockDim.x,s_y,S.x,S.y)] = 
			dev_u[flatten(k_x+blockDim.x,k_y,dev_Ld[0],dev_Ld[1])];
	}

	if (threadIdx.y < radius) {
		sh_rho[flatten(s_x,s_y-radius,S.x,S.y)] = 
			dev_rho[flatten(k_x,k_y-radius,dev_Ld[0], dev_Ld[1])];
		sh_u[flatten(s_x-radius,s_y,S.x,S.y)] = 
			dev_u[flatten(k_x-radius,k_y,dev_Ld[0], dev_Ld[1])];
		sh_rho[flatten(s_x,s_y+blockDim.y,S.x,S.y)] = 
			dev_rho[flatten(k_x,k_y+blockDim.y,dev_Ld[0],dev_Ld[1])];
		sh_u[flatten(s_x,s_y+blockDim.y,S.x,S.y)] = 
			dev_u[flatten(k_x,k_y+blockDim.y,dev_Ld[0],dev_Ld[1])];
	}
	
	__syncthreads();
	
	float2 stencil[NUS][2];
	
	for (int nu = 0 ; nu < NUS; ++nu) {
		stencil[nu][0].x = sh_rho[flatten(s_x-(nu+1),s_y,S.x,S.y)] * 
								sh_u[flatten(s_x-(nu+1),s_y,S.x,S.y)].x ;
		stencil[nu][1].x = sh_rho[flatten(s_x+(nu+1),s_y,S.x,S.y)] * 
								sh_u[flatten(s_x+(nu+1),s_y,S.x,S.y)].x ;
		stencil[nu][0].y = sh_rho[flatten(s_x,s_y-(nu+1),S.x,S.y)] * 
								sh_u[flatten(s_x,s_y-(nu+1),S.x,S.y)].x ;
		stencil[nu][1].y = sh_rho[flatten(s_x,s_y+(nu+1),S.x,S.y)] * 
								sh_u[flatten(s_x,s_y+(nu+1),S.x,S.y)].x ;
	}
	
	float divrhouval { dev_div2( stencil ) }; // value of the divergence of rho * u
	
	__syncthreads();
	
	dev_rho[k] += (-1.f)*dev_Deltat[0]*divrhouval;
	
}

__global__ void convectKernelc(float *dev_rho, float2 *dev_u) {
	constexpr int NUS = 1;
	constexpr int radius = NUS;
	
	extern __shared__ float2 sh_rhou[] ; // shared rho u, shared mass density times u
	
	// global indices
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x ;
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y ; 
	if ((k_x >= dev_Ld[0]) || (k_y >= dev_Ld[1]) ) return;
	const int k = flatten(k_x,k_y,dev_Ld[0],dev_Ld[1]) ;
	// local width and height
	const int2 S = { static_cast<int>(blockDim.x + 2 * radius), 
						static_cast<int>(blockDim.y + 2 * radius) };
	
	// local indices
	const int s_x = threadIdx.x + radius;
	const int s_y = threadIdx.y + radius;
	const int s_k = flatten(s_x,s_y,S.x,S.y);
	// assign default color values for d_out (black)
	
	// Load regular cells
	sh_rhou[s_k].x = dev_rho[k] * dev_u[k].x ; 
	sh_rhou[s_k].y = dev_rho[k] * dev_u[k].y; 
	// Load halo cels
	if (threadIdx.x < radius) {
		sh_rhou[flatten(s_x-radius,s_y,S.x,S.y)].x = 
			dev_rho[flatten(k_x-radius,k_y,dev_Ld[0], dev_Ld[1])] * 
				dev_u[flatten(k_x-radius,k_y,dev_Ld[0], dev_Ld[1])].x ; 
		sh_rhou[flatten(s_x-radius,s_y,S.x,S.y)].y = 
			dev_rho[flatten(k_x-radius,k_y,dev_Ld[0], dev_Ld[1])] * 
				dev_u[flatten(k_x-radius,k_y,dev_Ld[0], dev_Ld[1])].y ;
		sh_rhou[flatten(s_x+blockDim.x,s_y,S.x,S.y)].x = 
			dev_rho[flatten(k_x+blockDim.x,k_y,dev_Ld[0],dev_Ld[1])] * 
				dev_u[flatten(k_x+blockDim.x,k_y,dev_Ld[0],dev_Ld[1])].x;
		sh_rhou[flatten(s_x+blockDim.x,s_y,S.x,S.y)].y = 
			dev_rho[flatten(k_x+blockDim.x,k_y,dev_Ld[0],dev_Ld[1])] * 
				dev_u[flatten(k_x+blockDim.x,k_y,dev_Ld[0],dev_Ld[1])].y;
	}

	if (threadIdx.y < radius) {
		sh_rhou[flatten(s_x,s_y-radius,S.x,S.y)].x = 
			dev_rho[flatten(k_x,k_y-radius,dev_Ld[0], dev_Ld[1])] * 
				dev_u[flatten(k_x-radius,k_y,dev_Ld[0], dev_Ld[1])].x;
		sh_rhou[flatten(s_x-radius,s_y,S.x,S.y)].y = 
			dev_rho[flatten(k_x,k_y-radius,dev_Ld[0], dev_Ld[1])] * 
				dev_u[flatten(k_x-radius,k_y,dev_Ld[0], dev_Ld[1])].y;
		sh_rhou[flatten(s_x,s_y+blockDim.y,S.x,S.y)].x = 
			dev_rho[flatten(k_x,k_y+blockDim.y,dev_Ld[0],dev_Ld[1])] * 
				dev_u[flatten(k_x,k_y+blockDim.y,dev_Ld[0],dev_Ld[1])].x;
		sh_rhou[flatten(s_x,s_y+blockDim.y,S.x,S.y)].y = 
			dev_rho[flatten(k_x,k_y+blockDim.y,dev_Ld[0],dev_Ld[1])] * 
				dev_u[flatten(k_x,k_y+blockDim.y,dev_Ld[0],dev_Ld[1])].y;
	}
	
	__syncthreads();
	
	float2 stencil[NUS][2];
	
	for (int nu = 0 ; nu < NUS; ++nu) {
/*
		stencil[nu][0].x = sh_rho[flatten(s_x-(nu+1),s_y,S.x,S.y)] * 
								sh_u[flatten(s_x-(nu+1),s_y,S.x,S.y)].x ;
		stencil[nu][1].x = sh_rho[flatten(s_x+(nu+1),s_y,S.x,S.y)] * 
								sh_u[flatten(s_x+(nu+1),s_y,S.x,S.y)].x ;
		stencil[nu][0].y = sh_rho[flatten(s_x,s_y-(nu+1),S.x,S.y)] * 
								sh_u[flatten(s_x,s_y-(nu+1),S.x,S.y)].x ;
		stencil[nu][1].y = sh_rho[flatten(s_x,s_y+(nu+1),S.x,S.y)] * 
								sh_u[flatten(s_x,s_y+(nu+1),S.x,S.y)].x ;
*/
		stencil[nu][0].x = sh_rhou[flatten(s_x-(nu+1),s_y,S.x,S.y)].x  ;
		stencil[nu][1].x = sh_rhou[flatten(s_x+(nu+1),s_y,S.x,S.y)].x ;
		stencil[nu][0].y = sh_rhou[flatten(s_x,s_y-(nu+1),S.x,S.y)].y  ;
		stencil[nu][1].y = sh_rhou[flatten(s_x,s_y+(nu+1),S.x,S.y)].y ;

	}
	
	float divrhouval { dev_div1( stencil ) }; // value of the divergence of rho * u
	
	__syncthreads();
	
	dev_rho[k] -= dev_Deltat[0]*divrhouval;
	
}


__global__ void convect_fd_naive_sh( float* dev_rho, float* dev_rho_out, float2* dev_u ) {
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
	
	dev_rho_out[k] +=  dev_Deltat[0] * (-1.f) * div_value ;		
			
//	__syncthreads();		
			
}



__global__ void convect_fd_naive_sh2( float* dev_rho, float2* dev_u ) {

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
			
	__syncthreads();		
			
}


__global__ void convect_fd_naive_sh2b( float* dev_rho, float* dev_rho_out, float2* dev_u ) {

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
	
	dev_rho_out[k] +=  dev_Deltat[0] * (-1.f) * div_value ;		
			
	__syncthreads();		
			
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
	
	float2 stencil[NU][2] ; 
	
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







