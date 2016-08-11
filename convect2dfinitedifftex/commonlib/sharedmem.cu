/* sharedmem.cu
 * shared memory to a 1-dim. grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160710
 */
#include "sharedmem.h"

#include "errors.h" // HANDLE_ERROR()


namespace sharedmem {

__constant__ int RAD[1] ; 
	
__device__ int idxClip( int idx, int idxMax) {
			return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}
		
__device__ int flatten(int k_x, int k_y, int k_z, int L_x, int L_y, int L_z) {
	return idxClip( k_x, L_x) + idxClip( k_y,L_y)*L_x + idxClip(k_z, L_z)*L_x*L_y ;
}	
	
__device__ void sh_dev_div1( float* dev_rho, float3* dev_u, float &result ) {
	extern __shared__ float sh_rho[ ];
	extern __shared__ float3 sh_u[ ];

	// global indices
	const int k_x = threadIdx.x + blockIdx.x * blockDim.x; 
	const int k_y = threadIdx.y + blockIdx.y * blockDim.y; 
	const int k_z = threadIdx.z + blockIdx.z * blockDim.z; 
	
	// EY : 20160718 check this; I want nothing to be done, but I return 0.f	
	if (k_x >= dev_Ld[0] || k_y >= dev_Ld[1] || k_z >= dev_Ld[2] ) { return ;  }

	const int k = flatten( k_x,k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2]);

	constexpr int NUS = 1;

	// local S
	const int3 S { static_cast<int>(blockDim.x + 2*RAD[0]) , 
					static_cast<int>(blockDim.y + 2*RAD[0]) , 
					static_cast<int>(blockDim.z + 2*RAD[0]) } ;

	// local s_i
	const int s_x = threadIdx.x + RAD[0];
	const int s_y = threadIdx.y + RAD[0];
	const int s_z = threadIdx.z + RAD[0];

	const int s_k = flatten( s_x,s_y,s_z, S.x, S.y, S.z );

	// Load regular cells
	sh_rho[s_k] = dev_rho[k];
	sh_u[s_k]   = dev_u[k];
	
	// Load halo cells
	if (threadIdx.x < RAD[0]) {
		sh_rho[ flatten( s_x -RAD[0],s_y,s_z,S.x,S.y,S.z)] = 
			dev_rho[ flatten(k_x-RAD[0],k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		sh_u[ flatten( s_x -RAD[0],s_y,s_z,S.x,S.y,S.z)] = 
			dev_u[ flatten(k_x-RAD[0],k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		
		sh_rho[flatten(s_x+blockDim.x,s_y,s_z,S.x,S.y,S.z)] = 
			dev_rho[flatten(k_x+blockDim.x,k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])];
		sh_u[flatten(s_x+blockDim.x,s_y,s_z,S.x,S.y,S.z)]   = 
			dev_u[flatten(k_x+blockDim.x,k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])]  ;
	}
	
	if (threadIdx.y < RAD[0]) {
		sh_rho[ flatten( s_x,s_y-RAD[0],s_z,S.x,S.y,S.z)] = 
			dev_rho[ flatten(k_x,k_y-RAD[0],k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		sh_u[ flatten( s_x ,s_y-RAD[0],s_z,S.x,S.y,S.z)] = 
			dev_u[ flatten(k_x,k_y-RAD[0],k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		
		sh_rho[flatten(s_x,s_y+blockDim.y,s_z,S.x,S.y,S.z)] 
				= dev_rho[flatten(k_x,k_y+blockDim.y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])];
		sh_u[flatten(s_x,s_y+blockDim.y,s_z,S.x,S.y,S.z)]   = 
				dev_u[flatten(k_x,k_y+blockDim.y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])]  ;
	}

	if (threadIdx.z < RAD[0]) {
		sh_rho[ flatten( s_x ,s_y,s_z-RAD[0],S.x,S.y,S.z)] = 
			dev_rho[ flatten(k_x,k_y,k_z-RAD[0],dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		sh_u[ flatten( s_x,s_y,s_z-RAD[0],S.x,S.y,S.z)] = 
			dev_u[ flatten(k_x,k_y,k_z-RAD[0],dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		
		sh_rho[flatten(s_x,s_y,s_z+blockDim.z,S.x,S.y,S.z)] = 
			dev_rho[flatten(k_x,k_y,k_z+blockDim.z,dev_Ld[0],dev_Ld[1],dev_Ld[2])];
		sh_u[flatten(s_x,s_y,s_z+blockDim.z,S.x,S.y,S.z)]   = 
			dev_u[flatten(k_x,k_y,k_z+blockDim.z,dev_Ld[0],dev_Ld[1],dev_Ld[2])]  ;
	}

	__syncthreads();

//	__shared__ float3 stencil[NUS][2];

	float3 stencil[NUS][2];
		
	
	for (int nu = 0; nu < NUS; ++nu ) {
		stencil[nu][0].x = sh_rho[flatten(s_x-(nu+1),s_y,s_z,S.x,S.y,S.z)]*
							sh_u[flatten(s_x-(nu+1),s_y,s_z,S.x,S.y,S.z)].x;
		stencil[nu][1].x = sh_rho[flatten(s_x+(nu+1),s_y,s_z,S.x,S.y,S.z)]*
							sh_u[flatten(s_x+(nu+1),s_y,s_z,S.x,S.y,S.z)].x;
		stencil[nu][0].y = sh_rho[flatten(s_x,s_y-(nu+1),s_z,S.x,S.y,S.z)]*
							sh_u[flatten(s_x,s_y-(nu+1),s_z,S.x,S.y,S.z)].y;
		stencil[nu][1].y = sh_rho[flatten(s_x,s_y+(nu+1),s_z,S.x,S.y,S.z)]*
							sh_u[flatten(s_x,s_y+(nu+1),s_z,S.x,S.y,S.z)].y;
		stencil[nu][0].z = sh_rho[flatten(s_x,s_y,s_z-(nu+1),S.x,S.y,S.z)]*
							sh_u[flatten(s_x,s_y,s_z-(nu+1),S.x,S.y,S.z)].z;
		stencil[nu][1].z = sh_rho[flatten(s_x,s_y,s_z+(nu+1),S.x,S.y,S.z)]*
							sh_u[flatten(s_x,s_y,s_z+(nu+1),S.x,S.y,S.z)].z;
	}
	float div_value { dev_div1( stencil ) };
	
//	return div_value;
	result = div_value;
}

	
__device__ float sh_dev_div2( float* dev_rho, float3* dev_u ) {
	extern __shared__ float sh_rho[ ];
	extern __shared__ float3 sh_u[ ];

	// global indices
	const int k_x = threadIdx.x + blockIdx.x * blockDim.x; 
	const int k_y = threadIdx.y + blockIdx.y * blockDim.y; 
	const int k_z = threadIdx.z + blockIdx.z * blockDim.z; 
	
	// EY : 20160718 check this; I want nothing to be done, but I return 0.f	
	if (k_x >= dev_Ld[0] || k_y >= dev_Ld[1] || k_z >= dev_Ld[2] ) { return 0.f;  }

	const int k = flatten( k_x,k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2]);


	constexpr int NUS = 2;


	// local S
	const int3 S { static_cast<int>(blockDim.x + 2*RAD[0]) , 
					static_cast<int>(blockDim.y + 2*RAD[0]) , 
					static_cast<int>(blockDim.z + 2*RAD[0]) } ;

	// local s_i
	const int s_x = threadIdx.x + RAD[0];
	const int s_y = threadIdx.y + RAD[0];
	const int s_z = threadIdx.z + RAD[0];

	const int s_k = flatten( s_x,s_y,s_z, S.x, S.y, S.z );

	// Load regular cells
	sh_rho[s_k] = dev_rho[k];
	sh_u[s_k]   = dev_u[k];
	
	// Load halo cells
	if (threadIdx.x < RAD[0]) {
		sh_rho[ flatten( s_x -RAD[0],s_y,s_z,S.x,S.y,S.z)] = 
			dev_rho[ flatten(k_x-RAD[0],k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		sh_u[ flatten( s_x -RAD[0],s_y,s_z,S.x,S.y,S.z)] = 
			dev_u[ flatten(k_x-RAD[0],k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		
		sh_rho[flatten(s_x+blockDim.x,s_y,s_z,S.x,S.y,S.z)] = 
			dev_rho[flatten(k_x+blockDim.x,k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])];
		sh_u[flatten(s_x+blockDim.x,s_y,s_z,S.x,S.y,S.z)]   = 
			dev_u[flatten(k_x+blockDim.x,k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])]  ;
	}
	
	if (threadIdx.y < RAD[0]) {
		sh_rho[ flatten( s_x,s_y-RAD[0],s_z,S.x,S.y,S.z)] = 
			dev_rho[ flatten(k_x,k_y-RAD[0],k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		sh_u[ flatten( s_x ,s_y-RAD[0],s_z,S.x,S.y,S.z)] = 
			dev_u[ flatten(k_x,k_y-RAD[0],k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		
		sh_rho[flatten(s_x,s_y+blockDim.y,s_z,S.x,S.y,S.z)] 
				= dev_rho[flatten(k_x,k_y+blockDim.y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])];
		sh_u[flatten(s_x,s_y+blockDim.y,s_z,S.x,S.y,S.z)]   = 
				dev_u[flatten(k_x,k_y+blockDim.y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])]  ;
	}

	if (threadIdx.z < RAD[0]) {
		sh_rho[ flatten( s_x ,s_y,s_z-RAD[0],S.x,S.y,S.z)] = 
			dev_rho[ flatten(k_x,k_y,k_z-RAD[0],dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		sh_u[ flatten( s_x,s_y,s_z-RAD[0],S.x,S.y,S.z)] = 
			dev_u[ flatten(k_x,k_y,k_z-RAD[0],dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		
		sh_rho[flatten(s_x,s_y,s_z+blockDim.z,S.x,S.y,S.z)] = 
			dev_rho[flatten(k_x,k_y,k_z+blockDim.z,dev_Ld[0],dev_Ld[1],dev_Ld[2])];
		sh_u[flatten(s_x,s_y,s_z+blockDim.z,S.x,S.y,S.z)]   = 
			dev_u[flatten(k_x,k_y,k_z+blockDim.z,dev_Ld[0],dev_Ld[1],dev_Ld[2])]  ;
	}

	__syncthreads();

	//__shared__ float3 stencil[NUS][2];
	float3 stencil[NUS][2];
	
	
	for (int nu = 0; nu < NUS; ++nu ) {
		stencil[nu][0].x = sh_rho[flatten(s_x-(nu+1),s_y,s_z,S.x,S.y,S.z)]*
							sh_u[flatten(s_x-(nu+1),s_y,s_z,S.x,S.y,S.z)].x;
		stencil[nu][1].x = sh_rho[flatten(s_x+(nu+1),s_y,s_z,S.x,S.y,S.z)]*
							sh_u[flatten(s_x+(nu+1),s_y,s_z,S.x,S.y,S.z)].x;
		stencil[nu][0].y = sh_rho[flatten(s_x,s_y-(nu+1),s_z,S.x,S.y,S.z)]*
							sh_u[flatten(s_x,s_y-(nu+1),s_z,S.x,S.y,S.z)].y;
		stencil[nu][1].y = sh_rho[flatten(s_x,s_y+(nu+1),s_z,S.x,S.y,S.z)]*
							sh_u[flatten(s_x,s_y+(nu+1),s_z,S.x,S.y,S.z)].y;
		stencil[nu][0].z = sh_rho[flatten(s_x,s_y,s_z-(nu+1),S.x,S.y,S.z)]*
							sh_u[flatten(s_x,s_y,s_z-(nu+1),S.x,S.y,S.z)].z;
		stencil[nu][1].z = sh_rho[flatten(s_x,s_y,s_z+(nu+1),S.x,S.y,S.z)]*
							sh_u[flatten(s_x,s_y,s_z+(nu+1),S.x,S.y,S.z)].z;
	}
	
	
	float div_value { dev_div2( stencil ) };
	
		
	return div_value;
}	
	
__device__ float sh_dev_div3( float* dev_rho, float3* dev_u ) {
	extern __shared__ float sh_rho[ ];
	extern __shared__ float3 sh_u[ ];

	// global indices
	const int k_x = threadIdx.x + blockIdx.x * blockDim.x; 
	const int k_y = threadIdx.y + blockIdx.y * blockDim.y; 
	const int k_z = threadIdx.z + blockIdx.z * blockDim.z; 
	
	// EY : 20160718 check this; I want nothing to be done, but I return 0.f	
	if (k_x >= dev_Ld[0] || k_y >= dev_Ld[1] || k_z >= dev_Ld[2] ) { return 0.f;  }

	const int k = flatten( k_x,k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2]);


	constexpr int NUS = 3;


	// local S
	const int3 S { static_cast<int>(blockDim.x + 2*RAD[0]) , 
					static_cast<int>(blockDim.y + 2*RAD[0]) , 
					static_cast<int>(blockDim.z + 2*RAD[0]) } ;

	// local s_i
	const int s_x = threadIdx.x + RAD[0];
	const int s_y = threadIdx.y + RAD[0];
	const int s_z = threadIdx.z + RAD[0];

	const int s_k = flatten( s_x,s_y,s_z, S.x, S.y, S.z );

	// Load regular cells
	sh_rho[s_k] = dev_rho[k];
	sh_u[s_k]   = dev_u[k];
	
	// Load halo cells
	if (threadIdx.x < RAD[0]) {
		sh_rho[ flatten( s_x -RAD[0],s_y,s_z,S.x,S.y,S.z)] = 
			dev_rho[ flatten(k_x-RAD[0],k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		sh_u[ flatten( s_x -RAD[0],s_y,s_z,S.x,S.y,S.z)] = 
			dev_u[ flatten(k_x-RAD[0],k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		
		sh_rho[flatten(s_x+blockDim.x,s_y,s_z,S.x,S.y,S.z)] = 
			dev_rho[flatten(k_x+blockDim.x,k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])];
		sh_u[flatten(s_x+blockDim.x,s_y,s_z,S.x,S.y,S.z)]   = 
			dev_u[flatten(k_x+blockDim.x,k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])]  ;
	}
	
	if (threadIdx.y < RAD[0]) {
		sh_rho[ flatten( s_x,s_y-RAD[0],s_z,S.x,S.y,S.z)] = 
			dev_rho[ flatten(k_x,k_y-RAD[0],k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		sh_u[ flatten( s_x ,s_y-RAD[0],s_z,S.x,S.y,S.z)] = 
			dev_u[ flatten(k_x,k_y-RAD[0],k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		
		sh_rho[flatten(s_x,s_y+blockDim.y,s_z,S.x,S.y,S.z)] 
				= dev_rho[flatten(k_x,k_y+blockDim.y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])];
		sh_u[flatten(s_x,s_y+blockDim.y,s_z,S.x,S.y,S.z)]   = 
				dev_u[flatten(k_x,k_y+blockDim.y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])]  ;
	}

	if (threadIdx.z < RAD[0]) {
		sh_rho[ flatten( s_x ,s_y,s_z-RAD[0],S.x,S.y,S.z)] = 
			dev_rho[ flatten(k_x,k_y,k_z-RAD[0],dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		sh_u[ flatten( s_x,s_y,s_z-RAD[0],S.x,S.y,S.z)] = 
			dev_u[ flatten(k_x,k_y,k_z-RAD[0],dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		
		sh_rho[flatten(s_x,s_y,s_z+blockDim.z,S.x,S.y,S.z)] = 
			dev_rho[flatten(k_x,k_y,k_z+blockDim.z,dev_Ld[0],dev_Ld[1],dev_Ld[2])];
		sh_u[flatten(s_x,s_y,s_z+blockDim.z,S.x,S.y,S.z)]   = 
			dev_u[flatten(k_x,k_y,k_z+blockDim.z,dev_Ld[0],dev_Ld[1],dev_Ld[2])]  ;
	}

	__syncthreads();

//	__shared__ float3 stencil[NUS][2];
	float3 stencil[NUS][2];
	
	
	for (int nu = 0; nu < NUS; ++nu ) {
		stencil[nu][0].x = sh_rho[flatten(s_x-(nu+1),s_y,s_z,S.x,S.y,S.z)]*
							sh_u[flatten(s_x-(nu+1),s_y,s_z,S.x,S.y,S.z)].x;
		stencil[nu][1].x = sh_rho[flatten(s_x+(nu+1),s_y,s_z,S.x,S.y,S.z)]*
							sh_u[flatten(s_x+(nu+1),s_y,s_z,S.x,S.y,S.z)].x;
		stencil[nu][0].y = sh_rho[flatten(s_x,s_y-(nu+1),s_z,S.x,S.y,S.z)]*
							sh_u[flatten(s_x,s_y-(nu+1),s_z,S.x,S.y,S.z)].y;
		stencil[nu][1].y = sh_rho[flatten(s_x,s_y+(nu+1),s_z,S.x,S.y,S.z)]*
							sh_u[flatten(s_x,s_y+(nu+1),s_z,S.x,S.y,S.z)].y;
		stencil[nu][0].z = sh_rho[flatten(s_x,s_y,s_z-(nu+1),S.x,S.y,S.z)]*
							sh_u[flatten(s_x,s_y,s_z-(nu+1),S.x,S.y,S.z)].z;
		stencil[nu][1].z = sh_rho[flatten(s_x,s_y,s_z+(nu+1),S.x,S.y,S.z)]*
							sh_u[flatten(s_x,s_y,s_z+(nu+1),S.x,S.y,S.z)].z;
	}
	
	
	float div_value { dev_div3( stencil ) };
	
		
	return div_value;
}	

__device__ float sh_dev_div4( float* dev_rho, float3* dev_u ) {
	extern __shared__ float sh_rho[ ];
	extern __shared__ float3 sh_u[ ];

	// global indices
	const int k_x = threadIdx.x + blockIdx.x * blockDim.x; 
	const int k_y = threadIdx.y + blockIdx.y * blockDim.y; 
	const int k_z = threadIdx.z + blockIdx.z * blockDim.z; 
	
	// EY : 20160718 check this; I want nothing to be done, but I return 0.f	
	if (k_x >= dev_Ld[0] || k_y >= dev_Ld[1] || k_z >= dev_Ld[2] ) { return 0.f;  }

	const int k = flatten( k_x,k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2]);


	constexpr int NUS = 4;


	// local S
	const int3 S { static_cast<int>(blockDim.x + 2*RAD[0]) , 
					static_cast<int>(blockDim.y + 2*RAD[0]) , 
					static_cast<int>(blockDim.z + 2*RAD[0]) } ;

	// local s_i
	const int s_x = threadIdx.x + RAD[0];
	const int s_y = threadIdx.y + RAD[0];
	const int s_z = threadIdx.z + RAD[0];

	const int s_k = flatten( s_x,s_y,s_z, S.x, S.y, S.z );

	// Load regular cells
	sh_rho[s_k] = dev_rho[k];
	sh_u[s_k]   = dev_u[k];
	
	// Load halo cells
	if (threadIdx.x < RAD[0]) {
		sh_rho[ flatten( s_x -RAD[0],s_y,s_z,S.x,S.y,S.z)] = 
			dev_rho[ flatten(k_x-RAD[0],k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		sh_u[ flatten( s_x -RAD[0],s_y,s_z,S.x,S.y,S.z)] = 
			dev_u[ flatten(k_x-RAD[0],k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		
		sh_rho[flatten(s_x+blockDim.x,s_y,s_z,S.x,S.y,S.z)] = 
			dev_rho[flatten(k_x+blockDim.x,k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])];
		sh_u[flatten(s_x+blockDim.x,s_y,s_z,S.x,S.y,S.z)]   = 
			dev_u[flatten(k_x+blockDim.x,k_y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])]  ;
	}
	
	if (threadIdx.y < RAD[0]) {
		sh_rho[ flatten( s_x,s_y-RAD[0],s_z,S.x,S.y,S.z)] = 
			dev_rho[ flatten(k_x,k_y-RAD[0],k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		sh_u[ flatten( s_x ,s_y-RAD[0],s_z,S.x,S.y,S.z)] = 
			dev_u[ flatten(k_x,k_y-RAD[0],k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		
		sh_rho[flatten(s_x,s_y+blockDim.y,s_z,S.x,S.y,S.z)] 
				= dev_rho[flatten(k_x,k_y+blockDim.y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])];
		sh_u[flatten(s_x,s_y+blockDim.y,s_z,S.x,S.y,S.z)]   = 
				dev_u[flatten(k_x,k_y+blockDim.y,k_z,dev_Ld[0],dev_Ld[1],dev_Ld[2])]  ;
	}

	if (threadIdx.z < RAD[0]) {
		sh_rho[ flatten( s_x ,s_y,s_z-RAD[0],S.x,S.y,S.z)] = 
			dev_rho[ flatten(k_x,k_y,k_z-RAD[0],dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		sh_u[ flatten( s_x,s_y,s_z-RAD[0],S.x,S.y,S.z)] = 
			dev_u[ flatten(k_x,k_y,k_z-RAD[0],dev_Ld[0],dev_Ld[1],dev_Ld[2])] ;
		
		sh_rho[flatten(s_x,s_y,s_z+blockDim.z,S.x,S.y,S.z)] = 
			dev_rho[flatten(k_x,k_y,k_z+blockDim.z,dev_Ld[0],dev_Ld[1],dev_Ld[2])];
		sh_u[flatten(s_x,s_y,s_z+blockDim.z,S.x,S.y,S.z)]   = 
			dev_u[flatten(k_x,k_y,k_z+blockDim.z,dev_Ld[0],dev_Ld[1],dev_Ld[2])]  ;
	}

	__syncthreads();

//	__shared__ float3 stencil[NUS][2];
	float3 stencil[NUS][2];
		
	for (int nu = 0; nu < NUS; ++nu ) {
		stencil[nu][0].x = sh_rho[flatten(s_x-(nu+1),s_y,s_z,S.x,S.y,S.z)]*
							sh_u[flatten(s_x-(nu+1),s_y,s_z,S.x,S.y,S.z)].x;
		stencil[nu][1].x = sh_rho[flatten(s_x+(nu+1),s_y,s_z,S.x,S.y,S.z)]*
							sh_u[flatten(s_x+(nu+1),s_y,s_z,S.x,S.y,S.z)].x;
		stencil[nu][0].y = sh_rho[flatten(s_x,s_y-(nu+1),s_z,S.x,S.y,S.z)]*
							sh_u[flatten(s_x,s_y-(nu+1),s_z,S.x,S.y,S.z)].y;
		stencil[nu][1].y = sh_rho[flatten(s_x,s_y+(nu+1),s_z,S.x,S.y,S.z)]*
							sh_u[flatten(s_x,s_y+(nu+1),s_z,S.x,S.y,S.z)].y;
		stencil[nu][0].z = sh_rho[flatten(s_x,s_y,s_z-(nu+1),S.x,S.y,S.z)]*
							sh_u[flatten(s_x,s_y,s_z-(nu+1),S.x,S.y,S.z)].z;
		stencil[nu][1].z = sh_rho[flatten(s_x,s_y,s_z+(nu+1),S.x,S.y,S.z)]*
							sh_u[flatten(s_x,s_y,s_z+(nu+1),S.x,S.y,S.z)].z;
	}
	
	
	float div_value { dev_div4( stencil ) };
	
		
	return div_value;
}	
	
	

}
