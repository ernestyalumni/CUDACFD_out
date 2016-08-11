/* Euler2d.cu
 * 2-dim. Euler eq. (heat eq.) by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160804
 */
#include "Euler2d.h"

__constant__ float dev_Deltat[1]; // Deltat

__constant__ float dev_gas_params[2] ; // dev_gas_params[0] = heat capacity ratio


int blocksNeeded( int N_i, int M_i) { return (N_i+M_i-1)/M_i; }

__device__ unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n);}

__device__ int idxClip( int idx, int idxMax) {
	return idx > (idxMax - 1) ? (idxMax - 1): (idx < 0 ? 0 : idx);
}

__device__ int flatten(int col, int row, int width, int height) {
	return idxClip(col, width) + idxClip(row,height)*width ;
}

__device__ float pressure( float energy, float rho, float2 u ) {
	float pressure_val {0.f};
	float usqr { u.x*u.x+u.y+u.y };
	pressure_val = (dev_gas_params[0] - 1.f)*(energy-0.5f*rho*usqr)
	return pressure_val;
}

__global__ void EulerKernel(float *dev_rho, 
						float2 *dev_u,
						float2 *dev_p,
						float *dev_E ) {
	constexpr int NUS = 1;
	constexpr int radius = NUS;
	
	extern __shared__ float sh_rho[];
	extern __shared__ float2 sh_u[];
	extern __shared__ float2 sh_p[];
	extern __shared__ float sh_E[];
	
	// global indices
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
	if ((k_x >= dev_Ld[0]) || (k_y >= dev_Ld[1]) ) return;
	const int k = flatten(k_x,k_y,dev_Ld[0],dev_Ld[1]);
	// local width and height
	const int2 S = { static_cast<int>(blockDim.x + 2 * radius),
						static_cast<int>(blockDim.y + 2 * radius) };
	
	// local indices
	const int s_x = threadIdx.x + radius;
	const int s_y = threadIdx.y + radius;
	
	const int s_k = flatten(s_x,s_y, S.x, S.y) ;

	// Load regular cells
	sh_rho[s_k] = dev_rho[k];
	sh_u[s_k] = dev_u[k];
	sh_p[s_k] = dev_p[k];
	sh_E[s_k] = dev_E[k];
	// Load halo cells
	if (threadIdx.x < radius) {
		sh_rho[flatten(s_x-radius,s_y,S.x,S.y)] = 
			dev_rho[flatten(k_x-radius,k_y,dev_Ld[0],dev_Ld[1])] ;
		sh_rho[flatten(s_x+blockDim.x,s_y,S.x,S.y)] = 
			dev_rho[flatten(k_x+blockDim.x,k_y,dev_Ld[0],dev_Ld[1])] ; 
		
	}
	
	
	__syncthreads();
	
	
	
}
	
	
