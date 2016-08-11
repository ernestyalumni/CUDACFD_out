/* Euler2d.cu
 * 2-dim. Euler eq. (heat eq.) by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160804
 */
#include "Euler2d.h"

__global__ void copyouttoin(float *dev_rho, float *dev_rho_out) {
	const int k_x = threadIdx.x + blockDim.x * blockDim.x ; 
	const int k_y = threadIdx.y + blockDim.y * blockDim.y ; 

	const int k = k_x + k_y * blockDim.x * gridDim.x; 
	
	__syncthreads(); 
	dev_rho[k] = dev_rho_out[k] ; 
	__syncthreads();
}

void kernelLauncher(uchar4 *d_out, float *dev_rho, float *dev_rho_out, 
									float2 *dev_u, 
									dim3 Ld, dim3 M_in) {
//	constexpr int radius {1 };

	const dim3 gridSize(blocksNeeded(Ld.x,M_in.x), blocksNeeded(Ld.y, M_in.y) );
	
//	const size_t smSz = (M_in.x + 2*radius)*(M_in.y+2*radius)*(sizeof(float) + sizeof(float2));
	
//	convectKernel<<<gridSize,M_in,smSz>>>(dev_rho, dev_rho_out, dev_u) ;	
	convect_fd_naive_sh<<<gridSize,M_in>>>(dev_rho, dev_rho_out, dev_u) ;	

	
	float_to_char<<<gridSize,M_in>>>(d_out,dev_rho_out);
	
	copyouttoin<<<gridSize,M_in>>>(dev_rho,dev_rho_out);
}

void kernelLauncherb(uchar4 *d_out, float *dev_rho,  
									float2 *dev_u, 
									dim3 Ld, dim3 M_in) {
	constexpr int radius {2 };

	const dim3 gridSize(blocksNeeded(Ld.x,M_in.x), blocksNeeded(Ld.y, M_in.y) );
	
	const size_t smSz = (M_in.x + 2*radius)*(M_in.y+2*radius)*(sizeof(float) + sizeof(float2));
	
	convectKernelb<<<gridSize,M_in,smSz>>>(dev_rho, dev_u) ;	

	
	float_to_char<<<gridSize,M_in>>>(d_out,dev_rho);
	
}

void kernelLauncherc(uchar4 *d_out, float *dev_rho,  
									float2 *dev_u, 
									dim3 Ld, dim3 M_in) {
	constexpr int radius {1 };

	const dim3 gridSize(blocksNeeded(Ld.x,M_in.x), blocksNeeded(Ld.y, M_in.y) );
	
	const size_t smSz = (M_in.x + 2*radius)*(M_in.y+2*radius)*(sizeof(float2));
	
	convectKernelc<<<gridSize,M_in,smSz>>>(dev_rho, dev_u) ;	

	
	float_to_char<<<gridSize,M_in>>>(d_out,dev_rho);
	
}

void kernelLauncher2(uchar4 *d_out, float *dev_rho,  
									float2 *dev_u, 
									dim3 Ld, dim3 M_in) {
//	constexpr int radius {2 };

	const dim3 gridSize(blocksNeeded(Ld.x,M_in.x), blocksNeeded(Ld.y, M_in.y) );
	
//	const size_t smSz = (M_in.x + 2*radius)*(M_in.y+2*radius)*(sizeof(float) + sizeof(float2));
	
//	convectKernel<<<gridSize,M_in,smSz>>>(dev_rho, dev_rho_out, dev_u) ;	
	convect_fd_naive_sh2<<<gridSize,M_in>>>(dev_rho, dev_u) ;	
//	convect_fd2<<<gridSize,M_in>>>(dev_rho, dev_u) ;	

	
	float_to_char<<<gridSize,M_in>>>(d_out,dev_rho);
	
}

void kernelLauncher2b(uchar4 *d_out, float *dev_rho, float *dev_rho_out, 
									float2 *dev_u, 
									dim3 Ld, dim3 M_in) {
//	constexpr int radius {2 };

	const dim3 gridSize(blocksNeeded(Ld.x,M_in.x), blocksNeeded(Ld.y, M_in.y) );
	
//	const size_t smSz = (M_in.x + 2*radius)*(M_in.y+2*radius)*(sizeof(float) + sizeof(float2));
	
//	convectKernel<<<gridSize,M_in,smSz>>>(dev_rho, dev_rho_out, dev_u) ;	
	convect_fd_naive_sh2b<<<gridSize,M_in>>>(dev_rho, dev_rho_out, dev_u) ;	

	
	float_to_char<<<gridSize,M_in>>>(d_out,dev_rho_out);
	
	copyouttoin<<<gridSize,M_in>>>(dev_rho,dev_rho_out);
}
