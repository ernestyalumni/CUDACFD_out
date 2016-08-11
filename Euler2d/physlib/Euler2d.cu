/* Euler2d.cu
 * 2-dim. Euler eq. (heat eq.) by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160804
 */
#include "Euler2d.h"


void kernelLauncher(uchar4 *d_out, float *dev_rho,  
									float2 *dev_u, 
									dim3 Ld, dim3 M_in) {

	const dim3 gridSize(blocksNeeded(Ld.x,M_in.x), blocksNeeded(Ld.y, M_in.y) );
	
//	convect_fd<<<gridSize,M_in>>>(dev_rho, dev_u) ;	
	convect_fd_sh<<<gridSize,M_in>>>(dev_rho, dev_u) ;	

	float_to_char<<<gridSize,M_in>>>(d_out,dev_rho);
	
}


void kernelLauncher2(uchar4 *d_out, float *dev_rho,  
									float2 *dev_u, 
									dim3 Ld, dim3 M_in) {

	const dim3 gridSize(blocksNeeded(Ld.x,M_in.x), blocksNeeded(Ld.y, M_in.y) );
	
	convect_fd2<<<gridSize,M_in>>>(dev_rho, dev_u) ;	

	
	float_to_char<<<gridSize,M_in>>>(d_out,dev_rho);
	
}

void EulerLauncher(uchar4 *d_out, float *dev_rho, float2 *dev_p, 
									float2 *dev_u, 
									float *dev_E,
									dim3 Ld, dim3 M_in) {

	const dim3 gridSize(blocksNeeded(Ld.x,M_in.x), blocksNeeded(Ld.y, M_in.y) );
	
	Euler2dp1<<<gridSize,M_in>>>(dev_rho, dev_p, dev_u, dev_E) ;	

	
	float_to_char<<<gridSize,M_in>>>(d_out,dev_rho);
	
}
