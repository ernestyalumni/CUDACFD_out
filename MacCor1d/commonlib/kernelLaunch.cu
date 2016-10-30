/* kernelLaunch.cu
 * 1-dim. MacCormack method with global memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161028
 */
#include "kernelLaunch.h"

void kernelLauncher(float *f, float *foverline, float *f_out, dim3 Ld, dim3 M_in) {
	const dim3 gridSize( blocksNeeded( Ld.x, M_in.x) );
	
	MacCor_global_predict<<<gridSize,M_in>>>(f, foverline, Ld.x ) ;

	MacCor_global_correctupdate<<<gridSize,M_in>>>(f,foverline,f_out,Ld.x);

	swap<<<gridSize,M_in>>>( f, f_out, Ld.x ) ;

	BoundaryCondcheck_end<<<gridSize,M_in>>>(f, Ld.x) ;

}

int blocksNeeded( int N_i, int M_i) { return (N_i + M_i - 1)/M_i ; }
