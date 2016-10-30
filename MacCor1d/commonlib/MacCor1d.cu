/* MacCor1d.cu
 * 1-dim. MacCormack method
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161026
 * */
#include "MacCor1d.h"

__constant__ float dev_sigmaconst[1] ; // sigma

// MacCormack method

// MacCormack method, using (only) global memory
//__global__ void MacCor_global_predict(float *f, float *foverline, const float sigmaconst ) {
__global__ void MacCor_global_predict(float *f, float *foverline, const int L_x ) {

	// global indices
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x;

	int right = k_x + 1; 	
		
	if ( k_x >= L_x ) {
		return; }
		
	// predictor
	float prediction;
	prediction = f[k_x] - dev_sigmaconst[0]*(f[right]-f[k_x]) ;
	foverline[k_x] = prediction;
	
}

// MacCormack method; corrector and updating steps
__global__ void MacCor_global_correctupdate( float *f_in, float *foverline, float *f_out, 
												const int L_x) {
	
	// global indices
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x ;
		
	int left = k_x - 1;
	left = max( left, 0 );

	if (k_x >= L_x) {
		return ; }
	
	// boundary condition at x = 0
	if (k_x == 0 ) {
		f_out[k_x] = 0; 
		return; 
	}
	
	float fdbloverline = 0.f; // \overline{\overline{f}}
	fdbloverline = f_in[k_x] - dev_sigmaconst[0] * (foverline[k_x] - foverline[left]) ;

	__syncthreads() ;


	f_out[k_x] = 0.5f * ( foverline[k_x] + fdbloverline ) ;

	__syncthreads();

}	

__global__ void swap( float *f_in, float *f_out, const int L_x ) {
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (k_x >= L_x) {
		return ; }
		
	float tempval = f_out[k_x] ; 
	f_in[k_x] = tempval; 
	
}

__global__ void BoundaryCondcheck_end(float *f, const int L_x) {
	const int k_x = threadIdx.x + blockDim.x * blockDim.x;
	
	if (k_x >= L_x) {
		return ; }
		
	if (k_x == L_x - 1) {
		float tempval = 2.f*f[k_x -1] - f[k_x-2] ;
		f[k_x] = tempval; }

}
