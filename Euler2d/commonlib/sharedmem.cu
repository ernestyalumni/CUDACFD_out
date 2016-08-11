/* sharedmem.cu
 * shared shared memory routines
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160804
 */
#include "sharedmem.h"

int blocksNeeded( int N_i, int M_i) { return (N_i+M_i-1)/M_i; }

__device__ unsigned char clip(const float val) { 
	// Set limits, range
	const float scale = 0.999f ;
	const float newval = val/scale;
	
	int n = 256 * newval ;
	if (n==256) { n=256; }
	
	return n > 255 ? 255 : (n < 0 ? 0 : n);
}

__device__ int idxClip( int idx, int idxMax) {
	return idx > (idxMax - 1) ? (idxMax - 1): (idx < 0 ? 0 : idx);
}

__device__ int flatten(int col, int row, int width, int height) {
	return idxClip(col, width) + idxClip(row,height)*width ;
}
/*
__device__ float pressure( float energy, float rho, float2 u ) {
	float pressure_val {0.f};
	float usqr { u.x*u.x+u.y+u.y };
	pressure_val = (dev_gas_params[0] - 1.f)*(energy-0.5f*rho*usqr)
	return pressure_val;
}
* */

__global__ void float_to_char( uchar4* dev_out, const float* outSrc) {
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x;
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y;
	
	const int k   = k_x + k_y * blockDim.x*gridDim.x ; 

	dev_out[k].x = 0;
	dev_out[k].z = 0;
	dev_out[k].y = 0;
	dev_out[k].w = 255;

	float value = outSrc[k] ;

	const unsigned char intensity = clip( value ) ;
	dev_out[k].x = intensity ;       // higher temp -> more red
	dev_out[k].z = 255 - intensity ; // lower temp -> more blue
	
}
