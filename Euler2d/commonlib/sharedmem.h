/* sharedmem.h
 * shared shared memory routines
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160804
 */
#ifndef __SHAREDMEM_H__
#define __SHAREDMEM_H__ 

int blocksNeeded( int N_i, int M_i); 

__device__ unsigned char clip(int n) ;  

__device__ int idxClip( int idx, int idxMax) ;

__device__ int flatten(int col, int row, int width, int height) ;

//__device__ float pressure( float energy, float rho, float2 u ) ;

__global__ void float_to_char( uchar4* dev_out, const float* outSrc) ;

#endif // __SHAREDMEM_H__
