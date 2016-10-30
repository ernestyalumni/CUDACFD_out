/* MacCor1d.h
 * 1-dim. MacCormack method
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161029
 * */
#ifndef __MACCOR1D_H__
#define __MACCOR1D_H__

extern __constant__ float dev_sigmaconst[1] ; // sigma

__global__ void MacCor_global_predict(float *f, float *foverline,
										const int L_x); 

__global__ void MacCor_global_correctupdate( float *f_in, float *foverline, float *f_out,
												const int L_x);

__global__ void swap( float *f_in, float *f_out, const int L_x ) ;

__global__ void BoundaryCondcheck_end(float *f, const int L_x) ;

#endif // __MACCOR1D_H__ 
