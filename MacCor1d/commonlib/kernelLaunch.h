/* kernelLaunch.h
 * 1-dim. MacCormack method with global memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161030
 */
#ifndef __KERNELLAUNCH_H__
#define __KERNELLAUNCH_H__

#include "./MacCor1d.h" 

void kernelLauncher(float *, float *, float *, dim3, dim3 ) ;

int blocksNeeded( int N_i, int M_i) ;

#endif // __KERNELLAUNCH_H__
