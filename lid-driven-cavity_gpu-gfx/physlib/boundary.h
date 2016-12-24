/** boundary.h
 * \file boundary.h
 * fix boundary conditions at the "boundary strip" cf. Griebel, Dornseifer, & Neunhoeffer (1997)
 * \author Ernest Yeung  
 * \email ernestyalumni@gmail.com
 * \date 20161207
 * 
 * compilation tip: (compile separately)
 * nvcc -std=c++11 -c ./physlib/boundary.cu -o boundary.o
 * 
 */
#ifndef __BOUNDARY_H__
#define __BOUNDARY_H__

#include "../physlib/uvp.h"       // flatten




/* --------------------------------------------------------------- */
/* Setting the boundary conditions at the boundary strip.      	   */
// on the host CPU memory
/* --------------------------------------------------------------- */

void set_BCs_host (float* u, float* v, 
	const int imax, const int jmax) ;

/* --------------------------------------------------------------- */
/* Setting the boundary conditions at the boundary strip.      	   */
// on the device GPU memory
/* --------------------------------------------------------------- */

__global__
void set_BCs (float* u, float* v,
	const int imax, const int jmax) ;

////////////////////////////////////////////////////////////////////////
__global__ 
void set_horz_pres_BCs (float* pres_red, float* pres_black,
	const int imax, const int jmax) ;
	
////////////////////////////////////////////////////////////////////////
__global__
void set_vert_pres_BCs (float* pres_red, float* pres_black,
	const int imax, const int jmax) ;


#endif // __BOUNDARY_H__

