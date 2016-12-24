/** boundary.cu
 * \file boundary.cu
 * fix boundary conditions at the "boundary strip" cf. Griebel, Dornseifer, & Neunhoeffer (1997)
 * \author Ernest Yeung  
 * \email ernestyalumni@gmail.com
 * \date 20161207
 * 
 * compilation tip: (compile separately)
 * nvcc -std=c++11 -c ./physlib/boundary.cu -o boundary.o
 * 
 */
#include "boundary.h"

/* --------------------------------------------------------------- */
/* Setting the boundary conditions at the boundary strip.      	   */
// on the host CPU memory
/* --------------------------------------------------------------- */
///////////////////////////////////////////////////////////////////////////////

void set_BCs_host (float* u, float* v, 
	const int imax, const int jmax) 
{
	int ind;

	// loop through rows and columns
	for (ind = 0; ind < jmax + 2; ++ind) {

		// left boundary
		u[flatten(0, ind,jmax+2)] = 0.0;
		v[flatten(0, ind,jmax+2)] = -v[flatten(1, ind,jmax+2)];

		// right boundary
		u[flatten(imax, ind,jmax+2)] = 0.0;
		v[flatten(imax + 1, ind, jmax+2)] = -v[flatten(imax, ind, jmax+2)];

		// bottom boundary
		u[flatten(ind, 0, jmax+2)] = -u[flatten(ind, 1,jmax+2)];
		v[flatten(ind, 0, jmax+2)] = 0.0;

		// top boundary
		u[flatten(ind, jmax + 1, jmax+2)] = 2.0 - u[flatten(ind, jmax,jmax+2)];
		v[flatten(ind, jmax, jmax+2)] = 0.0;

		if (ind == jmax) {
			// left boundary
			u[flatten(0, 0, jmax+2)] = 0.0;
			v[flatten(0, 0, jmax+2)] = -v[flatten(1, 0, jmax + 2)];
			u[flatten(0, jmax + 1, jmax+2)] = 0.0;
			v[flatten(0, jmax + 1, jmax+2)] = -v[flatten(1, jmax + 1, jmax+2)];

			// right boundary
			u[flatten(imax, 0, jmax)] = 0.0;
			v[flatten(imax + 1, 0, jmax)] = -v[flatten(imax, 0, jmax + 2)];
			u[flatten(imax, jmax + 1, jmax+2)] = 0.0;
			v[flatten(imax + 1, jmax + 1, jmax+2)] = -v[flatten(imax, jmax + 1, jmax + 2)];

			// bottom boundary
			u[flatten(0, 0, jmax + 2)] = -u[flatten(0, 1, jmax + 2)];
			v[flatten(0, 0, jmax +2)] = 0.0;
			u[flatten(imax + 1, 0, jmax + 2)] = -u[flatten(imax + 1, 1, jmax + 2) ];
			v[flatten(imax + 1, 0, jmax + 2) ] = 0.0;

			// top boundary
			u[flatten(0, jmax + 1, jmax + 2) ] = 2.0 - u[flatten(0, jmax, jmax + 2)];
			v[flatten(0, jmax, jmax + 2) ] = 0.0;
			u[flatten(imax + 1, jmax + 1, jmax + 2) ] = 2.0 - u[flatten(imax + 1, jmax, jmax + 2) ];
			v[ flatten(ind, jmax + 1, jmax + 2)] = 0.0;
		} // end if

	} // end for

} // end set_BCs_host


/* --------------------------------------------------------------- */
/* Setting the boundary conditions at the boundary strip.      	   */
// on the device GPU memory
/* --------------------------------------------------------------- */
///////////////////////////////////////////////////////////////////////////////
__global__
void set_BCs (float* u, float* v,
	const int imax, const int jmax) 
{
	int ind = (blockIdx.x * blockDim.x) + threadIdx.x + 1;

	// left boundary
	u[flatten(0, ind,jmax+2)] = 0.0;
	v[flatten(0, ind,jmax+2)] = -v[flatten(1, ind,jmax+2)];

	// right boundary
	u[flatten(imax, ind,jmax+2)] = 0.0;
	v[flatten(imax + 1, ind, jmax+2)] = -v[flatten(imax, ind, jmax+2)];

	// bottom boundary
	u[flatten(ind, 0, jmax+2)] = -u[flatten(ind, 1,jmax+2)];
	v[flatten(ind, 0, jmax+2)] = 0.0;

	// top boundary
	u[flatten(ind, jmax + 1, jmax+2)] = 2.0 - u[flatten(ind, jmax,jmax+2)];
	v[flatten(ind, jmax, jmax+2)] = 0.0;

	if (ind == jmax) {
		// left boundary
		u[flatten(0, 0, jmax+2)] = 0.0;
		v[flatten(0, 0, jmax+2)] = -v[flatten(1, 0, jmax + 2)];
		u[flatten(0, jmax + 1, jmax+2)] = 0.0;
		v[flatten(0, jmax + 1, jmax+2)] = -v[flatten(1, jmax + 1, jmax+2)];

		// right boundary
		u[flatten(imax, 0, jmax)] = 0.0;
		v[flatten(imax + 1, 0, jmax)] = -v[flatten(imax, 0, jmax + 2)];
		u[flatten(imax, jmax + 1, jmax+2)] = 0.0;
		v[flatten(imax + 1, jmax + 1, jmax+2)] = -v[flatten(imax, jmax + 1, jmax + 2)];

		// bottom boundary
		u[flatten(0, 0, jmax + 2)] = -u[flatten(0, 1, jmax + 2)];
		v[flatten(0, 0, jmax +2)] = 0.0;
		u[flatten(imax + 1, 0, jmax + 2)] = -u[flatten(imax + 1, 1, jmax + 2) ];
		v[flatten(imax + 1, 0, jmax + 2) ] = 0.0;

		// top boundary
		u[flatten(0, jmax + 1, jmax + 2) ] = 2.0 - u[flatten(0, jmax, jmax + 2)];
		v[flatten(0, jmax, jmax + 2) ] = 0.0;
		u[flatten(imax + 1, jmax + 1, jmax + 2) ] = 2.0 - u[flatten(imax + 1, jmax, jmax + 2) ];
		v[ flatten(ind, jmax + 1, jmax + 2)] = 0.0;
	} // end if

} // end set_BCs

///////////////////////////////////////////////////////////////////////////////
__global__ 
void set_horz_pres_BCs (float* pres_red, float* pres_black,
	const int imax, const int jmax) 
{
	int col = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
	col = (col * 2) - 1;

	int NUM_2 = jmax >> 1;

	// p_i,0 = p_i,1
	pres_black[flatten(col, 0, NUM_2 + 2)] = pres_red[flatten(col, 1, NUM_2 + 2)];
	pres_red[flatten(col + 1, 0, NUM_2 + 2)] = pres_black[ flatten(col + 1, 1, NUM_2 + 2)];

	// p_i,jmax+1 = p_i,jmax
	pres_red[flatten(col, NUM_2 + 1, NUM_2 + 2)] = pres_black[flatten(col, NUM_2, NUM_2 + 2)];
	pres_black[flatten(col + 1, NUM_2 + 1, NUM_2 + 2) ] = pres_red[flatten(col + 1, NUM_2, NUM_2 + 2)];

} // end set_horz_pres_BCs

//////////////////////////////////////////////////////////////////////////////

__global__
void set_vert_pres_BCs (float* pres_red, float* pres_black,
	const int imax, const int jmax) 
{
	int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;

	int NUM_2 = jmax >> 1;

	// p_0,j = p_1,j
	pres_black[flatten(0, row, NUM_2 + 2)] = pres_red[flatten(1, row, NUM_2 + 2)];
	pres_red[flatten(0, row, NUM_2 + 2)] = pres_black[flatten(1, row, NUM_2 + 2)];

	// p_imax+1,j = p_imax,j
	pres_black[flatten(imax + 1, row, NUM_2 + 2)] = pres_red[flatten(imax, row, NUM_2 + 2)];
	pres_red[flatten(imax + 1, row, NUM_2 + 2)] = pres_black[flatten(imax, row, NUM_2 + 2)];

} // end set_pressure_BCs

///////////////////////////////////////////////////////////////////////////////
