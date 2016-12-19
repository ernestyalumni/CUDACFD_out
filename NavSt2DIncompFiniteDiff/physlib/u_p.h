/* u_p.h
 * computation of u, p for 2-dim. incompressible Navier-Stokes equation with finite difference
 * \author Ernest Yeung  
 * \email ernestyalumni@gmail.com
 * \date 20161209
 * 
 * compilation tip: (compile separately)
 * nvcc -std=c++11 -c ./physlib/u_p.cu -o u_p.o
 * 
 */
#ifndef __U_P_H__
#define __U_P_H__

#include <thrust/device_vector.h>

#include "../commonlib/checkerror.h"  // checkCudaErrors


/*------------------------------------------------------------------- */
/* Computation of tentative velocity field (F,G) -------------------- */
/*------------------------------------------------------------------- */

__global__ void compute_F(const float deltat, 
	const float* u, const float* v, float* F,
	const int imax, const int jmax, const float deltax, const float deltay,
	const float gamma, const float Re) ; 

__global__ void compute_G(const float deltat, 
	const float* u, const float* v, float* G,
	const int imax, const int jmax, const float deltax, const float deltay,
	const float gamma, const float Re) ;


////////////////////////////////////////////////////////////////////////
/* __host__ void copy_press_int( thrust::device_vector<float> p_all, 
	thrust::device_vector<float> & p_int,
	const int imax, const int jmax) ; 
void copy_press_int( const float* p_all, float* p_int,
	const int imax, const int jmax);

__global__ void sum_pressure( cudaSurfaceObject_t pSurfObj, 
	const int imax, const int jmax, float* pres_sum);
*/
/*------------------------------------------------------------------- */
/* Computation of the right hand side of the pressure equation ------ */
/*------------------------------------------------------------------- */


__global__ void compute_RHS( const float* F, const float* G, 
	float* RHS, 
	const int imax, const int jmax, 
	const float deltat, const float deltax, const float deltay);

/*------------------------------------------------------------------- */
/* SOR iteration for the Poisson equation for the pressure
/*------------------------------------------------------------------- */

__global__ void poisson( const float* p, const float* RHS, 
	float* p_temp, 
	const int imax, const int jmax,
	const float deltax, const float deltay, 
	const float omega) ;



__global__ void poisson_redblack( float* p, const float* RHS, 
	const int imax, const int jmax,
	const float deltax, const float deltay, 
	const float omega) ;

/*------------------------------------------------------------------- */
/* computation of residual */
/*------------------------------------------------------------------- */

__global__ void compute_residual( const float* p, const float* RHS, 
	const int imax, const int jmax,
	const float deltax, const float deltay, 
	float* residualsq_Array) ;

/* cannot do this:
	thrust::device_vector<float> & residualsq);
	* cf. http://stackoverflow.com/questions/5510715/thrust-inside-user-written-kernels
	* */
	
	
/*------------------------------------------------------------------- */
/* computation of new velocity values */
/*------------------------------------------------------------------- */

__global__ void calculate_u( float* u, const float* p, const float* F, 
	const int imax, const int jmax, const float deltat, const float deltax ) ;


__global__ void calculate_v( float* v, const float* p, const float* G, 
	const int imax, const int jmax, const float deltat, const float deltay ) ;




#endif // __U_P_H__

