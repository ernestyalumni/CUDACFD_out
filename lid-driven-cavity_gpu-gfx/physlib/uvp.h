/** uvp.h
 * \file uvp.h
 * computation of u, v, p for 2-dim. incompressible Navier-Stokes equation with finite difference
 * \author Ernest Yeung  
 * \email ernestyalumni@gmail.com
 * \date 20161209
 * 
 * compilation tip: (compile separately)
 * nvcc -std=c++11 -c ./physlib/u_p.cu -o u_p.o
 * 
 */
#ifndef __UVP_H__
#define __UVP_H__

#include "../commonlib/checkerror.h"  // checkCudaErrors

__host__ __device__ int flatten( int, int, int );

/*------------------------------------------------------------------- */
/* Computation of tentative velocity field (F,G) -------------------- */
/*------------------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////
__global__ void calculate_F ( const float* u, const float* v,
				  float* F,
				  const float mix_param, const float Re_num, float gx,
  				  const float dt, const float dx, const float dy,
				  const int imax, const int jmax, const int M_x) ;
				  
__global__ void calculate_G ( const float* u, const float* v,
				  float* G, 
				  const float mix_param, const float Re_num, float gy,
  				  const float dt, const float dx, const float dy,
				  const int imax, const int jmax, const int M_x) ;

__global__ void sum_pressure (const float* pres_red, const float* pres_black, 
					float* pres_sum, const int imax, const int jmax, const int M_x) ;
				  
/*------------------------------------------------------------------- */
/* SOR iteration for the Poisson equation for the pressure
/*------------------------------------------------------------------- */
///////////////////////////////////////////////////////////////////////////////
__global__ void red_kernel ( const float* F, 
				 const float* G, const float* pres_black,
				 float* pres_red, 
				const float omega, 
				  const float dt, const float dx, const float dy,
				  const int imax, const int jmax, const int M_x		 ); 				  
				  
__global__ void black_kernel ( const float* F, 
				   const float* G, const float* pres_red, 
				   float* pres_black,
				const float omega, 
				  const float dt, const float dx, const float dy,
				  const int imax, const int jmax, const int M_x) ;
			  

/*------------------------------------------------------------------- */
/* computation of residual */
/*------------------------------------------------------------------- */
__global__ void calc_residual ( const float* F, const float* G, 
					const float* pres_red, const float* pres_black,
					float* res_array,
				  const float dt, const float dx, const float dy,
				  const int imax, const int jmax, const int M_x);



/*------------------------------------------------------------------- */
/* computation of new velocity values */
/*------------------------------------------------------------------- */ 

__global__ void calculate_u ( const float* F, 
				  const float* pres_red, const float* pres_black, 
				  float* u, float* max_u, 
  				  const float dt, const float dx, const float dy,
				  const int imax, const int jmax, const int M_x);


__global__ void calculate_v ( const float* G, 
				  const float* pres_red, const float* pres_black, 
				  float* v, float* max_v,
  				  const float dt, const float dx, const float dy,
				  const int imax, const int jmax, const int M_x);


#endif // __UVP_H__
