/* finitediff.h
 * finite difference methods on a grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160710
 */
#ifndef __FINITEDIFF_H__
#define __FINITEDIFF_H__ 

#include "errors.h"

// I fixed the size of dev_c to be 4 because I don't anticipate that we'd need more accuracy 
extern __constant__ float2 dev_cnus[4]; // i = x,y; j = 1,2,3,4

// for the Laplacian and double derivatives
extern __constant__ float2 dev_cnus2[4]; // i = x,y; j = 1,2,3,4 

// for the divergence (and gradient)

void set1DerivativeParameters(const float hd_i[2] );

void set2DerivativeParameters(const float hd_i[2] );

void set3DerivativeParameters(const float hd_i[2] );

void set4DerivativeParameters(const float hd_i[2] );


__device__ float dev_dirder1(float stencil[1][2], float c_nus[4]);

__device__ float dev_dirder2(float stencil[2][2], float c_nus[4]);

__device__ float dev_dirder3(float stencil[3][2], float c_nus[4]);

__device__ float dev_dirder4(float stencil[4][2], float c_nus[4]);

// GRADIENT (GRAD) in 1,2,3,4 stencils, central difference

__device__ float2 dev_grad1( float2 stencil[1][2] ) ;

__device__ float2 dev_grad2( float2 stencil[2][2] ) ;

__device__ float2 dev_grad3( float2 stencil[3][2] ) ;

__device__ float2 dev_grad4( float2 stencil[4][2] ) ;


// DIVERGENCE (DIV) in 1, 2, 3, 4 stencils, central difference

__device__ float dev_div1( float2 stencil[1][2]  ) ;

__device__ float dev_div2( float2 stencil[2][2]  ) ; 

__device__ float dev_div3( float2 stencil[3][2]  ) ;

__device__ float dev_div4( float2 stencil[4][2]  ) ;

// For the Laplacian and double derivatives; central difference for order = 2 double derivatives

void set1dblDerivativeParameters(const float hd_i[2] );

void set2dblDerivativeParameters(const float hd_i[2] );

void set3dblDerivativeParameters(const float hd_i[2] );

void set4dblDerivativeParameters(const float hd_i[2] );

// For the Laplacian and double derivatives; order = 2 i.e. double derivatives

__device__ float dev_dirdblder1(float centerval, float stencil[1][2], float c_nus[4]);

__device__ float dev_dirdblder2(float centerval, float stencil[2][2], float c_nus[4]);

__device__ float dev_dirdblder3(float centerval, float stencil[3][2], float c_nus[4]);

__device__ float dev_dirdblder4(float centerval, float stencil[4][2], float c_nus[4]);

// LAPLACIAN (LAP) in 1, 2, 3, 4 stencils, central difference

__device__ float dev_lap1( float centerval, float2 stencil[1][2]  ) ;

__device__ float dev_lap2( float centerval, float2 stencil[2][2]  ) ; 

__device__ float dev_lap3( float centerval, float2 stencil[3][2]  ) ;

__device__ float dev_lap4( float centerval, float2 stencil[4][2]  ) ;


#endif // __FINITEDIFF_H__
