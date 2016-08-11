/* finitediff.cu
 * finite difference methods on a grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160707
 */
#include "finitediff.h"
#include "errors.h"

__constant__ float2 dev_cnus[4]; // i = x,y; j = 1,2,3,4

// for the Laplacian, and double derivatives
__constant__ float2 dev_cnus2[4]; // i = x,y; j = 1,2,3,4

void set1DerivativeParameters(const float hd_i[2] )
{
	float unscaled_cnu[4] { 1.f/2.f, 0.f, 0.f, 0.f };
	
	float2 *cnus = new float2[4];

	for (int nu = 0; nu < 4; ++nu ) {
		cnus[nu].x = unscaled_cnu[nu]*(1.f/hd_i[0] );
		cnus[nu].y = unscaled_cnu[nu]*(1.f/hd_i[1] );
	}
	
	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_cnus, cnus, sizeof(float2)*4, 0, cudaMemcpyHostToDevice) 
	); // offset from start is 0
	
	delete[] cnus;		
}

void set2DerivativeParameters(const float hd_i[2]  )
{
	float unscaled_cnu[4] { 2.f/3.f, -1.f/12.f, 0.f, 0.f, };
	
	float2 *cnus = new float2[4];

	for (int nu = 0; nu < 4; ++nu ) {
		cnus[nu].x = unscaled_cnu[nu]*(1.f/hd_i[0] );
		cnus[nu].y = unscaled_cnu[nu]*(1.f/hd_i[1] );
	}
	
	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_cnus, cnus, sizeof(float2)*4, 0, cudaMemcpyHostToDevice) 
	); // offset from start is 0
	
	delete[] cnus;		
}



void set3DerivativeParameters(const float hd_i[2] )
{
	float unscaled_cnu[4] { 3.f/4.f, -3.f/20.f, 1.f/60.f, 0.f };
	
	float2 *cnus = new float2[4];

	for (int nu = 0; nu < 4; ++nu ) {
		cnus[nu].x = unscaled_cnu[nu]*(1.f/hd_i[0] );
		cnus[nu].y = unscaled_cnu[nu]*(1.f/hd_i[1] );
	}
	
	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_cnus, cnus, sizeof(float2)*4, 0, cudaMemcpyHostToDevice) 
	); // offset from start is 0
	
	delete[] cnus;		

		
}

void set4DerivativeParameters(const float hd_i[2]  )
{
	float unscaled_cnu[4] { 4.f/5.f, -1.f/5.f, 4.f/105.f, -1.f/280.f  };
	
	float2 *cnus = new float2[4];

	for (int nu = 0; nu < 4; ++nu ) {
		cnus[nu].x = unscaled_cnu[nu]*(1.f/hd_i[0] );
		cnus[nu].y = unscaled_cnu[nu]*(1.f/hd_i[1] );
	}
	
	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_cnus, cnus, sizeof(float2)*4, 0, cudaMemcpyHostToDevice) 
	); // offset from start is 0
	
	delete[] cnus;		
}

__device__ float dev_dirder1(float stencil[1][2], float c_nus[4]) {
	float tempvalue {0.f};

	tempvalue += c_nus[0]*( stencil[0][1] - stencil[0][0] );

	return tempvalue;
}

__device__ float dev_dirder2(float stencil[2][2], float c_nus[4]) {
	int NU {2};
	float tempvalue {0.f};

	for (int nu = 0; nu < NU; ++nu ) {
		tempvalue += c_nus[nu]*( stencil[nu][1] - stencil[nu][0] );
	}
	return tempvalue;
}

__device__ float dev_dirder3(float stencil[3][2], float c_nus[4]) {
	int NU {3};
	float tempvalue {0.f};
		
	for (int nu = 0; nu < NU; ++nu ) {
		tempvalue += c_nus[nu]*( stencil[nu][1] - stencil[nu][0] );
	}
	return tempvalue;
}

__device__ float dev_dirder4(float stencil[4][2], float c_nus[4]) {
	int NU {4};
	float tempvalue {0.f};

	for (int nu = 0; nu < NU; ++nu ) {
		tempvalue += c_nus[nu]*( stencil[nu][1] - stencil[nu][0] );
	}
	return tempvalue;
}

// GRADIENT (GRAD) in 1,2,3,4 stencils, central difference

__device__ float2 dev_grad1( float2 stencil[1][2] ) {
	float stencilx[1][2] { { stencil[0][0].x, stencil[0][1].x } };
	float stencily[1][2] { { stencil[0][0].y, stencil[0][1].y } };
	
	float c_nusx[4] { dev_cnus[0].x, dev_cnus[1].x, dev_cnus[2].x, dev_cnus[3].x };
	float c_nusy[4] { dev_cnus[0].y, dev_cnus[1].y, dev_cnus[2].y, dev_cnus[3].y };

	float2 grad_value { dev_dirder1( stencilx, c_nusx ) , 
						dev_dirder1( stencily, c_nusy ) } ; 
	return grad_value;
}

__device__ float2 dev_grad2( float2 stencil[2][2]  ) {
	float stencilx[2][2] { { stencil[0][0].x, stencil[0][1].x }, { stencil[1][0].x, stencil[1][1].x } };
	float stencily[2][2] { { stencil[0][0].y, stencil[0][1].y }, { stencil[1][0].y, stencil[1][1].y } };

	float c_nusx[4] { dev_cnus[0].x, dev_cnus[1].x, dev_cnus[2].x, dev_cnus[3].x };
	float c_nusy[4] { dev_cnus[0].y, dev_cnus[1].y, dev_cnus[2].y, dev_cnus[3].y };
	
	float2 grad_value { dev_dirder2( stencilx, c_nusx ) , 
						dev_dirder2( stencily, c_nusy ) } ;

	return grad_value;
}

__device__ float2 dev_grad3( float2 stencil[3][2]  ) {
	float stencilx[3][2] { { stencil[0][0].x, stencil[0][1].x }, { stencil[1][0].x, stencil[1][1].x }, { stencil[2][0].x, stencil[2][1].x } };
	float stencily[3][2] { { stencil[0][0].y, stencil[0][1].y }, { stencil[1][0].y, stencil[1][1].y }, { stencil[2][0].y, stencil[2][1].y } };

	float c_nusx[4] { dev_cnus[0].x, dev_cnus[1].x, dev_cnus[2].x, dev_cnus[3].x };
	float c_nusy[4] { dev_cnus[0].y, dev_cnus[1].y, dev_cnus[2].y, dev_cnus[3].y };
	
	float2 grad_value { dev_dirder3( stencilx, c_nusx ) , 
						dev_dirder3( stencily, c_nusy ) } ;

	return grad_value;
}

__device__ float2 dev_grad4( float2 stencil[4][2]  ) {
	float stencilx[4][2] { { stencil[0][0].x, stencil[0][1].x }, { stencil[1][0].x, stencil[1][1].x }, { stencil[2][0].x, stencil[2][1].x }, { stencil[3][0].x, stencil[3][1].x } };
	float stencily[4][2] { { stencil[0][0].y, stencil[0][1].y }, { stencil[1][0].y, stencil[1][1].y }, { stencil[2][0].y, stencil[2][1].y }, { stencil[3][0].y, stencil[3][1].y } };

	float c_nusx[4] { dev_cnus[0].x, dev_cnus[1].x, dev_cnus[2].x, dev_cnus[3].x };
	float c_nusy[4] { dev_cnus[0].y, dev_cnus[1].y, dev_cnus[2].y, dev_cnus[3].y };
	
	float2 grad_value { dev_dirder4( stencilx, c_nusx ) , 
						dev_dirder4( stencily, c_nusy ) } ;

	return grad_value;
}


// DIVERGENCE (DIV) in 1,2,3,4 stencils, central difference

__device__ float dev_div1( float2 stencil[1][2]  ) {
	float stencilx[1][2] { { stencil[0][0].x, stencil[0][1].x } };
	float stencily[1][2] { { stencil[0][0].y, stencil[0][1].y } };

	float c_nusx[4] { dev_cnus[0].x, dev_cnus[1].x, dev_cnus[2].x, dev_cnus[3].x };
	float c_nusy[4] { dev_cnus[0].y, dev_cnus[1].y, dev_cnus[2].y, dev_cnus[3].y };
	
	float div_value { dev_dirder1( stencilx, c_nusx ) } ;
	div_value += dev_dirder1( stencily, c_nusy ) ;

	return div_value;
}

__device__ float dev_div2( float2 stencil[2][2]  ) {
	float stencilx[2][2] { { stencil[0][0].x, stencil[0][1].x }, { stencil[1][0].x, stencil[1][1].x } };
	float stencily[2][2] { { stencil[0][0].y, stencil[0][1].y }, { stencil[1][0].y, stencil[1][1].y } };

	float c_nusx[4] { dev_cnus[0].x, dev_cnus[1].x, dev_cnus[2].x, dev_cnus[3].x };
	float c_nusy[4] { dev_cnus[0].y, dev_cnus[1].y, dev_cnus[2].y, dev_cnus[3].y };
	
	float div_value { dev_dirder2( stencilx, c_nusx ) } ;
	div_value += dev_dirder2( stencily, c_nusy ) ;

	return div_value;
}

__device__ float dev_div3( float2 stencil[3][2]  ) {
	float stencilx[3][2] { { stencil[0][0].x, stencil[0][1].x }, { stencil[1][0].x, stencil[1][1].x }, { stencil[2][0].x, stencil[2][1].x } };
	float stencily[3][2] { { stencil[0][0].y, stencil[0][1].y }, { stencil[1][0].y, stencil[1][1].y }, { stencil[2][0].y, stencil[2][1].y } };

	float c_nusx[4] { dev_cnus[0].x, dev_cnus[1].x, dev_cnus[2].x, dev_cnus[3].x };
	float c_nusy[4] { dev_cnus[0].y, dev_cnus[1].y, dev_cnus[2].y, dev_cnus[3].y };
	
	float div_value { dev_dirder3( stencilx, c_nusx ) } ;
	div_value += dev_dirder3( stencily, c_nusy ) ;

	return div_value;
}

__device__ float dev_div4( float2 stencil[4][2]  ) {
	float stencilx[4][2] { { stencil[0][0].x, stencil[0][1].x }, { stencil[1][0].x, stencil[1][1].x }, { stencil[2][0].x, stencil[2][1].x }, { stencil[3][0].x, stencil[3][1].x } };
	float stencily[4][2] { { stencil[0][0].y, stencil[0][1].y }, { stencil[1][0].y, stencil[1][1].y }, { stencil[2][0].y, stencil[2][1].y }, { stencil[3][0].y, stencil[3][1].y } };

	float c_nusx[4] { dev_cnus[0].x, dev_cnus[1].x, dev_cnus[2].x, dev_cnus[3].x };
	float c_nusy[4] { dev_cnus[0].y, dev_cnus[1].y, dev_cnus[2].y, dev_cnus[3].y };
	
	float div_value { dev_dirder4( stencilx, c_nusx ) } ;
	div_value += dev_dirder4( stencily, c_nusy ) ;

	return div_value;
}

// For the Laplacian and double derivatives; central difference for order = 2 double derivatives

void set1dblDerivativeParameters(const float hd_i[2] )
{
	float unscaled_cnu[4] { 1.f, 0.f, 0.f, 0.f };
	
	float2 *cnus = new float2[4];

	for (int nu = 0; nu < 4; ++nu ) {
		cnus[nu].x = unscaled_cnu[nu]*(1.f/(hd_i[0]*hd_i[0]) );
		cnus[nu].y = unscaled_cnu[nu]*(1.f/(hd_i[1]*hd_i[1]) );
	}
	
	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_cnus2, cnus, sizeof(float2)*4, 0, cudaMemcpyHostToDevice) 
	); // offset from start is 0
	
	delete[] cnus;		
}

void set2dblDerivativeParameters(const float hd_i[2]  )
{
	float unscaled_cnu[4] { 4.f/3.f, -1.f/12.f, 0.f, 0.f, };
	
	float2 *cnus = new float2[4];

	for (int nu = 0; nu < 4; ++nu ) {
		cnus[nu].x = unscaled_cnu[nu]*(1.f/(hd_i[0]*hd_i[0]) );
		cnus[nu].y = unscaled_cnu[nu]*(1.f/(hd_i[1]*hd_i[1]) );
	}
	
	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_cnus2, cnus, sizeof(float2)*4, 0, cudaMemcpyHostToDevice) 
	); // offset from start is 0
	
	delete[] cnus;		
}



void set3dblDerivativeParameters(const float hd_i[2] )
{
	float unscaled_cnu[4] { 3.f/2.f , -3.f/20.f, 1.f/90.f, 0.f };
	
	float2 *cnus = new float2[4];

	for (int nu = 0; nu < 4; ++nu ) {
		cnus[nu].x = unscaled_cnu[nu]*(1.f/(hd_i[0]*hd_i[0]) );
		cnus[nu].y = unscaled_cnu[nu]*(1.f/(hd_i[1]*hd_i[1]) );
	}
	
	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_cnus2, cnus, sizeof(float2)*4, 0, cudaMemcpyHostToDevice) 
	); // offset from start is 0
	
	delete[] cnus;		
	
}

void set4dblDerivativeParameters(const float hd_i[2]  )
{
	float unscaled_cnu[4] { 8.f/5.f, -1.f/5.f, 8.f/315.f, -1.f/560.f  };
	
	float2 *cnus = new float2[4];

	for (int nu = 0; nu < 4; ++nu ) {
		cnus[nu].x = unscaled_cnu[nu]*(1.f/(hd_i[0]*hd_i[0]) );
		cnus[nu].y = unscaled_cnu[nu]*(1.f/(hd_i[1]*hd_i[1]) );
	}
	
	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_cnus2, cnus, sizeof(float2)*4, 0, cudaMemcpyHostToDevice) 
	); // offset from start is 0
	
	delete[] cnus;		
}

// For the Laplacian and double derivatives; order = 2 i.e. double derivatives


__device__ float dev_dirdblder1(float centerval, float stencil[1][2], float c_nus[4]) {
	float tempvalue {0.f};

	tempvalue += c_nus[0]*( stencil[0][1] + stencil[0][0] -2*centerval );

	return tempvalue;
}

__device__ float dev_dirdblder2(float centerval, float stencil[2][2], float c_nus[4]) {
	int NU {2};
	float tempvalue {0.f};

	for (int nu = 0; nu < NU; ++nu ) {
		tempvalue += c_nus[nu]*( stencil[nu][1] + stencil[nu][0] - 2*centerval );
	}
	return tempvalue;
}

__device__ float dev_dirdblder3(float centerval, float stencil[3][2], float c_nus[4]) {
	int NU {3};
	float tempvalue {0.f};
		
	for (int nu = 0; nu < NU; ++nu ) {
		tempvalue += c_nus[nu]*( stencil[nu][1] + stencil[nu][0] -2*centerval);
	}
	return tempvalue;
}

__device__ float dev_dirdblder4(float centerval, float stencil[4][2], float c_nus[4]) {
	int NU {4};
	float tempvalue {0.f};

	for (int nu = 0; nu < NU; ++nu ) {
		tempvalue += c_nus[nu]*( stencil[nu][1] + stencil[nu][0] -2*centerval);
	}
	return tempvalue;
}

// LAPLACIAN (LAP) in 1,2,3,4 stencils, central difference

__device__ float dev_lap1( float centerval, float2 stencil[1][2]  ) {
	float stencilx[1][2] { { stencil[0][0].x, stencil[0][1].x } };
	float stencily[1][2] { { stencil[0][0].y, stencil[0][1].y } };

	float c_nusx[4] { dev_cnus[0].x, dev_cnus[1].x, dev_cnus[2].x, dev_cnus[3].x };
	float c_nusy[4] { dev_cnus[0].y, dev_cnus[1].y, dev_cnus[2].y, dev_cnus[3].y };
	
	float lap_value { dev_dirdblder1(centerval, stencilx, c_nusx ) } ;
	lap_value += dev_dirdblder1( centerval, stencily, c_nusy ) ;

	return lap_value;
}

__device__ float dev_lap2(float centerval, float2 stencil[2][2]  ) {
	float stencilx[2][2] { { stencil[0][0].x, stencil[0][1].x }, { stencil[1][0].x, stencil[1][1].x } };
	float stencily[2][2] { { stencil[0][0].y, stencil[0][1].y }, { stencil[1][0].y, stencil[1][1].y } };

	float c_nusx[4] { dev_cnus[0].x, dev_cnus[1].x, dev_cnus[2].x, dev_cnus[3].x };
	float c_nusy[4] { dev_cnus[0].y, dev_cnus[1].y, dev_cnus[2].y, dev_cnus[3].y };
	
	float lap_value { dev_dirdblder2( centerval, stencilx, c_nusx ) } ;
	lap_value += dev_dirdblder2( centerval, stencily, c_nusy ) ;

	return lap_value;
}

__device__ float dev_lap3( float centerval, float2 stencil[3][2]  ) {
	float stencilx[3][2] { { stencil[0][0].x, stencil[0][1].x }, { stencil[1][0].x, stencil[1][1].x }, { stencil[2][0].x, stencil[2][1].x } };
	float stencily[3][2] { { stencil[0][0].y, stencil[0][1].y }, { stencil[1][0].y, stencil[1][1].y }, { stencil[2][0].y, stencil[2][1].y } };

	float c_nusx[4] { dev_cnus[0].x, dev_cnus[1].x, dev_cnus[2].x, dev_cnus[3].x };
	float c_nusy[4] { dev_cnus[0].y, dev_cnus[1].y, dev_cnus[2].y, dev_cnus[3].y };
	
	float lap_value { dev_dirdblder3(centerval, stencilx, c_nusx ) } ;
	lap_value += dev_dirdblder3( centerval,stencily, c_nusy ) ;

	return lap_value;
}

__device__ float dev_lap4( float centerval, float2 stencil[4][2]  ) {
	float stencilx[4][2] { { stencil[0][0].x, stencil[0][1].x }, { stencil[1][0].x, stencil[1][1].x }, { stencil[2][0].x, stencil[2][1].x }, { stencil[3][0].x, stencil[3][1].x } };
	float stencily[4][2] { { stencil[0][0].y, stencil[0][1].y }, { stencil[1][0].y, stencil[1][1].y }, { stencil[2][0].y, stencil[2][1].y }, { stencil[3][0].y, stencil[3][1].y } };

	float c_nusx[4] { dev_cnus[0].x, dev_cnus[1].x, dev_cnus[2].x, dev_cnus[3].x };
	float c_nusy[4] { dev_cnus[0].y, dev_cnus[1].y, dev_cnus[2].y, dev_cnus[3].y };
	
	float lap_value { dev_dirdblder4( centerval, stencilx, c_nusx ) } ;
	lap_value += dev_dirdblder4( centerval, stencily, c_nusy ) ;

	return lap_value;
}
