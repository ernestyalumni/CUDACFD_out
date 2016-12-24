/** uvp.cu
 * \file uvp.cu
 * \brief computation of u, v, p for 2-dim. incompressible Navier-Stokes equation with finite difference
 * \author Ernest Yeung  
 * \email ernestyalumni@gmail.com
 * \date 20161209
 * 
 * compilation tip: (compile separately)
 * nvcc -std=c++11 -c ./physlib/u_p.cu -o u_p.o
 * 
 */
#include "uvp.h"

__host__ __device__ int flatten( int i, int j, int stride) {
	return i * stride + j ;
}

/*------------------------------------------------------------------- */
/* Computation of tentative velocity field (F,G) -------------------- */
/*------------------------------------------------------------------- */

__global__ 
void calculate_F ( const float* u, const float* v,
				  float* F, 
					const float mix_param, const float Re_num, float gx,
  				  const float dt, const float dx, const float dy,
				  const int imax, const int jmax, const int M_x)
{	
	int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
	int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;
	
	if (col == imax) {
		// right boundary, F_ij = u_ij
		// also do left boundary
		F[ 0 * ( jmax + 2 ) + row ]    = u[ 0 * ( jmax + 2 ) + row ] ;
		F[ imax * ( jmax + 2 ) + row ] = u[ imax * ( jmax + 2 ) + row ] ;
	} else {
		
		// u velocities
		float u_ij   = u[ col * ( jmax + 2 ) + row ] ; 
		float u_ip1j = u[ (col+1) * ( jmax + 2 ) + row ] ; 
		float u_ijp1 = u[ col * ( jmax + 2 ) + row + 1 ] ; 
		float u_im1j = u[ (col - 1 ) * ( jmax + 2 ) + row ] ; 
		float u_ijm1 = u[ col * ( jmax + 2 ) + row - 1 ] ; 

		// v velocities
		float v_ij   = v[ col * ( jmax + 2 ) + row ] ; 
		float v_ip1j = v[ (col + 1) * ( jmax + 2 ) + row ] ; 
		float v_ijm1 = v[ col * ( jmax + 2 ) + row -1 ] ; 
		float v_ip1jm1 = v[ (col + 1 ) * ( jmax + 2 ) + row - 1] ; 


		// finite differences
		float du2dx, duvdy, d2udx2, d2udy2;

		du2dx = (((u_ij + u_ip1j) * (u_ij + u_ip1j) - (u_im1j + u_ij) * (u_im1j + u_ij))
				+ mix_param * (fabs(u_ij + u_ip1j) * (u_ij - u_ip1j)
				- fabs(u_im1j + u_ij) * (u_im1j - u_ij))) / (4.0 * dx);
		duvdy = ((v_ij + v_ip1j) * (u_ij + u_ijp1) - (v_ijm1 + v_ip1jm1) * (u_ijm1 + u_ij)
				+ mix_param * (fabs(v_ij + v_ip1j) * (u_ij - u_ijp1)
				- fabs(v_ijm1 + v_ip1jm1) * (u_ijm1 - u_ij))) / (4.0 * dy);
		d2udx2 = (u_ip1j - (2.0 * u_ij) + u_im1j) / (dx * dx);
		d2udy2 = (u_ijp1 - (2.0 * u_ij) + u_ijm1) / (dy * dy);

		F[ col * ( jmax + 2 ) + row ] = 
			u_ij + dt * (((d2udx2 + d2udy2) / Re_num) - du2dx - duvdy + gx);
		
	} // end if
		
} // end calculate_F

///////////////////////////////////////////////////////////////////////////////

__global__ 
void calculate_G ( const float* u, const float* v,
				  float* G, 
					const float mix_param, const float Re_num, float gy,
  				  const float dt, const float dx, const float dy,
				  const int imax, const int jmax, const int M_x)

{
	int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
	int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;
	
	if (row == jmax) {
		// top and bottom boundaries
		G[ col * ( jmax + 2 ) + 0 ]    = v[ col * ( jmax + 2 ) + 0 ] ;
		G[ col * ( jmax + 2 ) + jmax ] = v[ col * ( jmax + 2 ) + jmax ] ;
	} else {
		
		// u velocities
		float u_ij   = u[ col * ( jmax + 2 ) + row ] ; 
		float u_ijp1 = u[ col  * ( jmax + 2 ) + row + 1 ] ; 
		float u_im1j = u[ (col-1) * ( jmax + 2 ) + row  ] ; 
		float u_im1jp1 = u[ (col - 1 ) * ( jmax + 2 ) + row + 1] ; 

		// v velocities
		float v_ij   = v[ col * ( jmax + 2 ) + row ] ; 
		float v_ijp1 = v[ col * ( jmax + 2 ) + row +1 ] ; 
		float v_ip1j = v[ (col+1) * ( jmax + 2 ) + row  ] ; 
		float v_ijm1 = v[ col  * ( jmax + 2 ) + row -1 ] ; 
		float v_im1j = v[ (col - 1) * ( jmax + 2 ) + row  ] ; 

		
		// finite differences
		float dv2dy, duvdx, d2vdx2, d2vdy2;

		dv2dy = ((v_ij + v_ijp1) * (v_ij + v_ijp1) - (v_ijm1 + v_ij) * (v_ijm1 + v_ij)
				+ mix_param * (fabs(v_ij + v_ijp1) * (v_ij - v_ijp1)
				- fabs(v_ijm1 + v_ij) * (v_ijm1 - v_ij))) / (4.0 * dy);
		duvdx = ((u_ij + u_ijp1) * (v_ij + v_ip1j) - (u_im1j + u_im1jp1) * (v_im1j + v_ij)
				+ mix_param * (fabs(u_ij + u_ijp1) * (v_ij - v_ip1j) 
				- fabs(u_im1j + u_im1jp1) * (v_im1j - v_ij))) / (4.0 * dx);
		d2vdx2 = (v_ip1j - (2.0 * v_ij) + v_im1j) / (dx * dx);
		d2vdy2 = (v_ijp1 - (2.0 * v_ij) + v_ijm1) / (dy * dy);


		G[ col * ( jmax + 2 ) + row ] = 
			v_ij + dt * (((d2vdx2 + d2vdy2) / Re_num) - dv2dy - duvdx + gy);
			
	} // end if
		
} // end calculate_G

///////////////////////////////////////////////////////////////////////////////

// M_x is the number of threads in a block along the x-dimension
__global__ 
void sum_pressure (const float* pres_red, const float* pres_black, 
					float* pres_sum, 
					const int imax, const int jmax, const int M_x) 
{
	int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
	int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

	// shared memory for block's sum
	extern __shared__ float sum_cache[];
			
	int NUM_2 = jmax >> 1;

	float pres_r = pres_red[ col * ( NUM_2 + 2 ) + row ] ;
	float pres_b = pres_black[ col * ( NUM_2 + 2 ) + row ] ;

	// add squared pressure
	sum_cache[threadIdx.x] = (pres_r * pres_r) + (pres_b * pres_b);

	// synchronize threads in block to ensure all thread values stored
	__syncthreads();

	// add up values for block
	int i = M_x >> 1;
	while (i != 0) {
		if (threadIdx.x < i) {
			sum_cache[threadIdx.x] += sum_cache[threadIdx.x + i];
		}
		__syncthreads();
		i >>= 1;
	}

	// store block's summed values
	if (threadIdx.x == 0) {
		pres_sum[blockIdx.y + (gridDim.y * blockIdx.x)] = sum_cache[0];
	}

} // end sum_pressure

/*------------------------------------------------------------------- */
/* SOR iteration for the Poisson equation for the pressure
/*------------------------------------------------------------------- */
///////////////////////////////////////////////////////////////////////////////

/** Function to update pressure for red cells
 * 
 * \param[in]		dt			time-step size
 * \param[in]		F			array of discretized x-momentum eqn terms
 * \param[in]		G			array of discretized y-momentum eqn terms
 * \param[in]		pres_black	pressure values of black cells
 * \param[inout]	pres_red	pressure values of red cells
 */
__global__
void red_kernel ( const float* F, 
				 const float* G, const float* pres_black,
				 float* pres_red,
				const float omega, 
				  const float dt, const float dx, const float dy,
				  const int imax, const int jmax, const int M_x)

{
	int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
	int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;
	
	int NUM_2 = jmax >> 1;			
	
	float p_ij = 	pres_red[ col * ( NUM_2 + 2 ) + row ] ;
		
	float p_im1j = 	pres_black[ (col - 1) * ( NUM_2 + 2 ) + row ] ;
	float p_ip1j = 	pres_black[ (col + 1) * ( NUM_2 + 2 ) + row ] ;
	float p_ijm1 = 	pres_black[ col * ( NUM_2 + 2 ) + row - (col & 1 ) ] ;
	float p_ijp1 = 	pres_black[ col * ( NUM_2 + 2 ) + row + ((col + 1) & 1 ) ] ;
	
	// right-hand side
	float rhs = ((
	(
	F[ col * ( jmax + 2 ) + (2 * row) - (col & 1)  ]
	 - F[ (col - 1) * ( jmax + 2 ) + (2 * row) - (col & 1)  ] 
	 ) / dx)
			  + (
	(
	G[ col * ( jmax + 2 ) + (2 * row) - (col & 1)  ]
    - G[ col * ( jmax + 2 ) + (2 * row) - (col & 1) - 1 ]
	   ) / dy)) / dt;

	
	pres_red[ col * ( NUM_2 + 2 ) + row ] = p_ij * (1.0 - omega) + omega * 
		(((p_ip1j + p_im1j) / (dx * dx)) + ((p_ijp1 + p_ijm1) / (dy * dy)) - 
		rhs) / ((2.0 / (dx * dx)) + (2.0 / (dy * dy)));
	
} // end red_kernel

///////////////////////////////////////////////////////////////////////////////

/** Function to update pressure for black cells
 * 
 * \param[in]		dt			time-step size
 * \param[in]		F			array of discretized x-momentum eqn terms
 * \param[in]		G			array of discretized y-momentum eqn terms
 * \param[in]		pres_red	pressure values of red cells
 * \param[inout]	pres_black	pressure values of black cells
 */
__global__ 
void black_kernel ( const float* F, 
				   const float* G, const float* pres_red, 
				   float* pres_black,
				   const float omega,
				   const float dt, const float dx, const float dy,
				  const int imax, const int jmax, const int M_x) 
{
	int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
	int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;
	
	int NUM_2 = jmax >> 1;
	
	
	float p_ij = 	pres_black[ col * ( NUM_2 + 2 ) + row ] ;
		
	float p_im1j = 	pres_red[ (col - 1) * ( NUM_2 + 2 ) + row ] ;
	float p_ip1j = 	pres_red[ (col + 1) * ( NUM_2 + 2 ) + row ] ;
	float p_ijm1 = 	pres_red[ col * ( NUM_2 + 2 ) + row - ((col + 1) & 1 ) ] ;
	float p_ijp1 = 	pres_red[ col * ( NUM_2 + 2 ) + row + (col & 1 ) ] ;
	
	// right-hand side
	float rhs = ((
	(
	F[ col * ( jmax + 2 ) + (2 * row) - ((col + 1) & 1)  ]
	 - F[ (col - 1) * ( jmax + 2 ) + (2 * row) - ((col + 1) & 1)  ] 
	 ) / dx)
			  + (
	(
	G[ col * ( jmax + 2 ) + (2 * row) - ((col + 1) & 1)  ]
    - G[ col * ( jmax + 2 ) + (2 * row) - ((col+1) & 1) - 1 ]
	   ) / dy)) / dt;

	pres_black[ col * ( NUM_2 + 2 ) + row ] 
		 = p_ij * (1.0 - omega) + omega * 
		(((p_ip1j + p_im1j) / (dx * dx)) + ((p_ijp1 + p_ijm1) / (dy * dy)) - 
		rhs) / ((2.0 / (dx * dx)) + (2.0 / (dy * dy)));
		
} // end black_kernel

/*------------------------------------------------------------------- */
/* computation of residual */
/*------------------------------------------------------------------- */
///////////////////////////////////////////////////////////////////////////////

__global__
void calc_residual ( const float* F, const float* G, 
					const float* pres_red, const float* pres_black,
					float* res_array,
				  const float dt, const float dx, const float dy,
				  const int imax, const int jmax, const int M_x)

{
	int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
	int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;
	
	int NUM_2 = jmax >> 1;

	float p_ij, p_im1j, p_ip1j, p_ijm1, p_ijp1, rhs, res, res2;

	// red point
	p_ij = pres_red[flatten(col, row, (NUM_2+2))];
	
	p_im1j = pres_black[flatten(col-1,row,NUM_2+2)] ; 
	p_ip1j = pres_black[flatten(col+1,row,NUM_2+2)] ; 
	p_ijm1 = pres_black[flatten(col,row - (col & 1),NUM_2+2)] ; 
	p_ijp1 = pres_black[flatten(col,row+((col + 1) & 1),NUM_2+2)] ;

	rhs = (((
		F[ flatten(col, (2*row)-(col & 1),jmax + 2)] - 
		F[ flatten(col-1, (2*row)- (col & 1) , jmax + 2)] )
//		F(col, (2 * row) - (col & 1)) - F(col - 1, (2 * row) - (col & 1))) 
	/ dx)
		+  ((
		G[flatten(col, (2 * row) - (col & 1), jmax + 2)] - 
		G[flatten(col, (2 * row) - (col & 1)-1, jmax + 2)] )
//		G(col, (2 * row) - (col & 1)) - G(col, (2 * row) - (col & 1) - 1)) 
		/ dy)) / dt;

	// calculate residual
	res = ((p_ip1j - (2.0 * p_ij) + p_im1j) / (dx * dx))
		+ ((p_ijp1 - (2.0 * p_ij) + p_ijm1) / (dy * dy)) - rhs;

	// black point
	p_ij = pres_black[flatten(col, row, NUM_2 + 2)];

	p_im1j = pres_red[flatten(col - 1, row, NUM_2 + 2)];
	p_ip1j = pres_red[flatten(col + 1, row, NUM_2 + 2)];
	p_ijm1 = pres_red[flatten(col, row - ((col + 1) & 1) , NUM_2 + 2)];
	p_ijp1 = pres_red[flatten(col, row + (col & 1) , NUM_2 + 2)];
	
	// right-hand side
	rhs = ((
		(
		F[flatten(col, (2 * row) - ((col + 1) & 1),jmax +2)] - 
		F[flatten(col - 1, (2 * row) - ((col + 1) & 1), jmax + 2) ] 
		) / dx)
		+  (
		(
		G[ flatten(col, (2 * row) - ((col + 1) & 1), jmax + 2) ] - 
		G[ flatten( col, (2 * row) - ((col + 1) & 1) - 1, jmax + 2) ]
		) / dy
		)) / dt;

	// calculate residual
	res2 = ((p_ip1j - (2.0 * p_ij) + p_im1j) / (dx * dx))
		 + ((p_ijp1 - (2.0 * p_ij) + p_ijm1) / (dy * dy)) - rhs;

	// shared memory for block's sum
	extern __shared__ float sum_cache[];

	sum_cache[threadIdx.x] = (res * res) + (res2 * res2);

	// synchronize threads in block to ensure all residuals stored
	__syncthreads();

	// add up squared residuals for block
	int i = M_x >> 1;
	while (i != 0) {
		if (threadIdx.x < i) {
			sum_cache[threadIdx.x] += sum_cache[threadIdx.x + i];
		}
		__syncthreads();
		i >>= 1;
	}

	// store block's summed residuals
	if (threadIdx.x == 0) {
		res_array[blockIdx.y + (gridDim.y * blockIdx.x)] = sum_cache[0];
	}
} 


/*------------------------------------------------------------------- */
/* computation of new velocity values */
/*------------------------------------------------------------------- */
///////////////////////////////////////////////////////////////////////////////

__global__ 
void calculate_u ( const float* F, 
				  const float* pres_red, const float* pres_black, 
				  float* u, float* max_u, 
  				  const float dt, const float dx, const float dy,
				  const int imax, const int jmax, const int M_x)

{
	int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
	int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;
	
	// allocate shared memory to store max velocities
	extern __shared__ float max_cache[];
	max_cache[threadIdx.x] = 0.0;
	
	int NUM_2 = jmax >> 1;
	float new_u = 0.0;

	if (col != jmax) {

		float p_ij, p_ip1j, new_u2;

		// red point
		p_ij = pres_red[ flatten(col, row, NUM_2 + 2)];
		p_ip1j = pres_black[ flatten(col + 1, row, NUM_2 + 2) ];

		new_u = F[ flatten(col, (2 * row) - (col & 1), jmax + 2)] - (dt * (p_ip1j - p_ij) / dx);
		u[ flatten(col, (2 * row) - (col & 1), jmax + 2)] = new_u;

		// black point
		p_ij = pres_black[ flatten(col, row, NUM_2 + 2)];
		p_ip1j = pres_red[ flatten(col + 1, row, NUM_2 + 2) ];

		new_u2 = F[ flatten(col, (2 * row) - ((col + 1) & 1), jmax + 2) ] - (dt * (p_ip1j - p_ij) / dx);
		u[ flatten(col, (2 * row) - ((col + 1) & 1), jmax + 2) ] = new_u2;

		// check for max of these two
		new_u = fmax(fabs(new_u), fabs(new_u2));

		if ((2 * row) == jmax) {
			// also test for max velocity at vertical boundary
			new_u = fmax(new_u, fabs( u[ flatten(col, jmax + 1, jmax + 2) ] ));
		}
	} else {
		// check for maximum velocity in boundary cells also
		new_u = fmax(fabs( u[ flatten(imax, (2 * row), jmax + 2) ] ), 
			fabs( u[ flatten(0, (2 * row), jmax + 2)] ));
		new_u = fmax(fabs( u[flatten(imax, (2 * row) - 1, jmax + 2) ]), new_u);
		new_u = fmax(fabs( u[flatten(0, (2 * row) - 1, jmax + 2) ] ), new_u);

		new_u = fmax(fabs( u[flatten(imax + 1, (2 * row), jmax + 2)] ), new_u);
		new_u = fmax(fabs( u[ flatten(imax + 1, (2 * row) - 1, jmax + 2)] ), new_u);

	} // end if

	// store maximum u for block from each thread
	max_cache[threadIdx.x] = new_u;

	// synchronize threads in block to ensure all velocities stored
	__syncthreads();

	// calculate maximum for block
	int i = M_x >> 1;
	while (i != 0) {
		if (threadIdx.x < i) {
			max_cache[threadIdx.x] = fmax(max_cache[threadIdx.x], max_cache[threadIdx.x + i]);
		}
		__syncthreads();
		i >>= 1;
	}

	// store block's maximum
	if (threadIdx.x == 0) {
		max_u[blockIdx.y + (gridDim.y * blockIdx.x)] = max_cache[0];
	}

	
} // end calculate_u

///////////////////////////////////////////////////////////////////////////////

__global__ 
void calculate_v ( const float* G, 
				  const float* pres_red, const float* pres_black, 
				  float* v, float* max_v,
				  const float dt, const float dx, const float dy,
				  const int imax, const int jmax, const int M_x)
{
	int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
	int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;
	
	// allocate shared memory to store maximum velocities
	extern __shared__ float max_cache[];
	max_cache[threadIdx.x] = 0.0;

	int NUM_2 = jmax >> 1;
	float new_v = 0.0;
	
	if (row != NUM_2) {
		float p_ij, p_ijp1, new_v2;

		// red pressure point
		p_ij = pres_red[ flatten(col, row, NUM_2 + 2)];
		p_ijp1 = pres_black[flatten(col, row + ((col + 1) & 1), NUM_2 + 2)];
	
		new_v = G[ flatten(col, (2 * row) - (col & 1), jmax + 2)] - (dt * (p_ijp1 - p_ij) / dy);
		v[ flatten(col, (2 * row) - (col & 1), jmax + 2)] = new_v;


		// black pressure point
		p_ij = pres_black[ flatten(col, row, NUM_2 + 2)];
		p_ijp1 = pres_red[ flatten(col, row + (col & 1), NUM_2 + 2)];
		
		new_v2 = G[ flatten(col, (2 * row) - ((col + 1) & 1), jmax + 2)] - (dt * (p_ijp1 - p_ij) / dy);
		v[ flatten(col, (2 * row) - ((col + 1) & 1) , jmax + 2)] = new_v2;


		// check for max of these two
		new_v = fmax(fabs(new_v), fabs(new_v2));

		if (col == imax) {
			// also test for max velocity at vertical boundary
			new_v = fmax(new_v, fabs( v[ flatten(imax + 1, (2 * row), jmax + 2) ] ));
		}

	} else {

		if ((col & 1) == 1) {
			// black point is on boundary, only calculate red point below it
			float p_ij = pres_red[ flatten(col, row, NUM_2 +2) ];
			float p_ijp1 = pres_black[ flatten(col, row + ((col + 1) & 1), NUM_2 + 2)];
		
			new_v = G[ flatten(col, (2 * row) - (col & 1), jmax + 2) ] - (dt * (p_ijp1 - p_ij) / dy);
			v[ flatten(col, (2 * row) - (col & 1), jmax + 2 ) ] = new_v;

		} else {
			// red point is on boundary, only calculate black point below it
			float p_ij = pres_black[ flatten(col, row, NUM_2 + 2)];
			float p_ijp1 = pres_red[ flatten(col, row + (col & 1), NUM_2 + 2)];
		
			new_v = G[ flatten(col, (2 * row) - ((col + 1) & 1), jmax + 2) ] - (dt * (p_ijp1 - p_ij) / dy);
			v[ flatten(col, (2 * row) - ((col + 1) & 1), jmax + 2) ] = new_v;
		}

		// get maximum v velocity
		new_v = fabs(new_v);

		// check for maximum velocity in boundary cells also
		new_v = fmax(fabs( v[flatten(col, jmax, jmax + 2)] ), new_v);
		new_v = fmax(fabs( v[flatten(col, 0, jmax + 2)] ), new_v);

		new_v = fmax(fabs( v[flatten(col, jmax + 1, jmax + 2) ] ), new_v);

	} // end if
		
	// store absolute value of velocity
	max_cache[threadIdx.x] = new_v;
	
	// synchronize threads in block to ensure all velocities stored
	__syncthreads();

	// calculate maximum for block
	int i = M_x >> 1;
	while (i != 0) {
		if (threadIdx.x < i) {
			max_cache[threadIdx.x] = fmax(max_cache[threadIdx.x], max_cache[threadIdx.x + i]);
		}
		__syncthreads();
		i >>= 1;
	}

	// store block's summed residuals
	if (threadIdx.x == 0) {
		max_v[blockIdx.y + (gridDim.y * blockIdx.x)] = max_cache[0];
	}
	
} // end calculate_v

///////////////////////////////////////////////////////////////////////////////



 
