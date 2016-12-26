/** u_p.cu
 * \file u_p.cu
 * \brief computation of u, p for 2-dim. incompressible Navier-Stokes equation with finite difference
 * \author Ernest Yeung  
 * \email ernestyalumni@gmail.com
 * \date 20161209
 * 
 * compilation tip: (compile separately)
 * nvcc -std=c++11 -c ./physlib/u_p.cu -o u_p.o
 * 
 */

/*------------------------------------------------------------------- */
/* Computation of tentative velocity field (F,G) -------------------- */
/*------------------------------------------------------------------- */

__global__ void compute_F(const float deltat, 
	const float* u, const float* v, float* F,
	const int imax, const int jmax, const float deltax, const float deltay,
	const float gamma, const float Re) {
	
	const int i = threadIdx.x  + (blockDim.x*blockIdx.x) ; // be forewarned; Niemeyer has this "reversed" as the row as indexed differently as (blockIdx.x * blockDim.x)+threadIdx.x + 1
	const int j = threadIdx.y  + (blockDim.y*blockIdx.y) ; // be forewarned; Niemeyer has this "reversed" as the col as indexed differently as (blockIdx.y * blockDim.y)+threadIdx.y + 1
	if ((i > (imax+1)) || (j > (jmax+1))) { return; }

	if ((i>0)&&(i<=(imax-1))&&(j>0)&&(j<=jmax)) {
	// velocities u 
	float u_ij   = u[i + (imax+2)*j];
	float u_ip1j = u[i+1 + (imax+2)*j];
	float u_im1j = u[i-1 + (imax+2)*j];
	float u_ijp1 = u[i + (imax+2)*(j+1)];
	float u_ijm1 = u[i + (imax+2)*(j-1)];

	// velocities v 
	float v_ij     = v[i + (imax+2)*j];
	float v_ip1j   = v[i+1 + (imax+2)*j];
	float v_ijm1   = v[i + (imax+2)*(j-1)];
	float v_ip1jm1 = v[i+1 + (imax+2)*(j-1)];

	// finite differences
	float du2dx, duvdy;

	du2dx = ( ((u_ij + u_ip1j) * (u_ij + u_ip1j) - (u_im1j + u_ij) * (u_im1j + u_ij) ) + 
				gamma * ( fabs( u_ij + u_ip1j) * (u_ij - u_ip1j) - 
							fabs( u_im1j + u_ij) * (u_im1j - u_ij) ) )/(4.0*deltax) ; 
							
	duvdy = ( ((v_ij + v_ip1j)*(u_ij + u_ijp1) - (v_ijm1 + v_ip1jm1)*(u_ijm1 + u_ij) ) + 
				gamma * ( fabs( v_ij + v_ip1j ) * (u_ij - u_ijp1) - 
							fabs( v_ijm1 + v_ip1jm1 ) * ( u_ijm1 - u_ij ) ))/ (4.0 * deltay) ;
		
	// calculate the Laplacian, d^2udx^2 + d^2vdy^2, Laplacianu, i.e. Niemeyer's d2udx2, d2udy2
	float Laplacianu = (u_ip1j - 2.0*u_ij + u_im1j)/deltax/deltax + 
						(u_ijp1 - 2.0*u_ij + u_ijm1)/deltay/deltay ;
	
	float temp_F = u_ij + deltat * (Laplacianu/Re - du2dx - duvdy );
	F[i + (imax+2)*j] = temp_F;
	}

	/* F at external boundary  */
	/* ----------------------- */
	// eastern boundary or right boundary, F_ij = u_ij
	if (i == imax) {
		if ((j>=1)&&(j<=jmax)) {
			float u_imaxj = u[ imax + (imax+2)*j] ; 
			F[imax+(imax+2)*j] = u_imaxj ; } 
	}		
			
	// western boundary or left boundary
	if (i == 0) {
		if ((j>=1)&&(j<=jmax)) {
			float u_0j = u[ 0 + (imax+2)*j] ; 
			F[0 + (imax+2)*j] = u_0j ; }
	}	
} 	// end of compute_F

__global__ void compute_G(const float deltat, 
	const float* u, const float* v, float* G,
	const int imax, const int jmax, const float deltax, const float deltay,
	const float gamma, const float Re) {

	const int i = threadIdx.x  + (blockDim.x*blockIdx.x) ; // be forewarned; Niemeyer has this "reversed" as the row as indexed differently as (blockIdx.x * blockDim.x)+threadIdx.x + 1
	const int j = threadIdx.y  + (blockDim.y*blockIdx.y) ; // be forewarned; Niemeyer has this "reversed" as the col as indexed differently as (blockIdx.y * blockDim.y)+threadIdx.y + 1
	if ((i > (imax+1)) || (j > (jmax+1))) { return; }

	if ((i>0)&&(i<=imax)&&(j>0)&&(j<=(jmax-1)) ) {
	// velocities u 
	float u_ij   = u[i + (imax+2)*j];
	float u_im1j = u[i-1 + (imax+2)*j];
	float u_ijp1 = u[i + (imax+2)*(j+1)];
	float u_im1jp1 = u[i-1 + (imax+2)*(j+1)];

	// velocities v 
	float v_ij     = v[i + (imax+2)*j];
	float v_ip1j   = v[i+1 + (imax+2)*j];
	float v_im1j   = v[i-1 + (imax+2)*j];
	float v_ijp1 = v[i + (imax+2)*(j+1)];
	float v_ijm1   = v[i + (imax+2)*(j-1)];

	float duvdx = (( (u_ij + u_ijp1)*(v_ij + v_ip1j) - (u_im1j + u_im1jp1)*(v_im1j+v_ij)) 
					+ gamma * ( fabs(u_ij   + u_ijp1)   * (v_ij   - v_ip1j) - 
								fabs(u_im1j + u_im1jp1) * (v_im1j - v_ij) ))/(4.0*deltax);
	
	float dv2dy = (((v_ij + v_ijp1 )*( v_ij + v_ijp1 ) - ( v_ijm1 + v_ij )*( v_ijm1 + v_ij)) 
					+ gamma * ( fabs( v_ij   + v_ijp1) * (v_ij   - v_ijp1) - 
								fabs( v_ijm1 + v_ij  ) * ( v_ijm1 - v_ij) ))/(4.0*deltay);
		
	// calculate the Laplacian, d^2vdx^2 + d^2vdy^2, Laplacianu, i.e. Niemeyer's d2vdx2, d2vdy2
	float Laplacianv = (v_ip1j  -2.0*v_ij + v_im1j)/deltax/deltax + 
						(v_ijp1 -2.0*v_ij + v_ijm1)/deltay/deltay ;
	
	float temp_G  = v_ij + deltat * (Laplacianv/Re - dv2dy - duvdx );
	G[i + (imax+2)*j] = temp_G;
}

	/* G at external boundary  */
	/* ----------------------- */
	// northern boundary or top boundary
	if (j == jmax) {
		if ((i>=1)&&(i<=imax)) {
			float v_ijmax = v[ i + (imax+2)*jmax] ; 
			G[i + (imax+2)*jmax] = v_ijmax ; }
	}
			
	// southern boundary or bottom boundary
	if (j == 0) {
		if ((i>=1)&&(i<=imax)) {
			float v_i0 = v[ i + 0 ] ;
			G[i + 0 ] = v_i0; }
	}
} 	// end of compute_G

////////////////////////////////////////////////////////////////////////
// copy_press_int /brief copy interior pressure values
/*__host__ void copy_press_int( thrust::device_vector<float> p_all, 
	thrust::device_vector<float> & p_int,
	const int imax, const int jmax) {
	
	for (auto j = 0; j < (jmax+2); ++j) {
		for (auto i = 0; i < (imax+2); ++i) {
			if ((i>0)&&(i<(imax+1)) && (j>0) && (j<(jmax+1))) {
				const int k = (i-1) + imax * (j-1) ; 
				p_int[k] = p_all[ i + (imax+2)*j ] ; 
			}
		}
	}
}

void copy_press_int( const float* p_all, float* p_int,
	const int imax, const int jmax) {
	
	for (auto j = 0; j < (jmax+2); ++j) {
		for (auto i = 0; i < (imax+2); ++i) {
			if ((i>0)&&(i<(imax+1)) && (j>0) && (j<(jmax+1))) {
				const int k = (i-1) + imax * (j-1) ; 
				p_int[k] = p_all[ i + (imax+2)*j ] ; 
			}
		}
	}
}


__global__ void sum_pressure( cudaSurfaceObject_t pSurfObj, 
	const int imax, const int jmax, float* pres_sum) {

	const int k_x = threadIdx.x  + (blockDim.x*blockIdx.x) ; // be forewarned; Niemeyer has this "reversed" as the row as indexed differently as (blockIdx.x * blockDim.x)+threadIdx.x + 1
	const int k_y = threadIdx.y  + (blockDim.y*blockIdx.y) ; // be forewarned; Niemeyer has this "reversed" as the col as indexed differently as (blockIdx.y * blockDim.y)+threadIdx.y + 1
	const int k   = k_x + (imax+2)*k_y; // take note of the "striding" here, or i.e. choice of "stride" as imax+2 here

	if ((k_x > (imax+1)) || (k_y > (jmax+1))) { return; }

	float temp_val = 0.0; // temporary value
	float psq    = 0.0; // residual, squared

	if ((k_x >= 1)&&(k_x<=imax)) {
		if ((k_y >=1)&&(k_y<=jmax)) {
			surf2Dread(&temp_val, pSurfObj, k_x*4,    k_y);
			psq = temp_val * temp_val;			
			pres_sum[k] = psq; }
	}

	if ((k_x == 0) || (k_y == 0) || (k_x == (imax+1)) || (k_y == (jmax+1))  ) {
			pres_sum[k] = 0.0 ; }
} 
*/


/*------------------------------------------------------------------- */
/* Computation of the right hand side of the pressure equation ------ */
/* it's the "RHS" of the Poisson equation involving pressure p        */
/*------------------------------------------------------------------- */

__global__ void compute_RHS( const float* F, const float* G, 
	float* RHS, 
	const int imax, const int jmax, 
	const float deltat, const float deltax, const float deltay)  {

	// "striding" needed to "flatten" (i,j) indices 
	const int Nx = imax+2;
	
	const int i = threadIdx.x  + (blockDim.x*blockIdx.x) ; // be forewarned; Niemeyer has this "reversed" as the row as indexed differently as (blockIdx.x * blockDim.x)+threadIdx.x + 1
	const int j = threadIdx.y  + (blockDim.y*blockIdx.y) ; // be forewarned; Niemeyer has this "reversed" as the col as indexed differently as (blockIdx.y * blockDim.y)+threadIdx.y + 1
	if ((i > (imax+1)) || (j > (jmax+1))) { return; }

	if ((i >=1)&&(i<=imax)) { 
		if ((j >=1)&&(j<=jmax)) {
			float F_ij   = F[ i + Nx * j] ; 
			float F_im1j = F[ i-1 + Nx * j] ;
			float G_ij   = G[ i + Nx * j] ; 
			float G_ijm1 = G[ i + Nx * (j-1)] ;
			
			float temp_RHS_val = ((F_ij-F_im1j)/deltax + (G_ij-G_ijm1)/deltay)/deltat;
			RHS[ i + Nx * j ] = temp_RHS_val; 
		}
	}
}

/*------------------------------------------------------------------- */
/* SOR iteration for the Poisson equation for the pressure
/*------------------------------------------------------------------- */

__global__ void poisson( const float* p, const float* RHS, 
	float* p_temp, 
	const int imax, const int jmax,
	const float deltax, const float deltay, 
	const float omega) {
	
	// "striding" needed to "flatten" (i,j) indices
	const int Nx = imax+2;
	
	const int i = threadIdx.x  + (blockDim.x*blockIdx.x) ; // be forewarned; Niemeyer has this "reversed" as the row as indexed differently as (blockIdx.x * blockDim.x)+threadIdx.x + 1
	const int j = threadIdx.y  + (blockDim.y*blockIdx.y) ; // be forewarned; Niemeyer has this "reversed" as the col as indexed differently as (blockIdx.y * blockDim.y)+threadIdx.y + 1
	

	if ((i > (imax+1)) || (j > (jmax+1))) { return; }

	if ((i >= 1)&&(i <=imax)) {
		if ((j >=1)&&(  j <=jmax)) {
			float p_ij   = p[ i + Nx *j] ; 
			float p_ip1j = p[ i +1 + Nx *j] ;
			float p_im1j = p[ i -1 + Nx *j] ;
			float p_ijp1 = p[ i + Nx * (j+1) ] ; 
			float p_ijm1 = p[ i + Nx * (j-1) ] ;

			float RHS_ij = RHS[ i + Nx * j];
			
			float temp_val = (1.0-omega)*p_ij + 
				(omega/(2.0*(1./deltax/deltax+1./deltay/deltay))) * 
					( (p_ip1j + p_im1j)*(1./deltax/deltax) + 
						(p_ijp1 + p_ijm1)*(1./deltay/deltay) - 
							 RHS_ij ) ; // temp_val is RHS of poisson equation for pressure, in this case
			p_temp[ i+Nx*j] = temp_val ; }
	}
}


__global__ void poisson_redblack( float* p, const float* RHS, 
	const int imax, const int jmax,
	const float deltax, const float deltay, 
	const float omega) {
	
	// "striding" needed to "flatten" (i,j) indices
	const int Nx = imax+2;
	
	const int i = threadIdx.x  + (blockDim.x*blockIdx.x) ; // be forewarned; Niemeyer has this "reversed" as the row as indexed differently as (blockIdx.x * blockDim.x)+threadIdx.x + 1
	const int j = threadIdx.y  + (blockDim.y*blockIdx.y) ; // be forewarned; Niemeyer has this "reversed" as the col as indexed differently as (blockIdx.y * blockDim.y)+threadIdx.y + 1
	

	if ((i > (imax+1)) || (j > (jmax+1))) { return; }

	if ((i >= 1)&&(i <=imax)) {
		if ((j >=1)&&(  j <=jmax)) {
			if ( ((i+j) % 2) ==0 )  // red
				{ 
			float p_ij   = p[ i + Nx *j] ; 
			float p_ip1j = p[ i +1 + Nx *j] ;
			float p_im1j = p[ i -1 + Nx *j] ;
			float p_ijp1 = p[ i + Nx * (j+1) ] ; 
			float p_ijm1 = p[ i + Nx * (j-1) ] ;

			float RHS_ij = RHS[ i + Nx * j];
			
			float temp_val = (1.0-omega)*p_ij + 
				(omega/(2.0*(1./deltax/deltax+1./deltay/deltay))) * 
					( (p_ip1j + p_im1j)*(1./deltax/deltax) + 
						(p_ijp1 + p_ijm1)*(1./deltay/deltay) - 
							 RHS_ij ) ; // temp_val is RHS of poisson equation for pressure, in this case
			p[ i+Nx*j] = temp_val ; } }
	}
	__syncthreads();
	
	if ((i >= 1)&&(i <=imax)) {
		if ((j >=1)&&(  j <=jmax)) {
			if ( ((i+j) % 2) == 1 )  // black
				{ 
			float p_ij   = p[ i + Nx *j] ; 
			float p_ip1j = p[ i +1 + Nx *j] ;
			float p_im1j = p[ i -1 + Nx *j] ;
			float p_ijp1 = p[ i + Nx * (j+1) ] ; 
			float p_ijm1 = p[ i + Nx * (j-1) ] ;

			float RHS_ij = RHS[ i + Nx * j];
			
			float temp_val = (1.0-omega)*p_ij + 
				(omega/(2.0*(1./deltax/deltax+1./deltay/deltay))) * 
					( (p_ip1j + p_im1j)*(1./deltax/deltax) + 
						(p_ijp1 + p_ijm1)*(1./deltay/deltay) - 
							 RHS_ij ) ; // temp_val is RHS of poisson equation for pressure, in this case
			p[ i+Nx*j] = temp_val ; } }
	}	
}

/* ------------------------------------------------------------------ */
/* Computation of residual  */
/* ------------------------------------------------------------------ */
 
__global__ void compute_residual( const float* p, const float* RHS, 
	const int imax, const int jmax,
	const float deltax, const float deltay, 
	float* residualsq_Array) {

/* can't do this
	thrust::device_vector<float> & residualsq) { 	
	* cf. http://stackoverflow.com/questions/5510715/thrust-inside-user-written-kernels
	* */

	// "striding" needed to "flatten" (i,j) indices
	const int Nx = imax+2;
	
	const int i = threadIdx.x  + (blockDim.x*blockIdx.x) ; // be forewarned; Niemeyer has this "reversed" as the row as indexed differently as (blockIdx.x * blockDim.x)+threadIdx.x + 1
	const int j = threadIdx.y  + (blockDim.y*blockIdx.y) ; // be forewarned; Niemeyer has this "reversed" as the col as indexed differently as (blockIdx.y * blockDim.y)+threadIdx.y + 1
	

	if ((i > (imax+1)) || (j > (jmax+1))) { return; }

	if ((i >= 1)&&(i <=imax)) {
		if ((j >=1)&&(  j <=jmax)) {
			float p_ij   = p[ i + Nx *j] ; 
			float p_ip1j = p[ i +1 + Nx *j] ;
			float p_im1j = p[ i -1 + Nx *j] ;
			float p_ijp1 = p[ i + Nx * (j+1) ] ; 
			float p_ijm1 = p[ i + Nx * (j-1) ] ;

			float RHS_ij = RHS[ i + Nx * j ] ;

			// "temp_val" here is residual or res or "res2" for Griebel, et. al. and/or Niemeyer
			float temp_val = (((p_ip1j - p_ij)-(p_ij-p_im1j))/deltax/deltax + 
						((p_ijp1 - p_ij)-(p_ij-p_ijm1))/deltay/deltay) - RHS_ij;
			float ressq = temp_val * temp_val;			// residual, squared
			residualsq_Array[i + Nx * j] = ressq; }
	}

	if ((i == 0) || (j == 0) || (i == (imax+1)) || (j == (jmax+1))  ) {
			residualsq_Array[i + Nx *j] = 0.0 ; }
} 

/*------------------------------------------------------------------- */
/* computation of new velocity values */
/*------------------------------------------------------------------- */

__global__ void calculate_u( float* u, const float* p, const float* F, 
	const int imax, const int jmax, const float deltat, const float deltax ) {

	// "striding" needed to "flatten" (i,j) indices
	const int Nx = imax+2;
	
	const int i = threadIdx.x  + (blockDim.x*blockIdx.x) ; // be forewarned; Niemeyer has this "reversed" as the row as indexed differently as (blockIdx.x * blockDim.x)+threadIdx.x + 1
	const int j = threadIdx.y  + (blockDim.y*blockIdx.y) ; // be forewarned; Niemeyer has this "reversed" as the col as indexed differently as (blockIdx.y * blockDim.y)+threadIdx.y + 1

	if ((i > (imax+1)) || (j > (jmax+1))) { return; }

	float new_u = 0.0;
	if ((i >= 1)&&(i <=(imax-1))) {
		if ((j >=1)&&(  j <=jmax)) {
			float p_ij   = p[ i + Nx *j] ; 
			float p_ip1j = p[ i +1 + Nx *j] ;

			float F_ij = F[ i + Nx * j ] ;
			new_u = F_ij - (p_ip1j - p_ij)*deltat/deltax ; 
			u[ i + Nx *j] = new_u ; }
	}
}

__global__ void calculate_v( float* v, const float* p, const float* G, 
	const int imax, const int jmax, const float deltat, const float deltay ) {

	// "striding" needed to "flatten" (i,j) indices
	const int Nx = imax+2;
	
	const int i = threadIdx.x  + (blockDim.x*blockIdx.x) ; // be forewarned; Niemeyer has this "reversed" as the row as indexed differently as (blockIdx.x * blockDim.x)+threadIdx.x + 1
	const int j = threadIdx.y  + (blockDim.y*blockIdx.y) ; // be forewarned; Niemeyer has this "reversed" as the col as indexed differently as (blockIdx.y * blockDim.y)+threadIdx.y + 1

	if ((i > (imax+1)) || (j > (jmax+1))) { return; }

	float new_v = 0.0;
	if ((i >= 1)&&(i <=imax)) {
		if ((j >=1)&&(  j <=(jmax-1))) {
			float p_ij   = p[ i + Nx *j] ; 
			float p_ijp1 = p[ i  + Nx *(j+1)] ;

			float G_ij = G[ i + Nx * j ] ;
			new_v = G_ij - (p_ijp1 - p_ij)*deltat/deltay ; 
			v[ i + Nx *j] = new_v ; }
	}
}

/* --------------------------------------------------------- */
/* Routines to assist in the 								 */		
/* Computation of adaptive time stepsize satisfying  		 */
/* the CFL stability criteria								 */
/* and set the flag "write" if some data has to be written   */
/* into a file.												 */
/* --------------------------------------------------------- */

/*void calculate_max_uv( thrust::device_vector<float> & max_u_vec, thrust::device_vector<float> & max_v_vec, 
	const thrust::device_vector<float> u_vec, const thrust::device_vector<float> v_vec ) {
		
}
* */

	
	 


