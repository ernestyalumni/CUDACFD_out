/** main.cu
 * \file main.cu
 * Navier-Stokes equation solver in 2-dimensions, incompressible flow, by finite difference
 * \author Ernest Yeung  
 * \email ernestyalumni@gmail.com
 * \date 20161206
 * 
 * Compilation tips if you're not using a make file
 * 
 * nvcc -std=c++11 -c ./physlib/R2grid.cpp -o R2grid.o  // or 
 * g++ -std=c++11 -c ./physlib/R2grid.cpp -o R2grid.o
 * 
 * nvcc -std=c++11 -c ./physlib/dev_R2grid.cu -o dev_R2grid.o
 * nvcc -std=c++11 main.cu R2grid.o dev_R2grid.o o main.exe
 * 
 */
/*
 * cf. Kyle e. Niemeyer, Chih-Jen Sung.  
 * Accelerating reactive-flow simulations using graphics processing units.  
 * AIAA 2013-0371  American Institute of Aeronautics and Astronautics.  
 * http://dx.doi.org/10.5281/zenodo.44333
 * 
 * Michael Griebel, Thomas Dornsheifer, Tilman Neunhoeffer. 
 * Numerical Simulation in Fluid Dynamics: A Practical Introduction (Monographs on Mathematical Modeling and Computation). 
 * SIAM: Society for Industrial and Applied Mathematics (December 1997). 
 * ISBN-13:978-0898713985 QA911.G718 1997
 * 
 * */ 

#include <iomanip>					// std::setprecision
#include <iostream> 				// std::cout
#include <cmath>    				// std::sqrt, std::fmax , std::fmin

#include "./physlib/R2grid.h"      	// Grid2d
#include "./physlib/dev_R2grid.h"  	// Dev_Grid2d
#include "./physlib/uvp.h"          // compute_F, compute_G, etc.
#include "./physlib/boundary.h"     // set_BCs_host, set_BCs
#include "./commonlib/checkerror.h" // checkCudaErrors

#include "./commonlib/tex_anim2d.h" // GPUAnim2dTex

#define GL_GLEXT_PROTOTYPES // needed for identifier glGenBuffer, glBindBuffer, glBufferData, glDeleteBuffers

#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h> // or #include "cuda_gl_interop.h"

#include <array>  				// std::array
#include <vector>				// std::vector
#include <functional>



int main( int argc, char *argv[] ) {

	// ################################################################
	// ####################### Initialization #########################
	// ######################## of global-scope variables and objects #
	// ################################################################

	// ################################################################
	// ####################### Initialization #########################
	// ################################################################

	// discretization (parameters) <==> graphical (parameters)
	const int L_X { 256 };  			// WIDTH   // I've tried values 32.  128, 32, 0.5 works; 256, 32, 0.25 works (even though 256, 64 doesn't); 512, 64, doesn't work, neither does 512,32; 512, 16 works
	const int L_Y { 256 };  			// HEIGHT  // I've tried values 32,  128, 32, 0.5 works

	// "real", physical parameters
	/** try domain size (non-dimensional) */
	constexpr const float l_X = 1.0;  	// length (in x-direction)
	constexpr const float l_Y = 1.0; 	// height (in y-direction)

	// physics (on device); Euclidean (spatial) space
	dim3 dev_L2 { static_cast<unsigned int>(L_X), 
					static_cast<unsigned int>(L_Y) };

	Dev_Grid2d dev_grid2d( dev_L2); 

	// physics (on host); Euclidean (spatial) space
	constexpr std::array<int,2> LdS { L_X, L_Y } ;
	constexpr std::array<float,2> ldS { l_X, l_Y };

	//Grid2d grid2d{LdS, ldS};
	Grid2d grid2d(LdS, ldS);

	// dynamics (parameters)
	const dim3 M_i { 32, 32 }; 	// number of threads per block, i.e. Niemeyer's BLOCK_SIZE // I've tried values 4,4

	float t = 0.0 ;
	int cycle = 0;
	
	// iterations for SOR successive over relaxation
	int iter = 0;
	int itermax = 20000;  // I tried values such as 10000, Griebel, et. al. = 100

	/* READ the parameters of the problem                 */
	/* -------------------------------------------------- */ 

	/** Safety factor for time step modification; safety factor for time stepsize control */
	constexpr const float tau = 0.5; 

	/** SOR relaxation parameter; omg is Griebel's notation */
	constexpr const float omega = 1.7;  
	
	/** Discretization mixture parameter (gamma); gamma:upwind differencing factor is Griebel's notation */
	constexpr const float gamma_mix_param = 0.9;

	/** Reynolds number */
	constexpr const float Re_num = 1000.0;

	// SOR iteration tolerance
	const float tol = 0.001;  // Griebel, et. al., and Niemeyer has this at 0.001
	
	// time range
	const float time_start = 0.0;
	const float time_end = 0.01;  // L_X=L_Y=128, M_i=32, t_f=0.5 works
	
	// initial time step size
	float deltat = 0.02; // I've tried values 0.002

	////////////////////////////////////////	
	// block and grid dimensions

	// boundary conditions kernel
	dim3 block_bcs (M_i.x, 1);
	dim3 grid_bcs (grid2d.Ld[0] / M_i.x, 1);

	// pressure kernel
	dim3 block_pr (M_i.y, 1);
	dim3 grid_pr (grid2d.Ld[1] / (2 * M_i.y), grid2d.Ld[0]);

	// block and grid dimensions for F
	dim3 block_F (M_i.y, 1);
	dim3 grid_F (grid2d.Ld[1] / M_i.y, grid2d.Ld[0]);

	// block and grid dimensions for G
	dim3 block_G (M_i.y, 1);
	dim3 grid_G (grid2d.Ld[1] / M_i.y, grid2d.Ld[0]);
	
	// horizontal pressure boundary conditions
	dim3 block_hpbc (M_i.x, 1);
	dim3 grid_hpbc (grid2d.Ld[0] / (2 * M_i.x), 1);

	// vertical pressure boundary conditions
	dim3 block_vpbc (M_i.y, 1);
	dim3 grid_vpbc (grid2d.Ld[1] / (2 * M_i.y), 1);

	// "internal" cells only, so-called fluid cells from Griebel, et. al.
	dim3 inter_gridSize ( (grid2d.Ld[0] + M_i.x-1)/M_i.x , 
							(grid2d.Ld[1]+M_i.y-1)/M_i.y ) ;
	////////////////////////////////////////

	// residual variable
	//constexpr const int size_res = grid_pr.x * grid_pr.y;  doesn't work in std::array, expression must have a constant value
	constexpr const int size_res = L_Y/2 * L_X;
	std::array<float, size_res> res_array;

	/* delt satisfying CFL conditions */
	/* ------------------------------ */
	float max_u = 1.0e-10;
	float max_v = 1.0e-10;

	// variables to store maximum velocities
	constexpr const int size_max = L_Y/2 * L_X ;
	std::array<float, size_max> max_u_array;
	std::array<float, size_max> max_v_array;

	// pressure sum
	std::array<float, size_res> pres_sum_array;



	// time-step size based on grid and Reynolds number
	float dt_Re = 0.5 * Re_num / ((1.0 / (grid2d.hd[0] * grid2d.hd[0])) + (1.0 / (grid2d.hd[1] * grid2d.hd[1])));




	// set initial BCs
	set_BCs_host (grid2d.u , grid2d.v, grid2d.Ld[0], grid2d.Ld[1]);

	// get max velocity for initial values (including BCs)
	#pragma unroll
	for (int col = 0; col < grid2d.Ld[0] + 2; ++col) {
		#pragma unroll
		for (int row = 1; row < grid2d.Ld[1] + 2; ++row) {
			max_u = fmax(max_u, fabs( grid2d.u[flatten(col, row, grid2d.Ld[1]+2)] ));
		}
	}

	#pragma unroll
	for (int col = 1; col < grid2d.Ld[0] + 2; ++col) {
		#pragma unroll
		for (int row = 0; row < grid2d.Ld[1] + 2; ++row) {
			max_v = fmax(max_v, fabs( grid2d.v[ flatten(col, row, grid2d.Ld[1] + 2) ] ));
		}
	}

	////////////////////////////////////////
	// allocate and transfer device memory
	float* pres_sum_d;
	float* res_d;

	float* max_u_d;
	float* max_v_d;

	cudaMalloc ((void**) &pres_sum_d, size_res * sizeof(float));
	cudaMalloc ((void**) &res_d, size_res * sizeof(float));
	cudaMalloc ((void**) &max_u_d, size_max * sizeof(float));
	cudaMalloc ((void**) &max_v_d, size_max * sizeof(float));
	
	// copy to device memory
	checkCudaErrors(
		cudaMemcpy (dev_grid2d.u, grid2d.u, grid2d.staggered_NFLAT() * sizeof(float), cudaMemcpyHostToDevice) );
	cudaMemcpy (dev_grid2d.F, grid2d.F, grid2d.staggered_NFLAT() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy (dev_grid2d.v, grid2d.v, grid2d.staggered_NFLAT() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy (dev_grid2d.G, grid2d.G, grid2d.staggered_NFLAT() * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErrors(
		cudaMemcpy (dev_grid2d.pres_red, grid2d.pres_red, 
			(grid2d.Ld[0]/2+2)*(grid2d.Ld[1]+2) * sizeof(float), cudaMemcpyHostToDevice) );
	cudaMemcpy (dev_grid2d.pres_black, grid2d.pres_black, (grid2d.Ld[0]/2+2 )*(grid2d.Ld[1]+2)* sizeof(float), cudaMemcpyHostToDevice);
	////////////////////////////////////////

	
	
	// ################################################################
	// #######################               ##########################
	// #######################   MAIN LOOP   ##########################
	// #######################               ##########################
	// ################################################################
	
				/* t i m e    l o o p */
				/* ------------------ */
				/* time iteration loop      */
	for (t=time_start, cycle = 0; t< time_end; cycle++) {
		
		// calculate time step based on stability and CFL
		deltat = std::fmin((grid2d.hd[0] / max_u), (grid2d.hd[1] / max_v));
		deltat = tau * std::fmin(dt_Re, deltat);

		if ((t + deltat)>= time_end) {
			deltat = time_end - t;
		}

		// calculate F and G		
		calculate_F <<<grid_F, block_F>>> ( dev_grid2d.u, dev_grid2d.v, dev_grid2d.F, 
			gamma_mix_param, Re_num, 0.0,
			deltat, grid2d.hd[0], grid2d.hd[1], grid2d.Ld[0], grid2d.Ld[1], M_i.x);
		calculate_G <<<grid_G, block_G>>> (dev_grid2d.u, dev_grid2d.v, dev_grid2d.G, 
			gamma_mix_param, Re_num, 0.0,
			deltat, grid2d.hd[0], grid2d.hd[1], grid2d.Ld[0], grid2d.Ld[1], M_i.x);


		// get L2 norm of initial pressure
		sum_pressure <<<grid_pr, block_pr, M_i.x * sizeof(float) >>> (dev_grid2d.pres_red, dev_grid2d.pres_black, pres_sum_d, 
				grid2d.Ld[0], grid2d.Ld[1], M_i.x);
		checkCudaErrors(
			cudaMemcpy (
				pres_sum_array.data(), pres_sum_d, size_res * sizeof(float), cudaMemcpyDeviceToHost) );

		float p0_norm = 0.0;
		#pragma unroll
		for (int i = 0; i < size_res; ++i) {
			p0_norm += pres_sum_array[i];
		}
		
		p0_norm = sqrt(p0_norm / ((float)(grid2d.NFLAT())));
		if (p0_norm < 0.0001) {
		   p0_norm = 1.0;
		}

		// ensure all kernels are finished
		cudaDeviceSynchronize();

		float norm_L2;
		
		// calculate new pressure
		// red-black Gauss-Seidel with SOR iteration loop
		for (iter = 1; iter <= itermax; ++iter) {

			// set pressure boundary conditions
			set_horz_pres_BCs <<<grid_hpbc, block_hpbc>>> (dev_grid2d.pres_red, dev_grid2d.pres_black,
				grid2d.Ld[0], grid2d.Ld[1] );
			set_vert_pres_BCs <<<grid_vpbc, block_hpbc>>> (dev_grid2d.pres_red, dev_grid2d.pres_black,
				grid2d.Ld[0], grid2d.Ld[1] );

			// ensure kernel finished
			cudaDeviceSynchronize();

			// update red cells
			red_kernel <<<grid_pr, block_pr>>> (dev_grid2d.F, dev_grid2d.G, 
				dev_grid2d.pres_black, dev_grid2d.pres_red, 
				omega,
				deltat, grid2d.hd[0], grid2d.hd[1], grid2d.Ld[0], grid2d.Ld[1], M_i.x);

			// ensure red kernel finished
			cudaDeviceSynchronize();
			
			// update black cells
			black_kernel <<<grid_pr, block_pr>>> (dev_grid2d.F, dev_grid2d.G, 
				dev_grid2d.pres_red, dev_grid2d.pres_black, 
				omega,
				deltat, grid2d.hd[0], grid2d.hd[1], grid2d.Ld[0], grid2d.Ld[1], M_i.x);

			// ensure red kernel finished
			cudaDeviceSynchronize();
			
			// calculate residual values
			calc_residual <<<grid_pr, block_pr, M_i.x * sizeof(float) >>> (dev_grid2d.F, dev_grid2d.G, 
				dev_grid2d.pres_red, dev_grid2d.pres_black, res_d,
				deltat, grid2d.hd[0], grid2d.hd[1], grid2d.Ld[0], grid2d.Ld[1], M_i.x);

			// transfer residual value(s) back to CPU
			checkCudaErrors(
				cudaMemcpy (res_array.data(), res_d, size_res * sizeof(float), cudaMemcpyDeviceToHost) );

			norm_L2 = 0.0;
			#pragma unroll
			for (int i = 0; i < size_res; ++i) {
				norm_L2 += res_array[i];
			}
			
			// calculate residual
			norm_L2 = sqrt(norm_L2 / ((float)( grid2d.NFLAT() ))) / p0_norm;
			
			// if tolerance has been reached, end SOR iterations
			if (norm_L2 < tol) {
				break;
			}	
		} // end for loop for SOR iteration

		std::cout << "Time = " << t+deltat << ", deltat = " << deltat << ", iter = " <<
			iter << ", res (i.e. norm_L2) = " << std::setprecision(4) << norm_L2 << 
			", max_u : " << max_u << ", max_v : " << max_v << ", p0_norm : " << p0_norm << std::endl;
		
		// calculate new velocities and transfer maximums back

		calculate_u <<<grid_pr, block_pr, M_i.x * sizeof(float) >>> ( dev_grid2d.F, dev_grid2d.pres_red, dev_grid2d.pres_black, 
			dev_grid2d.u, max_u_d,
			deltat, grid2d.hd[0], grid2d.hd[1], grid2d.Ld[0], grid2d.Ld[1], M_i.x);
		
		checkCudaErrors(
			cudaMemcpy (max_u_array.data(), max_u_d, size_max * sizeof(float), cudaMemcpyDeviceToHost) );

		calculate_v <<<grid_pr, block_pr, M_i.x * sizeof(float)>>> (dev_grid2d.G, dev_grid2d.pres_red, dev_grid2d.pres_black, 
			dev_grid2d.v, max_v_d,
			deltat, grid2d.hd[0], grid2d.hd[1], grid2d.Ld[0], grid2d.Ld[1], M_i.x);

		checkCudaErrors( 
			cudaMemcpy (max_v_array.data(), max_v_d, size_max * sizeof(float), cudaMemcpyDeviceToHost) );
	
		// get maximum u- and v- velocities
		max_v = 1.0e-10;
		max_u = 1.0e-10;

		#pragma unroll
		for (int i = 0; i < size_max; ++i) {
			float test_u = max_u_array[i];
			max_u = std::fmax(max_u, test_u);

			float test_v = max_v_array[i];
			max_v = std::fmax(max_v, test_v);
		}

		// set velocity boundary conditions
		set_BCs <<<grid_bcs, block_bcs>>> (dev_grid2d.u, dev_grid2d.v, grid2d.Ld[0], grid2d.Ld[1]);

		cudaDeviceSynchronize();


		// increase time
		t += deltat;

		

	} // for time loop, END

	
	
	
	
	// free device memory
	checkCudaErrors( cudaFree( dev_grid2d.u )); 
	checkCudaErrors( cudaFree( dev_grid2d.v )); 
	checkCudaErrors( cudaFree( dev_grid2d.F )); 
	checkCudaErrors( cudaFree( dev_grid2d.G )); 
	
	checkCudaErrors( cudaFree( dev_grid2d.pres_red )); 
	checkCudaErrors( cudaFree( dev_grid2d.pres_black )); 

	checkCudaErrors( cudaFree( max_u_d )); 
	checkCudaErrors( cudaFree( max_v_d )); 
	checkCudaErrors( cudaFree( pres_sum_d )); 
	checkCudaErrors( cudaFree( res_d )); 

	cudaDeviceReset();
	return 0;
} // END of main
