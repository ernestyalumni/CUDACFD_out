/* main.cu
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
#include <cmath>    				// std::sqrt, std::fmax 

#include "./physlib/R2grid.h"      	// Grid2d
#include "./physlib/dev_R2grid.h"  	// Dev_Grid2d
#include "./physlib/u_p.h"          // compute_F, compute_G, compute_RHS, etc.
#include "./physlib/boundary.h"     // set_BConditions_host, set_BConditions, set_lidcavity_BConditions_host, set_lidcavity_BConditions
#include "./commonlib/checkerror.h" // checkCudaErrors

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>  // thrust::max_element (thrust::min_element)  
#include <thrust/copy.h>  // thrust::copy
#include <thrust/execution_policy.h> // thrust::device

// File I/O includes
#include <fstream> 		// std::ofstream

#include <array> 		// std::array

int main(int argc, char* argv[]) {
	// ################################################################
	// ####################### Initialization #########################
	// ################################################################
	
	// discretization (parameters) <==> graphical (parameters)
	const int L_X { 512 };  			// WIDTH   // I've tried values 32.  128, 32, 0.5 works; 256, 32, 0.25 works (even though 256, 64 doesn't); 512, 64, doesn't work, neither does 512,32; 512, 16 works
	const int L_Y { 512 };  			// HEIGHT  // I've tried values 32,  128, 32, 0.5 works

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

	Grid2d grid2d{LdS, ldS};

	// dynamics (parameters)
	const dim3 M_i { 32, 32 }; 	// number of threads per block, i.e. Niemeyer's BLOCK_SIZE // I've tried values 4,4

	float t = 0.0 ;
	int cycle = 0;
	
	// iterations for SOR successive over relaxation
	int iter = 0;
	int itermax = 10000;  // I tried values such as 10000, Griebel, et. al. = 100

	/* READ the parameters of the problem                 */
	/* -------------------------------------------------- */ 

	/** Safety factor for time step modification; safety factor for time stepsize control */
	constexpr const float tau = 0.5; 

	/** SOR relaxation parameter; omg is Griebel's notation */
	constexpr const float omega = 1.7;  
	
	/** Discretization mixture parameter (gamma); gamma:upwind differencing factor is Griebel's notation */
	constexpr const float gamma = 0.9;

	/** Reynolds number */
	constexpr const float Re_num = 1000.0;

	// SOR iteration tolerance
	const float tol = 0.001;  // Griebel, et. al., and Niemeyer has this at 0.001
	
	// time range
	const float time_start = 0.0;
	const float time_end = 1.4000;  // L_X=L_Y=128, M_i=32, t_f=0.5 works
	
	// initial time step size
	float deltat = 0.02; // I've tried values 0.002
	
	// set initial BCs on host CPU
	set_BConditions_host( grid2d );
	set_lidcavity_BConditions_host( grid2d );

	set_BConditions( dev_grid2d );
	set_lidcavity_BConditions( dev_grid2d );

	/* delt satisfying CFL conditions */
	/* ------------------------------ */
	float max_u = 1.0e-10;
	float max_v = 1.0e-10;

	// variables to store maximum velocities
	thrust::device_vector<float> max_u_vec(grid2d.NFLAT());
	thrust::device_vector<float> max_v_vec(grid2d.NFLAT());

	// get max velocity for initial values (including BCs)
	#pragma unroll
	for (auto j = 1; j < grid2d.staggered_Ld[1]; ++j) {
		#pragma unroll
		for (auto i = 0; i < grid2d.staggered_Ld[0]; ++i) {
			max_u = std::fmax( max_u, 
				fabs( dev_grid2d.u[ i + grid2d.staggered_Ld[0] * j] )  ) ;
		}
	}	
	#pragma unroll
	for (auto j = 0; j < grid2d.staggered_Ld[1]; ++j) {
		#pragma unroll
		for (auto i = 1; i < grid2d.staggered_Ld[0]; ++i) {
			max_v = std::fmax( max_v, 
				fabs( dev_grid2d.v[ i + grid2d.staggered_Ld[0] * j] )  ) ;
		}
	}	
	

// This is why you can't do (dev_grid2d->u).end()
// cf. http://stackoverflow.com/questions/13104138/error-expression-must-have-a-pointer-type-when-using-the-this-keyword
/*	thrust::device_vector<float>::iterator max_u_iter = 
		thrust::max_element( dev_grid2d.u.begin(), dev_grid2d.u.end() );
	max_u = std::fmax( *max_u_iter, max_u ) ;

	thrust::device_vector<float>::iterator max_v_iter = 
		thrust::max_element( dev_grid2d.v.begin(), dev_grid2d.v.end() );
	max_v = std::fmax( *max_v_iter, max_v ) ;
*/	

	////////////////////////////////////////	
	// block and grid dimensions
	// "default" gridSize is number of blocks on a grid along a dimension
	dim3 gridSize ( (grid2d.staggered_Ld[0] + M_i.x -1)/M_i.x, 
						(grid2d.staggered_Ld[1] + M_i.y - 1)/M_i.y) ;
	
	// pressure kernel, alternative way to launch threads in blocks, so called sPencil
/*	dim3 block_press( M_i.x, 1 ) ; 
	dim3 grid_press( ( grid2d.staggered_Ld[0] + block_press.x - 1)/block_press.x , grid2d.staggered_Ld[1] ) ;
	* */
	////////////////////////////////////////

	// residual variable
	// residualsquared thrust device vector
	thrust::device_vector<float> residualsq(grid2d.staggered_SIZE() );
	float* residualsq_Array = thrust::raw_pointer_cast( residualsq.data() );

	
	// pressure sum 
	/* Note that the pressure summation needed to normalize to the pressure magnitude for 
	 * relative tolerance is, in Griebel, et. al's implementation, the first part of the 
	 * POISSON routine, and used at the very end of POISSON, here in the GPU implementation
	 * it's separated */ 
	thrust::device_vector<float> pres_sum_vec(grid2d.NFLAT());
	float* pres_sum_Arr = thrust::raw_pointer_cast( pres_sum_vec.data() );
	

	// time-step size based on grid and Reynolds number
	float dt_Re = 0.5 * Re_num / ((1.0 / (grid2d.hd[0] * grid2d.hd[0])) + (1.0 / (grid2d.hd[1] * grid2d.hd[1])));
	
	// ################################################################
	// #######################               ##########################
	// #######################   MAIN LOOP   ##########################
	// #######################               ##########################
	// ################################################################
	
				/* t i m e    l o o p */
				/* ------------------ */
	// time iteration loop
	for (t=time_start,cycle=0; t < time_end; cycle++) {

		// calculate time step based on stability and CFL
		deltat = std::fmin( (grid2d.hd[0] / max_u), ( grid2d.hd[1]/ max_v) );
		deltat = tau * std::fmin( dt_Re, deltat);

		if ((t+deltat) >= time_end) {
			deltat = time_end - t; }
	
	
		/* Compute tentative velocity field (F,G) */
		// i.e. calculate F and G
		/* -------------------------------------- */			
		compute_F<<<gridSize,M_i>>>( deltat, 
			dev_grid2d.u_arr, dev_grid2d.v_arr, dev_grid2d.F_arr,
			grid2d.Ld[0], grid2d.Ld[1], grid2d.hd[0], grid2d.hd[1],
			gamma, Re_num); 

		compute_G<<<gridSize,M_i>>>( deltat, 
			dev_grid2d.u_arr, dev_grid2d.v_arr, dev_grid2d.G_arr,
			grid2d.Ld[0], grid2d.Ld[1], grid2d.hd[0], grid2d.hd[1],
			gamma, Re_num); 

	

	// get L2 norm of initial pressure
	for (auto j = 0; j < dev_grid2d.staggered_Ld.y; ++j) {
		for (auto i = 0; i < dev_grid2d.staggered_Ld.x; ++i) {
			if ((i>0)&&(i<(dev_grid2d.Ld.x+1)) && (j>0) && (j<(dev_grid2d.Ld.y+1))) {
				int k = (i-1) + dev_grid2d.Ld.x * (j-1) ; 
				pres_sum_vec[k] = dev_grid2d.p[ dev_grid2d.staggered_flatten(i,j) ] ; 
			}
		}
	}
	float p0_norm = 0.0;
	p0_norm = thrust::reduce( pres_sum_vec.begin(), pres_sum_vec.end(), 0, thrust::plus<float>() );
	
	p0_norm =sqrt(p0_norm / (static_cast<float>( grid2d.NFLAT() ) ));
	
	if (p0_norm < 0.0001) {
		p0_norm = 1.0;
	}
	
	// ensure all kernels are finished
	cudaDeviceSynchronize();
	
	/* Compute right hand side for pressure equation */
	/* --------------------------------------------- */
	compute_RHS<<<gridSize,M_i>>>( dev_grid2d.F_arr, dev_grid2d.G_arr,
		dev_grid2d.RHS_arr, 
		dev_grid2d.Ld.x, dev_grid2d.Ld.y, 
		deltat, grid2d.hd[0], grid2d.hd[1] );
	
	float norm_L2; // residual; res for Griebel, et. al., norm_L2 for Niemeyer
	
	/* Solve the pressure equation by successive over relaxation */
	/* ---------------------------------------------------------- */
	// calculate new pressure
	for (iter = 1; iter <= itermax; iter++) {
		// set pressure boundary conditions
		set_horiz_press_BCs( dev_grid2d ) ;
		set_vert_press_BCs( dev_grid2d ) ;
	
		// ensure kernel finished
		cudaDeviceSynchronize();
	
		// operations needed to do poisson; poisson and thrust::swap
/*
		poisson<<<gridSize, M_i>>>( dev_grid2d.p_arr, dev_grid2d.RHS_arr, 
			dev_grid2d.p_temp_arr, 
			grid2d.Ld[0], grid2d.Ld[1], grid2d.hd[0], grid2d.hd[1], omega) ; 

		thrust::copy( thrust::device, dev_grid2d.p_temp.begin(), dev_grid2d.p_temp.end(), dev_grid2d.p.begin() );
			// dev_grid2d.p is no a copy of dev_grid2d.p_temp
*/
		// END of operations needed to do poisson; poisson and thrust::swap


		
		poisson_redblack<<<gridSize, M_i>>>( dev_grid2d.p_arr, dev_grid2d.RHS_arr, 
			grid2d.Ld[0], grid2d.Ld[1], grid2d.hd[0], grid2d.hd[1], omega) ; 


		// calculate residual values
		compute_residual<<<gridSize, M_i>>>( dev_grid2d.p_arr, dev_grid2d.RHS_arr, 
			grid2d.Ld[0], grid2d.Ld[1], grid2d.hd[0], grid2d.hd[1], 
			residualsq_Array) ; 

		
		norm_L2 = thrust::reduce( residualsq.begin(), residualsq.end(), 0, thrust::plus<float>() );
		
		// calculate residual
		norm_L2 = sqrt( norm_L2/ ( static_cast<float>( grid2d.NFLAT() )) ) / p0_norm;


		// if tolerance has been reached, end SOR iterations
		if (norm_L2 < tol) {
			break;
		}				
	} // END for loop, to solve the pressure equation by SOR

	std::cout << "Time = " << t + deltat << ", delta t = " << deltat << ", iter = " 
		<< iter << 	", res (or norm_L2) = " << std::setprecision(9) << norm_L2 << 
		", max_u : " << max_u << ", max_v : " << max_v << ", p0_norm : " << p0_norm << 
		", cycle = " << cycle << std::endl; 

		/* Compute the new velocity field */
		// i.e. calculate new velocities
		/* ------------------------------ */

		calculate_u<<<gridSize,M_i>>>( dev_grid2d.u_arr, dev_grid2d.p_arr,
			dev_grid2d.F_arr, grid2d.Ld[0], grid2d.Ld[1], deltat, grid2d.hd[0] );

		calculate_v<<<gridSize,M_i>>>( dev_grid2d.v_arr, dev_grid2d.p_arr,
			dev_grid2d.G_arr, grid2d.Ld[0], grid2d.Ld[1], deltat, grid2d.hd[1] );


		// get maximum u- and v- velocities
		max_v = 1.0e-10;
		max_u = 1.0e-10;
	
		for (auto j = 0 ; j < grid2d.Ld[1] ; ++j) { 
			for (auto i = 0; i < grid2d.Ld[0]; ++i) {
				max_u_vec[i + grid2d.Ld[0]*j] = dev_grid2d.u[ (i+1) + grid2d.staggered_Ld[0] * (j+1) ] ;  
				max_v_vec[i + grid2d.Ld[0]*j] = dev_grid2d.v[ (i+1) + grid2d.staggered_Ld[0] * (j+1) ] ;
			}
		}

		thrust::device_vector<float>::iterator max_u_iter = 
			thrust::max_element( max_u_vec.begin(), max_u_vec.end() );
		max_u = std::fmax( *max_u_iter, max_u);

		thrust::device_vector<float>::iterator max_v_iter = 
			thrust::max_element( max_v_vec.begin(), max_v_vec.end() );
		max_v = std::fmax( *max_v_iter, max_v);



		// set velocity boundary conditions
		/* Set boundary conditions */
		/* ----------------------- */
		set_BConditions( dev_grid2d ) ;
		
		/* Set special boundary conditions */
		/* Overwrite preset default values */
		/* ------------------------------- */
		set_lidcavity_BConditions( dev_grid2d  );

		cudaDeviceSynchronize();

		// increase time
		t += deltat;

	} // END end for loop, time iteration loop 

	// sanity check
	std::cout << " Right after time iteration loop, final p : " << std::endl;
	std::cout << "\n dev_grid2d.p : " << std::endl; 
//		for (auto j = (grid2d.staggered_Ld[1]-1); j >= 0; --j) {
//			for (auto i = 0; i < grid2d.staggered_Ld[0]; ++i) {
		for (auto j = (grid2d.staggered_Ld[1]-1)/2; j >= 0; --j) {
			for (auto i = 0; i < grid2d.staggered_Ld[0]/2; ++i) {
				std::cout << std::setprecision(3) << dev_grid2d.p[i+(grid2d.staggered_Ld[0])*j] << " " ; }
			std::cout << std::endl ; }
	// END of sanity check

	// ################################################################
	// ########################## File I/O ############################
	// ################################################################

	std::ofstream ofile; // no file mode is set
	ofile.open("velocity_gpu.dat"); // mode implicitly out and trunc

	ofile << "#x\ty\tu\tv\n" ;
	
	std::array<int,2> ix_ind;
	std::array<float,2> x_ind;
	
	for (auto j = 0; j <= grid2d.Ld[1]; j++) {
		for (auto i = 0; i <= grid2d.Ld[0]; i++) {
			float u_ij = dev_grid2d.u[ i + grid2d.staggered_Ld[0] * j ] ; 
			float v_ij = dev_grid2d.v[ i + grid2d.staggered_Ld[0] * j ] ; 
			
			ix_ind[0] = i; 
			ix_ind[1] = j; 
			x_ind = grid2d.gridpt_to_space( ix_ind ) ;
			
			ofile << x_ind[0] << "\t" << x_ind[1] << "\t" << u_ij << "\t" << v_ij << std::endl;
		}
	}
	ofile.close();

	
	std::cout << " End of program " << std::endl;
	return 0;
} 

