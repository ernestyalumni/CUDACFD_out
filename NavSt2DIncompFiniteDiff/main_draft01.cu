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
 * nvcc -std=c++11 -c ./commonlib/surfObj2d.cu -o surfObj2d.o
 * nvcc -std=c++11 main.cu R2grid.o dev_R2grid.o surfObj2d.o -o main.exe
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
 
#include <iostream> 				// std::cout
#include <cmath>    				// std::sqrt, std::fmax 

#include "./physlib/R2grid.h"      	// Grid2d
#include "./physlib/dev_R2grid.h"  	// Dev_Grid2d
#include "./physlib/boundary.h"     // set_BConditions_host, set_BConditions, set_lidcavity_BConditions_host, set_lidcavity_BConditions
#include "./commonlib/checkerror.h" // checkCudaErrors

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>  // thrust::max_element (thrust::min_element)  


int main(int argc, char* argv[]) {
	// ################################################################
	// ####################### Initialization #########################
	// ################################################################
	
	// discretization (parameters) <==> graphical (parameters)
	const int L_X { 64 };  			// WIDTH   // I've tried values 32
	const int L_Y { 64 };  			// HEIGHT  // I've tried values 32

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
	const dim3 M_i { 16, 16 }; 	// number of threads per block, i.e. Niemeyer's BLOCK_SIZE // I've tried values 4,4

	float t = 0.0 ;
	int cycle = 0;
	
	// iterations for SOR successive over relaxation
	int iter = 0;
	int itermax = 10000;  // I tried values such as 10000, 100

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
	const float time_end = 0.01;
	
	// initial time step size
	float deltat = 0.002; // I've tried values 0.002
	
	// set initial BCs on host CPU
	set_BConditions_host( grid2d );
	set_lidcavity_BConditions_host( grid2d );

	set_BConditions( dev_grid2d );
	set_lidcavity_BConditions( dev_grid2d );

	/* delt satisfying CFL conditions */
	/* ------------------------------ */
	float max_u = 1.0e-10;
	float max_v = 1.0e-10;

	thrust::device_vector<float>::iterator max_u_iter = 
		thrust::max_element( dev_grid2d.u.begin(), dev_grid2d.u.end() );
	max_u = std::fmax( *max_u_iter, max_u ) ;

	thrust::device_vector<float>::iterator max_v_iter = 
		thrust::max_element( dev_grid2d.v.begin(), dev_grid2d.v.end() );
	max_v = std::fmax( *max_v_iter, max_v ) ;
	

	////////////////////////////////////////	
	// block and grid dimensions
	// "default" blockSize is number of blocks on a grid along a dimension
	dim3 blockSize ( (grid2d.staggered_Ld[0] + M_i.x -1)/M_i.x, 
						(grid2d.staggered_Ld[1] + M_i.y - 1)/M_i.y) ;
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
	thrust::device_vector<float> pres_sum_vec(grid2d.staggered_SIZE());
	float* pres_sum_Arr = thrust::raw_pointer_cast( pres_sum_vec.data() );
	
	// sanity check
	std::cout << " residualsq.size() : " << residualsq.size() << std::endl;
	std::cout << " pres_sum_vec.size() : " << pres_sum_vec.size() << std::endl;
	
	

	// time-step size based on grid and Reynolds number
	float dt_Re = 0.5 * Re_num / ((1.0 / (grid2d.hd[0] * grid2d.hd[0])) + (1.0 / (grid2d.hd[1] * grid2d.hd[1])));
	
				/* t i m e    l o o p */
				/* ------------------ */
	// time iteration loop
	for (t=time_start,cycle=0; t < time_end; cycle++) {

		// calculate time step based on stability and CFL
		deltat = std::fmin( (grid2d.hd[0] / max_u), ( grid2d.hd[1]/ max_v) );
		deltat = tau * std::fmin( dt_Re, deltat);
	
		if ((t+deltat) >= time_end) {
			deltat = time_end - t; }
	

		t += deltat;

	}


	
	std::cout << " End of program " << std::endl;
	return 0;
} 

