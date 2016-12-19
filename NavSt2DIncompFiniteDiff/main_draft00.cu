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

	// sanity check
	thrust::device_vector<float>::iterator max_u0_iter = 
		thrust::max_element( dev_grid2d.u.begin(), dev_grid2d.u.end() );
	float max_u0 = std::fmax( *max_u0_iter, static_cast<float>(0.0) ) ;

	thrust::device_vector<float>::iterator max_v0_iter = 
		thrust::max_element( dev_grid2d.v.begin(), dev_grid2d.v.end() );
	float max_v0 = std::fmax( *max_v0_iter, static_cast<float>(0.0) ) ;
	
	std::cout << " max_u0 : " << max_u0 << " max_v0 : " << max_v0 << std::endl;
	
	set_lidcavity_BConditions( dev_grid2d );

	/* delt satisfying CFL conditions */
	/* ------------------------------ */

	float max_u = 1.0e-10;
	float max_v = 1.0e-10;
	// get max velocity for initial values (including BCs)
	// cf. read Farzad's answer in http://stackoverflow.com/questions/22278631/using-pragma-unroll-in-cuda
/*	#pragma unroll
	for (auto j=1; j <=(grid2d.Ld[1]+1); ++j) {
		#pragma unroll
		for (auto i=0; i<=(grid2d.Ld[1]+1); ++i) {
			max_u = std::fmax( max_u, std::abs( grid2d.u[ grid2d.staggered_flatten(i,j) ] ) );
		}
	}

	#pragma unroll
	for (auto j=0; j <=(grid2d.Ld[1]+1); ++j) {
		#pragma unroll
		for (auto i=1; i<=(grid2d.Ld[1]+1); ++i) {
			max_v = std::fmax( max_v, std::abs( grid2d.v[ grid2d.staggered_flatten(i,j) ] ) );
		}
	}
*/
	

	// sanity check
	thrust::device_vector<float>::iterator max_u_iter = 
		thrust::max_element( dev_grid2d.u.begin(), dev_grid2d.u.end() );
	max_u = std::fmax( *max_u_iter, max_u ) ;

	thrust::device_vector<float>::iterator max_v_iter = 
		thrust::max_element( dev_grid2d.v.begin(), dev_grid2d.v.end() );
	max_v = std::fmax( *max_v_iter, max_v ) ;
	
	std::cout << " This is max_u : " << max_u << std::endl;
	std::cout << " This is max_v : " << max_v << std::endl;



	
	std::cout << " End of program " << std::endl;
	return 0;
} 

