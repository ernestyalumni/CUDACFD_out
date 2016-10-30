/* main.cu
 * 1-dim. MacCormack method
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161026
 */
#include <iostream>

#include <cmath>  // std::sin
#include <cstdio> // FILE fopen
#include <vector> // std::vector

#include "./physlib/dev_R1grid.h"
#include "./physlib/R1grid.h"

#include "./commonlib/errors.h"
#include "./commonlib/kernelLaunch.h"

// device sanity checks
__global__ void sanity_checks_constants() {
	// dev_sigmaconst is the so-called Courant number, i.e. CFL (Courant Friedrichs-Lewy) number,
	// on the device GPU
	printf(" dev_sigmaconst : %2.8f \n", dev_sigmaconst[0] ) ; 
}

// host error checks
// error_check_1d : error check 1-dimensional
float error_check_1d(float *f, float *f_ref, const int L) {
	float error_val { 0.f }; 
	for (auto i = 0; i < L; ++i) {
		error_val += (f[i]-f_ref[i])*(f[i]-f_ref[i]) ; 
	}
	return error_val;
}

void traveling_sine( std::vector<float>& f, Grid1d grid1d_in, float t_now ) {
	const float PI_CONST { std::acos(-1.f) };

	std::array<int,1> ix_in { 0 }; // i index variable for initial values
	std::array<float,1> x_init{  0.f  }; // x variable for initial values
	for (auto i = 0; i < grid1d_in.NFLAT(); ++i) {
		ix_in[0] = i;
		x_init = grid1d_in.gridpt_to_space( ix_in ) ;
		if ((x_init[0] >= t_now) && ( x_init[0] <= (t_now + 1.f) )) {
			f[i] = std::sin( 2.f * PI_CONST * (x_init[0] - t_now ) ) ;
		}
		else {
			f[i] = 0.f;
		}
	}
}
		

int main(int argc, char** argv) {
	// math (constants)
	const float PI_CONST { std::acos(-1.f) };

	// computation constants
	constexpr const float Deltat {  0.001f };

	// Calculate the total number of iterations, related to time interval T
	constexpr const float timeT { 4.f };
	const int iterations { static_cast<int>(timeT/ Deltat ) } ; 
	// sanity check
	std::cout << " iterations to do : " << iterations << std::endl;

	constexpr const int WIDTH  { 500 };

	dim3 L_x(WIDTH);
	dim3 M_x(2);
	
	// physics
	//  on the host, set up the spatial manifold, Euclidean R1

	constexpr std::array<int,1> LdS { WIDTH  };
	constexpr std::array<float,1> ldS { 5.f };
	
	Grid1d grid1d( LdS, ldS);
	
	// sanity check
	std::cout << " This is hd[0], i.e. Delta x : " << grid1d.hd[0] << std::endl;
	
	//  on device, set up the spatial manifold, Euclidean R1
	dim3 dev_L1 { static_cast<unsigned int>(WIDTH) };
	dev_Grid1d dev_grid1d( dev_L1 );
	
	// initial conditions (on the host)
	std::array<int,1> ix_in { 0 }; // i index variable for initial values
	std::array<float,1> x_init{  0.f  }; // x variable for initial values
	for (int i=0; i<(grid1d.Ld[0]); ++i) {
		ix_in[0] = i;
		x_init = grid1d.gridpt_to_space( ix_in ) ;
		if ((x_init[0] >= 0.f) && ( x_init[0] <= 1.f )) {
			grid1d.f[i] = std::sin( 2.f * PI_CONST * x_init[0] ) ;
		}
		else {
			grid1d.f[i] = 0.f;
		}
	}
	
	// copy initial conditions (to the device)
	HANDLE_ERROR(
		cudaMemcpy( dev_grid1d.dev_f, grid1d.f, grid1d.NFLAT() * sizeof(float),
			cudaMemcpyHostToDevice)
	);

	// computational constants
	// Courant number, or CFL (Courant-Friedrichs-Lewy) number, denoted sigma
	float sigmaconst[1] = { Deltat/grid1d.hd[0] };
	// sanity check
	std::cout << " This is Delta t    : " << Deltat        << std::endl;
	std::cout << " This is sigmaconst : " << sigmaconst[0] << std::endl;
	std::cout << " sigmaconst denotes the so-called Courant number, \
					i.e. CFL (Courant-Friedrichs-Lewy) number " << std::endl;
	HANDLE_ERROR(
		cudaMemcpyToSymbol(
			dev_sigmaconst, sigmaconst, sizeof(float)*1, 0, cudaMemcpyHostToDevice)
	);
	// on device, sanity check
	sanity_checks_constants<<<1,1>>>();
	
	for (auto i = 0; i<iterations; ++i) {
//	for (auto i = 0; i<3000; ++i) {
		kernelLauncher( dev_grid1d.dev_f, dev_grid1d.dev_foverline, dev_grid1d.dev_f_out,
			L_x, M_x);
	}

	// copy to the host the results from the device
	HANDLE_ERROR(
		cudaMemcpy( grid1d.f , dev_grid1d.dev_f, grid1d.NFLAT() * sizeof(float),
			cudaMemcpyDeviceToHost)
	);

	std::vector<float> f_ref ;
//	traveling_sine( f_ref, grid1d, timeT ) ;

	FILE *outfile = fopen("results.csv", "w");
	for (int i = 1; i < grid1d.NFLAT(); ++i) {
		ix_in[0] = i;
		x_init = grid1d.gridpt_to_space( ix_in ); 
//		fprintf(outfile, "%d,%f,%f,%f\n", i, x_init[0], grid1d.f[i], f_ref[i] );
		fprintf(outfile, "%d,%f,%f\n", i, x_init[0], grid1d.f[i]  );
	}
	fclose(outfile);
	
	
	// 
	

	HANDLE_ERROR(
		cudaFree( dev_grid1d.dev_f )
	);
	
	HANDLE_ERROR(
		cudaFree( dev_grid1d.dev_foverline )
	);

	HANDLE_ERROR(
		cudaFree( dev_grid1d.dev_f_out )
	);

	
	
}
