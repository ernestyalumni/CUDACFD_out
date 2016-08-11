/*
 * main.cu
 * 3-dimensional convection with time-independent velocty vector field using
 * CUDA C/C++ implementing finite difference
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160723
*/
#include <functional>

#include "./commonlib/tex_anim2d.h"
#include "./commonlib/errors.h"
#include "./commonlib/finitediff.h"
#include "./commonlib/sharedmem.h"

#include "./physlib/convect.h"
#include "./physlib/R3grid.h"
#include "./physlib/dev_R3grid.h"

#include "math.h" // CUDA C/C++ math.h

#define GL_GLEXT_PROTOTYPES // needed for identifier glGenBuffer, glBindBuffer, glBufferData, glDeleteBuffers

#include <GL/glut.h>

#include <cuda_runtime.h>

#include <cuda_gl_interop.h>


const float Deltat[1] { 0.00001f } ; 

// physics
const int W { 680 } ;
const int H { 680 } ;
const int DEPTH { 320 } ;

dim3 dev_L3 { static_cast<unsigned int>(W), 
				static_cast<unsigned int>(H),
				static_cast<unsigned int>(DEPTH) };
				
dev_Grid3d dev_grid3d( dev_L3 );		


// graphics + physics

const dim3 M_i { 2, 2, 2 };

const int iters_per_render { 5 } ;


GPUAnim2dTex animtexmap( W, H );
GPUAnim2dTex* texptr = &animtexmap;


// struct DataBlock for recording, benchmarking events, iterations
struct DataBlock {
	cudaEvent_t  start, stop;
	float        totalTime;
	float        frames; 
};

DataBlock databenchmarks ; 



void make_render(int iters_per_render ) {
	uchar4* dev_out=0; 
	
	HANDLE_ERROR(
		cudaGraphicsMapResources(1, &(texptr->cuda_pixbufferObj_resource), 0 ) );
	
	HANDLE_ERROR(
		cudaGraphicsResourceGetMappedPointer((void **)&dev_out, NULL, 
			texptr->cuda_pixbufferObj_resource ) 
	);
	
	dim3 grids( (dev_L3.x+M_i.x-1)/M_i.x,(dev_L3.y+M_i.y-1)/M_i.y,(dev_L3.z+M_i.z-1)/M_i.z) ;
	
/*	
	constexpr int radius { 2 };
	const size_t smSz = (M_i.x + 2*radius)*(M_i.y + 2*radius)*(M_i.z+2*radius)*sizeof(float) ;
	*/
	
	cudaEventRecord( databenchmarks.start, 0 );
	
	for (int i = 0 ; i < iters_per_render; ++i ) {
		convect_fd_naive_sh<<<grids,M_i>>>(
			dev_grid3d.dev_rho, dev_grid3d.dev_u ); 

//		convect_sh<<<grids,M_i, smSz>>>( dev_grid3d.dev_rho, dev_grid3d.dev_u ) ;

//		float_to_color3d<<<grids,M_i>>>(dev_out, dev_grid3d.dev_rho ) ;


//		float_to_char<<<grids,M_i>>>(dev_out, dev_grid3d.dev_rho ) ;

	}
	//float_to_color3d<<<grids,M_i>>>(dev_out, dev_grid3d.dev_rho ) ;

	float_to_char<<<grids,M_i>>>(dev_out, dev_grid3d.dev_rho ) ;


	// Recording time for rough benchmarking, only
	cudaEventRecord( databenchmarks.stop, 0 );
	cudaEventSynchronize( databenchmarks.stop );

	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, databenchmarks.start, databenchmarks.stop );

	databenchmarks.totalTime += elapsedTime;
	++databenchmarks.frames;

//	printf("Iteration complete : ticks %d \n ", ticks );
	printf("Average Time per frame: %3.1f ms \n", databenchmarks.totalTime/databenchmarks.frames );
	// END of Recording time for rough benchmarking, only, END


	HANDLE_ERROR(
		cudaGraphicsUnmapResources( 1, &texptr->cuda_pixbufferObj_resource, 0 ));


	char title[128];
	sprintf(title, "mass density Visualizer - Iterations=%4d, ", iterationCount );
	
	glutSetWindowTitle(title);
}; 

std::function<void()> render = std::bind( make_render, iters_per_render ) ;



std::function<void()> draw_texture = std::bind( make_draw_texture, W,H) ;

void display() {
	render();
	draw_texture()   ;


	glutSwapBuffers();
}




int main(int argc, char** argv) {
	// physics
	constexpr float rho_0 { 0.956 };
	constexpr std::array<int,3> LdS { W, H, DEPTH} ; 
	constexpr std::array<float,3> ldS { 1.f, 1.f, 1.f };

	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_Deltat, Deltat, sizeof(float)*1,0,cudaMemcpyHostToDevice) );

	const int Ld_to_const[3] { LdS[0], LdS[1], LdS[2] };
	
	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_Ld, Ld_to_const, sizeof(int)*3,0,cudaMemcpyHostToDevice) );


	// set radius for shared memory "tiling", i.e. for the "halo" cells
	const int radius[1] { 2 };
	HANDLE_ERROR(
		cudaMemcpyToSymbol( sharedmem::RAD, radius, sizeof(int)*1,0,cudaMemcpyHostToDevice) );


	Grid3d grid3d( LdS, ldS);

	const float hds[3] { grid3d.hd[0], grid3d.hd[1], grid3d.hd[2] };

	set2DerivativeParameters(hds );

	// data benchmarking, cuda timing of events
	databenchmarks.totalTime = 0;
	databenchmarks.frames = 0 ;

	cudaEventCreate( &databenchmarks.start );
	cudaEventCreate( &databenchmarks.stop );
	


	// graphics setup
	
	texptr->initGLUT(&argc,argv) ;
	 
	glutKeyboardFunc( keyboard_func );
	glutMouseFunc( mouse_func );
	glutIdleFunc( idle );
	glutDisplayFunc( display) ;
	texptr->initPixelBuffer();
		

	// initial conditions

	for (int k=0; k<(grid3d.Ld[2]); ++k) {
		for (int j=0; j<(grid3d.Ld[1]); ++j) {
			for (int i=0;i<(grid3d.Ld[0]); ++i) {

				grid3d.u[ grid3d.flatten(i,j,k) ].x = 25.0;  // meters/second
				grid3d.u[ grid3d.flatten(i,j,k) ].y = 25.0;  // meters/second
				grid3d.u[ grid3d.flatten(i,j,k) ].z = 12.0;  // meters/second

			}
		}
	}

	std::array<int,3> ix_in { 0, 0, 0 };
	std::array<float,3> b_0 { 0.25f*grid3d.ld[0], 0.25f*grid3d.ld[1], 0.5f*grid3d.ld[2]  };
	
	for (int k=0; k<(grid3d.Ld[2]); ++k) {
		for (int j=0; j<(grid3d.Ld[1]); ++j) {
			for (int i=0; i<(grid3d.Ld[0]); ++i) {
				ix_in[0] = i;
				ix_in[1] = j;
				ix_in[2] = k;
				grid3d.rho[ grid3d.flatten(i,j,k) ] = 
						gaussian3d( rho_0, 0.05, b_0,grid3d.gridpt_to_space(ix_in));

			}
		}
	}


	HANDLE_ERROR(
		cudaMemcpy( dev_grid3d.dev_rho, grid3d.rho, grid3d.NFLAT()*sizeof(float), cudaMemcpyHostToDevice)
		);

	HANDLE_ERROR(
		cudaMemcpy( dev_grid3d.dev_u, grid3d.u, grid3d.NFLAT()*sizeof(float3), cudaMemcpyHostToDevice)
		);


	glutMainLoop();

	HANDLE_ERROR(
		cudaFree( dev_grid3d.dev_rho )  );

	HANDLE_ERROR(
		cudaFree( dev_grid3d.dev_u ) );
		
	texptr->exitfunc(); 
}
	
