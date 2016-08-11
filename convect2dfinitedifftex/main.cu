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

#include "./physlib/convect.h"
#include "./physlib/R2grid.h"
#include "./physlib/dev_R2grid.h"

#include "math.h" // CUDA C/C++ math.h

#define GL_GLEXT_PROTOTYPES // needed for identifier glGenBuffer, glBindBuffer, glBufferData, glDeleteBuffers

#include <GL/glut.h>

#include <cuda_runtime.h>

#include <cuda_gl_interop.h>


const float Deltat[1] { 0.00001f } ; 

// physics
const int W { 680 } ;
const int H { 680 } ;

dim3 dev_L2 { static_cast<unsigned int>(W), 
				static_cast<unsigned int>(H)
				 };
				
dev_Grid2d dev_grid2d( dev_L2 );		


// graphics + physics

const dim3 M_i { 2, 2  };

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
	
	dim3 grids( (dev_L2.x+M_i.x-1)/M_i.x,(dev_L2.y+M_i.y-1)/M_i.y ) ;

	
	cudaEventRecord( databenchmarks.start, 0 );
	
	for (int i = 0 ; i < iters_per_render; ++i ) {
		convect_fd_naive_sh<<<grids,M_i>>>(
			dev_grid2d.dev_rho, dev_grid2d.dev_u ); 

//		convect_sh<<<grids,M_i, smSz>>>( dev_grid3d.dev_rho, dev_grid3d.dev_u ) ;

//		float_to_color3d<<<grids,M_i>>>(dev_out, dev_grid3d.dev_rho ) ;


//		float_to_char<<<grids,M_i>>>(dev_out, dev_grid3d.dev_rho ) ;

	}
	//float_to_color3d<<<grids,M_i>>>(dev_out, dev_grid3d.dev_rho ) ;

	float_to_char<<<grids,M_i>>>(dev_out, dev_grid2d.dev_rho ) ;


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
	constexpr std::array<int,2> LdS { W, H} ; 
	constexpr std::array<float,2> ldS { 1.f, 1.f };

	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_Deltat, Deltat, sizeof(float)*1,0,cudaMemcpyHostToDevice) );

	const int Ld_to_const[2] { LdS[0], LdS[1]  };
	
	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_Ld, Ld_to_const, sizeof(int)*2,0,cudaMemcpyHostToDevice) );



	Grid2d grid2d( LdS, ldS);

	const float hds[2] { grid2d.hd[0], grid2d.hd[1] };

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

	for (int j=0; j<(grid2d.Ld[1]); ++j) {
		for (int i=0;i<(grid2d.Ld[0]); ++i) {
			grid2d.u[ grid2d.flatten(i,j) ].x = 25.0;  // meters/second
			grid2d.u[ grid2d.flatten(i,j) ].y = 25.0;  // meters/second
		}
	}

	std::array<int,2> ix_in { 0, 0  };
	std::array<float,2> b_0 { 0.25f*grid2d.ld[0], 0.25f*grid2d.ld[1]   };
	
	for (int j=0; j<(grid2d.Ld[1]); ++j) {
		for (int i=0; i<(grid2d.Ld[0]); ++i) {
			ix_in[0] = i;
			ix_in[1] = j;
			grid2d.rho[ grid2d.flatten(i,j) ] = 
					gaussian2d( rho_0, 0.05, b_0,grid2d.gridpt_to_space(ix_in));

		}
	}
	


	HANDLE_ERROR(
		cudaMemcpy( dev_grid2d.dev_rho, grid2d.rho, grid2d.NFLAT()*sizeof(float), cudaMemcpyHostToDevice)
		);

	HANDLE_ERROR(
		cudaMemcpy( dev_grid2d.dev_u, grid2d.u, grid2d.NFLAT()*sizeof(float2), cudaMemcpyHostToDevice)
		);


	glutMainLoop();

	HANDLE_ERROR(
		cudaFree( dev_grid2d.dev_rho )  );

	HANDLE_ERROR(
		cudaFree( dev_grid2d.dev_u ) );
		
	texptr->exitfunc(); 
}
	
