/* main.cpp
 * 2-dim. Euler Eq. by finite difference with shared memory
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160802
 */
#include <functional>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>


#include "./physlib/dev_R2grid.h"
#include "./physlib/R2grid.h"
#include "./physlib/dev_phys_param.h"
#include "./physlib/Euler2d.h"
#include "./physlib/convect.h"

#include "./commonlib/sharedmem.h"
#include "./commonlib/errors.h"
#include "./commonlib/tex_anim2d.h"
#include "./commonlib/finitediff.h"

#include "math.h" // CUDA C/C++ math.h

#define GL_GLEXT_PROTOTYPES // needed for identifier glGenBuffer, glBindBuffer, glBufferData, glDeleteBuffers

#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h> // or #include "cuda_gl_interop.h"

const float Deltat[1] { 0.0000015f };

// physics
const int WIDTH  { 640 } ;
const int HEIGHT { 640 } ;

dim3 dev_L2 { static_cast<unsigned int>(WIDTH), 
				static_cast<unsigned int>(HEIGHT) };

dev_Grid2d dev_grid2d( dev_L2 );		

// graphics + physics

const dim3 M_i { 2 , 2  };

const int iters_per_render { 10 };

GPUAnim2dTex bitmap( WIDTH, HEIGHT );
GPUAnim2dTex* testGPUAnim2dTex = &bitmap; 


void make_render(dim3 Ld_in, int iters_per_render_in, GPUAnim2dTex* texmap  ) {
	uchar4 *d_out = 0;
	cudaGraphicsMapResources(1, &texmap->cuda_pixbufferObj_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,
		texmap->cuda_pixbufferObj_resource);

	for (int i = 0; i < iters_per_render_in; ++i) {
//		kernelLauncher(d_out, dev_grid2d.dev_rho, dev_grid2d.dev_u, Ld_in, M_i );
//		kernelLauncher2(d_out, dev_grid2d.dev_rho, dev_grid2d.dev_u, Ld_in, M_i );
		EulerLauncher2(d_out, dev_grid2d.dev_rho, 
				dev_grid2d.dev_p, dev_grid2d.dev_u, dev_grid2d.dev_E, Ld_in, M_i) ;

				cudaDeviceSynchronize();
	}

	cudaGraphicsUnmapResources(1, &texmap->cuda_pixbufferObj_resource, 0);
	
	char title[128];
	sprintf(title, "Mass density Visualizer - Iterations=%4d, ", 
				   iterationCount );
	glutSetWindowTitle(title);

	cudaDeviceSynchronize();
}	



std::function<void()> render = std::bind( make_render, dev_L2, iters_per_render, testGPUAnim2dTex);	

std::function<void()> draw_texture = std::bind( make_draw_texture, WIDTH, HEIGHT) ;

void display() {
	render();
	draw_texture();
	glutSwapBuffers();
}



int main(int argc, char** argv) {
	// physics
	constexpr std::array<int,2> LdS {WIDTH, HEIGHT };
	constexpr std::array<float,2> ldS {1.f, 1.f };
	
	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_Deltat, Deltat, sizeof(float)*1,0,cudaMemcpyHostToDevice) );

	const float gas_params[3] { 
								 1.4f,
								 1.f, 1.f } ; // \gamma  
										// heat capacity for constant volume, per volume
										// M


	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_gas_params, gas_params, sizeof(float)*3,0,cudaMemcpyHostToDevice) );
	
	const int Ld_to_const[2] { LdS[0], LdS[1] } ;
	
	HANDLE_ERROR(
		cudaMemcpyToSymbol( dev_Ld, Ld_to_const, sizeof(int)*2,0,cudaMemcpyHostToDevice) );
	
	Grid2d grid2d( LdS, ldS);
	
	const float hds[2] { grid2d.hd[0], grid2d.hd[1] } ;
	
	// sanity check
	std::cout << " hds : .x : " << hds[0] << " .y : " << hds[1] << std::endl;
	
	//	set1DerivativeParameters(hds);
	set2DerivativeParameters(hds);
//	set3DerivativeParameters(hds);
//	set4DerivativeParameters(hds);
	
	// initial conditions	
	constexpr float rho_0 { 0.956 };
	
	constexpr float tau_0 { 0.001f };
	
	for (int j=0; j<(grid2d.Ld[1]); ++j) {
		for (int i=0;i<(grid2d.Ld[0]); ++i) {
			grid2d.u[ grid2d.flatten(i,j) ].x = 25.0;  // meters/second
			grid2d.u[ grid2d.flatten(i,j) ].y = 25.0;  // meters/second

		}
	}

	std::array<int,2> ix_in { 0, 0 };
	std::array<float,2> b_0 { 0.25f*grid2d.ld[0], 0.25f*grid2d.ld[1]  };

	for (int j=0; j<(grid2d.Ld[1]); ++j) {
		for (int i=0; i<(grid2d.Ld[0]); ++i) {
			ix_in[0] = i;
			ix_in[1] = j;
			grid2d.rho[ grid2d.flatten(i,j) ] = 
				gaussian2d( rho_0, 0.025, b_0,grid2d.gridpt_to_space(ix_in));

		}
	}

	for (int j=0; j<(grid2d.Ld[1]); ++j) {
		for (int i=0; i<(grid2d.Ld[0]); ++i) {
			grid2d.p[grid2d.flatten(i,j)].x = grid2d.rho[grid2d.flatten(i,j)]*
										grid2d.u[grid2d.flatten(i,j)].x ;
			grid2d.p[grid2d.flatten(i,j)].y = grid2d.rho[grid2d.flatten(i,j)]*
										grid2d.u[grid2d.flatten(i,j)].y ;
		}
	}

	int k { 0 } ;
	for (int j=0; j<(grid2d.Ld[1]); ++j) {
		for (int i=0; i<(grid2d.Ld[0]); ++i) {
			k = grid2d.flatten(i,j);
			grid2d.E[k] = grid2d.rho[k]*
				(tau_0*gas_params[1]/gas_params[2] + 
					0.5f*(grid2d.u[k].x*grid2d.u[k].x+grid2d.u[k].y*grid2d.u[k].y) );
		}
	}
	
	
	HANDLE_ERROR( 
		cudaMemcpy( dev_grid2d.dev_rho , grid2d.rho, grid2d.NFLAT()*sizeof(float), 
			cudaMemcpyHostToDevice)
	);

	HANDLE_ERROR( 
		cudaMemcpy( dev_grid2d.dev_u , grid2d.u, grid2d.NFLAT()*sizeof(float2), 
			cudaMemcpyHostToDevice)
	);
	
	HANDLE_ERROR( 
		cudaMemcpy( dev_grid2d.dev_p , grid2d.p, grid2d.NFLAT()*sizeof(float2), 
			cudaMemcpyHostToDevice)
	);

	HANDLE_ERROR( 
		cudaMemcpy( dev_grid2d.dev_E , grid2d.E, grid2d.NFLAT()*sizeof(float), 
			cudaMemcpyHostToDevice)
	);
	
	printInstructions();


	testGPUAnim2dTex->initGLUT(&argc, argv);

	glutKeyboardFunc(keyboard_func);
	glutMouseFunc(mouse_func);
	glutIdleFunc(idle);

	glutDisplayFunc(display);

	testGPUAnim2dTex->initPixelBuffer();

	glutMainLoop();

	HANDLE_ERROR(
		cudaFree( dev_grid2d.dev_rho ) );

	HANDLE_ERROR(
		cudaFree( dev_grid2d.dev_rho_out ) );

	HANDLE_ERROR(
		cudaFree( dev_grid2d.dev_u ) );

	HANDLE_ERROR(
		cudaFree( dev_grid2d.dev_u_out ) );

	HANDLE_ERROR(
		cudaFree( dev_grid2d.dev_p ) );

	HANDLE_ERROR(
		cudaFree( dev_grid2d.dev_p_out ) );

	HANDLE_ERROR(
		cudaFree( dev_grid2d.dev_E ) );

	HANDLE_ERROR(
		cudaFree( dev_grid2d.dev_E_out ) );

	//	HANDLE_ERROR(
	//	     cudaDeviceSynchronize()
	//	     );
	
	return 0;
} 

	
