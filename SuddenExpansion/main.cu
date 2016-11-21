/* main.cu
 * Sudden Expansion, in 2-dimensions
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161120
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
#include <functional>

#include <iostream> // std::cout
#include <fstream>  // std::ofstream
#include <cmath>    // std::sqrt

#include "./physlib/R2grid.h" // Grid2d
#include "./physlib/dev_R2grid.h" // dev_Grid2d
#include "./commonlib/surfObj2d.h" // SurfObj2d
#include "./commonlib/checkerror.h" // checkCudaErrors
#include "./commonlib/tex_anim2d.h" // GPUAnim2dTex, iterationCount, keyboard_func, mouse_func, idle, display

// discretization (parameters) <=> graphical (parameters)
const int L_X  { 640 } ;  // WIDTH
const int L_Y  { 640 } ;  // HEIGHT

// "real", physical parameters
constexpr const float l_X = 10.f; // length (in x-direction)
constexpr const float l_Y = 2.f ; // height (in y-direction)
constexpr const float h_val { l_Y * 0.75f } ; // inlet radius

// physics (on device); Euclidean (spatial) space
dim3 dev_L2 { static_cast<unsigned int>(L_X), 
				static_cast<unsigned int>(L_Y) };

dev_Grid2d dev_grid2d( dev_L2 );

// Create surface object to "bind" to corresponding cudaArray
SurfObj2d surf_rho( dev_grid2d.cuArr_f) ;  
SurfObj2d surf_rho_out( dev_grid2d.cuArr_f_out) ;  
SurfObj2d surf_u( dev_grid2d.cuArr_u) ;  
SurfObj2d surf_u_out( dev_grid2d.cuArr_u_out) ;  



// physics (on host); Euclidean (spatial) space
constexpr std::array<int,2> LdS { L_X, L_Y } ; 
constexpr std::array<float,2> ldS { l_X, l_Y };
	
Grid2d grid2d{ LdS, ldS };

const float2 hds { grid2d.hd[0], grid2d.hd[1] }; 

// dynamics (parameters)
const dim3 M_i { 4, 4 };

// graphics + dynamics (parameters)
const int iters_per_render { 10 } ;
// graphics
GPUAnim2dTex bitmap( L_X, L_Y );
GPUAnim2dTex* eleGPUAnim2dTex = &bitmap; 

void make_render(dim3 Ld_in, int iters_per_render_in, GPUAnim2dTex* texmap ) {
	uchar4 *d_out = 0;
	cudaGraphicsMapResources(1, &texmap->cuda_pixbufferObj_resource, 0) ;
	cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,
		texmap->cuda_pixbufferObj_resource);
		
	const dim3 N_i {  (Ld_in.x + M_i.x - 1)/M_i.x, (Ld_in.y + M_i.y - 1)/M_i.y  } ;
		
	for (int i =0; i < iters_per_render_in; ++i) {
		d_BoundaryConditions<<<N_i, M_i>>>( hds, surf_u.surfObj, Ld_in.x, Ld_in.y, h_val ) ;
		
		floatux_to_char<<<N_i,M_i>>>( d_out, surf_u.surfObj, Ld_in.x, Ld_in.y ) ;
		cudaDeviceSynchronize() ;
	}
	
	cudaGraphicsUnmapResources(1, &texmap->cuda_pixbufferObj_resource,0);
	
	char title[128];
	sprintf( title, "velocity (in x-direction) Visualizer - Iterations=%4d, ",
		iterationCount );
	glutSetWindowTitle(title);
	
	cudaDeviceSynchronize() ;
	
}

std::function<void()> render = std::bind( make_render, dev_L2, iters_per_render, eleGPUAnim2dTex) ;

std::function<void()> draw_texture = std::bind( make_draw_texture, L_X, L_Y );

void display() {
	render() ;
	draw_texture() ;
	glutSwapBuffers();
}


int main(int argc, char* argv[]) {

	// initial condition 
	std::array<int,2> ix_in {0,0};
	std::array<float,2> x_in {0.f, 0.f };
	float radius_pos { 0.f } ; 
	float tempux { 0.f } ;
	float tempuy { 0.f } ;
	
	for (auto j = 0; j < grid2d.Ld[1] ; ++j) { 
		for (auto i = 0; i < grid2d.Ld[0]; ++i ) {
			ix_in[0] = i;
			ix_in[1] = j;
			x_in = grid2d.gridpt_to_space( ix_in ) ; 
			grid2d.f[ grid2d.flatten(i,j) ] = 0.0f ;  // kg/m^3 
			grid2d.u[ grid2d.flatten(i,j) ].x = 0.1f ; 
			grid2d.u[ grid2d.flatten(i,j) ].y = 0.f ;
			radius_pos = (x_in[0]-l_X/2.f)*(x_in[0]-l_X/2.f)+(x_in[1]-l_Y/2.f)*(x_in[1]-l_Y/2.f) ; 
			radius_pos = std::sqrt( radius_pos ) ;
			if (radius_pos < l_Y/5.f ) {
				tempux = (l_Y/5.f - radius_pos)*(-1.f*( x_in[1]-l_Y/2.f ) ) ; 
				tempuy = (l_Y/5.f - radius_pos)*( x_in[0]-l_X/2.f ) ;
				grid2d.u[ grid2d.flatten(i,j)].x = tempux ; 
				grid2d.u[ grid2d.flatten(i,j)].y = tempuy ; 
				
			}
			if (x_in[0] == 0 ) {
				grid2d.f[ grid2d.flatten(i,j) ] = 1.f ;  // kg/m^3
			}	
		}
	}

	checkCudaErrors(
		cudaMemcpyToArray( dev_grid2d.cuArr_f,0,0,
			(grid2d.f).data(), sizeof(float)*grid2d.NFLAT(), cudaMemcpyHostToDevice) );

	checkCudaErrors(
		cudaMemcpyToArray( dev_grid2d.cuArr_u,0,0,
			(grid2d.u).data(), sizeof(float2)*grid2d.NFLAT(), cudaMemcpyHostToDevice) );





	eleGPUAnim2dTex->initGLUT(&argc, argv) ;
	
	glutKeyboardFunc(keyboard_func);
	glutMouseFunc(mouse_func);
	glutIdleFunc(idle);

	glutDisplayFunc(display); 
	
	eleGPUAnim2dTex->initPixelBuffer();
	
	glutMainLoop();


	return 0;
	
}
