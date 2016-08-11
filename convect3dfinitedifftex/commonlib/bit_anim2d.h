/* bit_anim2d.h
 * 2-dim. GPU bit animation 
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160722
 */
#ifndef __BIT_ANIM2D_H__
#define __BIT_ANIM2D_H__

#define GL_GLEXT_PROTOTYPES // needed for identifier glGenBuffer, glBindBuffer, glBufferData, glDeleteBuffers

#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "errors.h"

#include <stdlib.h>
// #include <cstdlib> // std::exit

extern int iterationCount  ;

struct GPUAnim2dBit {
	GLuint pixbufferObj ; // OpenGL pixel buffer object

	cudaGraphicsResource *cuda_pixbufferObj_resource;
 
	int width, height;
 
	GPUAnim2dBit( int w, int h ) {
		width  = w;
		height = h;

		pixbufferObj = 0 ;
	}
/*
	~GPUAnim2dBit() {
		exitfunc();
	}
*/

	void initGLUT(int *argc, char **argv) {
		glutInit(argc, argv);
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
		glutInitWindowSize(width, height);
		glutCreateWindow("Mass Density. Vis.");

	}
	
	void initPixelBuffer() {
		glGenBuffers(1, &pixbufferObj );
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pixbufferObj);
		glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, 
			GL_DYNAMIC_DRAW_ARB);

		HANDLE_ERROR(
			cudaGraphicsGLRegisterBuffer( &cuda_pixbufferObj_resource, pixbufferObj, 
			cudaGraphicsMapFlagsNone) );
	}
	
	void exitfunc() {
		if (pixbufferObj) {
		HANDLE_ERROR( 
			cudaGraphicsUnregisterResource( cuda_pixbufferObj_resource) );

		glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 ) ;

		glDeleteBuffers(1, &pixbufferObj);
	}
	}
};	

// interactions

void keyboard_func( unsigned char key, int x, int y) ; 
	
void mouse_func( int button, int state, int x, int y ) ;

// make* functions make functions to pass into OpenGL (note OpenGL is inherently a C API)

void make_display_func(int width, int height) ;



__global__ void float_to_color3d( uchar4* optr, const float* outSrc) ;


#endif // # __BIT_ANIM2D_H__
