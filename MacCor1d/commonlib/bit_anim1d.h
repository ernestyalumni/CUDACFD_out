/* bit_anim1d.h
 * 1-dim. GPU bitmap animation
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161026
 * */
#ifndef __BIT_ANIM1D_H__
#define __BIT_ANIM1D_H__

#include <stdio.h>

#define GL_GLEXT_PROTOTYPES // needed for identifier glGenBuffer, glBindBuffer, glBufferData, glDeleteBuffers

#include <GL/glut.h> // GLuint

//#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "errors.h"

#include <stdlib.h> // exit

extern int iterationCount ;

struct GPUAnim1dBit {
	GLuint bufferObj ; // OpenGL buffer object

	cudaGraphicsResource *cuda_resource;
	
	int width, height; 
	
	GPUAnim1dBit( int w, int h) {
		width  = w;
		height = h;

	}

	~GPUAnim1dBit() {
		exitfunc();
	}
	
	void initGLUT(int *argc, char **argv) {
		// trick GLUT into thinking we're passing an argument; hence argc, argv as argument inputs
		glutInit(argc, argv);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
		glutInitWindowSize(width, height);
		
		glutCreateWindow("1-dim. bitmap"); // to draw our results; "bitmap" name is arbitrary
	}
	
	void initPixelBuffer() {
		// create pixel buffer object in OpenGL, store in our global variable GLuint bufferObj
		glGenBuffers(1, &bufferObj );
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
		glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0 , 
			GL_DYNAMIC_DRAW_ARB);
		
		HANDLE_ERROR(
			cudaGraphicsGLRegisterBuffer( &cuda_resource, bufferObj, cudaGraphicsMapFlagsNone) );
	}
	
	void exitfunc() {
		// clean up OpenGL and CUDA
		if(bufferObj) {
			HANDLE_ERROR(
				cudaGraphicsUnregisterResource( cuda_resource ) );

			glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			glDeleteBuffers(1, &bufferObj);
		}
	}
	
};

// interactions

void keyboard_func( unsigned char key, int x, int y) ;

void mouse_func( int button, int state, int x, int y);

void idle();

void printInstructions() ;

// make* functions make functions to pass into OpenGL (note OpenGL is inherently a C API)

void make_draw_bitmap(int w, int h);

#endif // # __BIT_ANIM1D_H__

		
		
