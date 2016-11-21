/* tex_anim2d.cu
 * 2-dim. GPU texture animation 
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160720
 */
#include "tex_anim2d.h"
  
int iterationCount = 0 ;


// interactions

void keyboard_func( unsigned char key, int x, int y) {

	if (key==27) {
//		std::exit(0) ;
		exit(0);
	}
	glutPostRedisplay();
}
	
void mouse_func( int button, int state, int x, int y ) {
	glutPostRedisplay();
}

void idle() {
	++iterationCount;
	glutPostRedisplay();
}

void printInstructions() {
	printf("2 dim. texture animation \n"

			"Exit                           : Esc\n"
	
	);
}

// make* functions make functions to pass into OpenGL (note OpenGL is inherently a C API

void make_draw_texture(int w, int h) {
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, 
		GL_UNSIGNED_BYTE, NULL);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0,0);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0,h);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(w,h);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(w,0);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}	

// from physical scalar values to color intensities on an OpenGL bitmap
__global__ void floatux_to_char( uchar4* dev_out, cudaSurfaceObject_t uSurf, 
									const int L_x, const int L_y) {
	const int k_x = threadIdx.x + blockDim.x * blockIdx.x ; 
	const int k_y = threadIdx.y + blockDim.y * blockIdx.y ; 
	
	const int k = k_x + k_y * blockDim.x *gridDim.x ; 
	if ((k_x >= L_x ) || (k_y >= L_y)) {
		return ; }
	
	dev_out[k].x = 0;
	dev_out[k].y = 0;
	dev_out[k].z = 0;
	dev_out[k].w = 255;
	 
	float2 tempu; 
	surf2Dread(&tempu, uSurf, k_x * 8, k_y );

	// clipping part
	const float scale = 2.f ;
	const float newval = tempu.x / scale; 
	 
	int n = 256 * newval ; 
	n = max( min( n, 255) , 0 ) ; 
	// END of clipping part
	 
	const unsigned char intensity = n ; 
	dev_out[k].x = intensity ;  // higher magnitude -> more red
	dev_out[k].z = 255 - intensity ; // lower magnitude -> more blue
}
