/* bit_anim1d.cu
 * 1-dim. GPU bitmap animation
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20161026
 * */
#include "bit_anim1d.h"

int iterationCount = 0;

// interactions

void keyboard_func( unsigned char key, int x, int y) {
	
	if (key==27) {
		exit(0);
	}
	glutPostRedisplay();
} 

void mouse_func( int button, int state, int x, int y) {
	glutPostRedisplay();
}

void idle() {
	++iterationCount;
	glutPostRedisplay();
}

void printInstructions() {
	printf("1 dim. bitmap animation \n"
	
			"Exit                        : Esc \n"
	);
}

// make* functions make functions to pass into OpenGL (note OpenGL is inherently a C APU)

void make_draw_bitmap(int w, int h) {
	glClearColor( 0.0, 0.0, 0.0, 1.0) ;
	glClear( GL_COLOR_BUFFER_BIT) ;
	glDrawPixels( w, h, 
					GL_RGBA, GL_UNSIGNED_BYTE,0);
	glutSwapBuffers();
}
