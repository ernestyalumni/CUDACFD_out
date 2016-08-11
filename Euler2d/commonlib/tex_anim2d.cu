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
