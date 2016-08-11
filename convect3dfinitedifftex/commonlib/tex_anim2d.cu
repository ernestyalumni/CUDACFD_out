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
		exit(0) ;
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
	printf("3 dim. texture animation \n");

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


__device__ unsigned char clip(const float val ) {
	// Set limits, range
	const float scale = 0.999f ;
	const float newval = val/scale;

	int n = 256 * newval ; 
	if (n==256) {n=256; }
	return n > 255 ? 255 : (n < 0 ? 0 : n); 
}

__global__ void float_to_char( uchar4* dev_out, const float* outSrc) {
	// map from threadIdx/BlockIdx to pixel position
	int k_x = threadIdx.x + blockIdx.x*blockDim.x ;
	int k_y = threadIdx.y + blockIdx.y*blockDim.y ;
	
	// choose at which z coordinate to make the slice in x-y plane
	int zcoordslice = blockDim.z*gridDim.z/2*1; 

	int offset = k_x + k_y*blockDim.x*gridDim.x ;
	int fulloffset = offset + zcoordslice*blockDim.x*gridDim.x*blockDim.y*gridDim.y ;

	dev_out[offset].x = 0;
	dev_out[offset].y = 0;
	dev_out[offset].z = 0;
	dev_out[offset].w = 255;
	


	float value = outSrc[fulloffset];

	const unsigned char intensity = clip(value) ;

	dev_out[offset].x = intensity ; // higher mass density -> more red

	dev_out[offset].z = 255 - intensity ; // lower mass density -> more blue
	
}


__global__ void float_to_color3d( uchar4* optr, const float* outSrc) {
	// map from threadIdx/BlockIdx to pixel position
	int k_x = threadIdx.x + blockIdx.x*blockDim.x ;
	int k_y = threadIdx.y + blockIdx.y*blockDim.y ;

	// choose at which z coordinate to make the slice in x-y plane
	int zcoordslice = blockDim.z*gridDim.z/2*1; 

	int offset = k_x + k_y*blockDim.x*gridDim.x ;
	int fulloffset = offset + zcoordslice*blockDim.x*gridDim.x*blockDim.y*gridDim.y ;
	float value = outSrc[fulloffset];
	
	// Be aware of the "hard-coded" (numerical) constants for 
	// maximum and minimum scalar values that'll be assigned white and black, respectively
	if (value < 0.0001 ) { value = 0; }
	else if (value > 1.0 ) { value = 1.; } 

	// convert to long rainbow RGB* 
	value = value/0.20;
	int valueint  = ((int) floorf( value )); // this is the integer part 
	int valuefrac = ((int) floorf(255*(value-valueint)) );
	
	switch( valueint )
	{
		case 0:	optr[offset].x = 255; optr[offset].y = valuefrac; optr[offset].z = 0; 
		optr[offset].w = 255; 
		break;
		case 1:	optr[offset].x = 255-valuefrac; optr[offset].y = 255; optr[offset].z = 0; 
		optr[offset].w = 255; 
		break;
		case 2:	optr[offset].x = 0; optr[offset].y = 255; optr[offset].z = valuefrac; 
		optr[offset].w = 255; 
		break;
		case 3:	optr[offset].x = 0; optr[offset].y = 255-valuefrac; optr[offset].z = 255; 
		optr[offset].w = 255; 
		break;
		case 4:	optr[offset].x = valuefrac; optr[offset].y = 0; optr[offset].z = 255; 
		optr[offset].w = 255; 
		break;
		case 5:	optr[offset].x = 255; optr[offset].y = 0; optr[offset].z = 255; 
		optr[offset].w = 255; 
		break;
	}
}



