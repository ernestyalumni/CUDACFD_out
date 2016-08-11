/*
 * gpu_anim.cu
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160716
 */
#include "gpu_anim.h"

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
