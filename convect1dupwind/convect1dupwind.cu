/*
 * convect1dupwind.cu
 * 1-dimensional convection with time-independent velocty vector field using
 * CUDA C/C++ on GPU implementing "Upwind" interpolation scheme	
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160618
*/

#include "commonlib/gpu_anim.h"
#include "cuda.h"
#include "math.h" // CUDA C/C++ math.h

#define Nthreads 400
#define DELTAt 1./(10.)
#define RHO0 0.656
#define L_0 1.0 
#define DELTAx L_0/((float) Nthreads)
#define DIMY 400

__global__ void convect( float* mavg, float* u ) {
	int k_x = threadIdx.x + blockIdx.x * blockDim.x ;
	
	int left     = k_x - 1;
	int right    = k_x + 1;
	
	// boundary conditions
	if (k_x == 0) {
		left++;
	}
	else if (k_x == (Nthreads - 1)) {
		right--;
	}	
	float flux_right;
	if (u[k_x] > 0. ) {
		flux_right = mavg[k_x]*u[k_x]; }
	else {
		flux_right = mavg[right]*u[k_x]; }
	float flux_left;
	if (u[left] > 0.) {
		flux_left = mavg[left]*u[left]; }
	else {
		flux_left = mavg[k_x]*u[left]; }
	mavg[k_x] += (-1.)*(DELTAt/ DELTAx)*( flux_right - flux_left);

}

float gaussian( float x, float A, float k, float x_0) 
{
	return A*exp(-k*(x-x_0)*(x-x_0)); 
}

struct DataBlock {
	float 				*dev_mavg;
	float 				*dev_u;
	GPUAnimBitmap 		*bitmap;
// The following are for time-keeping
	cudaEvent_t 		start, stop; 
	float 				totalTime;
	float 				frames;
};


__global__ void float_to_1dimplot( uchar4* optr, const float* outSrc) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x ;
	
	int ivalue = ((int) (outSrc[x]*((float) DIMY) ));
	
	// remember optr is a pointer to the buffer that OpenGL and CUDA SHARES
	for (int j = 0 ; j < DIMY ; ++j ) {
		int offset = x + j *gridDim.x * blockDim.x ;
		if (j < ivalue ) {
			optr[offset].x = 0 ;
			optr[offset].y = 255 ;
			optr[offset].z = 0;
			optr[offset].w = 255;
		} else {
			optr[offset].x = 255 ;
			optr[offset].y = 0;
			optr[offset].z = 0;
			optr[offset].w = 255;
		}
	}
}


void anim_gpu(uchar4* outputBitmap, DataBlock *d, int ticks) {
	cudaEventRecord( d->start, 0 ) ;
	
	/* change the 1000 time steps per frame manually */
	for (int i =0; i< 1000; ++i) {
		convect<<<Nthreads/20,20>>>( d->dev_mavg, d->dev_u);
	}
		
	float_to_1dimplot<<<Nthreads/20,20>>>(outputBitmap,d->dev_mavg);
	
	// Recording time for rough benchmarking only	
	cudaEventRecord( d->stop, 0) ;
	cudaEventSynchronize( d->stop ) ;
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, d->start, d->stop);
	
	d->totalTime += elapsedTime;
	++d->frames;
	printf( "Average Time per frame:  %3.1f ms\n", d->totalTime/d->frames );
// END of Recording time for rough benchmarking only, END
}

int main() {
	DataBlock data;
	GPUAnimBitmap bitmap( Nthreads, DIMY, &data);
	data.bitmap = &bitmap;
	data.totalTime = 0;
	data.frames = 0;
	// END of GPU animation setup END
	cudaEventCreate( &data.start );
	cudaEventCreate( &data.stop  );
	
	
	float mavg[Nthreads];
	float u[Nthreads];
	
	cudaMalloc((void**)&data.dev_mavg, Nthreads*sizeof(float));
	cudaMalloc((void**)&data.dev_u, Nthreads*sizeof(float));
	
	// fill in host memory for time-independent u 
	for(int j = 0; j<Nthreads; ++j) {
		u[j] = 1.0; // m/s
	}
	// fill in host memory for initial mavg, which I chose to be a gaussian here
	// for some reason nvcc doesn't like C++11 lambda features
	/* auto gaussian = [](float x, float A, float k, float x_0) { return A*exp(-k*(x-x_0)*(x-x_0)); }; 
	for (int j = 0; j <= Nthreads; ++j) {
		rho.push_back( gaussian( ((float) j)*DELTAx, RHO0, 1./sqrt(0.00001),0.25) );
	}
	* nvcc didn't like inline either
	* */
	for (int j = 0; j < Nthreads; ++j) {
		mavg[j] =  gaussian( ((float) j)*DELTAx, RHO0, 1./sqrt(0.0001),0.25); 
	}

	cudaMemcpy( data.dev_mavg, mavg, Nthreads*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy( data.dev_u, u, Nthreads*sizeof(float), cudaMemcpyHostToDevice);
	
//	delete [] rho;
//	delete [] u;
	
	bitmap.anim_and_exit( (void (*)(uchar4* , void*,int))anim_gpu, NULL);
}
