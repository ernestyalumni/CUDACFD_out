/*
 * convect3dupwind_shared.cu
 * 3-dimensional convection with time-independent velocty vector field using
 * CUDA C/C++ implementing "Upwind" interpolation
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160704
*/

#include "./commonlib/gpu_anim.h"
#include "./commonlib/errors.h"
#include "./commonlib/gpu_bitmap.h"

#include "./physlib/R3grid.h"
#include "./physlib/dev_R3grid.h"

#include "math.h" // CUDA C/C++ math.h

#define N_x 1920 // total number of blocks in the grid in x-direction
#define N_y 1920 // total number of blocks in the grid in y-direction
#define N_z 32 // total number of blocks in the grid in z-direction

#define M_x 16 // number of threads per block in x-direction
#define M_y 16 // number of threads per block in y-direction
#define M_z 4 // number of threads per block in z-direction

#define DELTAt 1./(100000.)
#define RHO0 0.956 // 0.656
#define L_0X 1.0
#define L_0Y 1.0
#define L_0Z 1.0
#define DELTAx L_0X/((float) N_x)
#define DELTAy L_0Y/((float) N_y)
#define DELTAz L_0Z/((float) N_z)


__global__ void convect( float *rho, float *ux, float *uy, float *uz) {
	// Experimental 
	dev_Grid3d blockgrid3d {blockDim.x*gridDim.x, blockDim.y*gridDim.y, blockDim.z*gridDim.z};

	// map from threadIdx/blockIdx to x grid position
	int k_x = threadIdx.x + blockIdx.x * blockDim.x;
	int k_y = threadIdx.y + blockIdx.y * blockDim.y;
	int k_z = threadIdx.z + blockIdx.z * blockDim.z;
	int offset = k_x + k_y*blockDim.x*gridDim.x + k_z*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

	int left     = offset - 1;
	int right    = offset + 1;
	int down     = offset + N_x;  // on a bitmap, up down is "backwards"
	int up       = offset - N_x;
	int top      = offset + N_x*N_y;
	int bottom   = offset - N_x*N_y;

	if (k_x == 0) {
		left++;
	}
	if (k_x==(N_x-1)) {
		right--; }
	if (k_y == 0) { up += N_x; }
	if (k_y == (N_y - 1 )) { down -= N_x ; }
	if (k_z == 0) { bottom += N_x*N_y; }
	if (k_z == (N_z - 1) ) { top -= N_x*N_y ; }

	float flux_right;
	if (ux[offset]>0.) { flux_right = DELTAy*DELTAz*rho[offset]*ux[offset] ; }
	else { flux_right = DELTAy*DELTAz*rho[right]*ux[offset] ; }
	float flux_left;
	if (ux[left] > 0.) { flux_left = (-1.)*DELTAy*DELTAz*rho[left]*ux[left]; }
	else { flux_left = (-1.)*DELTAy*DELTAz*rho[offset]*ux[left] ; }
	
	float flux_up;
	if (uy[up] > 0.) { flux_up = (-1.)*DELTAx*DELTAz*rho[up]*uy[up] ; }
	else { flux_up = (-1.)*DELTAx*DELTAz*rho[offset]*uy[up] ; }
	
	float flux_down;
	if (uy[offset] >0.) { flux_down = DELTAx*DELTAz*rho[offset]*uy[offset]; }
	else { flux_down = DELTAx*DELTAz*rho[down]*uy[offset]; }
		
	float flux_top;
	if (uz[offset]>0.) { flux_top = DELTAx*DELTAy*rho[offset]*uz[offset] ; }
	else { flux_top = DELTAx*DELTAy*rho[top]*uz[offset]; }
	float flux_bottom;
	if (uz[bottom]>0. ) { flux_bottom = (-1.)*DELTAx*DELTAy*rho[bottom]*uz[bottom] ; }
	else { flux_bottom = (-1.)*DELTAx*DELTAy*rho[offset]*uz[bottom] ; }
		
	rho[offset] += (-1.)*(DELTAt/(DELTAx*DELTAy*DELTAz))*( 
					flux_right+flux_left+flux_up+flux_down +flux_top+flux_bottom);
}
	
float gaussian3d( float x, float y, float z, float A, float k, float x_0, float y_0, float z_0) 
{
	return A*exp(-k*((x-x_0)*(x-x_0)+(y-y_0)*(y-y_0) +(z-z_0)*(z-z_0))); 
}


// globals needed by the update routine
struct DataBlock {
	float         *dev_rho;
	float         *dev_ux;
	float         *dev_uy;
	float         *dev_uz;
	GPUAnimBitmap *bitmap;
	cudaEvent_t   start, stop;
	float         totalTime;
	float         frames;
};

__global__ void profile2d( uchar4* optr , const float* outSrc) {
	// map from threadIdx/BlockIdx to pixel position
	int k_x = threadIdx.x + blockIdx.x*blockDim.x ;
	int k_y = threadIdx.y + blockIdx.y*blockDim.y ;

	// choose at which z coordinate to make the slice in x-y plane
	int zcoordslice = blockDim.z*gridDim.z/2*1; 
	int offset = k_x + k_y*blockDim.x*gridDim.x ;
	int fulloffset = offset + zcoordslice*blockDim.x*gridDim.x*blockDim.y*gridDim.y ;
	float value = outSrc[fulloffset];
	
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

void anim_gpu(uchar4* outputBitmap, DataBlock *d, int ticks) {
	cudaEventRecord( d-> start,0 );
	
	dim3 grids((N_x+M_x-1)/M_x,(N_y+M_x-1)/M_y,(N_z+M_z-1)/M_z);
	dim3 threads(M_x,M_y,M_z);
//	/* change the 1000 time steps per frame manually 
	for (int i = 0; i < 2; ++i ) {
		convect<<<grids,threads>>>( d->dev_rho, d->dev_ux, d->dev_uy, d->dev_uz);
	}
	
	float_to_color3d<<<grids,threads>>>( outputBitmap, d->dev_rho);
	
	// Recording time for rough benchmarking, only
	cudaEventRecord( d-> stop, 0);
	cudaEventSynchronize(d-> stop);
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, d->start, d->stop);
	
	d->totalTime += elapsedTime;
	++d->frames;
	printf("Average Time per frame:  %3.1f ms\n",d->totalTime/d->frames );
	// END of Recording time for rough benchmarking, only, END
}

int main( void ) {
	// sanity check
	Grid3d grid3dsanity { N_x,N_y,N_z };
	std::cout << " These are the default h_i values : " << grid3dsanity.h_is[0] << " " 
		<< grid3dsanity.h_is[1] << " " << grid3dsanity.h_is[2] << std::endl;

	Grid3d grid3d { N_x,N_y,N_z, DELTAx, DELTAy, DELTAz };
	
	DataBlock data;

	GPUAnimBitmap bitmap( grid3d.N_is[0], grid3d.N_is[1], &data );

	data.bitmap = &bitmap;
	data.totalTime = 0;
	data.frames = 0;
	// END of GPU animation setup END

	cudaEventCreate( &data.start );
	cudaEventCreate( &data.stop  );
	
	float* rho = new float[grid3d.NFLAT()];
	float* ux  = new float[grid3d.NFLAT()];  
	float* uy  = new float[grid3d.NFLAT()];  
	float* uz  = new float[grid3d.NFLAT()];  


	HANDLE_ERROR( 
		cudaMalloc((void**)&data.dev_rho,grid3d.NFLAT()*sizeof(float))
		);
	HANDLE_ERROR(
		cudaMalloc((void**)&data.dev_ux,grid3d.NFLAT()*sizeof(float))
			);
	HANDLE_ERROR(
		cudaMalloc((void**)&data.dev_uy,grid3d.NFLAT()*sizeof(float))
		);
	HANDLE_ERROR(
		cudaMalloc((void**)&data.dev_uz,grid3d.NFLAT()*sizeof(float))
		);

	// initial conditions

	for (int k=0; k<(grid3d.N_is[2]); ++k) {
		for (int j=0; j<(grid3d.N_is[1]); ++j) {
			for (int i=0;i<(grid3d.N_is[0]); ++i) {
				ux[ grid3d.flatten(i,j,k)] = 20.0; // meters/second

				uy[ grid3d.flatten(i,j,k) ] = 20.0; // meters/second
				uz[ grid3d.flatten(i,j,k) ] = 16.0 ;
			}
		}
	}

	for (int k=0; k<(grid3d.N_is[2]); ++k) {
		for (int j=0; j<(grid3d.N_is[1]); ++j) {
			for (int i=0; i<(grid3d.N_is[0]); ++i) {
				rho[ grid3d.flatten(i,j,k) ] = 
					gaussian3d( ((float) i)*grid3d.h_is[0],
								((float) j)*grid3d.h_is[1], ((float) k)*grid3d.h_is[2],
									RHO0, 1./sqrt(0.0001),0.25,0.25, 0.5);
			}
		}
	}
	
	HANDLE_ERROR(
		cudaMemcpy( data.dev_rho, rho, grid3d.NFLAT()*sizeof(float), cudaMemcpyHostToDevice)
		);
	HANDLE_ERROR(
		cudaMemcpy( data.dev_ux, ux, grid3d.NFLAT()*sizeof(float), cudaMemcpyHostToDevice)
		);
	HANDLE_ERROR(
		cudaMemcpy( data.dev_uy, uy, grid3d.NFLAT()*sizeof(float), cudaMemcpyHostToDevice)
		);
	HANDLE_ERROR(
		cudaMemcpy( data.dev_uz, uz, grid3d.NFLAT()*sizeof(float), cudaMemcpyHostToDevice)
		);
	
	delete[] rho;
	delete[] ux;
	delete[] uy;
	delete[] uz;
	
	bitmap.anim_and_exit((void (*)(uchar4*,void*,int))anim_gpu, NULL);
}
	
