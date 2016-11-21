/* dev_R2grid.cu
 * R3 under discretization (discretize functor) to a grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 2016115
 * 
 * compilation tip: (compile separately)
 * nvcc -std=c++11 -c ./physlib/dev_R2grid.cu -o dev_R2grid.o
 * 
 */
#include "dev_R2grid.h"

//__constant__ int dev_Ld[2];

// constructor
__host__ dev_Grid2d::dev_Grid2d( dim3 Ld_in) : Ld(Ld_in)
{
	checkCudaErrors(
		cudaMalloc((void**)&this->dev_f, this->NFLAT()*sizeof(float)) );

	checkCudaErrors(
		cudaMalloc((void**)&this->dev_f_out, this->NFLAT()*sizeof(float)) );

	checkCudaErrors(
		cudaMalloc((void**)&this->dev_u, this->NFLAT()*sizeof(float2)) );

	checkCudaErrors(
		cudaMalloc((void**)&this->dev_u_out, this->NFLAT()*sizeof(float2)) );


	(this->channelDesc_f) = cudaCreateChannelDesc( 32, 0, 0, 0, 
													cudaChannelFormatKindFloat);

	// 8 bits * 4 bytes in float (sizeof(float)) = 32
/*	(this->channelDesc_f2) = cudaCreateChannelDesc( 32, 32, 0, 0, 
								cudaChannelFormatKindFloat);*/ // This gave a Segmentation Fault 
	(this->channelDesc_f2) = cudaCreateChannelDesc<float2>();
	

	checkCudaErrors(
		cudaMallocArray(&(this->cuArr_f), &(this->channelDesc_f), (this->Ld).x, (this->Ld).y, 
						cudaArraySurfaceLoadStore) ); 

	checkCudaErrors(
		cudaMallocArray(&(this->cuArr_f_out), &(this->channelDesc_f), (this->Ld).x, (this->Ld).y,
						cudaArraySurfaceLoadStore) ); 

	checkCudaErrors(
		cudaMallocArray(&(this->cuArr_u), &(this->channelDesc_f2), (this->Ld).x, (this->Ld).y,
						cudaArraySurfaceLoadStore) ); 

	checkCudaErrors(
		cudaMallocArray(&(this->cuArr_u_out), &(this->channelDesc_f2), (this->Ld).x, (this->Ld).y,
						cudaArraySurfaceLoadStore) ); 

}

// destructor

__host__ dev_Grid2d::~dev_Grid2d() {

	checkCudaErrors(
		cudaFree( this->dev_f ) );

	checkCudaErrors(
		cudaFree( this->dev_f_out ) );


	checkCudaErrors(
		cudaFree( this->dev_u ) );

	checkCudaErrors(
		cudaFree( this->dev_u_out ) );


	checkCudaErrors(
		cudaFreeArray( this->cuArr_f ));
		
	checkCudaErrors(
		cudaFreeArray( this->cuArr_f_out ));


	checkCudaErrors(
		cudaFreeArray( this->cuArr_u )); 

	checkCudaErrors(
		cudaFreeArray( this->cuArr_u_out )); 

}


__host__ int dev_Grid2d :: NFLAT() {
	return Ld.x*Ld.y;
}	

__global__ void d_BoundaryConditions( float2 hds, 
										cudaSurfaceObject_t uSurf , 
										const int L_x, const int L_y,
										const float h_val) {
	const int k_x = threadIdx.x + blockIdx.x * blockDim.x ;
	const int k_y = threadIdx.y + blockIdx.y * blockDim.y ;
	if ((k_x >= L_x) || (k_y >= L_y)) { 
		return ; }

	// real values on Euclidean space
	float2 xreal ; 
	float2 lreal ;
	xreal.x = hds.x * k_x; 
	xreal.y = hds.y * k_y; 
	lreal.x = hds.x * L_x; 
	lreal.y = hds.y * L_y; 
	
	// stencil values, or halo cells
	float2 tempu, b, bb;

	// wall boundary condition, y=0
	if (k_y == 0) {
		// no slip condition
		tempu.x = 0.f;
		tempu.y = 0.f;
		surf2Dwrite( tempu, uSurf, k_x * 8 , 0) ; }

	// symmetry plane boundary condition, y=H
	else if (k_y == (L_y - 1)) {
		surf2Dread(&b, uSurf, k_x * 8, L_y - 2) ;  
		surf2Dread(&bb, uSurf, k_x * 8, L_y - 3) ;  
		tempu.x = (4.f/3.f) * b.x - (1.f/3.f) * bb.x ;
		tempu.y = 0.f ;
		surf2Dwrite( tempu, uSurf, k_x * 8, L_y-1) ; }
	
	// inlet condition at x=0
	else if (k_x == 0) {
		tempu.y = 0.f ;
		
		if ( xreal.y <= (lreal.y -h_val )) {
			tempu.x = 0.f ; }
		else if (( xreal.y > (lreal.y -h_val)) && (xreal.y <= lreal.y)) {
			float tempux {0.f };
			tempux = 1.5f*( 2.f *(xreal.y-lreal.y+h_val)/h_val  
							-(xreal.y-lreal.y+h_val)*(xreal.y-lreal.y+h_val)/(h_val*h_val)) ; 
			tempu.x = tempux;
		}
		surf2Dwrite( tempu, uSurf, 0 , k_y ) ; }
			
}	
	



