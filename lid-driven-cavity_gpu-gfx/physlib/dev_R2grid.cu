/* dev_R2grid.cu
 * R3 under discretization (discretize functor) to a grid
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160728
 */
#include "dev_R2grid.h"


__host__ Dev_Grid2d::Dev_Grid2d( dim3 Ld_in) : Ld(Ld_in)
{
	checkCudaErrors(
		cudaMalloc((void**)&this->u, this->staggered_NFLAT()*sizeof(float) ) );

	checkCudaErrors(
		cudaMalloc((void**)&this->v, this->staggered_NFLAT()*sizeof(float) ) );

	checkCudaErrors(
		cudaMalloc((void**)&this->F, this->staggered_NFLAT()*sizeof(float) ) );

	checkCudaErrors(
		cudaMalloc((void**)&this->G, this->staggered_NFLAT()*sizeof(float) ) );

	int size_pres = ((Ld.x / 2 ) + 2 ) * (Ld.y + 2);

	checkCudaErrors(
		cudaMalloc((void**)&this->pres_red, size_pres*sizeof(float) ) );

	checkCudaErrors(
		cudaMalloc((void**)&this->pres_black, size_pres*sizeof(float) ) );

}

// destructor
/*
__host__ dev_Grid3d::~dev_Grid3d() {
	HANDLE_ERROR( 
		cudaFree( this->dev_rho ) );
	HANDLE_ERROR(
		cudaFree( this->dev_E ) );
	HANDLE_ERROR(
		cudaFree( this->dev_u ) );
	
}
* */

__host__ int Dev_Grid2d :: NFLAT() {
	return Ld.x*Ld.y;
}	

__host__ int Dev_Grid2d :: staggered_NFLAT() {
	return (Ld.x+2)*(Ld.y+2);
}	


	
